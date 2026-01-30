# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import shutil
import numpy as np
from ase.io import write
from ase.calculators.calculator import Calculator
from mace.calculators import MACECalculator as MACECalculatorOriginal

from mlps_finetuning.energy_ref import get_corrected_energy
from mlps_finetuning.utilities import RedirectOutput

# -------------------------------------------------------------------------------------
# MACE CALCULATOR
# -------------------------------------------------------------------------------------

class MACECalculator(MACECalculatorOriginal):

    def __init__(
        self,
        model_name: bool = None,
        logfile: str = None,
        **kwargs: dict,
    ):
        # Load pretrained model.
        if "models" not in kwargs and "model_paths" not in kwargs:
            kwargs["models"] = [get_pretrained_mace_model(
                return_raw_model=True,
                model_name=model_name,
                **kwargs,
            )]
        # Initialize.
        with RedirectOutput(logfile=logfile):
            super().__init__(**kwargs)
        # Counter to keep track of the number of single-point evaluations.
        self.counter = 0
        self.info = {}
    
    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        self.counter += 1

# -------------------------------------------------------------------------------------
# GET PRETRAINED MACE MODEL
# -------------------------------------------------------------------------------------

def get_pretrained_mace_model(
    model_name: str = None,
    return_raw_model: bool = False,
    **kwargs: dict,
):
    """
    Get pretrained MACE model.
    """
    from mace.calculators.foundations_models import mace_mp_names
    if model_name == "mace_anicc":
        from mace.calculators import mace_anicc
        return mace_anicc(return_raw_model=return_raw_model, **kwargs)
    elif model_name == "mace_off":
        from mace.calculators import mace_off
        return mace_off(return_raw_model=return_raw_model, **kwargs)
    elif model_name == "mace_omol":
        from mace.calculators import mace_omol
        return mace_omol(return_raw_model=return_raw_model, **kwargs)
    elif model_name in mace_mp_names:
        from mace.calculators import mace_mp
        return mace_mp(model=model_name, return_raw_model=return_raw_model, **kwargs)
    else:
        raise NameError("Wrong model_name.")

# -------------------------------------------------------------------------------------
# TRAIN MACE MODEL
# -------------------------------------------------------------------------------------

def train_MACE_model(
    label: str,
    directory: str,
    seed: int = 42,
    logfile: str = None,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    batch_size: int = 4,
    energies_ref: str = "average",
    multiheads_finetuning: bool = False,
    **kwargs: dict,
):
    """
    Train MACE model.
    """
    import sys
    import torch
    import warnings
    from mace.cli.run_train import main as mace_main
    # Suppress warnings.
    warnings.filterwarnings("ignore", module="torch.jit")
    # Training parameters.
    parameters = {
        "name": label,
        "seed": seed,
        "work_dir": directory,
        "max_num_epochs": epochs,
        "lr": learning_rate,
        "batch_size": batch_size,
        "E0s": energies_ref,
        "multiheads_finetuning": str(multiheads_finetuning),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    }
    argv = [""]
    for key, arg in parameters.items():
        if arg is True:
            argv.append(f"--{key}")
        elif arg not in (False, None):
            argv.append(f"--{key}={arg}")
    # Train the model.
    sys.argv = argv
    with RedirectOutput(logfile=logfile):
        mace_main()
    # Return checkpoint path.
    if "swa" in parameters:
        return os.path.join(directory, f"{label}_stagetwo.model")
    else:
        return os.path.join(directory, f"{label}.model")

# -------------------------------------------------------------------------------------
# FINETUNE MACE MODEL
# -------------------------------------------------------------------------------------

def finetune_MACE_model(
    atoms_list: list,
    calc: Calculator = None,
    directory: str = "finetuning",
    label: str = "model",
    energy_corr_dict: dict = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.0,
    seed: int = 42,
    atoms_tasks: list = None,
    logfile: str = None,
    clean_directory: bool = False,
    energies_ref: str = "average",
    multiheads_finetuning: bool = False,
    calc_kwargs: dict = {},
    **kwargs: dict,
):
    """
    Fine-tune MACE model from ase Atoms data.
    """
    # Remove old directory.
    if clean_directory is True and os.path.isdir(directory):
        shutil.rmtree(directory)
    # Start from the model in the calculator.
    if calc is not None and "checkpoint_path" in calc.info:
        kwargs["restart_latest"] = True
        checkpoint_path_new = os.path.join(directory, f"{label}.model")
        if checkpoint_path_new != calc.info["checkpoint_path"]:
            shutil.copyfile(calc.info["checkpoint_path"], checkpoint_path_new)
    # Prepare reference energies.
    energies_ref = prepare_energies_ref(
        energies_ref=energies_ref,
        atoms_list=atoms_list,
        directory=directory,
    )
    # Prepare train and test files.
    file_train_path, file_val_path, file_test_path = prepare_train_val_test_files(
        atoms_list=atoms_list,
        directory=directory,
        test_fraction=test_fraction,
        seed=seed,
        atoms_tasks=atoms_tasks,
        energy_corr_dict=energy_corr_dict,
    )
    # Run the fine-tuning.
    checkpoint_path_new = train_MACE_model(
        label=label,
        directory=directory,
        logfile=logfile,
        train_file=file_train_path,
        valid_file=file_val_path,
        test_file=file_test_path,
        energies_ref=energies_ref,
        multiheads_finetuning=multiheads_finetuning,
        **kwargs,
    )
    # Return calculator.
    calc = MACECalculator(model_paths=[checkpoint_path_new], **calc_kwargs)
    calc.info["checkpoint_path"] = checkpoint_path_new
    return calc

# -------------------------------------------------------------------------------------
# PREPARE TRAIN VAL TEST
# -------------------------------------------------------------------------------------

def prepare_train_val_test_files(
    atoms_list: list,
    directory: str,
    val_fraction: float = 0.1,
    test_fraction: float = 0.0,
    seed: int = 42,
    atoms_tasks: list = None,
    energy_corr_dict: dict = None,
) -> list:
    """
    Write the list of atoms into train, val (and test) text files.
    """
    # Get tasks and atoms tasks.
    tasks = ["train", "val", "test"]
    if atoms_tasks is None:
        train_fraction = 1. - val_fraction - test_fraction
        # Shuffle the list of atoms.
        if round(train_fraction) < 1.0:
            atoms_list = atoms_list[:]
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(atoms_list)
        n_data = len(atoms_list)
        aa = int(round(n_data * train_fraction))
        bb = int(round(n_data * (train_fraction + val_fraction)))
        atoms_tasks = [atoms_list[:aa], atoms_list[aa:bb], atoms_list[bb:]]
    # Write to the databases.
    file_path_list = []
    for task, atoms_list in zip(tasks, atoms_tasks):
        # Do not create an empty database.
        if len(atoms_list) == 0:
            file_path_list.append(None)
            continue
        # Create directory.
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{task}.xyz")
        file_path_list.append(file_path)
        # Prepare database.
        for atoms in atoms_list:
            # Apply energy correction.
            atoms.info["REF_energy"] = get_corrected_energy(
                atoms=atoms,
                energy_corr_dict=energy_corr_dict,
            )
            atoms.arrays["REF_forces"] = atoms.get_forces(apply_constraint=False)
        # Write xyz file.
        write(filename=file_path, images=atoms_list)
    # Return paths of xyz files.
    return file_path_list

# -------------------------------------------------------------------------------------
# PREPARE ENERGIES REF
# -------------------------------------------------------------------------------------

def prepare_energies_ref(
    energies_ref: str,
    atoms_list: list,
    directory: str,
):
    """
    Prepare reference energies for training MACE.
    """
    # Returns if energies_ref is a string.
    if isinstance(energies_ref, str):
        return energies_ref
    # Zero reference energies if energies_ref is None.
    if energies_ref is None:
        energies_ref = {
            int(key): 0. for atoms in atoms_list for key in atoms.get_atomic_numbers()
        }
    # Write reference energies to json file if energies_ref is a dictionary.
    if isinstance(energies_ref, dict):
        import json
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, "E0s.json")
        with open(filename, "w") as fileobj:
            json.dump(energies_ref, fileobj)
    # Return json file.
    return filename

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------