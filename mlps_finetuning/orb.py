# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import shutil
import yaml
import numpy as np
from ase.calculators.calculator import Calculator
from orb_models.forcefield import pretrained, ORB_PRETRAINED_MODELS
from orb_models.forcefield.calculator import ORBCalculator as ORBCalculatorOriginal

from mlps_finetuning.energy_ref import get_corrected_energy
from mlps_finetuning.utilities import RedirectOutput

# -------------------------------------------------------------------------------------
# ORB CALCULATOR
# -------------------------------------------------------------------------------------

class ORBCalculator(ORBCalculatorOriginal):

    def __init__(
        self,
        model: bool = None,
        model_name: bool = None,
        device: str = "cuda",
        logfile: str = None,
        **kwargs: dict,
    ):
        # Load pretraoned model.
        # TODO:
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
# TRAIN ORB MODEL
# -------------------------------------------------------------------------------------

def train_ORB_model(
    directory: str = "training",
    checkpoint_path: str = None,
    label: str = "model",
    logfile: str = None,
    **kwargs,
) -> str:
    """
    Train ORB model.
    """
    from dataclasses import dataclass
    from orbmodels.finetune import run as orb_run # TODO: check this!
    # Training arguments dataclass.
    @dataclass
    class TrainingArgs:
        random_seed: int = 42
        device_id: int = 0
        wandb: bool = True
        wandb_entity: str = "orbitalmaterials"
        dataset: str = "mp-traj"
        data_path: str = data_path # TODO:
        num_workers: int = 8
        batch_size: int = 8
        gradient_clip_val: float = 0.5
        max_epochs: int = 100
        save_every_x_epochs: int = 5
        num_steps: int = 1000
        checkpoint_path: str = checkpoint_path
        lr: float = 3e-4
        base_model: str = "orb_v3_conservative_inf_omat"
    # Training arguments container.
    args = TrainingArgs(**kwargs)
    # Train the model.
    with RedirectOutput(logfile=logfile):
        orb_run(args=args)
    # Return path of the new checkpoint.
    return os.path.join(...) # TODO:

# -------------------------------------------------------------------------------------
# PREPARE TRAIN VAL TEST DBS
# -------------------------------------------------------------------------------------

def prepare_train_val_test_dbs(
    atoms_list: list,
    directory: str,
    val_fraction: float = 0.1,
    test_fraction: float = 0.0,
    seed: int = 42,
    atoms_tasks: list = None,
    energy_corr_dict: dict = None,
) -> list:
    """
    Write the list of atoms into train, validation (and test) databases.
    """
    from ase.db import connect
    # Get tasks and atoms tasks.
    tasks = ["train", "val", "test"]
    if atoms_tasks is None:
        train_fraction = 1. - val_fraction - test_fraction
        # Shuffle the list of atoms.
        if train_fraction < 1.:
            atoms_list = atoms_list[:]
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(atoms_list)
        n_data = len(atoms_list)
        aa = int(n_data * train_fraction)
        bb = int(n_data * (train_fraction + val_fraction))
        atoms_tasks = [atoms_list[:aa], atoms_list[aa:bb], atoms_list[bb:]]
    # Write to the databases.
    db_path_list = []
    for task, atoms_list in zip(tasks, atoms_tasks):
        # Do not create an empty database.
        if len(atoms_list) == 0:
            db_path_list.append(None)
            continue
        # Create directory.
        os.makedirs(os.path.join(directory, task), exist_ok=True)
        db_path = os.path.join(directory, task, f"{task}.db")
        db_path_list.append(db_path)
        # Prepare database.
        with connect(db_path, append=False) as db_ase:
            natoms = []
            for atoms in atoms_list:
                # Apply energy correction.
                if energy_corr_dict is not None:
                    atoms.calc.results["energy"] = get_corrected_energy(
                        atoms=atoms,
                        energy_corr_dict=energy_corr_dict,
                    )
                # Write to database.
                db_ase.write(atoms)
                natoms.append(len(atoms))
    # Return paths of databases.
    return db_path_list

# -------------------------------------------------------------------------------------
# FINETUNE ORB MODEL
# -------------------------------------------------------------------------------------

def finetune_ORB_model(
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
    config_dict: dict = None,
    checkpoint_path: str = None,
    model_name: str = None,
    local_cache: str = None,
    config_yaml_path: str = None,
    delete_config_keys: list = "default",
    update_config_keys: dict = {},
    kwargs_cmd: dict = {},
    calc_kwargs = {"seed": 42},
    **kwargs: dict,
) -> Calculator:
    """
    Fine-tune OCP model from ase Atoms data.
    """
    # Remove old directory.
    if clean_directory is True and os.path.isdir(directory):
        shutil.rmtree(directory)
    # Start from the model in the calculator.
    if calc is not None:
        config_dict = calc.config
    # Prepare train, val, and test databases.
    db_train_path, db_val_path, db_test_path = prepare_train_val_test_dbs(
        atoms_list=atoms_list,
        directory=directory,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        atoms_tasks=atoms_tasks,
        energy_corr_dict=energy_corr_dict,
    )
    # Run the fine-tuning.
    checkpoint_path_new = train_ORB_model(
        directory=directory,
        db_train_path=db_train_path,
        db_val_path=db_val_path,
        db_test_path=db_test_path,
        config_dict=config_dict,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        local_cache=local_cache,
        delete_config_keys=delete_config_keys,
        update_config_keys=update_config_keys,
        config_yaml_path=config_yaml_path,
        kwargs_cmd=kwargs_cmd,
        label=label,
        logfile=logfile,
        **kwargs,
    )
    # Return calculator.
    return ORBCalculator(
        checkpoint_path=checkpoint_path_new,
        **calc_kwargs,
    )

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------