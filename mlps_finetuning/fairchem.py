# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import inspect
import numpy as np
from pathlib import Path
from ase.calculators.calculator import Calculator
from fairchem.core import FAIRChemCalculator as FAIRChemCalculatorOriginal
from fairchem.core.calculate.pretrained_mlip import (
    get_predict_unit,
    load_predict_unit,
)

from mlps_finetuning.energy_ref import get_corrected_energy
from mlps_finetuning.utilities import RedirectOutput

# -------------------------------------------------------------------------------------
# FAIRCHEM CALCULATOR
# -------------------------------------------------------------------------------------

class FAIRChemCalculator(FAIRChemCalculatorOriginal):

    def __init__(
        self,
        predict_unit: object = None,
        checkpoint_path: str = None,
        logfile: str = None,
        adjust_sum_force: bool = False,
        **kwargs: dict,
    ):
        # Build predict unit from kwargs if not provided.
        if checkpoint_path is not None:
            unit_keys = inspect.getfullargspec(load_predict_unit).args
            kwargs_unit = {key: kwargs.pop(key) for key in unit_keys if key in kwargs}
            predict_unit = load_predict_unit(path=checkpoint_path, **kwargs_unit)
        elif predict_unit is None:
            unit_keys = inspect.getfullargspec(get_predict_unit).args
            kwargs_unit = {key: kwargs.pop(key) for key in unit_keys if key in kwargs}
            predict_unit = get_predict_unit(**kwargs_unit)
        # Initialize.
        with RedirectOutput(logfile=logfile):
            super().__init__(predict_unit=predict_unit, **kwargs)
        # Adjust forces so that their sum is equal to zero.
        self.adjust_sum_force = adjust_sum_force
        # Counter to keep track of the number of single-point evaluations.
        self.counter = 0
        self.info = {}
    
    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        if self.adjust_sum_force is True:
            self.results["forces"] -= self.results["forces"].mean(axis=0)
        self.counter += 1

# -------------------------------------------------------------------------------------
# TRAIN FAIRCHEM MODEL
# -------------------------------------------------------------------------------------

def train_FAIRChem_model(
    directory: str,
    db_train_path: str,
    db_val_path: str,
    db_test_path: str = None,
    dataset_name: str = "oc20",
    regression_tasks: str = "ef",
    base_model_name: str = "uma-s-1p1",
    checkpoint_path: str = None,
    label: str = "model_00",
    energy_coeff: float = None,
    forces_coeff: float = None,
    stress_coeff: float = None,
    normalizer_rmsd: float = 1.0,
    elem_refs: list = [0.] * 100,
    compute_references: bool = False,
    num_workers: int = 1,
    checkpoint_name: str = "inference_ckpt.pt",
    logfile: str = None,
    **kwargs: dict,
):
    """
    Train FAIRChem model.
    """
    from dataclasses import dataclass
    from fairchem.core._cli import main as fairchem_main
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    # Process training data.
    if compute_references is True:
        from fairchem.core.scripts.create_finetune_dataset import (
            compute_normalizer_and_linear_reference,
        )
        normalizer_rmsd, elem_refs = compute_normalizer_and_linear_reference(
            train_path=db_train_path,
            num_workers=num_workers,
        )
    # Create yaml file.
    config = create_config_yaml_files(
        directory=directory,
        dataset_name=dataset_name,
        regression_tasks=regression_tasks,
        db_train_path=db_train_path,
        db_val_path=db_val_path,
        db_test_path=db_test_path,
        base_model_name=base_model_name,
        checkpoint_path=checkpoint_path,
        label=label,
        energy_coeff=energy_coeff,
        forces_coeff=forces_coeff,
        stress_coeff=stress_coeff,
        normalizer_rmsd=normalizer_rmsd,
        elem_refs=elem_refs,
        **kwargs,
    )
    # Training arguments dataclass.
    @dataclass
    class TrainingArgs:
        config: str
    # Training arguments container.
    args = TrainingArgs(config=config)
    # Train the model.
    with RedirectOutput(logfile=logfile):
        fairchem_main(args=args, override_args=[])
    # Return path of the new checkpoint.
    return os.path.join(directory, label, "checkpoints", "final", checkpoint_name)

# -------------------------------------------------------------------------------------
# FAIRCHEM ROOT
# -------------------------------------------------------------------------------------

def fairchem_root():
    """
    Returns the root of the FAIRChem package.
    """
    import fairchem
    return Path(fairchem.__path__[0])

# -------------------------------------------------------------------------------------
# CREATE CONFIG YAML FILES
# -------------------------------------------------------------------------------------

def create_config_yaml_files(
    directory: str,
    dataset_name: str,
    regression_tasks: str,
    db_train_path: str,
    db_val_path: str,
    db_test_path: str = None,
    base_model_name: str = "uma-s-1",
    checkpoint_path: str = None,
    label: str = None,
    energy_coeff: float = None,
    forces_coeff: float = None,
    stress_coeff: float = None,
    normalizer_rmsd: float = 1.0,
    elem_refs: list = [0.] * 100,
    filename: str = "uma_sm_finetune.yaml",
    **kwargs: dict,
):
    """
    Create new config yaml files from FAIRChem templates.
    """
    # Dictionary of filenames associated to different regression tasks.
    task_yaml_dict = {
        "e": "uma_conserving_data_task_energy.yaml",
        "ef": "uma_conserving_data_task_energy_force.yaml",
        "efs": "uma_conserving_data_task_energy_force_stress.yaml",
    }
    # Fine-tuning and task template files.
    templates_dir = fairchem_root() / Path("configs/uma/finetune")
    filename_yaml = Path("uma_sm_finetune_template.yaml")
    filename_task = Path("data") / task_yaml_dict[regression_tasks]
    # Update fine-tuning file.
    with open(templates_dir / filename_yaml) as fileobj:
        config = yaml.safe_load(fileobj)
        config["base_model_name"] = str(base_model_name)
        config["defaults"][0]["data"] = str(filename_task.stem)
        config["job"]["run_dir"] = str(directory)
        if label is not None:
            config["job"]["timestamp_id"] = label
        if checkpoint_path is not None:
            config["runner"]["train_eval_unit"]["model"]["checkpoint_location"] = (
                checkpoint_path
            )
            del config["base_model_name"]
    config.update(**kwargs)
    with open(Path(directory) / filename, "w") as fileobj:
        yaml.dump(config, fileobj, default_flow_style=False, sort_keys=False)
    # Update task yaml file.
    with open(templates_dir / filename_task) as fileobj:
        config_task = yaml.safe_load(fileobj)
        config_task["dataset_name"] = str(dataset_name)
        config_task["normalizer_rmsd"] = float(normalizer_rmsd)
        config_task["elem_refs"] = list(elem_refs)
        config_task["train_dataset"]["splits"]["train"]["src"] = str(db_train_path)
        if db_val_path is not None:
            config_task["val_dataset"]["splits"]["val"]["src"] = str(db_val_path)
        if db_test_path is not None:
            config_task["val_dataset"]["splits"]["val"]["src"] = str(db_val_path)
        if energy_coeff is not None:
            config_task["tasks_list"][0]["loss_fn"]["coefficient"] = energy_coeff
        if forces_coeff is not None:
            config_task["tasks_list"][1]["loss_fn"]["coefficient"] = forces_coeff
        if stress_coeff is not None:
            config_task["tasks_list"][2]["loss_fn"]["coefficient"] = stress_coeff
    os.makedirs(directory / Path("data"), exist_ok=True)
    with open(Path(directory) / filename_task, "w") as fileobj:
        yaml.dump(config_task, fileobj, default_flow_style=False, sort_keys=False)
    # Return new fine-tuning yaml file path.
    return Path(directory) / filename

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
):
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
        db_path = os.path.join(directory, task, f"{task}.aselmdb")
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
        # Write metadata file.
        metadata_name = os.path.join(directory, task, "metadata.npz")
        np.savez_compressed(metadata_name, natoms=natoms)
    # Return paths of databases.
    return db_path_list

# -------------------------------------------------------------------------------------
# FINETUNE FAIRCHEM MODEL
# -------------------------------------------------------------------------------------

def finetune_FAIRChem_model(
    atoms_list: list,
    calc: Calculator = None,
    directory: str = "finetuning",
    val_fraction: float = 0.1,
    test_fraction: float = 0.0,
    seed: int = 42,
    atoms_tasks: list = None,
    energy_corr_dict: dict = None,
    base_model_name: str = "uma-s-1",
    checkpoint_path: str = None,
    dataset_name: str = "oc20",
    regression_tasks: str = "ef",
    label: str = "model_00",
    kwargs_calc: dict = {},
    logfile: str = None,
    **kwargs: dict,
):
    """
    Fine-tune FAIRChem model from ase Atoms data.
    """
    # Start from the model in the calculator.
    if calc is not None and "checkpoint_path" in calc.info:
        checkpoint_path = calc.info["checkpoint_path"]
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
    checkpoint_path_new = train_FAIRChem_model(
        directory=directory,
        base_model_name=base_model_name,
        checkpoint_path=checkpoint_path,
        dataset_name=dataset_name,
        regression_tasks=regression_tasks,
        label=label,
        db_train_path=db_train_path,
        db_val_path=db_val_path,
        db_test_path=db_test_path,
        logfile=logfile,
        **kwargs,
    )
    # Return calculator.
    calc = FAIRChemCalculator(
        checkpoint_path=checkpoint_path_new,
        task_name=dataset_name,
        **kwargs_calc,
    )
    calc.info["checkpoint_path"] = checkpoint_path_new
    return calc

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------