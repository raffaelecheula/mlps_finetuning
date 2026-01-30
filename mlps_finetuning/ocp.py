# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import shutil
import yaml
import numpy as np
import warnings
from copy import deepcopy
from ase.cluster import Icosahedron
from ase.calculators.calculator import Calculator
warnings.filterwarnings("ignore", category=FutureWarning)
from fairchem.core import OCPCalculator as OCPCalculatorOriginal

from mlps_finetuning.energy_ref import get_corrected_energy
from mlps_finetuning.utilities import RedirectOutput

# -------------------------------------------------------------------------------------
# OCP CALCULATOR
# -------------------------------------------------------------------------------------

class OCPCalculator(OCPCalculatorOriginal):

    def __init__(
        self,
        adjust_sum_force: bool = False,
        logfile: str = None,
        **kwargs: dict,
    ):
        # Initialize.
        with RedirectOutput(logfile=logfile):
            super().__init__(**kwargs)
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
# OCP CALCULATOR WITH REFERENCE
# -------------------------------------------------------------------------------------

class OCPCalculatorWithReference(OCPCalculatorOriginal):
        
    def __init__(
        self,
        adjust_sum_force: bool = False,
        atoms_ref = Icosahedron(symbol="Cu", noshells=2, latticeconstant=None),
        vacuum_ref: float = 50.,
        logfile: str = None,
        **kwargs: dict,
    ):
        # Initialize.
        with RedirectOutput(logfile=logfile):
            super().__init__(**kwargs)
        # Adjust forces so that their sum is equal to zero.
        self.adjust_sum_force = adjust_sum_force
        # Reference atoms needed for calculate gas phase molecules with GemNet.
        self.atoms_ref = atoms_ref
        self.vacuum_ref = vacuum_ref
        # Counter to keep track of the number of single-point evaluations.
        self.counter = 0
        self.info = {}
    
    def calculate(self, atoms, properties, system_changes):
        try:
            super().calculate(atoms, properties, system_changes)
        except RuntimeError:
            print("OCPCalculator: adding reference.")
            self.calculate_with_reference(atoms, properties, system_changes)
        if self.adjust_sum_force is True:
            self.results["forces"] -= self.results["forces"].mean(axis=0)
        self.counter += 1

    def calculate_with_reference(self, atoms, properties, system_changes):
        """
        Calculate results for structure with a reference (for GemNet-OC).
        """
        # Create a copy of the atoms and add vacuum.
        atoms_copy = atoms.copy()
        atoms_copy.cell[2,2] += 2 * self.vacuum_ref
        # Position the reference atoms in the cell.
        atoms_ref = self.atoms_ref.copy()
        atoms_ref.translate([0., 0., atoms.cell[2,2]+self.vacuum_ref])
        atoms_ref.cell = atoms_copy.cell.copy()
        # Calculate properties of reference atoms.
        super().calculate(atoms_ref, properties, system_changes)
        results_ref = self.results.copy()
        # Calculate properties of atoms plus reference atoms.
        atoms_copy += atoms_ref
        super().calculate(atoms_copy, properties, system_changes)
        # Update system specific properties.
        for prop in ["energy", "stress", "dipole", "magmom", "free_energy"]:
            if prop in self.results:
                self.results[prop] = self.results[prop] - results_ref[prop]
        # Update atom specific properties.
        for prop in ["forces", "stresses", "charges", "magmoms", "energies"]:
            if prop in self.results:
                self.results[prop] = self.results[prop][:len(atoms)]

# -------------------------------------------------------------------------------------
# TRAIN OCP MODEL
# -------------------------------------------------------------------------------------

def train_OCP_model(
    config_yaml_path: str,
    directory: str = "training",
    checkpoint_path: str = None,
    label: str = "model",
    checkpoint_name: str = "best_checkpoint.pt",
    logfile: str = None,
    **kwargs,
) -> str:
    """
    Train OCP model.
    """
    import torch
    from dataclasses import dataclass
    from fairchem.core._cli import main as ocp_main
    # Training arguments dataclass.
    @dataclass
    class TrainingArgs:
        mode: str = "train"
        config_yml: str = config_yaml_path
        identifier: str = ""
        debug: bool = False
        run_dir: str = directory
        print_every: int = 10
        seed: int = 42
        amp: bool = False
        checkpoint: str = checkpoint_path
        timestamp_id: str = label
        sweep_yml: str = None
        submit: bool = False
        summit: bool = False
        logdir: str = "logs"
        slurm_partition: str = None
        slurm_account: str = None
        slurm_qos: str = None
        slurm_mem: int = 80
        slurm_timeout: int = 72
        num_gpus: int = 1
        cpu: bool = False if torch.cuda.is_available() else True
        num_nodes: int = 1
        gp_gpus: int = None
    # Training arguments container.
    args = TrainingArgs(**kwargs)
    # Train the model.
    with RedirectOutput(logfile=logfile):
        ocp_main(args=args, override_args=[])
    # Return path of the new checkpoint.
    return os.path.join(directory, "checkpoints", label, checkpoint_name)

# -------------------------------------------------------------------------------------
# UPDATE CONFIG AND TRAIN OCP MODEL
# -------------------------------------------------------------------------------------

def update_config_and_train_OCP_model(
    directory: str,
    db_train_path: str,
    db_val_path: str = None,
    db_test_path: str = None,
    config_dict: dict = None,
    checkpoint_path: str = None,
    model_name: str = None,
    local_cache: str = None,
    delete_config_keys: list = "default",
    update_config_keys: dict = {},
    config_yaml_path: str = None,
    checkpoint_name: str = "best_checkpoint.pt",
    kwargs_cmd: dict = {},
    label: str = "model",
    logfile: str = None,
    **kwargs: dict,
) -> str:
    """
    Update config yaml file and train the OCP model.
    """
    # Get config and checkpoint path.
    if config_dict is not None:
        checkpoint_path = config_dict["checkpoint"]
    elif checkpoint_path is not None:
        config_dict = get_config_from_checkpoint(checkpoint_path=checkpoint_path)
    elif model_name is not None:
        config_dict = get_config_from_model_name(
            model_name=model_name,
            local_cache=local_cache,
        )
        checkpoint_path = config_dict["checkpoint"]
    else:
        raise KeyError(
            "config_dict, model_name, or checkpoint_path must be provided"
        )
    # Get update-config keys from kwargs config.
    update_config_keys = get_update_config_keys(
        config_dict=config_dict,
        checkpoint_path=checkpoint_path,
        db_train_path=db_train_path,
        db_val_path=db_val_path,
        db_test_path=db_test_path,
        **kwargs,
    )
    # Default delete keys.
    if delete_config_keys == "default":
        delete_config_keys = default_delete_config_keys()
    # Update config dict.
    config_dict_new = update_config_dict(
        config_dict=config_dict,
        delete_config_keys=delete_config_keys,
        update_config_keys=update_config_keys,
    )
    # Write config yaml file.
    if config_yaml_path is None:
        config_yaml_path = os.path.join(directory, "config.yaml")
    with open(config_yaml_path, "w") as fileobj:
        yaml.dump(config_dict_new, fileobj)
    # Run the training.
    checkpoint_path_new = train_OCP_model(
        checkpoint_path=checkpoint_path,
        config_yaml_path=config_yaml_path,
        directory=directory,
        label=label,
        checkpoint_name=checkpoint_name,
        logfile=logfile,
        **kwargs_cmd,
    )
    # Return new checkpoint path and config.
    return checkpoint_path_new, config_dict_new

# -------------------------------------------------------------------------------------
# GET CONFIG FROM CHECKPOINT
# -------------------------------------------------------------------------------------

def get_config_from_checkpoint(
    checkpoint_path: str,
) -> dict:
    """
    Get config from checkpoint path.
    """
    import torch
    from fairchem.core.common.utils import update_config
    data = torch.load(checkpoint_path, map_location="cpu")
    config_dict = data["config"]
    if "model_attributes" in config_dict:
        config_dict["model_attributes"]["name"] = config_dict.pop("model")
        config_dict["model"] = config_dict["model_attributes"]
    config_dict["model"]["otf_graph"] = True
    config_dict["checkpoint"] = str(checkpoint_path)
    config_dict["trainer"] = config_dict.get("trainer", "ocp")
    config_dict = update_config(config_dict)
    # Return config dict.
    return config_dict

# -------------------------------------------------------------------------------------
# READ CONFIG FROM MODEL NAME
# -------------------------------------------------------------------------------------

def get_config_from_model_name(
    model_name: str,
    local_cache: str,
) -> dict:
    """
    Get config from model name.
    """
    from fairchem.core.models.model_registry import model_name_to_local_file
    checkpoint_path = model_name_to_local_file(
        model_name=model_name,
        local_cache=local_cache,
    )
    config_dict = get_config_from_checkpoint(checkpoint_path=checkpoint_path)
    # Return config dict.
    return config_dict

# -------------------------------------------------------------------------------------
# GET UPDATE CONFIG KEYS
# -------------------------------------------------------------------------------------

def get_update_config_keys(
    config_dict: dict,
    checkpoint_path: str,
    db_train_path: str,
    db_val_path: str,
    db_test_path: str = None,
    db_format: str = "ase_db",
    gpus: int = 1,
    amp: bool = False,
    eval_every: int = 100,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    batch_size: int = 4,
    eval_batch_size: int = 4,
    num_workers: int = 8,
    energy_coeff: float = 30,
    forces_coeff: float = 60,
    stress_coeff: float = 10,
    primary_metric: str = "forces_mae",
    warmup_epochs: int = 5,
    warmup_steps: int = 10,
    logger: str = "tensorboard",
    **kwargs: dict,
) -> dict:
    """
    Get update-config keys.
    """
    update_config_keys = {
        "gpus": gpus,
        "amp": amp,
        "checkpoint": checkpoint_path,
        "optim.eval_every": eval_every,
        "optim.max_epochs": epochs,
        "optim.lr_initial": learning_rate,
        "optim.batch_size": batch_size,
        "optim.eval_batch_size": eval_batch_size,
        "optim.num_workers": num_workers,
        "optim.energy_coefficient": energy_coeff,
        "optim.force_coefficient": forces_coeff,
        "task.primary_metric": primary_metric,
        "logger": logger,
        "loss_functions.[0].energy.coefficient": energy_coeff,
        "loss_functions.[1].forces.coefficient": forces_coeff,
        "dataset.train.src": db_train_path,
        "dataset.train.format": db_format,
        "dataset.train.a2g_args.r_energy": True,
        "dataset.train.a2g_args.r_forces": True,
    }
    # Validation dataset.
    if db_val_path is not None:
        update_config_keys.update({
            "dataset.val.src": db_val_path,
            "dataset.val.format": db_format,
            "dataset.val.a2g_args.r_energy": True,
            "dataset.val.a2g_args.r_forces": True,
        })
    # Test dataset.
    if db_test_path is not None:
        update_config_keys.update({
            "dataset.test.src": db_test_path,
            "dataset.test.format": db_format,
            "dataset.test.a2g_args.r_energy": False,
            "dataset.test.a2g_args.r_forces": False,
        })
    # Update stress parameters.
    if "stress" in config_dict["outputs"]:
        update_config_keys.update({
            "dataset.train.a2g_args.r_stress": True,
            "loss_functions.[2].stress.coefficient": stress_coeff,
        })
        if db_val_path is not None:
            update_config_keys["dataset.val.a2g_args.r_stress"] = True
        if db_test_path is not None:
            update_config_keys["dataset.test.a2g_args.r_stress"] = False
    # Update scheduler parameters.
    if config_dict["optim"].get("scheduler", None) == "LambdaLR":
        update_config_keys.update({
            "optim.scheduler_params.epochs": epochs,
            "optim.scheduler_params.lr": learning_rate,
            "optim.scheduler_params.warmup_epochs": warmup_epochs,
        })
    elif "warmup_steps" in config_dict["optim"]:
        update_config_keys["optim.warmup_steps"] = warmup_steps
    # Return update-config keys.
    return update_config_keys

# -------------------------------------------------------------------------------------
# DEFAULT DELETE CONFIG KEYS
# -------------------------------------------------------------------------------------

def default_delete_config_keys() -> list:
    """
    Default keys to clean config yaml file.
    """
    delete_config_keys = [
        "slurm",
        "cmd",
        "logger",
        "task",
        "model_attributes",
        "optim.loss_force",
		"optim.load_balancing",
        "dataset",
        "test_dataset",
        "val_dataset",
    ]
    return delete_config_keys

# -------------------------------------------------------------------------------------
# UPDATE CONFIG DICT
# -------------------------------------------------------------------------------------

def update_config_dict(
    config_dict: dict,
    delete_config_keys: list = [],
    update_config_keys: dict = {},
) -> dict:
    """
    Update config dictionary.
    """
    # Delete config keys.
    for key in delete_config_keys:
        if key in config_dict:
            del config_dict[key]
        elif "." in key and key.split(".")[0] in config_dict:
            key_list = key.split(".")
            nested_dict = config_dict[key_list[0]]
            for key_ii in key_list[1:-1]:
                if key_ii in nested_dict:
                    nested_dict = nested_dict[key_ii]
            if key_list[-1] in nested_dict:
                del nested_dict[key_list[-1]]
    # Update config keys.
    for key in update_config_keys:
        if "." not in key:
            config_dict[key] = update_config_keys[key]
        else:
            key_list = key.split(".")
            if key_list[0] not in config_dict:
                config_dict[key_list[0]] = {}
            nested_dict = config_dict[key_list[0]]
            for key_ii in key_list[1:-1]:
                if key_ii[0] == "[":
                    key_ii = int(key_ii[1:-1])
                elif key_ii not in nested_dict:
                    nested_dict[key_ii] = {}
                nested_dict = nested_dict[key_ii]
            nested_dict[key_list[-1]] = update_config_keys[key]
    # Return updated config dict.
    return config_dict

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
    from ase.stress import full_3x3_to_voigt_6_stress
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
    db_path_list = []
    for task, atoms_list in zip(tasks, atoms_tasks):
        # Do not create an empty database.
        if atoms_list is None or len(atoms_list) == 0:
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
                # Copy atoms.
                calc = deepcopy(atoms.calc)
                atoms = atoms.copy()
                atoms.calc = calc
                # Get forces with no constraints.
                forces = atoms.get_forces(apply_constraint=False)
                # Apply energy correction.
                energy = get_corrected_energy(
                    atoms=atoms,
                    energy_corr_dict=energy_corr_dict,
                )
                # Convert stress to Voigt order.
                stress = atoms.calc.results.get("stress", np.zeros(6))
                if stress.shape == (3, 3):
                    stress = full_3x3_to_voigt_6_stress(stress)
                # Remove constraints and update results.
                atoms.constraints = []
                atoms.calc.atoms = atoms
                atoms.calc.results = {
                    "energy": energy,
                    "forces": forces,
                    "stress": stress,
                }
                # Write to database.
                db_ase.write(atoms)
                natoms.append(len(atoms))
            db_ase.metadata = {"natoms": natoms}
    # Return paths of databases.
    return db_path_list

# -------------------------------------------------------------------------------------
# FINETUNE OCP MODEL
# -------------------------------------------------------------------------------------

def finetune_OCP_model(
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
    checkpoint_name: str = "best_checkpoint.pt",
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
    # No best checkpoint without validation set.
    if db_val_path is None:
        checkpoint_name = "checkpoint.pt"
    # Run the fine-tuning.
    checkpoint_path_new, config_dict_new = update_config_and_train_OCP_model(
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
        checkpoint_name=checkpoint_name,
        kwargs_cmd=kwargs_cmd,
        label=label,
        logfile=logfile,
        **kwargs,
    )
    # Return calculator.
    return OCPCalculator(
        checkpoint_path=checkpoint_path_new,
        config_yml=config_dict_new,
        **calc_kwargs,
    )

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------