# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import numpy as np
from ase.cluster import Icosahedron
from fairchem.core import OCPCalculator as OCPCalculatorOriginal

from mlps_finetuning.energy_ref import get_corrected_energy

# -------------------------------------------------------------------------------------
# OCP CALCULATOR
# -------------------------------------------------------------------------------------

class OCPCalculator(OCPCalculatorOriginal):
        
    def __init__(
        self,
        adjust_sum_force: bool = False,
        atoms_ref = Icosahedron(symbol="Cu", noshells=2, latticeconstant=None),
        vacuum_ref: float = 50.,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Adjust forces so that their sum is equal to zero.
        self.adjust_sum_force = adjust_sum_force
        # Reference atoms needed for calculate gas phase molecules with GemNet.
        self.atoms_ref = atoms_ref
        self.vacuum_ref = vacuum_ref
        # Counter to keep track of the number of single-point evaluations.
        self.counter = 0
    
    def calculate(self, atoms, properties, system_changes):
        try:
            super().calculate(atoms, properties, system_changes)
        except RuntimeError:
            self.calculate_with_reference(atoms, properties, system_changes)
        if self.adjust_sum_force is True:
            self.results["forces"] -= self.results["forces"].mean(axis=0)
        self.counter += 1

    def calculate_with_reference(self, atoms, properties, system_changes):
        """Calculate results for structure with a reference."""
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
# FINETUNE OCP
# -------------------------------------------------------------------------------------

def finetune_ocp(
    checkpoint_path: str,
    mode: str = "train",
    config_yaml: str = "config.yaml",
    directory: str = "finetuning",
    **kwargs,
):
    """Finetune OCP model."""
    from fairchem.core._cli import main as fairchem_main
    # Default flags.
    flags = {
        "checkpoint": checkpoint_path,
        "mode": mode,
        "config_yml": config_yaml,
        "run_dir": directory,
        "identifier": "",
        "debug": False,
        "print_every": 10,
        "seed": 0,
        "amp": False,
        "timestamp_id": None,
        "sweep_yml": None,
        "submit": False,
        "summit": False,
        "logdir": "logs",
        "slurm_partition": None,
        "slurm_account": None,
        "slurm_qos": None,
        "slurm_mem": 80,
        "slurm_timeout": 72,
        "num_gpus": 1,
        "cpu": False,
        "num_nodes": 1,
        "gp_gpus": None,
    }
    flags.update(kwargs)
    # Create training arguments container.
    class TrainingArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    args = TrainingArgs(**flags)
    # Run the finetuning.
    fairchem_main(args=args, override_args=[])
    # Return path of best checkpoint.
    best_checkpoint_path = os.path.join(
        directory, "checkpoints", flags["timestamp_id"], "best_checkpoint.pt",
    )
    return best_checkpoint_path

# -------------------------------------------------------------------------------------
# DEFAULT DELETE KEYS
# -------------------------------------------------------------------------------------

def default_delete_keys():
    """Default delete keys to remove from config yaml files."""
    delete_keys = [
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
    return delete_keys

# -------------------------------------------------------------------------------------
# UPDATE CONFIG YAML
# -------------------------------------------------------------------------------------

def update_config_yaml(
    checkpoint_path: str = None,
    config_dict: dict = None,
    config_yaml: str = "config.yaml",
    delete_keys: tuple = (),
    update_keys: dict = {},
):
    """Generate a yaml config file from an existing checkpoint file."""
    # Read config dictionary.
    if config_dict is None:
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
    # Delete config keys.
    for key in delete_keys:
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
    for key in update_keys:
        if "." not in key:
            config_dict[key] = update_keys[key]
        else:
            key_list = key.split(".")
            if key_list[0] not in config_dict:
                config_dict[key_list[0]] = {}
            nested_dict = config_dict[key_list[0]]
            for key_ii in key_list[1:-1]:
                if key_ii not in nested_dict:
                    nested_dict[key_ii] = {}
                nested_dict = nested_dict[key_ii]
            nested_dict[key_list[-1]] = update_keys[key]
    # Write config yaml file.
    with open(config_yaml, 'w') as fileobj:
        yaml.dump(config_dict, fileobj)

## -------------------------------------------------------------------------------------
## SPLIT DATABASE
## -------------------------------------------------------------------------------------
#
#def split_database(
#    db_ase_name: str,
#    fractions: tuple = (0.8, 0.1, 0.1),
#    filenames: tuple = ("train.db", "test.db", "val.db"),
#    directory: str = ".",
#    seed: int = 42,
#    energy_corr_dict: dict = None,
#):
#    """Split an ase database into train, test and validation ase databases."""
#    from ase.db import connect
#    os.makedirs(directory, exist_ok=True)
#    # Get filenames (and delete them).
#    db_name_list = []
#    for name in filenames:
#        db_name = os.path.join(directory, name)
#        if os.path.exists(db_name):
#            os.remove(db_name)
#        db_name_list.append(db_name)
#    # Read source database.
#    db_ase = connect(db_ase_name)
#    n_data = db_ase.count()
#    # Set sum of fractions equal to 1 and get numbers of data.
#    fractions = np.array(fractions)
#    fractions /= fractions.sum()
#    n_data_array = np.array(np.round(fractions*n_data), dtype=int)
#    n_data_array[-1] = n_data-np.sum(n_data_array[:-1])
#    # Shuffle the database ids.
#    ids = np.arange(1, n_data+1)
#    rng = np.random.default_rng(seed=seed)
#    rng.shuffle(ids)
#    # Write new databases.
#    num = 0
#    for ii, db_name in enumerate(db_name_list):
#        with connect(db_name) as db_new:
#            natoms = []
#            for id in ids[num:num+n_data_array[ii]]:
#                row = db_ase.get(id=int(id))
#                atoms = row.toatoms()
#                if energy_corr_dict is not None:
#                    atoms.calc.results["energy"] = get_corrected_energy(
#                        atoms=atoms,
#                        energy_corr_dict=energy_corr_dict,
#                    )
#                db_new.write(atoms)
#                natoms.append(len(atoms))
#            db_new.metadata = {"natoms": natoms}
#        num += n_data_array[ii]

# -------------------------------------------------------------------------------------
# PREPARE TRAIN VAL DBS
# -------------------------------------------------------------------------------------

def prepare_train_val_dbs(
    atoms_list: list,
    train_ratio: float = 0.90,
    val_ratio: float = 0.10,
    use_test_set: bool = False,
    directory: str = ".",
    seed: int = 42,
    energy_corr_dict: dict = None,
):
    """Split an ase database into train, test and validation ase databases."""
    from ase.db import connect
    os.makedirs(directory, exist_ok=True)
    # Delete previous datasets.
    filenames = ["train.db", "val.db"]
    if use_test_set is True:
        filenames.append("test.db")
    db_name_list = []
    for name in filenames:
        db_name = os.path.join(directory, name)
        if os.path.exists(db_name):
            os.remove(db_name)
        db_name_list.append(db_name)
    # Shuffle the indices of the atoms.
    n_data = len(atoms_list)
    indices = np.arange(n_data)
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(indices)
    aa, bb = int(n_data*train_ratio), int(n_data*(train_ratio+val_ratio))
    indices_list = [indices[:aa], indices[aa:bb], indices[bb:]]
    # Write new databases.
    for ii, db_name in enumerate(db_name_list):
        with connect(db_name) as db_new:
            natoms = []
            for jj in indices_list[ii]:
                atoms = atoms_list[jj]
                if energy_corr_dict is not None:
                    atoms.calc.results["energy"] = get_corrected_energy(
                        atoms=atoms,
                        energy_corr_dict=energy_corr_dict,
                    )
                db_new.write(atoms)
                natoms.append(len(atoms))
            db_new.metadata = {"natoms": natoms}

# -------------------------------------------------------------------------------------
# FINETUNE OCP TRAIN VAL
# -------------------------------------------------------------------------------------

def finetune_ocp_train_val(
    atoms_list: list,
    config_dict: dict = None,
    checkpoint_path: str = None,
    update_keys: dict = {},
    delete_keys: list = default_delete_keys(),
    kwargs_main: dict = {},
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    seed: int = 42,
    energy_corr_dict: dict = None,
    use_test_set: bool = False,
    directory: str = "finetuning",
    **kwargs,
):
    """Finetune OCP model from ase Atoms data."""
    # Prepare train and val databases.
    prepare_train_val_dbs(
        atoms_list=atoms_list,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_test_set=use_test_set,
        directory=directory,
        seed=seed,
        energy_corr_dict=energy_corr_dict,
    )
    # Update kwargs trainer.
    update_keys.update({
        "dataset.train.src": os.path.join(directory, "train.db"),
        "dataset.train.format": "ase_db",
        "dataset.train.a2g_args.r_energy": True,
        "dataset.train.a2g_args.r_forces": True,
        "dataset.val.src": os.path.join(directory, "val.db"),
        "dataset.val.format": "ase_db",
        "dataset.val.a2g_args.r_energy": True,
        "dataset.val.a2g_args.r_forces": True,
    })
    if use_test_set is True:
        update_keys.update({
            "dataset.test.src": os.path.join(directory, "test.db"),
            "dataset.test.format": "ase_db",
            "dataset.test.a2g_args.r_energy": False,
            "dataset.test.a2g_args.r_forces": False,
        })
    # Get `config_dict` or `checkpoint_path` from `kwargs_trainer`.
    if config_dict is not None:
        checkpoint_path = config_dict["checkpoint"]
    # Update config.yaml file.
    config_yaml = os.path.join(directory, "config.yaml")
    update_config_yaml(
        checkpoint_path=checkpoint_path,
        config_yaml=config_yaml,
        delete_keys=delete_keys,
        update_keys=update_keys,
    )
    # Run the fine-tuning.
    finetune_ocp(
        checkpoint_path=checkpoint_path,
        config_yaml=config_yaml,
        directory=directory,
        **kwargs_main,
    )

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------