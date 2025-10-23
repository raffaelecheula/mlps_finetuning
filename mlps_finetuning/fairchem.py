# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import fairchem
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from fairchem.core import FAIRChemCalculator as FAIRChemCalculatorOriginal
from fairchem.core import pretrained_mlip

from mlps_finetuning.energy_ref import get_corrected_energy

# -------------------------------------------------------------------------------------
# FAIRCHEM CALCULATOR
# -------------------------------------------------------------------------------------

class FAIRChemCalculator(FAIRChemCalculatorOriginal):

    def __init__(
        self,
        adjust_sum_force: bool = False,
        **kwargs,
    ):
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
# FAIRCHEM ROOT
# -------------------------------------------------------------------------------------

def fairchem_root():
    """
    Returns the root of the FairChem package.
    """
    return Path(fairchem.__path__[0])

# -------------------------------------------------------------------------------------
# CREATE CONFIG YAML FILES
# -------------------------------------------------------------------------------------

def create_config_yaml_files(
    directory: str,
    dataset_name: str,
    regression_tasks: str,
    config_dict: dict,
    db_train_path: str,
    db_val_path: str,
    base_model_name: str,
    checkpoint_path: str = None,
    label: str = None,
    energy_coeff: float = None,
    forces_coeff: float = None,
    stress_coeff: float = None,
    normalizer_rmsd: float = 1.0,
    elem_refs: list = [0.] * 100,
    filename: str = "uma_sm_finetune.yaml",
):
    """
    Create new config yaml files from FairChem templates.
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
    config.update(config_dict)
    with open(Path(directory) / filename, "w") as fileobj:
        yaml.dump(config, fileobj, default_flow_style=False, sort_keys=False)
    # Update task yaml file.
    with open(templates_dir / filename_task) as fileobj:
        config_task = yaml.safe_load(fileobj)
        config_task["dataset_name"] = str(dataset_name)
        config_task["normalizer_rmsd"] = float(normalizer_rmsd)
        config_task["elem_refs"] = list(elem_refs)
        config_task["train_dataset"]["splits"]["train"]["src"] = str(db_train_path)
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
# FINETUNE FAIRCHEM
# -------------------------------------------------------------------------------------

def finetune_fairchem(
    directory: str = "finetuning",
    dataset_name: str = "oc20",
    regression_tasks: str = "ef",
    config_dict = {},
    db_train_path: str = "finetuning/train/train.db",
    db_val_path: str = "finetuning/val/val.db",
    base_model_name: str = "uma-s-1p1",
    checkpoint_path: str = None,
    label: str = "model_01",
    energy_coeff: float = None,
    forces_coeff: float = None,
    stress_coeff: float = None,
    normalizer_rmsd: float = 1.0,
    elem_refs: list = [0.] * 100,
    compute_references: bool = False,
    num_workers: int = 1,
):
    """
    Finetune FairChem model.
    """
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
        config_dict=config_dict,
        db_train_path=db_train_path,
        db_val_path=db_val_path,
        base_model_name=base_model_name,
        checkpoint_path=checkpoint_path,
        label=label,
        energy_coeff=energy_coeff,
        forces_coeff=forces_coeff,
        stress_coeff=stress_coeff,
        normalizer_rmsd=normalizer_rmsd,
        elem_refs=elem_refs,
    )
    # Create training arguments container.
    @dataclass
    class FinetuningArgs:
        config: str
    args = FinetuningArgs(config=config)
    # Run the finetuning.
    fairchem_main(args=args, override_args=[])
    # Return path of best checkpoint.
    checkpoint_path = os.path.join(
        directory, label, "checkpoints", "final", "inference_ckpt.pt",
    )
    return checkpoint_path

# -------------------------------------------------------------------------------------
# PREPARE TRAIN VAL DBS
# -------------------------------------------------------------------------------------

def prepare_train_val_dbs(
    atoms_list: list,
    train_ratio: float = 0.90,
    val_ratio: float = 0.10,
    use_test_set: bool = False,
    directory: str = "finetuning",
    seed: int = 42,
    energy_corr_dict: dict = None,
):
    """
    Split an ase database into train, test and validation ase databases.
    """
    from ase.db import connect
    tasks = ["train", "val", "test"] if use_test_set is True else ["train", "val"]
    # Shuffle the indices of the atoms.
    n_data = len(atoms_list)
    indices = np.arange(n_data)
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(indices)
    aa, bb = int(n_data*train_ratio), int(n_data*(train_ratio+val_ratio))
    indices_list = [indices[:aa], indices[aa:bb], indices[bb:]]
    # Write databases.
    db_name_list = []
    for ii, task in enumerate(tasks):
        # Create directory.
        os.makedirs(os.path.join(directory, task), exist_ok=True)
        db_name = os.path.join(directory, task, f"{task}.aselmdb")
        db_name_list.append(db_name)
        # Delete previous database.
        if os.path.exists(db_name):
            os.remove(db_name)
        # Write new database.
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
        # Write metadata file.
        metadata_name = os.path.join(directory, task, "metadata.npz")
        np.savez_compressed(metadata_name, natoms=natoms)
    # Return paths of databases.
    return db_name_list

# -------------------------------------------------------------------------------------
# FINETUNE FAIRCHEM TRAIN VAL
# -------------------------------------------------------------------------------------

def finetune_fairchem_train_val(
    atoms_list: list,
    config_dict: dict = {},
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    seed: int = 42,
    energy_corr_dict: dict = None,
    use_test_set: bool = False,
    directory: str = "finetuning",
    base_model_name: str = "uma-s-1",
    dataset_name: str = "oc20",
    regression_tasks: str = "ef",
    label: str = "model_00",
    required_properties: list = ["energy", "forces"],
    return_calculator: bool = True,
    kwargs_calc: dict = {},
    kwargs_unit: dict = {},
):
    """
    Finetune FairChem model from ase Atoms data.
    """
    from fairchem.core.units.mlip_unit import load_predict_unit
    # Filter atoms with required properties.
    atoms_list = [
        atoms for atoms in atoms_list
        if set(required_properties).issubset(atoms.calc.results)
    ]
    # Prepare train and val databases.
    db_train_path, db_val_path = prepare_train_val_dbs(
        atoms_list=atoms_list,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_test_set=use_test_set,
        directory=directory,
        seed=seed,
        energy_corr_dict=energy_corr_dict,
    )
    # Run the fine-tuning.
    checkpoint_path = finetune_fairchem(
        config_dict=config_dict,
        directory=directory,
        base_model_name=base_model_name,
        dataset_name=dataset_name,
        regression_tasks=regression_tasks,
        label=label,
        db_train_path=db_train_path,
        db_val_path=db_val_path,
    )
    # Return calculator or checkpoint path.
    if return_calculator:
        predict_unit = load_predict_unit(
            path=checkpoint_path,
            **kwargs_unit,
        )
        return FAIRChemCalculator(
            predict_unit=predict_unit,
            task_name=dataset_name,
            **kwargs_calc,
        )
    else:
        return checkpoint_path

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------