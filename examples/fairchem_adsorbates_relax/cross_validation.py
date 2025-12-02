# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase.db import connect

from mlps_finetuning.fairchem import FAIRChemCalculator, finetune_FAIRChem_model
from mlps_finetuning.energy_ref import get_energy_corrections
from mlps_finetuning.databases import (
    get_atoms_list_from_db,
    write_atoms_list_to_db,
)
from mlps_finetuning.crossvalidations import (
    get_crossvalidator,
    get_reference_energies_adsorbates,
    get_formation_energy_adsorbate,
    cross_validation_with_optimization,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # MLP model.
    calc_name = "FAIRChem"
    model_name = "uma-s-1p1"

    # Cross-validation parameters.
    stratified = True # Stratified cross-validation.
    group = False # Group cross-validation.
    key_groups = "dopant" # dopant | species
    key_stratify = "species" # dopant | species
    n_splits = 5 # Number of splits for cross-validation.
    finetuning = False # Fine-tune the MLP model.
    kwargs_init = {"index": 0} # {"relaxed": True} or {"index": 0}
    n_gas_added = 0 # Number of times to add gas molecules to training set.
    n_clean_added = 0 # Number of times to add clean surfaces to training set.
    n_max_train = 1000 # Maximum number of data in training set.
    only_n_folds = 1 # Run calculations only on the first n folds (for testing).
    only_active_dopants = True
    exclude_physisorbed = True
    random_state = 42

    # Formation energies.
    formation_energies = True
    calculate_ref_clean = True
    calculate_ref_gas = True

    # Ase calculator.
    calc = FAIRChemCalculator(
        model_name=model_name,
        device="cuda",
        cache_dir=os.getenv("PRETRAINED_MODELS", "."),
        task_name="oc20",
        seed=42,
    )

    # Ase database.
    db_ase_name = "../ZrO2_DFT.db"
    selection = "class=adsorbates"

    # Energy corrections database.
    db_corr_name = "../ZrO2_ref.db"
    yaml_corr_name = f"../ZrO2_ref_{calc_name}.yaml"

    # Trainer parameters.
    kwargs_trainer = {
        "val_fraction": 0.10,
        "logfile": "log.txt",
        "base_model_name": model_name,
        "dataset_name": "oc20",
        "regression_tasks": "ef",
        # kwargs.
        "epochs": 100,
        "lr": 1e-4,
        "steps": None,
        "batch_size": 4,
        "weight_decay": 1e-5,
        "evaluate_every_n_steps": 100,
        "checkpoint_every_n_steps": 100,
    }
    
    # Initialize ase database.
    db_ase = connect(name=db_ase_name)
    # Get list of initial atoms structures from database.
    atoms_init = get_atoms_list_from_db(db_ase, selection=selection, **kwargs_init)
    if only_active_dopants is True:
        active_dopants = ["Al3+", "Cd2+", "Ga3+", "In3+", "Zn2+"]
        atoms_init = [
            atoms for atoms in atoms_init if atoms.info["dopant"] in active_dopants
        ]
    if exclude_physisorbed is True:
        excluded = ["17_CH2O+OH+H", "19_CH2O+H2O", "23_CH2O+2H", "18_CH2O+OH+H_2"]
        atoms_init = [
            atoms for atoms in atoms_init if atoms.info["species"] not in excluded
        ]
    print(f"Number of structures: {len(atoms_init)}")
    
    # Initialize cross-validation.
    crossval = get_crossvalidator(
        stratified=stratified,
        group=group,
        n_splits=n_splits,
        random_state=random_state,
        shuffle=True,
    )
    
    # Additional atoms for training.
    atoms_train_added = []
    kwargs_match = {"class": "molecules"}
    atoms_train_added += get_atoms_list_from_db(db_ase, **kwargs_match) * n_gas_added
    kwargs_match = {"class": "adsorbates", "species": "00_clean"}
    atoms_train_added += get_atoms_list_from_db(db_ase, **kwargs_match) * n_clean_added
    
    # Get energy corrections.
    energy_corr_dict = get_energy_corrections(
        db_corr_name=db_corr_name,
        yaml_corr_name=yaml_corr_name,
        calc=calc,
    )
    
    # Get groups for splits.
    groups = [atoms.info[key_groups] for atoms in atoms_init]
    
    # Reference energies parameters.
    ref_energies_kwargs = {
        "references_gas": ["H2", "H2O", "CO2"],
        "kwargs_init": kwargs_init,
        "calculate_ref_clean": calculate_ref_clean,
        "calculate_ref_gas": calculate_ref_gas,
    }
    
    # Run cross-validation with optimization.
    results = cross_validation_with_optimization(
        atoms_init=atoms_init,
        db_ase=db_ase,
        key_groups=key_groups,
        key_stratify=key_stratify,
        finetune_MLP_fun=finetune_FAIRChem_model,
        calc=calc,
        crossval=crossval,
        kwargs_trainer=kwargs_trainer,
        energy_corr_dict=energy_corr_dict,
        finetuning=finetuning,
        atoms_train_added=atoms_train_added,
        formation_energies=formation_energies,
        formation_energy_fun=get_formation_energy_adsorbate,
        ref_energies_fun=get_reference_energies_adsorbates,
        ref_energies_kwargs=ref_energies_kwargs,
        n_max_train=n_max_train,
        only_n_folds=only_n_folds,
    )
    
    # Results database.
    model_tag = "finetuned" if finetuning is True else "pretrained"
    db_res_name = f"ZrO2_{calc_name}_{model_tag}.db"
    keys_store = ["class", "species", "surface", "dopant", "uid"]
    keys_match = ["uid"]
    
    # Store results into ase database.
    db_ase = connect(name=db_res_name, append=True)
    write_atoms_list_to_db(
        atoms_list=results["Atoms_list_MLP"],
        db_ase=db_ase,
        keys_store=keys_store,
        keys_match=keys_match,
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------