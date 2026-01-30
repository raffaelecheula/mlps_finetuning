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
from mlps_finetuning.cross_validations import (
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
    finetuning = True # Fine-tune the MLP model.
    n_gas_added = 0 # Number of times to add gas molecules to training set.
    n_clean_added = 0 # Number of times to add clean surfaces to training set.
    start_from_relaxed = False # Start relaxations from relaxed structures.
    train_on_relaxed = False # Use only relaxed structures in training set.
    use_ref_database = True # Use reference database for energy corrections.
    n_max_train = 800 # Maximum number of data in training set.
    only_n_folds = 1 # Run calculations only on the first n folds (for testing).
    random_state = 42 # Random state for cross-validator.
    required_properties = ["energy", "forces"] # Required properties of atoms.

    # Formation energies.
    formation_energies = True
    calculate_ref_clean = True
    calculate_ref_gas = True

    # ASE calculator.
    calc = FAIRChemCalculator(
        model_name=model_name,
        device="cuda",
        cache_dir=os.getenv("PRETRAINED_MODELS", "."),
        task_name="oc20",
        seed=42,
    )

    # ASE database.
    db_ase_name = "../../ZrO2_DFT.db"
    selection = "class=adsorbates"

    # Energy corrections database.
    db_ref_name = "../../ZrO2_ref.db"

    # Fine-tuning parameters.
    finetune_kwargs = {
        "epochs": 100,
        "learning_rate": 1e-4,
        "batch_size": 4,
        "val_fraction": 0.10,
        "logfile": "log.txt",
        "clean_directory": True,
        # Other parameters.
        "energy_coeff": 10,
        "forces_coeff": 10,
        "base_model_name": model_name,
        "dataset_name": "oc20",
        "regression_tasks": "ef",
        "weight_decay": 1e-5,
        "evaluate_every_n_steps": 100,
        "checkpoint_every_n_steps": 100,
    }
    
    # Initialize ase database.
    db_ase = connect(name=db_ase_name)
    # Get list of initial atoms structures from database.
    kwargs_init = {"relaxed": True} if start_from_relaxed else {"index": 0}
    atoms_init = get_atoms_list_from_db(db_ase, selection=selection, **kwargs_init)
    atoms_init = [atoms for atoms in atoms_init if atoms.info["species"] != "00_clean"]
    print(f"Number of initial structures (to relax): {len(atoms_init)}")
    # Get list of all atoms structures from database.
    kwargs_all = {"relaxed": True} if train_on_relaxed else {}
    atoms_all = get_atoms_list_from_db(db_ase, selection=selection, **kwargs_all)
    atoms_all = [atoms for atoms in atoms_all if atoms.info["species"] != "00_clean"]
    print(f"Number of all DFT data (used in training sets): {len(atoms_all)}")
    
    # Additional atoms for training.
    atoms_train_added = []
    kwargs_match = {"class": "molecules"}
    atoms_train_added += get_atoms_list_from_db(db_ase, **kwargs_match) * n_gas_added
    kwargs_match = {"class": "adsorbates", "species": "00_clean"}
    atoms_train_added += get_atoms_list_from_db(db_ase, **kwargs_match) * n_clean_added
    
    # Initialize cross-validation.
    crossval = get_crossvalidator(
        stratified=stratified,
        group=group,
        n_splits=n_splits,
        random_state=random_state,
        shuffle=True,
    )

    # Energy corrections parameters.
    if use_ref_database is True:
        db_ref = connect(name=db_ref_name)
        atoms_ref = get_atoms_list_from_db(db_ref)
        energy_corr_kwargs = {"fit_first": ["C", "O", "H"]}
    else:
        atoms_ref = None
        energy_corr_kwargs = {}
    
    # Reference energies parameters.
    ref_energies_kwargs = {
        "references_gas": ["H2", "H2O", "CO2"],
        "kwargs_init": kwargs_init,
        "calculate_ref_clean": calculate_ref_clean,
        "calculate_ref_gas": calculate_ref_gas,
        "kwargs_surface": {"species": "00_clean"},
        "kwargs_molecule": {"surface": "none"},
    }
    
    # Run cross-validation with optimization.
    results = cross_validation_with_optimization(
        atoms_init=atoms_init,
        atoms_all=atoms_all,
        db_ase=db_ase,
        crossval=crossval,
        key_groups=key_groups,
        key_stratify=key_stratify,
        calc=calc,
        finetuning=finetuning,
        finetune_MLP_fun=finetune_FAIRChem_model,
        finetune_kwargs=finetune_kwargs,
        atoms_ref=atoms_ref,
        energy_corr_kwargs=energy_corr_kwargs,
        atoms_train_added=atoms_train_added,
        formation_energies=formation_energies,
        formation_energy_fun=get_formation_energy_adsorbate,
        ref_energies_fun=get_reference_energies_adsorbates,
        ref_energies_kwargs=ref_energies_kwargs,
        required_properties=required_properties,
        n_max_train=n_max_train,
        only_n_folds=only_n_folds,
    )
    
    # Results database.
    model_tag = "finetuned" if finetuning is True else "pretrained"
    db_res_name = f"ZrO2_{calc_name}_{model_tag}.db"
    keys_store = ["class", "species", "surface", "dopant", "uid"]
    keys_match = ["uid"]
    
    # Store results into ase database.
    db_res = connect(name=db_res_name, append=False)
    write_atoms_list_to_db(
        atoms_list=results["atoms_list_MLP"],
        db_ase=db_res,
        keys_store=keys_store,
        keys_match=keys_match,
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Run script and measure execution time.
    import timeit
    time_start = timeit.default_timer()
    main()
    time_stop = timeit.default_timer()
    print(f"Execution time = {time_stop - time_start:6.1f} [s].")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------