# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.db import connect

from mlps_finetuning.chgnet import (
    CHGNetCalculator,
    finetune_chgnet_train_val,
)
from mlps_finetuning.energy_ref import get_energy_corrections
from mlps_finetuning.databases import (
    get_atoms_list_from_db,
    write_atoms_list_to_db,
)
from mlps_finetuning.workflow import (
    get_crossvalidator,
    get_reference_energies_adsorbates,
    get_formation_energy_adsorbate,
    cross_validation_with_optimization,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Cross-validation parameters.
    stratified = True # Stratified cross-validation.
    group = False # Group cross-validation.
    key_groups = "dopant" # dopant | species
    key_stratify = "species" # dopant | species
    n_splits = 5 # Number of splits for cross-validation.
    finetuning = True # Fine-tune the MLP model.
    kwargs_init = {"index": 0} # {"relaxed": True} or {"index": 0}
    n_gas_added = 0 # Number of times to add gas molecules to training set.
    n_clean_added = 0 # Number of times to add clean surfaces to training set.
    only_active_dopants = True
    exclude_physisorbed = True
    random_state = 42

    # Formation energies.
    formation_energies = True
    calculate_ref_clean = True
    calculate_ref_gas = True

    # Ase calculator.
    use_device = None
    calc = CHGNetCalculator(use_device=use_device)

    # Ase database.
    db_ase_name = "../ZrO2_dft.db"
    selection = "class=adsorbates"

    # Energy corrections database.
    db_corr_name = "../ZrO2_ref.db"
    yaml_corr_name = "../ZrO2_ref_chgnet.yaml"

    # Trainer parameters.
    kwargs_trainer = {
        "learning_rate": 1e-4,
        "epochs": 100,
        "batch_size": 8,
        "targets": "ef",
        "train_ratio": 0.90,
        "val_ratio": 0.10,
        "optimizer": "Adam",
        "scheduler": "CosLR",
        "criterion": "MSE",
        "use_device": use_device,
        "print_freq": 10,
        "wandb_path": "chgnet/crossval-adsorbates",
        "save_dir": None,
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
        finetune_mlp_fun=finetune_chgnet_train_val,
        calc=calc,
        crossval=crossval,
        kwargs_trainer=kwargs_trainer,
        energy_corr_dict=energy_corr_dict,
        finetuning=finetuning,
        atoms_train_added=atoms_train_added,
        formation_energies=formation_energies,
        ref_energies_fun=get_reference_energies_adsorbates,
        ref_energies_kwargs=ref_energies_kwargs,
        formation_energy_fun=get_formation_energy_adsorbate,
    )
    
    # Results database.
    model_tag = "finetuned" if finetuning is True else "pretrained"
    db_res_name = f"ZrO2_chgnet_{model_tag}.db"
    keys_store = ["class", "species", "surface", "dopant", "uid"]
    keys_match = ["uid"]
    
    # Store results into ase database.
    db_res = connect(name=db_res_name, append=True)
    write_atoms_list_to_db(
        atoms_list=results["atoms_pred"],
        db_ase=db_res,
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