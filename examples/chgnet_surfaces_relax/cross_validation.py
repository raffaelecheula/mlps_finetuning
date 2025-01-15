# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.db import connect
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    StratifiedGroupKFold,
)
from chgnet.model.dynamics import CHGNetCalculator

from mlps_finetuning.energy_ref import get_energy_corrections
from mlps_finetuning.chgnet import finetune_chgnet_train_val
from mlps_finetuning.databases import (
    get_atoms_list_from_db,
    write_atoms_list_to_db,
)
from mlps_finetuning.workflow import (
    get_reference_energies_adsorbates,
    get_formation_energy_adsorbate,
    cross_validation_with_optimization,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # TODO: test different cross-validators.
    # TODO: Test different values of n_gas.
    # TODO: Test different trainer parameters.

    # Cross-validation parameters.
    finetuning = False # Fine-tune the MLP model.
    crossval_name = "KFold" # Type of cross-validation.
    key_groups = "dopant" # "dopant" or "species"
    kwargs_init = {"relaxed": True} # {"relaxed": True} or {"index": 0}
    n_gas_added = 0 # Number of times to add gas molecules to training set.
    n_splits = 5
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
    selection = "class=surfaces"

    # Reference database.
    db_ref_name = "../ZrO2_ref.db"
    yaml_name = "../ZrO2_ref_chgnet.yaml"

    # Results database.
    db_res_name = "ZrO2_chgnet.db"
    keys_store = ["class", "species", "surface", "dopant", "uid"]
    keys_match = ["uid"]

    # Trainer parameters.
    kwargs_trainer = {
        "targets": "efm",
        "batch_size": 8,
        "train_ratio": 0.90,
        "val_ratio": 0.10,
        "optimizer": "Adam",
        "scheduler": "CosLR",
        "criterion": "MSE",
        "epochs": 100,
        "learning_rate": 1e-3,
        "use_device": use_device,
        "print_freq": 10,
        "wandb_path": "chgnet/crossval-surfaces",
        "save_dir": None,
    }
    
    # Initialize ase database.
    db_ase = connect(name=db_ase_name)

    # Get list of initial atoms structures from database.
    atoms_init = get_atoms_list_from_db(db_ase, selection=selection, **kwargs_init)
    print(f"Number of structures: {len(atoms_init)}")
    
    # Initialize cross-validation.
    if crossval_name == "KFold":
        crossval = KFold(n_splits, random_state=random_state, shuffle=True)
    elif crossval_name == "StratifiedKFold":
        crossval = StratifiedKFold(n_splits, random_state=random_state, shuffle=True)
    elif crossval_name == "GroupKFold":
        crossval = GroupKFold(n_splits)
    elif crossval_name == "StratifiedGroupKFold":
        crossval = StratifiedGroupKFold(n_splits)
    
    # Additional atoms for training.
    kwargs_match = {"class": "molecules"}
    atoms_train_added = get_atoms_list_from_db(db_ase, **kwargs_match) * n_gas_added
    
    # Get energy corrections.
    energy_corr_dict = get_energy_corrections(
        db_ref_name=db_ref_name,
        yaml_name=yaml_name,
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
        groups=groups,
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
    
    # Store results into ase database.
    db_ase = connect(name=db_res_name, append=True)
    write_atoms_list_to_db(
        atoms_list=results["atoms_pred"],
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