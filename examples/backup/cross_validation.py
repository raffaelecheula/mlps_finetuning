# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.db import connect
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneGroupOut
from chgnet.model.dynamics import CHGNetCalculator

from mlps_finetuning.energy_ref import get_corrected_energy, get_energy_corrections
from mlps_finetuning.chgnet import finetune_chgnet_train_val
from mlps_finetuning.databases import (
    get_atoms_from_db,
    get_atoms_list_from_db,
    write_atoms_list_to_db,
)
from mlps_finetuning.workflow import (
    optimize_atoms,
    get_reference_energies_adsorbates,
    get_formation_energy_adsorbate,
    print_energies,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Splits parameters.
    finetuning = False
    n_splits = 5
    random_state = 42
    key_groups = "dopant"
    kwargs_init = {"relaxed": True, "species": "09_OH+H"}

    # Formation energies.
    formation_energies = True
    calculate_ref_clean = True
    calculate_ref_gas = True

    # Device.
    use_device = None

    # Results database.
    db_res_name = "ZrO2_chgnet.db"
    keys_store = ["class", "species", "surface", "dopant", "uid"]

    # Ase database.
    db_ase_name = "../ZrO2_dft.db"
    selection = "class=adsorbates"

    # Reference database.
    db_ref_name = "../ZrO2_ref.db"
    yaml_name = "../ZrO2_ref_chgnet.yaml"

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
        "wandb_path": "chgnet",
        "save_dir": None,
    }
    
    # Gas references.
    references_gas = ["H2", "H2O", "CO2"]
    
    # Ase calculator.
    calc = CHGNetCalculator(use_device=use_device)
    
    # Initialize ase database.
    db_ase = connect(name=db_ase_name)

    # Get list of initial atoms structures from database.
    atoms_init = get_atoms_list_from_db(db_ase, selection=selection, **kwargs_init)
    print(f"Number of structures: {len(atoms_init)}")
    
    # Get list of all atoms structures from database.
    atoms_all = get_atoms_list_from_db(db_ase)
    
    # Get energy corrections.
    energy_corr_dict = get_energy_corrections(
        db_ref_name=db_ref_name,
        yaml_name=yaml_name,
        calc=calc,
    )
    
    # Initialize cross-validation.
    crossval = KFold(n_splits, random_state=random_state, shuffle=True)
    
    # Get groups for splits.
    groups = [atoms.info[key_groups] for atoms in atoms_init]
    
    # Additional atoms for training.
    atoms_train_added = []
    
    # Loop over splits.
    energy_true = []
    energy_pred = []
    e_form_pred = []
    e_form_true = []
    atoms_pred = []
    indices = list(range(len(atoms_init)))
    for ii, (indices_train, indices_test) in enumerate(
        crossval.split(indices, groups=groups)
    ):
        # Get training and test sets.
        uid_train = [atoms_init[ii].info["uid"] for ii in indices_train]
        atoms_train = [atoms for atoms in atoms_all if atoms.info["uid"] in uid_train]
        atoms_train += atoms_train_added
        atoms_test = np.array(atoms_init, dtype=object)[indices_test]
        # Run finetuning on train set.
        if finetuning is True:
            calc = finetune_chgnet_train_val(
                atoms_list=atoms_train,
                energy_corr_dict=energy_corr_dict,
                **kwargs_trainer,
            )
        # Get reference energies.
        if formation_energies is True:
            energies_ref, energies_ref_dft, compositions_ref = get_reference_energies_adsorbates(
                atoms_list=atoms_test,
                references_gas=references_gas,
                calc=calc,
                db_ase=db_ase,
                energy_corr_dict=energy_corr_dict,
                kwargs_init=kwargs_init,
                calculate_ref_clean=calculate_ref_clean,
                calculate_ref_gas=calculate_ref_gas,
            )
        # Relax structures.
        for atoms in atoms_test:
            # Optimize structure and get MLP energy.
            optimize_atoms(
                atoms=atoms,
                calc=calc,
                energy_corr_dict=energy_corr_dict,
            )
            # Get DFT energy.
            atoms_dft = get_atoms_from_db(db_ase, uid=atoms.info["uid"], relaxed=True)
            energy = atoms.get_potential_energy()
            energy_dft = atoms_dft.get_potential_energy()
            # Store results.
            energy_pred.append(energy)
            energy_true.append(energy_dft)
            atoms.info["energy"] = energy
            atoms.info["energy_dft"] = energy_dft
            # Get formation energies.
            if formation_energies is True:
                # Get MLP formation energy.
                e_form = get_formation_energy_adsorbate(
                    atoms=atoms,
                    energies_ref=energies_ref,
                    compositions_ref=compositions_ref,
                )
                # Get DFT formation energy.
                e_form_dft = get_formation_energy_adsorbate(
                    atoms=atoms_dft,
                    energies_ref=energies_ref_dft,
                    compositions_ref=compositions_ref,
                )
                # Store results.
                e_form_pred.append(e_form)
                e_form_true.append(e_form_dft)
                atoms.info["e_form"] = e_form
                atoms.info["e_form_dft"] = e_form_dft
            else:
                e_form = None
                e_form_dft = None
            # Print energies.
            print_energies(
                energy=energy,
                energy_dft=energy_dft,
                e_form=e_form,
                e_form_dft=e_form_dft,
            )
            # Append atoms to list.
            atoms_pred.append(atoms)
    # Calculate average errors.
    mae_energy = mean_absolute_error(y_true=energy_true, y_pred=energy_pred)
    print(f"MAE Energy: {mae_energy:.3f} eV")
    if formation_energies is True:
        mae_e_form = mean_absolute_error(y_true=e_form_true, y_pred=e_form_pred)
        print(f"MAE E_form: {mae_e_form:.3f} eV")
    # Store results into ase database.
    db_ase = connect(name=db_res_name, append=True)
    write_atoms_list_to_db(
        atoms_list=atoms_pred,
        db_ase=db_ase,
        keys_store=keys_store,
        keys_match=None,
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------