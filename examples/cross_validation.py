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
from mlps_finetuning.databases import get_atoms_from_db, get_atoms_list_from_db
from mlps_finetuning.workflow import (
    optimize_atoms,
    get_reference_energies,
    get_formation_energy,
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
    kwargs_init = {"relaxed": True}

    # Formation energies.
    formation_energies = True
    calculate_ref_clean = True
    calculate_ref_gas = False

    # Device.
    use_device = None

    # Ase database.
    db_ase_name = "ZrO2_dft.db"
    selection = "class=adsorbates"

    # Reference database.
    db_ref_name = "ZrO2_ref.db"
    yaml_name = "ZrO2_ref_chgnet.yaml"

    # Trainer parameters.
    kwargs_trainer = {
        "targets": "efm",
        "batch_size": 8,
        "train_ratio": 0.80,
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
    
    # Get list of all atoms structures from database.
    atoms_all = get_atoms_list_from_db(db_ase)
    
    # Get energy corrections.
    energy_corr_dict = get_energy_corrections(
        db_ref_name=db_ref_name,
        yaml_name=yaml_name,
        calc=calc,
    )
    
    # Initialize cross-validation.
    indices = list(range(len(atoms_init)))
    kfold = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    
    # Get groups for splits.
    groups = [atoms.info[key_groups] for atoms in atoms_init]
    
    # Additional atoms for training.
    atoms_train_added = []
    
    # Loop over splits.
    y_true = []
    y_pred = []
    atoms_pred = []
    for ii, (indices_train, indices_test) in enumerate(
        kfold.split(indices, groups=groups)
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
            references_surf = list(
                {atoms.info["surface"]: None for atoms in atoms_test}
            )
            energies_ref, energies_ref_dft, compositions_ref = get_reference_energies(
                references_surf=references_surf,
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
            optimize_atoms(
                atoms=atoms,
                calc=calc,
                energy_corr_dict=energy_corr_dict,
            )
            # Get dft energy.
            atoms_dft = get_atoms_from_db(db_ase, uid=atoms.info["uid"], relaxed=True)
            if formation_energies is True:
                e_form = get_formation_energy(
                    atoms=atoms,
                    energies_ref=energies_ref,
                    compositions_ref=compositions_ref,
                )
                e_form_dft = get_formation_energy(
                    atoms=atoms_dft,
                    energies_ref=energies_ref_dft,
                    compositions_ref=compositions_ref,
                )
                y_pred.append(e_form)
                y_true.append(e_form_dft)
                print(f"E form MLP: {e_form:+12.3f} eV")
                print(f"E form DFT: {e_form_dft:+12.3f} eV")
                print(f"E form dev: {e_form-e_form_dft:+12.3f} eV")
            else:
                energy = atoms.get_potential_energy()
                energy_dft = atoms_dft.get_potential_energy()
                y_pred.append(energy)
                y_true.append(energy_dft)
                print(f"Energy MLP: {energy:+12.3f} eV")
                print(f"Energy DFT: {energy_dft:+12.3f} eV")
                print(f"Energy dev: {energy-energy_dft:+12.3f} eV")
            atoms_pred.append(atoms)
    # Calculate errors.
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    print(f"MAE energies: {mae:.3f} eV")

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------