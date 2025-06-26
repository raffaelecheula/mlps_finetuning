# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

from ase.db import connect
from sklearn.model_selection import KFold

from mlps_finetuning.chgnet import CHGNetCalculator, finetune_chgnet_crossval
from mlps_finetuning.energy_ref import get_energy_corrections
from mlps_finetuning.databases import get_atoms_list_from_db

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # DFT database.
    db_dft_name = "ZrO2_dft.db"

    # Energy corrections database.
    db_corr_name = "ZrO2_ref.db"
    yaml_corr_name = "ZrO2_ref_chgnet.yaml"

    # Get energy corrections.
    calc = CHGNetCalculator()
    energy_corr_dict = get_energy_corrections(
        db_corr_name=db_corr_name,
        yaml_corr_name=yaml_corr_name,
        calc=calc,
    )
    
    # Get atoms from database.
    selection = "class=surfaces"
    db_dft = connect(db_dft_name)
    atoms_list = get_atoms_list_from_db(db_ase=db_dft, selection=selection)
    
    # Initialize cross-validator.
    crossval = KFold(n_splits=5, random_state=42, shuffle=True)
    
    # Run finetuning.
    finetune_chgnet_crossval(
        atoms_list=atoms_list,
        crossval=crossval,
        energy_corr_dict=energy_corr_dict,
        targets="efm",
        batch_size=8,
        optimizer="Adam",
        scheduler="CosLR",
        criterion="MSE",
        epochs=100,
        learning_rate=1e-5,
        use_device=None,
        print_freq=10,
        wandb_path="chgnet/crossval-singlepoints",
        save_dir=None,
        train_composition_model=False
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------