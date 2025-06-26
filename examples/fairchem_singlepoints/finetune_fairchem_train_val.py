# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

from ase.db import connect

from mlps_finetuning.fairchem import finetune_fairchem_train_val
from mlps_finetuning.databases import get_atoms_list_from_db

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Ase database.
    db_ase_name = "../ZrO2_dft.db"
    selection = "class=adsorbates"

    # Initialize ase database.
    db_ase = connect(name=db_ase_name)
    # Get list of initial atoms structures from database.
    atoms_list = get_atoms_list_from_db(db_ase, selection=selection)[:160]
    print(f"Number of structures: {len(atoms_list)}")

    # Trainer parameters.
    kwargs_trainer = {
        "directory": "finetuning",
        "base_model_name": "uma-s-1",
        "dataset_name": "oc20",
        "regression_tasks": "ef",
        "timestamp_id": "model_01",
        "config_dict": {
            "epochs": 100,
            "steps": None,
            "batch_size": 4,
            "lr": 1e-5,
            "weight_decay": 1e-5,
            "evaluate_every_n_steps": 100,
            "checkpoint_every_n_steps": 1000,
        },
    }
    
    # Finetune the model.
    finetune_fairchem_train_val(
        atoms_list=atoms_list,
        **kwargs_trainer,
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------