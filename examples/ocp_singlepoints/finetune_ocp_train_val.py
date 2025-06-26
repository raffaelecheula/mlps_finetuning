# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os

from mlps_finetuning.ocp import (
    OCPCalculator,
    split_database,
    update_config_yaml,
    finetune_ocp,
    default_delete_keys,
)
from mlps_finetuning.energy_ref import get_energy_corrections

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Parameters.
    db_tot_name = "../ZrO2_ref.db"
    directory = "finetuning"
    identifier = "ZnO2"
    timestamp_id = "model_1"
    
    # Energy corrections database.
    db_corr_name = "../ZrO2_ref.db"
    yaml_corr_name = "../ZrO2_ref_fairchem.yaml"
    
    # Model name and cache.
    model_name = "GemNet-OC-S2EFS-OC20+OC22"
    #model_name = "EquiformerV2-31M-S2EF-OC20-All+MD"
    #model_name = "EquiformerV2-lE4-lF100-S2EFS-OC22"
    #model_name = "eSCN-L6-M3-Lay20-S2EF-OC20-All+MD"
    local_cache = "pretrained_models"
    
    # Ase calculator.
    calc = OCPCalculator(
        model_name=model_name,
        local_cache=local_cache,
        cpu=False,
    )
    config_dict = calc.config
    checkpoint_path = config_dict["checkpoint"]
    config_yaml = os.path.join(directory, "config.yaml")

    # Get energy corrections.
    energy_corr_dict = get_energy_corrections(
        db_corr_name=db_corr_name,
        yaml_corr_name=yaml_corr_name,
        calc=calc,
    )

    # Split the database into train and val databases.
    if not os.path.isfile(os.path.join(directory, "train.db")):
        split_database(
            db_ase_name=db_tot_name,
            fractions=(0.8, 0.1, 0.1),
            filenames=("train.db", "test.db", "val.db"),
            directory=directory,
            seed=42,
            energy_corr_dict=energy_corr_dict,
        )
    
    # Update config file.
    update_keys = {
        # Train parameters.
        "gpus": 1,
        "optim.eval_every": 10, # 4
        "optim.max_epochs": 10,
        #"optim.lr_initial": 1e-5,
        "optim.batch_size": 4, # 1
        #"optim.num_workers": 4,
        "logger": "tensorboard",
        #"task.primary_metric": "forces_mae",
        # Train data.
        "dataset.train.src": os.path.join(directory, "train.db"),
        "dataset.train.format": "ase_db",
        "dataset.train.a2g_args.r_energy": True,
        "dataset.train.a2g_args.r_forces": True,
        # Test data.
        "dataset.test.src": os.path.join(directory, "test.db"),
        "dataset.test.format": "ase_db",
        "dataset.test.a2g_args.r_energy": False,
        "dataset.test.a2g_args.r_forces": False,
        # val data.
        "dataset.val.src": os.path.join(directory, "val.db"),
        "dataset.val.format": "ase_db",
        "dataset.val.a2g_args.r_energy": True,
        "dataset.val.a2g_args.r_forces": True,
    }
    update_config_yaml(
        config_dict=config_dict,
        config_yaml=config_yaml,
        delete_keys=default_delete_keys(),
        update_keys=update_keys,
    )
    
    # Run the fine-tuning.
    finetune_ocp(
        checkpoint_path=checkpoint_path,
        config_yaml=config_yaml,
        directory=directory,
        identifier=identifier,
        timestamp_id=timestamp_id,
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------