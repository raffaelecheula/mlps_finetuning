# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase.db import connect

from mlps_finetuning.mace import MACECalculator, finetune_MACE_model
from mlps_finetuning.energy_ref import get_energy_corrections
from mlps_finetuning.databases import get_atoms_list_from_db
from mlps_finetuning.utilities import filter_atoms_list
from mlps_finetuning.cross_validations import resample_atoms_list

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # MLP model.
    calc_name = "MACE"
    model_name = "medium-mpa-0"

    # Cross-validation parameters.
    n_max_data = 1000 # Maximum number of data.
    random_state = 42 # Random state for resampling.
    required_properties = ["energy", "forces"] # Required properties of atoms.

    # ASE calculator.
    calc = MACECalculator(
        model_name=model_name,
        default_dtype="float64",
        logfile="log.txt",
    )

    # ASE database.
    db_ase_name = "../../ZrO2_DFT.db"
    selection = "class=adsorbates"

    # Energy corrections database.
    db_corr_name = "../../ZrO2_ref.db"
    yaml_corr_name = f"../../ZrO2_ref_{calc_name}.yaml"

    # Fine-tuning parameters.
    finetune_kwargs = {
        "calc": None,
        "directory": "finetuning",
        "label": "model_00",
        "val_fraction": 0.1,
        "logfile": "log.txt",
        # kwargs.
        "max_num_epochs": 100,
        "lr": 1e-4,
        "batch_size": 4,
        "num_workers": 4,
        "multiheads_finetuning": "False",
        "foundation_model": model_name,
        "model": "MACE",
        "num_interactions": 3,
        "num_channels": 128,
        "E0s": "average",
        "max_L": 1,
        "correlation": 3,
        "r_max": 5.0,
        "ema": True,
        "device": "cuda",
        "save_cpu": True,
    }
    
    # Initialize ase database.
    db_ase = connect(name=db_ase_name)
    # Get list of initial atoms structures from database.
    atoms_list = get_atoms_list_from_db(db_ase, selection=selection)
    # Filter data with the required properties.
    if required_properties is not None:
        atoms_list = filter_atoms_list(
            atoms_list=atoms_list,
            required_properties=required_properties,
        )
    # Limit the number of data.
    if n_max_data is not None:
        atoms_list = resample_atoms_list(
            atoms_list=atoms_list,
            key_groups="uid",
            n_max_data=n_max_data,
            random_state=random_state,
            duplicates_ok=False,
        )
    print(f"Number of structures: {len(atoms_list)}")
    
    # Get energy corrections.
    energy_corr_dict = get_energy_corrections(
        db_corr_name=db_corr_name,
        yaml_corr_name=yaml_corr_name,
        calc=calc,
    )

    # Run fine-tuning.
    calc = finetune_MACE_model(
        atoms_list=atoms_list,
        energy_corr_dict=energy_corr_dict,
        **finetune_kwargs,
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