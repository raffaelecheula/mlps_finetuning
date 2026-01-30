# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase.db import connect

from mlps_finetuning.fairchem import FAIRChemCalculator, finetune_FAIRChem_model
from mlps_finetuning.energy_ref import get_energy_corrections
from mlps_finetuning.databases import get_atoms_list_from_db
from mlps_finetuning.cross_validations import resample_atoms_list

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # MLP model.
    calc_name = "FAIRChem"
    model_name = "uma-s-1p1"

    # Cross-validation parameters.
    n_max_data = 1000 # Maximum number of data.
    random_state = 42 # Random state for resampling.
    required_properties = ["energy", "forces"] # Required properties of atoms.

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
    db_corr_name = "../../ZrO2_ref.db"
    yaml_corr_name = f"../../ZrO2_ref_{calc_name}.yaml"

    # Fine-tuning parameters.
    finetune_kwargs = {
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
    atoms_list = get_atoms_list_from_db(db_ase, selection=selection)
    # Filter data with the required properties.
    if required_properties is not None:
        atoms_list = [
            atoms for atoms in atoms_list
            if all(key in atoms.calc.results for key in required_properties)
        ]
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
    calc = finetune_FAIRChem_model(
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