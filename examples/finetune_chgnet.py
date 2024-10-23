# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import json
import yaml
import numpy as np
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.data.dataset import StructureData, get_train_val_test_loader
from chgnet.model.dynamics import CHGNetCalculator

from mlps_finetuning.energy_ref import get_energy_corrections
from mlps_finetuning.chgnet import finetune_chgnet_train_val
from mlps_finetuning.databases import get_atoms_list_from_db

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # DFT database.
    db_dft_name = "ZrO2_dft.db"

    # Reference database.
    db_ref_name = "ZrO2_ref.db"
    yaml_name = "ZrO2_ref_chgnet.yaml"

    # Get energy corrections.
    calc = CHGNetCalculator()
    energy_corr_dict = get_energy_corrections(
        db_ref_name=db_ref_name,
        yaml_name=yaml_name,
        calc=calc,
    )
    
    selection = ""
    atoms_list = get_atoms_list_from_db(db_ase=db_dft_name, selection=selection)
    
    # Run finetuning.
    finetune_chgnet_train_val(
        atoms_list=atoms_list,
        energy_corr_dict=energy_corr_dict,
        targets="efm",
        batch_size=8,
        train_ratio=0.80,
        val_ratio=0.10,
        optimizer="Adam",
        scheduler="CosLR",
        criterion="MSE",
        epochs=10,
        learning_rate=1e-3,
        use_device="cpu",
        print_freq=10,
        wandb_path="chgnet/finetune",
        save_dir="run_0",
        train_composition_model=False,
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------