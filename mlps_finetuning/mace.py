# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from mace.calculators import mace_anicc, mace_mp, mace_off, mace_omol
from mace.calculators import MACECalculator as MACECalculatorOriginal

from mlps_finetuning.energy_ref import get_corrected_energy

# -------------------------------------------------------------------------------------
# FAIRCHEM CALCULATOR
# -------------------------------------------------------------------------------------

class MACECalculator(MACECalculatorOriginal):

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Counter to keep track of the number of single-point evaluations.
        self.counter = 0
        self.info = {}
    
    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        self.counter += 1

# -------------------------------------------------------------------------------------
# FINETUNE MACE TRAIN VAL
# -------------------------------------------------------------------------------------

def finetune_mace_train_val(
    atoms_list: list,
    directory: str = "finetuning",
    energy_corr_dict: dict = None,
    kwargs_main: dict = {},
    return_calculator: bool = True,
    kwargs_calc: dict = {},
):
    """Finetune MACE model from ase Atoms data."""
    import sys
    import logging
    from ase.io import write
    from mace.cli.run_train import main as mace_main
    # Set up logging.
    #logger = logging.getLogger()
    #if logger.hasHandlers():
    #    logger.handlers.clear()
    #logging.basicConfig(level=logging.INFO)
    # Prepare training file.
    for ii in range(len(atoms_list)):
        atoms_list[ii].info["REF_energy"] = get_corrected_energy(
            atoms=atoms_list[ii],
            energy_corr_dict=energy_corr_dict,
        )
        atoms_list[ii].arrays["REF_forces"] = atoms_list[ii].get_forces()
    # Change directory and write files.
    cwd = os.getcwd()
    os.makedirs(directory, exist_ok=True)
    os.chdir(directory)
    write(filename=kwargs_main["train_file"], images=atoms_list)
    # Set command line arguments.
    argv = [""]
    for key, arg in kwargs_main.items():
        if arg is True:
            argv.append(f"--{key}")
        elif arg not in (False, None):
            argv.append(f"--{key}={arg}")
    print(argv)
    # Train model.
    sys.argv = argv
    mace_main()
    os.chdir(cwd)
    # Path to checkpoint.
    model_name = kwargs_main["name"]
    checkpoint_path = os.path.join(directory, f"{model_name}.model")
    # Return calculator.
    if return_calculator:
        return MACECalculator(model_paths=[checkpoint_path], **kwargs_calc)
    else:
        return checkpoint_path

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------