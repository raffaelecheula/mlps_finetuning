# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import shutil
from ase.calculators.vasp import (
    Vasp as VaspOriginal,
    VaspInteractive as VaspInteractiveOriginal,
)

# -------------------------------------------------------------------------------------
# VASP
# -------------------------------------------------------------------------------------

class Vasp(VaspOriginal):
    
    def __init__(
        self,
        directory: str = "vasp",
        filename_yaml: str = None,
        clean_directory: bool = False,
        **kwargs: dict,
    ):
        # Read yaml file.
        if filename_yaml is not None and os.path.isfile(filename_yaml):
            with open(filename_yaml, mode="r") as fileobj:
                kwargs.update(yaml.safe_load(fileobj))
        # Clean directory with previous calculation outputs.
        if clean_directory is True and os.path.isdir(directory):
            shutil.rmtree(directory)
        # Initialize calculator.
        super().__init__(**kwargs, directory=directory)
        self.counter = 0
        self.info = {}
    
    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        self.counter += 1

# -------------------------------------------------------------------------------------
# VASP INTERACTIVE
# -------------------------------------------------------------------------------------

class VaspInteractive(VaspInteractiveOriginal):
    
    def __init__(
        self,
        directory: str = "vasp",
        filename_yaml: str = None,
        clean_directory: bool = False,
        **kwargs: dict,
    ):
        # Read yaml file.
        if filename_yaml is not None and os.path.isfile(filename_yaml):
            with open(filename_yaml, mode="r") as fileobj:
                kwargs.update(yaml.safe_load(fileobj))
        # Clean directory with previous calculation outputs.
        if clean_directory is True and os.path.isdir(directory):
            shutil.rmtree(directory)
        # Initialize calculator.
        super().__init__(**kwargs)
        self.counter = 0
        self.info = {}
    
    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        self.counter += 1

# -------------------------------------------------------------------------------------
# WRITE MODE TS MODECAR VASP
# -------------------------------------------------------------------------------------

def write_mode_TS_modecar_vasp(mode_TS: list):
    """
    Write TS mode to modecar for dimer calculation in vasp.
    """
    with open("MODECAR", "w+") as fileobj:
        for line in mode_TS:
            print("{0:20.10E} {1:20.10E} {2:20.10E}".format(*line), file=fileobj)

# -------------------------------------------------------------------------------------
# WRITE MODE TS POSCAR VASP
# -------------------------------------------------------------------------------------

def write_mode_TS_poscar_vasp(mode_TS: list):
    """
    Write TS mode to poscar for improved dimer calculation in vasp.
    """
    with open("POSCAR", "a") as fileobj:
        print("", file=fileobj)
        for line in mode_TS:
            print("{0:20.10E} {1:20.10E} {2:20.10E}".format(*line), file=fileobj)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------