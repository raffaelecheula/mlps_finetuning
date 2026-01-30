# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import uuid
import yaml
import shutil
from ase.calculators.espresso import EspressoProfile, Espresso as EspressoOriginal
from ase.calculators.socketio import SocketIOCalculator as SocketIOCalculatorOriginal

# -------------------------------------------------------------------------------------
# ESPRESSO
# -------------------------------------------------------------------------------------

class Espresso(EspressoOriginal):
    
    def __init__(
        self,
        directory: str = "espresso",
        command: str = "mpirun pw.x",
        pseudo_dir: str = None,
        filename_yaml: str = None,
        pseudopotentials: str = "auto",
        pseudo_yaml: str = "pseudopotentials.yaml",
        clean_directory: bool = False,
        **kwargs: dict,
    ):
        # Get pseudopotentials directory.
        if pseudo_dir is None:
            pseudo_dir = os.getenv("ESPRESSO_PSEUDO", ".")
        # Get pseudopotentials names.
        if pseudopotentials == "auto":
            pseudopotentials = get_pseudo_names(
                pseudo_dir=pseudo_dir,
                pseudo_yaml=pseudo_yaml,
            )
        # Read yaml file.
        if filename_yaml is not None and os.path.isfile(filename_yaml):
            with open(filename_yaml, mode="r") as fileobj:
                kwargs.update(yaml.safe_load(fileobj))
        # Clean directory with previous calculation outputs.
        if clean_directory is True and os.path.isdir(directory):
            shutil.rmtree(directory)
        # Espresso profile.
        profile = EspressoProfile(command=command, pseudo_dir=pseudo_dir)
        # Initialize calculator.
        super().__init__(
            profile=profile,
            directory=directory,
            pseudopotentials=pseudopotentials,
            **kwargs,
        )
        self.counter = 0
        self.info = {}
    
    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        self.counter += 1

    def to_socket(self, unixsocket: str = f"ase-qe-{uuid.uuid4().hex}"):
        calc_tmp = self.socketio(unixsocket=unixsocket)
        return SocketIOCalculator(
            launch_client=calc_tmp.launch_client,
            unixsocket=calc_tmp._unixsocket,
        )

# -------------------------------------------------------------------------------------
# SOCKET IO CALCULATOR
# -------------------------------------------------------------------------------------

class SocketIOCalculator(SocketIOCalculatorOriginal):
    
    def __init__(self, **kwargs: dict):
        super().__init__(**kwargs)
        self.counter = 0
        self.info = {}
    
    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        self.counter += 1

# -------------------------------------------------------------------------------------
# GET PSEUDO NAMES
# -------------------------------------------------------------------------------------

def get_pseudo_names(
    pseudo_dir: str,
    pseudo_yaml: str = "pseudopotentials.yaml",
) -> dict:
    """
    Get pseudopotentials dictionary.
    """
    if not os.path.isfile(os.path.join(pseudo_dir, pseudo_yaml)):
        # Create pseudopotentials yaml file.
        return read_pseudo_names(pseudo_dir=pseudo_dir, pseudo_yaml=pseudo_yaml)
    else:
        # Read pseudopotentials yaml file.
        return yaml.safe_load(open(os.path.join(pseudo_dir, pseudo_yaml)))

# -------------------------------------------------------------------------------------
# READ PSEUDO NAMES
# -------------------------------------------------------------------------------------

def read_pseudo_names(
    pseudo_dir: str,
    pseudo_yaml: str = "pseudopotentials.yaml",
) -> dict:
    """
    Read pseudopotentials from directory and save them in a yaml file.
    """
    import re
    pseudopotentials = {}
    for filename in os.listdir(pseudo_dir):
        if filename.lower().endswith(".upf"):
            element = re.split(r"[_.-]", filename, maxsplit=1)[0].capitalize()
            pseudopotentials[element] = filename
    # Save pseudopotentials in yaml file.
    if pseudo_yaml is not None:
        with open(os.path.join(pseudo_dir, pseudo_yaml), mode="w") as fileobj:
            yaml.safe_dump(pseudopotentials, fileobj)
    # Return pseudopotentials dictionary.
    return pseudopotentials

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------