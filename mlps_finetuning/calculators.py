# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator

# -------------------------------------------------------------------------------------
# DEFAULT NAMES DICTIONARIES
# -------------------------------------------------------------------------------------

# Aliases for CHGNet model names.
aliases_CHGNet = {
    "MPtrj": "0.3.0",
    "R2SCAN": "r2scan",
}
# Aliases for MACE model names.
aliases_MACE = {
    "MP-0": "medium",
    "MP-0b": "medium-0b",
    "MP-0b2": "medium-0b2",
    "MP-0b3": "medium-0b3",
    "MPA-0": "medium-mpa-0",
}
# Aliases for OCP model names.
aliases_OCP = {
    "GemNet-OC": "GemNet-OC-S2EF-OC20-All",
    "PaiNN": "PaiNN-S2EF-OC20-All",
    "SCN": "SCN-S2EF-OC20-All+MD",
    "EquiformerV2": "EquiformerV2-31M-S2EF-OC20-All+MD",
    "eSCN": "eSCN-L6-M3-Lay20-S2EF-OC20-All+MD",
    "eSEN": "eSEN-30M-OAM",
}
# Aliases for FAIRChem model names.
aliases_FAIRChem = {
    "UMA-s": "uma-s-1p1",
    "UMA-m": "uma-m-1p1",
    "eSEN-s": "esen-sm-conserving-all-oc25",
    "eSEN-m": "esen-md-direct-all-oc25",
}

# -------------------------------------------------------------------------------------
# GET CHGNET CALCULATOR
# -------------------------------------------------------------------------------------

def get_CHGNet_calculator(
    model_name: str,
    atoms: Atoms = None,
    **kwargs: dict,
) -> Calculator:
    """
    Get CHGNet calculator.
    """
    from mlps_finetuning.chgnet import CHGNetCalculator
    # Substitute model name.
    if model_name in aliases_CHGNet:
        model_name = aliases_CHGNet[model_name]
    # Default kwargs.
    kwargs = {
        "model_name": model_name,
        "use_device": "cuda" if torch.cuda.is_available() else "cpu",
        "verbose": False,
        **kwargs,
    }
    # Return calculator.
    return CHGNetCalculator(**kwargs)

# -------------------------------------------------------------------------------------
# GET MACE CALCULATOR
# -------------------------------------------------------------------------------------

def get_MACE_calculator(
    model_name: str,
    atoms: Atoms = None,
    logfile: str = None,
    **kwargs: dict,
) -> Calculator:
    """
    Get MACE calculator.
    """
    from mlps_finetuning.mace import MACECalculator
    # Substitute model name.
    if model_name in aliases_MACE:
        model_name = aliases_MACE[model_name]
    # Default kwargs.
    kwargs = {
        "model_name": model_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    }
    # Return calculator.
    return MACECalculator(**kwargs)

# -------------------------------------------------------------------------------------
# GET OCP CALCULATOR
# -------------------------------------------------------------------------------------

def get_OCP_calculator(
    model_name: str,
    atoms: Atoms = None,
    logfile: str = None,
    **kwargs: dict,
) -> Calculator:
    """
    Get OCP calculator.
    """
    from mlps_finetuning.ocp import OCPCalculator
    # Substitute model name.
    if model_name in aliases_OCP:
        model_name = aliases_OCP[model_name]
    # Default kwargs.
    kwargs = {
        "model_name": model_name,
        "local_cache": os.getenv("PRETRAINED_MODELS", "."),
        "cpu": False if torch.cuda.is_available() else True,
        "seed": 42,
        **kwargs,
    }
    # Return calculator.
    return OCPCalculator(**kwargs)

# -------------------------------------------------------------------------------------
# GET FAIRCHEM CALCULATOR
# -------------------------------------------------------------------------------------

def get_FAIRChem_calculator(
    model_name: str,
    atoms: Atoms = None,
    logfile: str = None,
    **kwargs: dict,
) -> Calculator:
    """
    Get FAIRChem calculator.
    """
    from mlps_finetuning.fairchem import FAIRChemCalculator
    # Substitute model name.
    if model_name in aliases_FAIRChem:
        model_name = aliases_FAIRChem[model_name]
    # Default kwargs.
    kwargs = {
        "model_name": model_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cache_dir": os.getenv("PRETRAINED_MODELS", "."),
        "task_name": "oc20",
        **kwargs,
    }
    # Return calculator.
    return FAIRChemCalculator(**kwargs)

# -------------------------------------------------------------------------------------
# GET ESPRESSO CALCULATOR
# -------------------------------------------------------------------------------------

def get_Espresso_calculator(
    atoms: Atoms,
    directory: str = "espresso",
    filename_yaml: str = "espresso.yaml",
    pseudo_dir: str = None,
    command: str = "mpirun pw.x",
    clean_directory: bool = False,
    basedir: str = "",
    **kwargs,
) -> Calculator:
    """
    Get Quantum Espresso calculator.
    """
    from ase.calculators.espresso import EspressoProfile, Espresso
    # Read yaml file.
    if filename_yaml is not None:
        import yaml
        filepath = os.path.join(basedir, filename_yaml)
        if os.path.isfile(filepath):
            with open(filepath, mode="r") as fileobj:
                kwargs.update(yaml.safe_load(fileobj))
    # Get pseudopotentials names.
    if kwargs.get("pseudopotentials", "auto") == "auto":
        from qe_toolkit.io import get_pseudopotentials_names
        kwargs["pseudopotentials"] = get_pseudopotentials_names()
    # Get pseudopotentials directory.
    if pseudo_dir is None:
        pseudo_dir = os.getenv("ESPRESSO_PSEUDO", ".")
    # Remove previous calculaton folder.
    if clean_directory is True and os.path.isdir(directory):
        import shutil
        shutil.rmtree(directory)
    # Return Quantum Espresso calculator.
    profile = EspressoProfile(command=command, pseudo_dir=pseudo_dir)
    calc = Espresso(profile=profile, directory=directory, **kwargs)
    calc.counter = 0
    calc.info = {}
    return calc

# -------------------------------------------------------------------------------------
# GET VASP CALCULATOR
# -------------------------------------------------------------------------------------

def get_VASP_calculator(
    atoms: Atoms,
    directory: str = "vasp",
    filename_yaml: str = "vasp.yaml",
    clean_directory: bool = False,
    interactive: bool = False,
    basedir: str = "",
    **kwargs,
) -> Calculator:
    """
    Get VASP calculator.
    """
    if interactive is True:
        from ase.calculators.vasp import VaspInteractive as Vasp
    else:
        from ase.calculators.vasp import Vasp
    # Read yaml file.
    if filename_yaml is not None:
        import yaml
        filepath = os.path.join(basedir, filename_yaml)
        if os.path.isfile(filepath):
            with open(filepath, mode="r") as fileobj:
                kwargs.update(yaml.safe_load(fileobj))
    # Remove previous calculaton folder.
    if clean_directory is True and os.path.isdir(directory):
        import shutil
        shutil.rmtree(directory)
    # Return VASP calculator.
    calc = Vasp(directory=directory, **kwargs)
    calc.counter = 0
    calc.info = {}
    return calc

# -------------------------------------------------------------------------------------
# GET CALCULATOR
# -------------------------------------------------------------------------------------

def get_calculator(
    calc_name: str,
    model_name: str = None,
    atoms: Atoms = None,
    **kwargs: dict,
) -> Calculator:
    """
    Get a calculator.
    """
    # Get model name from calc name.
    if "/" in calc_name:
        calc_name, model_name = model_name.split("/")
    # CHGNet calculator.
    if calc_name == "CHGNet":
        return get_CHGNet_calculator(model_name=model_name, atoms=atoms, **kwargs)
    # MACE calculator.
    elif calc_name == "MACE":
        return get_MACE_calculator(model_name=model_name, atoms=atoms, **kwargs)
    # OCP Calculator.
    elif calc_name == "OCP":
        return get_OCP_calculator(model_name=model_name, atoms=atoms, **kwargs)
    # FAIRChem Calculator.
    elif calc_name == "FAIRChem":
        return get_FAIRChem_calculator(model_name=model_name, atoms=atoms, **kwargs)
    # Espresso Calculator.
    elif calc_name == "Espresso":
        return get_Espresso_calculator(atoms=atoms, **kwargs)
    # VASP Calculator.
    elif calc_name == "VASP":
        return get_VASP_calculator(atoms=atoms, **kwargs)
    # No match found.
    else:
        raise NameError(f"{model_name} calculator not found!")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
