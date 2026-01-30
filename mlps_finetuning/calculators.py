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
    directory: str = "espresso",
    command: str = "mpirun pw.x",
    pseudo_dir: str = None,
    filename_yaml: str = "espresso.yaml",
    clean_directory: bool = False,
    socket: bool = False,
    basedir: str = "",
    **kwargs,
) -> Calculator:
    """
    Get Quantum Espresso calculator.
    """
    from mlps_finetuning.espresso import Espresso
    # Yaml file.
    if filename_yaml is not None:
        filename_yaml = os.path.join(basedir, filename_yaml)
    # Set up Espresso calculator.
    calc = Espresso(
        directory=directory,
        command=command,
        pseudo_dir=pseudo_dir,
        filename_yaml=filename_yaml,
        clean_directory=clean_directory,
        **kwargs,
    )
    # Socket calculator.
    if socket is True:
        calc = calc.to_socket()
    # Return calculator.
    return calc

# -------------------------------------------------------------------------------------
# GET VASP CALCULATOR
# -------------------------------------------------------------------------------------

def get_VASP_calculator(
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
        from mlps_finetuning.vasp import VaspInteractive
    else:
        from mlps_finetuning.vasp import Vasp
    # Yaml file.
    if filename_yaml is not None:
        filename_yaml = os.path.join(basedir, filename_yaml)
    # Return VASP calculator.
    calc = Vasp(
        directory=directory,
        filename_yaml=filename_yaml,
        **kwargs,
    )
    return calc

# -------------------------------------------------------------------------------------
# GET CALCULATOR
# -------------------------------------------------------------------------------------

def get_calculator(
    calc_name: str,
    model_name: str = None,
    **kwargs: dict,
) -> Calculator:
    """
    Get a calculator.
    """
    # Get model name from calc name.
    if "/" in calc_name:
        calc_name, model_name = calc_name.split("/")
    elif "_" in calc_name:
        calc_name, model_name = calc_name.split("_")
    # CHGNet calculator.
    if calc_name == "CHGNet":
        return get_CHGNet_calculator(model_name=model_name, **kwargs)
    # MACE calculator.
    elif calc_name == "MACE":
        return get_MACE_calculator(model_name=model_name, **kwargs)
    # OCP Calculator.
    elif calc_name == "OCP":
        return get_OCP_calculator(model_name=model_name, **kwargs)
    # FAIRChem Calculator.
    elif calc_name == "FAIRChem":
        return get_FAIRChem_calculator(model_name=model_name, **kwargs)
    # Espresso Calculator.
    elif calc_name == "Espresso":
        return get_Espresso_calculator(**kwargs)
    # VASP Calculator.
    elif calc_name == "VASP":
        return get_VASP_calculator(**kwargs)
    # No match found.
    else:
        raise NameError(f"{calc_name} calculator not found!")

# -------------------------------------------------------------------------------------
# GET FINETUNE FUNCTION
# -------------------------------------------------------------------------------------

def get_finetune_function(
    calc_name: str,
) -> callable:
    """
    Get the fine-tuning function correspondent to a calculator.
    """
    # Get model name from calc name.
    if "/" in calc_name:
        calc_name, model_name = calc_name.split("/")
    elif "_" in calc_name:
        calc_name, model_name = calc_name.split("_")
    # CHGNet calculator.
    if calc_name == "CHGNet":
        from mlps_finetuning.chgnet import finetune_CHGNet_model
        return finetune_CHGNet_model
    # MACE calculator.
    elif calc_name == "MACE":
        from mlps_finetuning.mace import finetune_MACE_model
        return finetune_MACE_model
    # OCP Calculator.
    elif calc_name == "OCP":
        from mlps_finetuning.ocp import finetune_OCP_model
        return finetune_OCP_model
    # FAIRChem Calculator.
    elif calc_name == "FAIRChem":
        from mlps_finetuning.fairchem import finetune_FAIRChem_model
        return finetune_FAIRChem_model
    # No match found.
    else:
        raise NameError(f"Fine-tuning function for {model_name} not found!")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
