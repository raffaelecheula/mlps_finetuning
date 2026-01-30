# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import json
import yaml
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read

# -------------------------------------------------------------------------------------
# GET SYMBOLS DICT
# -------------------------------------------------------------------------------------

def get_symbols_dict(atoms: Atoms) -> dict:
    """
    Return a dictionary with the counts of elements in the atoms.
    """
    symbol_list = atoms.get_chemical_symbols()
    return {
        symbol: int(symbol_list.count(symbol)) for symbol in dict.fromkeys(symbol_list)
    }

# -------------------------------------------------------------------------------------
# CALCULATE ENERGY CORRECTIONS
# -------------------------------------------------------------------------------------

def calculate_energy_corrections(
    atoms_list: list,
    calc: Calculator,
    fit_first: list = [],
    energy_corr_first: dict = {},
) -> dict:
    """
    Calculate energy correction per atomic species for fine-tuning of MLPs.
    """
    from sklearn.linear_model import LinearRegression
    # Screen atoms with energy.
    atoms_list = [atoms for atoms in atoms_list if "energy" in atoms.calc.results]
    # Fit first the parameters of some elements.
    fit_first = fit_first or []
    if fit_first:
        atoms_first = [
            atoms for atoms in atoms_list 
            if all(key in fit_first for key in get_symbols_dict(atoms=atoms))
        ]
        energy_corr_first = calculate_energy_corrections(
            atoms_list=atoms_first,
            calc=calc,
            fit_first=None,
        )
        atoms_list = [atoms for atoms in atoms_list if atoms not in atoms_first]
    # Collect data from atoms_list.
    delta_energies = []
    symbols_dicts = []
    for atoms in atoms_list:
        # Get symbols and energies.
        symbols = get_symbols_dict(atoms=atoms)
        energy_old = atoms.get_potential_energy()
        atoms_new = atoms.copy()
        atoms_new.calc = calc
        energy_new = atoms_new.get_potential_energy()
        # Apply first corrections.
        if energy_corr_first:
            for elem, coeff in energy_corr_first.items():
                energy_new -= symbols.pop(elem, 0) * coeff
        # Calculate energy difference.
        delta_energies.append(energy_new - energy_old)
        symbols_dicts.append(symbols)
    # Get list of all symbols.
    symbols_all = list({key: None for symbols in symbols_dicts for key in symbols})
    # Get lists of symbols for each atoms structure.
    symbols_lists = [
        [symbols.get(key, 0) for key in symbols_all] for symbols in symbols_dicts
    ]
    # Train a linear regression model.
    regr = LinearRegression(fit_intercept=False)
    regr.fit(X=symbols_lists, y=delta_energies)
    # Get energy correction dictionary.
    energy_corr_dict = {
        **energy_corr_first,
        **{str(elem): float(coeff) for elem, coeff in zip(symbols_all, regr.coef_)}
    }
    return energy_corr_dict

# -------------------------------------------------------------------------------------
# GET ENERGY CORRECTIONS
# -------------------------------------------------------------------------------------

def get_energy_corrections(
    atoms_list: list,
    calc: Calculator,
    yaml_corr_name: str = None,
    fit_first: list = [],
    train_on_relaxed: bool = False,
) -> dict:
    """
    Get energy corrections dictionary from ase database or yaml file.
    """
    # Train only on relaxed structures.
    if train_on_relaxed is True:
        atoms_list = [atoms for atoms in atoms_list if atoms.info["relaxed"] is True]
    # Get energy corrections dictionary.
    if yaml_corr_name is not None and os.path.isfile(yaml_corr_name):
        # Read yaml file.
        with open(yaml_corr_name, "r") as fileobj:
            energy_corr_dict = yaml.safe_load(fileobj)
    else:
        # Read dft output files.
        energy_corr_dict = calculate_energy_corrections(
            atoms_list=atoms_list,
            calc=calc,
            fit_first=fit_first,
        )
        # Write yaml file.
        if yaml_corr_name is not None:
            # Custom yaml representer for floats.
            def float_representer(dumper, value):
                return dumper.represent_scalar("tag:yaml.org,2002:float", repr(value))
            yaml.add_representer(float, float_representer)
            with open(yaml_corr_name, "w") as fileobj:
                yaml.dump(energy_corr_dict, fileobj)
    # Return energy corrections dictionary.
    return energy_corr_dict

# -------------------------------------------------------------------------------------
# GET CORRECTED ENERGY
# -------------------------------------------------------------------------------------

def get_corrected_energy(
    atoms: Atoms,
    energy_corr_dict: dict,
    reverse: bool = False,
    energy: float = None,
) -> float:
    """
    Get energy of atoms, corrected with energy_corr_dict.
    """
    if energy is None:
        energy = atoms.get_potential_energy()
    if energy_corr_dict is not None:
        for elem, num in get_symbols_dict(atoms=atoms).items():
            if reverse is True:
                energy -= energy_corr_dict[elem] * num
            else:
                energy += energy_corr_dict[elem] * num
    return float(energy)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------