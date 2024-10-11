# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import json
import yaml
import numpy as np
from ase.io import read

# -------------------------------------------------------------------------------------
# GET SYMBOLS DICT
# -------------------------------------------------------------------------------------

def get_symbols_dict(atoms):
    """Return a dictionary with the counts of elements in the atoms."""
    symbol_list = atoms.get_chemical_symbols()
    return {
        symbol: symbol_list.count(symbol) for symbol in dict.fromkeys(symbol_list)
    }

# -------------------------------------------------------------------------------------
# CALCULATE ENERGY CORRECTIONS
# -------------------------------------------------------------------------------------

def calculate_energy_corrections(atoms_list, calc):
    """Calculate energy correction per atomic species for finetuning of MLPs."""
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    # Collect data from atoms_list.
    data = {}
    for ii, atoms in enumerate(atoms_list):
        if "energy" not in atoms.calc.results:
            continue
        energy_old = atoms.get_potential_energy()
        atoms_new = atoms.copy()
        atoms_new.calc = calc
        energy_new = atoms_new.get_potential_energy()
        struct_dict = {"deltaE": energy_new-energy_old}
        struct_dict.update(get_symbols_dict(atoms))
        data.update({ii: struct_dict})
    # Create a Dataframe with the data.
    df = pd.DataFrame.from_dict(data=data, orient='index').fillna(value=0.)
    y_dep = df.pop('deltaE').to_numpy()
    X_indep = df.to_numpy()
    elements = df.columns.to_list()
    # Train a linear regression model.
    regr = LinearRegression(fit_intercept=False)
    regr.fit(X_indep, y_dep)
    return {str(elem): float(coeff) for elem, coeff in zip(elements, regr.coef_)}

# -------------------------------------------------------------------------------------
# GET ENERGY CORRECTIONS
# -------------------------------------------------------------------------------------

def get_energy_corrections(db_ref_name, yaml_name, calc):
    """Get energy_corr_dict from ase database or yaml file."""
    if os.path.isfile(yaml_name):
        # Read yaml file.
        with open(yaml_name, 'r') as fileobj:
            energy_corr_dict = yaml.safe_load(fileobj)
    else:
        # Read dft output files.
        energy_corr_dict = calculate_energy_corrections(
            atoms_list=read(db_ref_name, index=":"),
            calc=calc,
        )
        with open(yaml_name, 'w') as fileobj:
            yaml.dump(energy_corr_dict, fileobj)
    return energy_corr_dict

# -------------------------------------------------------------------------------------
# GET CORRECTED ENERGY
# -------------------------------------------------------------------------------------

def get_corrected_energy(atoms, energy_corr_dict, energy=None):
    """Get energy of atoms, corrected with energy_corr_dict."""
    if energy is None:
        energy = atoms.get_potential_energy()
    if energy_corr_dict is not None:
        for elem, num in get_symbols_dict(atoms).items():
            energy += energy_corr_dict[elem]*num
    return energy

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------