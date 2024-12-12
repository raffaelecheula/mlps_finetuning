# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer
from ase.db.core import Database
from ase.optimize import BFGS
from ase.formula import Formula

from mlps_finetuning.energy_ref import get_corrected_energy
from mlps_finetuning.databases import get_atoms_from_db

# -------------------------------------------------------------------------------------
# OPTIMIZE ATOMS
# -------------------------------------------------------------------------------------

def optimize_atoms(
    atoms: Atoms,
    calc: Calculator,
    energy_corr_dict: dict,
    optimizer: Optimizer = BFGS,
    fmax: float = 0.05,
    steps: int = 500,
):
    """Optimize atoms."""
    print("Relax: " + " ".join([
        atoms.info["class"], atoms.info["surface"], atoms.info["species"]
    ]))
    atoms.calc = calc
    opt = optimizer(atoms)
    opt.run(fmax=fmax, steps=steps)
    energy = get_corrected_energy(
        atoms=atoms,
        energy_corr_dict=energy_corr_dict,
        reverse=True,
    )
    atoms.calc.results["energy"] = energy

# -------------------------------------------------------------------------------------
# GET REFERENCE ENERGIES
# -------------------------------------------------------------------------------------

def get_reference_energies(
    references_surf: list,
    references_gas: list,
    calc: Calculator,
    db_ase: Database,
    energy_corr_dict: dict,
    kwargs_init: dict = {"index": 0},
    calculate_ref_clean: bool = True,
    calculate_ref_gas: bool = True,
) -> tuple:
    """Get energy references dictionaries."""
    # Calculate energy of reference surfaces.
    energies_ref = {}
    energies_ref_dft = {}
    compositions_ref = {}
    for surface in references_surf:
        kwargs_match = {"surface": surface, "species": "clean"}
        atoms = get_atoms_from_db(db_ase, **kwargs_init, **kwargs_match)
        atoms_dft = get_atoms_from_db(db_ase, uid=atoms.info["uid"], relaxed=True)
        energies_ref_dft[surface] = atoms_dft.get_potential_energy()
        compositions_ref[surface] = Formula(atoms.get_chemical_formula()).count()
        if calculate_ref_clean is True:
            optimize_atoms(
                atoms=atoms,
                calc=calc,
                energy_corr_dict=energy_corr_dict,
            )
            energies_ref[surface] = atoms.get_potential_energy()
        else:
            energies_ref[surface] = atoms_dft.get_potential_energy()
    # Calculate energy of reference gas.
    atoms_list = []
    atoms_dft_list = []
    for molecule in references_gas:
        kwargs_match = {"surface": "none", "species": molecule}
        atoms = get_atoms_from_db(db_ase=db_ase, **kwargs_init, **kwargs_match)
        atoms_dft = get_atoms_from_db(db_ase, uid=atoms.info["uid"], relaxed=True)
        atoms_dft_list.append(atoms_dft)
        if calculate_ref_gas is True:
            optimize_atoms(
                atoms=atoms,
                calc=calc,
                energy_corr_dict=energy_corr_dict,
            )
            atoms_list.append(atoms)
        else:
            atoms_list.append(atoms_dft)
    # Update energy references dictionaries.
    energies_ref.update(get_elem_energies_ref(atoms_list=atoms_list))
    energies_ref_dft.update(get_elem_energies_ref(atoms_list=atoms_dft_list))
    # Return energy references dictionary.
    return energies_ref, energies_ref_dft, compositions_ref

# -------------------------------------------------------------------------------------
# GET ELEM ENERGIES REF
# -------------------------------------------------------------------------------------

def get_elem_energies_ref(atoms_list: list) -> dict:
    """Get the reference energies of a set of linearly independent species."""
    energies = [atoms.get_potential_energy() for atoms in atoms_list]
    formulas = [Formula(atoms.get_chemical_formula()).count() for atoms in atoms_list]
    elements = list({elem: None for form in formulas for elem in form})
    comp_matrix = [[form.get(elem, 0) for elem in elements] for form in formulas]
    inv_matrix = np.linalg.inv(comp_matrix)
    energies_ref = np.dot(inv_matrix, energies)
    energies_ref = {elem: float(energies_ref[ii]) for ii, elem in enumerate(elements)}
    return energies_ref

# -------------------------------------------------------------------------------------
# GET FORMATION ENERGY
# -------------------------------------------------------------------------------------

def get_formation_energy(
    atoms: Atoms,
    energies_ref: dict,
    compositions_ref: dict,
) -> float:
    """Get the formation energy of a structure."""
    composition = Formula(atoms.get_chemical_formula()).count()
    composition_ref = compositions_ref[atoms.info["surface"]]
    elements = {elem: None for elem in composition}
    elements.update({elem: None for elem in composition_ref})
    composition_species = {
        elem: composition.get(elem, 0) - composition_ref.get(elem, 0)
        for elem in elements
    }
    composition_species = {
        elem: count for elem, count in composition_species.items() if count != 0
    }
    e_form = atoms.get_potential_energy()
    for elem, count in composition_species.items():
        e_form -= count * energies_ref[elem]
    return e_form

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------