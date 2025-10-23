# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator, all_properties
from ase.optimize.optimize import Optimizer
from ase.db.core import Database
from ase.optimize import BFGS
from ase.formula import Formula
from sklearn.model_selection import BaseCrossValidator

from mlps_finetuning.energy_ref import get_corrected_energy
from mlps_finetuning.databases import get_atoms_list_from_db, get_atoms_from_db

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
    keys_print: list = ["class", "surface", "species"],
):
    """
    Optimize atoms.
    """
    print("Relax: " + " - ".join([atoms.info[key] for key in keys_print]))
    # Relax structure.
    atoms.calc = calc
    opt = optimizer(atoms)
    opt.run(fmax=fmax, steps=steps)
    # Get corrected energy.
    atoms.calc.results["energy"] = get_corrected_energy(
        atoms=atoms,
        energy_corr_dict=energy_corr_dict,
        reverse=True,
    )
    # Update calculator.
    results = {
        pp: atoms.calc.results[pp] for pp in atoms.calc.results
        if pp in all_properties
    }
    atoms.calc = SinglePointCalculator(atoms=atoms, **results)

# -------------------------------------------------------------------------------------
# GET REFERENCE ENERGIES ADSORBATES
# -------------------------------------------------------------------------------------

def get_reference_energies_adsorbates(
    atoms_list: list,
    references_gas: list,
    calc: Calculator,
    db_ase: Database,
    energy_corr_dict: dict,
    kwargs_init: dict = {"index": 0},
    calculate_ref_clean: bool = True,
    calculate_ref_gas: bool = True,
) -> tuple:
    """
    Get energy references dictionaries.
    """
    # Calculate energy of reference surfaces.
    references_surf = list({atoms.info["surface"]: None for atoms in atoms_list})
    energies_ref = {}
    energies_ref_dft = {}
    compositions_ref = {}
    for surface in references_surf:
        kwargs_match = kwargs_init.copy()
        kwargs_match.update({"surface": surface, "species": "clean"})
        atoms = get_atoms_from_db(db_ase, **kwargs_match)
        atoms_dft = get_atoms_from_db(db_ase, uid=atoms.info["uid"], relaxed=True)
        energies_ref_dft[surface] = atoms_dft.get_potential_energy()
        compositions_ref[surface] = Formula(atoms.get_chemical_formula()).count()
        if calculate_ref_clean is True:
            optimize_atoms(
                atoms=atoms,
                calc=calc,
                energy_corr_dict=energy_corr_dict,
            )
            energy = atoms.get_potential_energy()
            energy_dft = atoms_dft.get_potential_energy()
            print_energies(energy, energy_dft)
            energies_ref[surface] = atoms.get_potential_energy()
        else:
            energies_ref[surface] = atoms_dft.get_potential_energy()
    # Calculate energy of reference gas.
    atoms_list = []
    atoms_dft_list = []
    for molecule in references_gas:
        kwargs_match = kwargs_init.copy()
        kwargs_match.update({"surface": "none", "species": molecule})
        atoms = get_atoms_from_db(db_ase=db_ase, **kwargs_match)
        atoms_dft = get_atoms_from_db(db_ase, uid=atoms.info["uid"], relaxed=True)
        atoms_dft_list.append(atoms_dft)
        if calculate_ref_gas is True:
            optimize_atoms(
                atoms=atoms,
                calc=calc,
                energy_corr_dict=energy_corr_dict,
            )
            energy = atoms.get_potential_energy()
            energy_dft = atoms_dft.get_potential_energy()
            print_energies(energy, energy_dft)
            atoms_list.append(atoms)
        else:
            atoms_list.append(atoms_dft)
    # Update energy references dictionaries.
    energies_ref.update(get_elem_energies_ref(atoms_list=atoms_list))
    energies_ref_dft.update(get_elem_energies_ref(atoms_list=atoms_dft_list))
    # Return energy references dictionary.
    return energies_ref, energies_ref_dft, compositions_ref

# -------------------------------------------------------------------------------------
# PRINT ENERGIES
# -------------------------------------------------------------------------------------

def print_energies(
    energy: float,
    energy_dft: float,
    e_form: float = None,
    e_form_dft: float = None,
):
    """
    Print energies.
    """
    print("-"*82)
    print(" | ".join([
        "Energy",
        f"{energy:+12.3f} eV (MLP)",
        f"{energy_dft:+12.3f} eV (DFT)",
        f"{energy-energy_dft:+12.3f} eV (MLP-DFT)",
    ]))
    if e_form is not None and e_form_dft is not None:
        print(" | ".join([
            "E_form",
            f"{e_form:+12.3f} eV (MLP)",
            f"{e_form_dft:+12.3f} eV (DFT)",
            f"{e_form-e_form_dft:+12.3f} eV (MLP-DFT)",
        ]))
    print("-"*82)

# -------------------------------------------------------------------------------------
# GET ELEM ENERGIES REF
# -------------------------------------------------------------------------------------

def get_elem_energies_ref(atoms_list: list) -> dict:
    """
    Get the reference energies of a set of linearly independent species.
    """
    energies = [atoms.get_potential_energy() for atoms in atoms_list]
    formulas = [Formula(atoms.get_chemical_formula()).count() for atoms in atoms_list]
    elements = list({elem: None for form in formulas for elem in form})
    comp_matrix = [[form.get(elem, 0) for elem in elements] for form in formulas]
    inv_matrix = np.linalg.inv(comp_matrix)
    energies_ref = np.dot(inv_matrix, energies)
    energies_ref = {elem: float(energies_ref[ii]) for ii, elem in enumerate(elements)}
    return energies_ref

# -------------------------------------------------------------------------------------
# GET FORMATION ENERGY ADSORBATE
# -------------------------------------------------------------------------------------

def get_formation_energy_adsorbate(
    atoms: Atoms,
    energies_ref: dict,
    compositions_ref: dict,
) -> float:
    """
    Get the formation energy of a structure.
    """
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
    e_form -= energies_ref[atoms.info["surface"]]
    return float(e_form)

# -------------------------------------------------------------------------------------
# GET CROSSVALIDATOR
# -------------------------------------------------------------------------------------

def get_crossvalidator(
    stratified: bool = False,
    group: bool = False,
    n_splits: int = 5,
    random_state: int = None,
    shuffle: bool = True,
) -> BaseCrossValidator:
    """
    Get Scikit-Learn crossvalidator.
    """
    from sklearn.model_selection import (
        KFold,
        StratifiedKFold,
        GroupKFold,
        StratifiedGroupKFold,
    )
    if (stratified, group) == (False, False):
        crossval_class = KFold
    elif (stratified, group) == (True, False):
        crossval_class = StratifiedKFold
    elif (stratified, group) == (False, True):
        crossval_class = GroupKFold
    elif (stratified, group) == (True, True):
        crossval_class = StratifiedGroupKFold
    # Return crossvalidator.
    return crossval_class(
        n_splits=n_splits,
        random_state=random_state,
        shuffle=shuffle
    )

# -------------------------------------------------------------------------------------
# CROSS VALIDATION WITH OPTIMIZATION
# -------------------------------------------------------------------------------------

def cross_validation_with_optimization(
    atoms_init: list,
    db_ase: Database,
    key_groups: str,
    key_stratify: str,
    finetune_mlp_fun: callable,
    calc: Calculator,
    crossval: BaseCrossValidator,
    kwargs_trainer: dict,
    energy_corr_dict: dict,
    finetuning: bool = False,
    atoms_train_added: list = [],
    formation_energies: bool = False,
    formation_energy_fun: callable = None,
    ref_energies_fun: callable = None,
    ref_energies_kwargs: dict = {},
    required_properties: list = ["energy", "forces"],
) -> dict:
    """
    Run cross-validation with optimization.
    """
    from sklearn.metrics import mean_absolute_error
    # Get list of all atoms structures from database.
    atoms_all = get_atoms_list_from_db(db_ase)
    atoms_all = [
        atoms for atoms in atoms_all
        if set(required_properties).issubset(atoms.calc.results)
    ]
    # Loop over splits.
    energy_true = []
    energy_pred = []
    e_form_pred = []
    e_form_true = []
    atoms_pred = []
    # Get indices, groups, and stratify lists.
    indices = list(range(len(atoms_init)))
    if key_groups is not None:
        groups = [atoms.info[key_groups] for atoms in atoms_init]
    else:
        groups = None
    if key_stratify is not None:
        stratify = [atoms.info[key_stratify] for atoms in atoms_init]
    else:
        stratify = None
    # Cross-validation.
    for ii, (indices_train, indices_test) in enumerate(
        crossval.split(X=indices, y=stratify, groups=groups)
    ):
        # Get training and test sets.
        uid_train = [atoms_init[ii].info["uid"] for ii in indices_train]
        atoms_train = [atoms for atoms in atoms_all if atoms.info["uid"] in uid_train]
        atoms_train += atoms_train_added
        atoms_test = np.array(atoms_init, dtype=object)[indices_test]
        # Run finetuning on train set.
        if finetuning is True:
            calc = finetune_mlp_fun(
                atoms_list=atoms_train,
                energy_corr_dict=energy_corr_dict,
                **kwargs_trainer,
            )
        # Get reference energies.
        if formation_energies is True:
            energies_ref, energies_ref_dft, compositions_ref = ref_energies_fun(
                atoms_list=atoms_test,
                calc=calc,
                db_ase=db_ase,
                energy_corr_dict=energy_corr_dict,
                **ref_energies_kwargs,
            )
        # Relax structures.
        for atoms in atoms_test:
            # Optimize structure and get MLP energy.
            optimize_atoms(
                atoms=atoms,
                calc=calc,
                energy_corr_dict=energy_corr_dict,
            )
            # Get DFT energy.
            atoms_dft = get_atoms_from_db(db_ase, uid=atoms.info["uid"], relaxed=True)
            energy = atoms.get_potential_energy()
            energy_dft = atoms_dft.get_potential_energy()
            # Store results.
            energy_pred.append(energy)
            energy_true.append(energy_dft)
            atoms.info["energy"] = energy
            atoms.info["energy_dft"] = energy_dft
            # Get formation energies.
            if formation_energies is True:
                # Get MLP formation energy.
                e_form = formation_energy_fun(
                    atoms=atoms,
                    energies_ref=energies_ref,
                    compositions_ref=compositions_ref,
                )
                # Get DFT formation energy.
                e_form_dft = formation_energy_fun(
                    atoms=atoms_dft,
                    energies_ref=energies_ref_dft,
                    compositions_ref=compositions_ref,
                )
                # Store results.
                e_form_pred.append(e_form)
                e_form_true.append(e_form_dft)
                atoms.info["e_form"] = e_form
                atoms.info["e_form_dft"] = e_form_dft
            else:
                e_form = None
                e_form_dft = None
            # Print energies.
            print_energies(
                energy=energy,
                energy_dft=energy_dft,
                e_form=e_form,
                e_form_dft=e_form_dft,
            )
            # Append atoms to list.
            atoms_pred.append(atoms)
    # Calculate average errors.
    mae_energy = mean_absolute_error(y_true=energy_true, y_pred=energy_pred)
    print(f"MAE Energy: {mae_energy:.3f} eV")
    if formation_energies is True:
        mae_e_form = mean_absolute_error(y_true=e_form_true, y_pred=e_form_pred)
        print(f"MAE E_form: {mae_e_form:.3f} eV")
    # Return list of atoms.
    results = {
        "atoms_pred": atoms_pred,
        "energy_true": energy_true,
        "energy_pred": energy_pred,
        "e_form_true": e_form_true,
        "e_form_pred": e_form_pred,
    }
    return results

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------