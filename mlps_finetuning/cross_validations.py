# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize.optimize import Optimizer
from ase.db.core import Database
from ase.optimize import BFGS
from ase.formula import Formula
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import mean_absolute_error

from mlps_finetuning.utilities import filter_results, filter_atoms_list, print_title
from mlps_finetuning.energy_ref import get_energy_corrections, get_corrected_energy
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
    # Print title.
    title = "Relax: " + " - ".join([atoms.info[key] for key in keys_print])
    print_title(string=title, width=84)
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
    results = filter_results(atoms.calc.results)
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
    kwargs_surface: dict = {"species": "clean"},
    kwargs_molecule: dict = {"surface": "none"},
    calculate_ref_clean: bool = True,
    calculate_ref_gas: bool = True,
) -> tuple:
    """
    Get energy references dictionaries.
    """
    # Calculate energy of reference surfaces.
    references_surf = list({atoms.info["surface"]: None for atoms in atoms_list})
    energies_ref = {}
    energies_ref_DFT = {}
    compositions_ref = {}
    for surface in references_surf:
        kwargs_match = {**kwargs_init, "surface": surface, **kwargs_surface}
        atoms = get_atoms_from_db(db_ase, **kwargs_match)
        atoms_DFT = get_atoms_from_db(db_ase, uid=atoms.info["uid"], relaxed=True)
        energies_ref_DFT[surface] = atoms_DFT.get_potential_energy()
        compositions_ref[surface] = Formula(atoms.get_chemical_formula()).count()
        if calculate_ref_clean is True:
            optimize_atoms(
                atoms=atoms,
                calc=calc,
                energy_corr_dict=energy_corr_dict,
            )
            energy_MLP = atoms.get_potential_energy()
            energy_DFT = atoms_DFT.get_potential_energy()
            print_energies(energy_MLP=energy_MLP, energy_DFT=energy_DFT)
            energies_ref[surface] = atoms.get_potential_energy()
        else:
            energies_ref[surface] = atoms_DFT.get_potential_energy()
    # Calculate energy of reference gas.
    atoms_list = []
    atoms_list_DFT = []
    for molecule in references_gas:
        kwargs_match = {**kwargs_init, "species": molecule, **kwargs_molecule}
        atoms = get_atoms_from_db(db_ase=db_ase, **kwargs_match)
        atoms_DFT = get_atoms_from_db(db_ase, uid=atoms.info["uid"], relaxed=True)
        atoms_list_DFT.append(atoms_DFT)
        if calculate_ref_gas is True:
            optimize_atoms(
                atoms=atoms,
                calc=calc,
                energy_corr_dict=energy_corr_dict,
            )
            energy_MLP = atoms.get_potential_energy()
            energy_DFT = atoms_DFT.get_potential_energy()
            print_energies(energy_MLP=energy_MLP, energy_DFT=energy_DFT)
            atoms_list.append(atoms)
        else:
            atoms_list.append(atoms_DFT)
    # Update energy references dictionaries.
    energies_ref.update(get_elem_energies_ref(atoms_list=atoms_list))
    energies_ref_DFT.update(get_elem_energies_ref(atoms_list=atoms_list_DFT))
    # Return energy references dictionary.
    return energies_ref, energies_ref_DFT, compositions_ref

# -------------------------------------------------------------------------------------
# PRINT ENERGIES
# -------------------------------------------------------------------------------------

def print_energies(
    energy_MLP: float,
    energy_DFT: float,
    e_form_MLP: float = None,
    e_form_DFT: float = None,
):
    """
    Print energies.
    """
    print("-" * 88)
    print(" | ".join([
        "Energy",
        f"{energy_MLP:+12.3f} [eV] (MLP)",
        f"{energy_DFT:+12.3f} [eV] (DFT)",
        f"{energy_MLP - energy_DFT:+12.3f} [eV] (MLP-DFT)",
    ]))
    if e_form_MLP is not None and e_form_DFT is not None:
        print(" | ".join([
            "E_form",
            f"{e_form_MLP:+12.3f} [eV] (MLP)",
            f"{e_form_DFT:+12.3f} [eV] (DFT)",
            f"{e_form_MLP - e_form_DFT:+12.3f} [eV] (MLP-DFT)",
        ]))
    print("-" * 88)

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
# IDENTITY 1 FOLD
# -------------------------------------------------------------------------------------

class Identity1Fold:
    """
    Cross-validator that does not split the data, but returns a single fold where 
    train indices = test indices = all indices (for calculating training errors).
    """
    def __init__(self, n_splits: int = 1, **kwargs: dict):
        assert n_splits == 1, "Identity1Fold only supports n_splits=1."
    
    def get_n_splits(self, X: list = None, y: list = None, groups: list = None):
        return 1
    
    def split(self, X: list, y: list = None, groups: list = None):
        yield range(len(X)), range(len(X))

# -------------------------------------------------------------------------------------
# GET CROSSVALIDATOR
# -------------------------------------------------------------------------------------

def get_crossvalidator(
    crossval_name: str = None,
    stratified: bool = None,
    group: bool = None,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True
) -> object:
    """
    Get cross-validator.
    """
    from sklearn.model_selection import (
        KFold,
        StratifiedKFold,
        GroupKFold,
        StratifiedGroupKFold,
    )
    # Prepare the cross-validation parameters.
    kwargs = {"shuffle": shuffle, "random_state": random_state}
    # Check the cross-validation type and create the object.
    if n_splits == 1:
        crossval = Identity1Fold(n_splits=n_splits)
    elif crossval_name == "KFold" or [group, stratified] == [False] * 2:
        crossval = KFold(n_splits=n_splits, **kwargs)
    elif crossval_name == "StratifiedKFold" or [group, stratified] == [False, True]:
        crossval = StratifiedKFold(n_splits=n_splits, **kwargs)
    elif crossval_name == "GroupKFold" or [group, stratified] == [True, False]:
        crossval = GroupKFold(n_splits=n_splits)
    elif crossval_name == "StratifiedGroupKFold" or [group, stratified] == [True] * 2:
        crossval = StratifiedGroupKFold(n_splits=n_splits, **kwargs)
    else:
        raise ValueError("Wrong cross-validation parameters.")
    # Return the cross-validation object.
    return crossval

# -------------------------------------------------------------------------------------
# CROSS VALIDATION WITH OPTIMIZATION
# -------------------------------------------------------------------------------------

def cross_validation_with_optimization(
    atoms_init: list,
    atoms_all: list,
    db_ase: Database,
    crossval: BaseCrossValidator,
    key_groups: str,
    key_stratify: str,
    calc: Calculator,
    finetuning: bool,
    finetune_MLP_fun: callable,
    finetune_kwargs: dict = {},
    atoms_ref: list = None,
    energy_corr_kwargs: dict = {},
    atoms_train_added: list = [],
    formation_energies: bool = False,
    formation_energy_fun: callable = None,
    ref_energies_fun: callable = None,
    ref_energies_kwargs: dict = {},
    required_properties: list = ["energy", "forces"],
    n_max_train: int = None,
    seed: int = 42,
    only_n_folds: int = None,
    directory: str = "split_{0:02d}",
    label: str = "model",
) -> dict:
    """
    Run cross-validation with optimization.
    """
    # Filter data with the required properties.
    if required_properties is not None:
        atoms_all = filter_atoms_list(
            atoms_list=atoms_all,
            required_properties=required_properties,
        )
    # Loop over splits.
    energy_list_DFT = []
    energy_list_MLP = []
    e_form_list_DFT = []
    e_form_list_MLP = []
    atoms_list_MLP = []
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
        # Print title.
        print_title(string=f"Split {ii}", width=84)
        # Get test set.
        atoms_test = np.array(atoms_init, dtype=object)[indices_test]
        # Get training set.
        uids = [atoms_init[ii].info["uid"] for ii in indices_train]
        atoms_train = [atoms for atoms in atoms_all if atoms.info["uid"] in uids]
        # Limit the number of training set.
        if n_max_train is not None:
            atoms_train = resample_atoms_list(
                atoms_list=atoms_train,
                key_groups="uid",
                n_max_data=n_max_train,
                duplicates_ok=True,
            )
        # Add data to the training set.
        atoms_train += atoms_train_added
        # Print number of training data.
        print(f"Number of training data: {len(atoms_train)}")
        # Get energy corrections from train data or reference atoms.
        energy_corr_dict = get_energy_corrections(
            atoms_list=atoms_ref or atoms_train,
            calc=calc,
            **energy_corr_kwargs,
        )
        # Run finetuning on training set.
        if finetuning is True:
            # Update kwargs trainer.
            if directory is not None:
                finetune_kwargs["directory"] = directory.format(ii)
            if label is not None:
                finetune_kwargs["label"] = label.format(ii)
            # Run fine-tuning.
            calc = finetune_MLP_fun(
                atoms_list=atoms_train,
                energy_corr_dict=energy_corr_dict,
                **finetune_kwargs,
            )
            energy_corr_dict = get_energy_corrections(
                atoms_list=atoms_ref or atoms_train,
                calc=calc,
                **energy_corr_kwargs,
            )
        # Get reference energies.
        if formation_energies is True:
            energies_ref, energies_ref_DFT, compositions_ref = ref_energies_fun(
                atoms_list=atoms_test,
                calc=calc,
                db_ase=db_ase,
                energy_corr_dict=energy_corr_dict,
                **ref_energies_kwargs,
            )
        # Relax structures.
        for atoms in atoms_test:
            # Get DFT energy.
            atoms_DFT = get_atoms_from_db(db_ase, uid=atoms.info["uid"], relaxed=True)
            energy_DFT = atoms_DFT.get_potential_energy()
            # Optimize structure and get MLP energy.
            optimize_atoms(
                atoms=atoms,
                calc=calc,
                energy_corr_dict=energy_corr_dict,
            )
            energy_MLP = atoms.get_potential_energy()
            # Store results.
            energy_list_MLP.append(energy_MLP)
            energy_list_DFT.append(energy_DFT)
            atoms.info["Energy_MLP"] = energy_MLP
            atoms.info["Energy_DFT"] = energy_DFT
            # Get formation energies.
            if formation_energies is True:
                # Get MLP formation energy.
                e_form_MLP = formation_energy_fun(
                    atoms=atoms,
                    energies_ref=energies_ref,
                    compositions_ref=compositions_ref,
                )
                # Get DFT formation energy.
                e_form_DFT = formation_energy_fun(
                    atoms=atoms_DFT,
                    energies_ref=energies_ref_DFT,
                    compositions_ref=compositions_ref,
                )
                # Store results.
                e_form_list_MLP.append(e_form_MLP)
                e_form_list_DFT.append(e_form_DFT)
                atoms.info["E_form_MLP"] = e_form_MLP
                atoms.info["E_form_DFT"] = e_form_DFT
            else:
                e_form_MLP = None
                e_form_DFT = None
            # Print energies.
            print_energies(
                energy_MLP=energy_MLP,
                energy_DFT=energy_DFT,
                e_form_MLP=e_form_MLP,
                e_form_DFT=e_form_DFT,
            )
            # Append atoms to list.
            atoms_list_MLP.append(atoms)
        # Stop after n folds (for testing).
        if only_n_folds is not None and ii + 1 == only_n_folds:
            break
    # Calculate average errors.
    mae_energy = mean_absolute_error(energy_list_DFT, energy_list_MLP)
    print(f"MAE Energy: {mae_energy:.3f} [eV]")
    if formation_energies is True:
        mae_e_form = mean_absolute_error(e_form_list_DFT, e_form_list_MLP)
        print(f"MAE E_form: {mae_e_form:.3f} [eV]")
    # Return list of atoms.
    results = {
        "atoms_list_MLP": atoms_list_MLP,
        "Energy_list_DFT": energy_list_DFT,
        "Energy_list_MLP": energy_list_MLP,
        "E_form_list_DFT": e_form_list_DFT,
        "E_form_list_MLP": e_form_list_MLP,
    }
    return results

# -------------------------------------------------------------------------------------
# RESAMPLE ATOMS LIST
# -------------------------------------------------------------------------------------

def resample_atoms_list(
    atoms_list: list,
    key_groups: str = "uid",
    n_max_data: int = 1000,
    random_state: int = 42,
    duplicates_ok: bool = False,
):
    """
    Uniformly resample a list of Atoms objects so that the sampled dataset
    contains approximately equal representation from each unique group.
    """
    from sklearn.utils import resample
    # Group the indices of atoms by their uids.
    indices_groups = {}
    for ii, atoms in enumerate(atoms_list):
        group = atoms.info[key_groups]
        indices_groups.setdefault(group, []).append(ii)
    # Determine how many samples to take per group.
    n_per_group = n_max_data // len(indices_groups)
    # Sample indices from each group.
    indices_all = []
    for indices in indices_groups.values():
        n_samples = n_per_group + 0
        replace = False
        if len(indices) < n_per_group and duplicates_ok is True:
            replace = True
        elif len(indices) < n_per_group and duplicates_ok is False:
            n_samples = len(indices)
        indices_all += resample(
            indices,
            n_samples=n_samples,
            replace=replace,
            random_state=random_state,
        )
    # Sample remaining indices randomly from the full list.
    n_remaining = n_max_data - len(indices_all)
    if n_remaining > 0:
        indices_all += resample(
            [ii for ii in range(len(atoms_list)) if ii not in indices_all],
            n_samples=n_remaining,
            replace=False,
            random_state=random_state,
        )
    # Return the resampled list of atoms.
    return [atoms_list[ii] for ii in indices_all]

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------