# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import sys
import logging
import warnings
import numpy as np
from copy import deepcopy
from ase import Atoms
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from ase.calculators.singlepoint import all_properties

# -------------------------------------------------------------------------------------
# OPTIMAL REORDER INDICES
# -------------------------------------------------------------------------------------

def optimal_reorder_indices(
    atoms: Atoms,
    atoms_ref: Atoms,
) -> list:
    """
    Calculate indices to reorder `atoms` to best match `atoms_ref`.
    """
    numbers = np.array([aa.number for aa in atoms])
    numbers_ref = np.array([aa.number for aa in atoms_ref])
    array = np.hstack([atoms.positions, numbers.reshape(-1, 1)])
    array_ref = np.hstack([atoms_ref.positions, numbers_ref.reshape(-1, 1)])
    # Compute pairwise Euclidean distance cost matrix between rows.
    cost_matrix = cdist(XA=array_ref, XB=array, metric="euclidean")
    # Solve the optimal assignment problem (Hungarian algorithm).
    indices_ref, indices = linear_sum_assignment(cost_matrix)
    # Return indices.
    return indices

# -------------------------------------------------------------------------------------
# REORDER ATOMS
# -------------------------------------------------------------------------------------

def reorder_atoms(
    atoms: Atoms,
    atoms_ref: Atoms = None,
    indices: list = None,
) -> None:
    """
    Reorder atoms from `indices` or `atoms_ref`.
    """
    if indices is None:
        indices = optimal_reorder_indices(atoms=atoms, atoms_ref=atoms_ref)
    # Reorder the atoms.
    n_atoms = len(indices)
    atoms.positions = np.vstack([atoms.positions[indices], atoms.positions[n_atoms:]])
    atoms.symbols = np.hstack([atoms.symbols[indices], atoms.symbols[n_atoms:]])
    # Reassign atoms to calculator to avoid new calculation.
    if atoms.calc:
        atoms.calc.atoms = atoms

# -------------------------------------------------------------------------------------
# REPEAT ATOMS WITH RESULTS
# -------------------------------------------------------------------------------------

def repeat_atoms_with_results(
    atoms: Atoms,
    repetitions: tuple,
) -> Atoms:
    """
    Repeat `atoms` object multiplying the results by the number of copies.
    """
    n_copies = np.prod(repetitions)
    atoms_rep = atoms.copy()
    atoms_rep *= repetitions
    atoms_rep.calc = deepcopy(atoms.calc)
    atoms_rep.calc.results = {}
    # Update system specific properties.
    for prop in ["energy", "stress", "dipole", "magmom", "free_energy"]:
        if prop in atoms.calc.results:
            atoms_rep.calc.results[prop] = atoms.calc.results[prop] * n_copies
    # Update atom specific properties.
    for prop in ["forces", "stresses", "charges", "magmoms", "energies"]:
        if prop in atoms.calc.results:
            atoms_rep.calc.results[prop] = np.vstack(
                [atoms.calc.results[prop]] * n_copies
            )
    atoms_rep.calc.atoms = atoms_rep
    return atoms_rep

# -------------------------------------------------------------------------------------
# FILTER ATOMS LIST
# -------------------------------------------------------------------------------------

def filter_atoms_list(
    atoms_list: list,
    required_properties: list = ["energy", "forces"],
):
    """
    Filter atoms in list with required properties in the results dictionary.
    """
    return [
        atoms for atoms in atoms_list
        if all(key in atoms.calc.results for key in required_properties)
    ]

# -------------------------------------------------------------------------------------
# FILTER RESULTS
# -------------------------------------------------------------------------------------

def filter_results(
    results: dict,
    properties: list = all_properties,
):
    """
    Filter results to only properties managed by ASE.
    """
    return {pp: results[pp] for pp in results if pp in properties}

# -------------------------------------------------------------------------------------
# FILTER CONSTRAINTS
# -------------------------------------------------------------------------------------

def filter_constraints(
    atoms: object,
):
    """
    Filter constraints to only constraints managed by ASE.
    """
    from ase.constraints import __all__
    for ii, constraint in reversed(list(enumerate(atoms.constraints))):
        if constraint.todict()["name"] not in __all__:
            del atoms.constraints[ii]

# -------------------------------------------------------------------------------------
# PRINT TITLE
# -------------------------------------------------------------------------------------

def print_title(
    string: str,
    width: int = 100,
):
    """
    Print title.
    """
    for text in ["-" * width, string.center(width), "-" * width]:
        print("#", text, "#")

# -------------------------------------------------------------------------------------
# REDIRECT OUTPUT
# -------------------------------------------------------------------------------------

class RedirectOutput:
    def __init__(self, logfile: str = None, mode: str = "a"):
        self.logfile = logfile
        self.mode = mode

    def __enter__(self):
        # Save old outputs and handlers.
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        # Save old logging handlers.
        self.old_handlers = logging.root.handlers[:]
        logging.root.handlers.clear()
        # Save old warning function.
        self.old_showwarning = warnings.showwarning
        # Redirect outputs.
        if self.logfile is not None:
            self.logfile_obj = open(file=self.logfile, mode=self.mode)
            sys.stdout = self.logfile_obj
            sys.stderr = sys.stdout
            self.redirect_warnings()

    def redirect_warnings(self):
        # Redirect warnings.
        def showwarning_new(message, category, filename, lineno, file=None, line=None):
            out = warnings.formatwarning(message, category, filename, lineno, line)
            print(out, file=self.logfile_obj)
        warnings.showwarning = showwarning_new

    def __exit__(self, exc_type, exc, tb):
        if self.logfile is not None:
            sys.stdout.close()
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr
        # Restore old logging handlers.
        logging.root.handlers.clear()
        for old in self.old_handlers:
            logging.root.addHandler(old)
        # Restore old warning function.
        warnings.showwarning = self.old_showwarning
        # Do not suppress exceptions.
        return False

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------