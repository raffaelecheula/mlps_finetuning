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

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# -------------------------------------------------------------------------------------
# OPTIMAL REORDER INDICES
# -------------------------------------------------------------------------------------

def optimal_reorder_indices(atoms, atoms_ref):
    """Calculate indices to reorder `atoms` to best match `atoms_ref`."""
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

def reorder_atoms(atoms, atoms_ref=None, indices=None):
    """Reorder atoms from `indices` or `atoms_ref`."""
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
# END
# -------------------------------------------------------------------------------------