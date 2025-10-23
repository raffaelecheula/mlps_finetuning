# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import warnings
import numpy as np
from copy import deepcopy
from ase import Atoms
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

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
# PARITY PLOT
# -------------------------------------------------------------------------------------

def parity_plot(
    y_true: list,
    y_pred: list,
    y_stds: list = None,
    ax: object = None,
    lims: list = [-5, +5],
    alpha: float = 0.20,
    color: str = "crimson",
    ms: float = 5,
    fmt: str = "o",
    capsize: float = 3,
    show_errors: bool = True,
    add_violin_plot: bool = True,
    kwargs_errorbar: dict = {},
    kwargs_violin: dict = {},
) -> object:
    """
    Parity plot of the results.
    """
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    # Plot parity line.
    ax.plot(lims, lims, "k--")
    # Check if data are outside the boundaries.
    y_all = np.hstack([y_true, y_pred])
    if np.any((y_all < lims[0]) | (y_all > lims[1])):
        warnings.warn("Some data points fall outside the plot limits!", UserWarning)
    # Plot data.
    ax.errorbar(
        x=y_true,
        y=y_pred,
        yerr=y_stds,
        ms=ms,
        fmt=fmt,
        alpha=alpha,
        color=color,
        capsize=capsize,
        **kwargs_errorbar,
    )
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("E$_{DFT}$ [eV]", fontdict={"fontsize": 16})
    ax.set_ylabel("E$_{model}$ [eV]", fontdict={"fontsize": 16})
    ax.tick_params(labelsize=13, width=1.5, length=6, direction="out")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # Calculate the MAE and the RMSE.
    if show_errors is True:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        ax.text(
            x=lims[0]+(lims[1]-lims[0])*0.23,
            y=lims[0]+(lims[1]-lims[0])*0.92,
            s=f"MAE = {mae:6.3f} [eV]\nRMSE = {rmse:6.3f} [eV]",
            fontsize=13,
            ha='center',
            va='center',
            bbox={
                "boxstyle": 'round,pad=0.5',
                "edgecolor": 'black',
                "facecolor": 'white',
                "linewidth": 1.5,
            },
        )
    # Add violin plot.
    if add_violin_plot is True:
        inset_ax = fig.add_axes([0.70, 0.13, 0.18, 0.25])
        violin_plot(
            y_true=y_true,
            y_pred=y_pred,
            ax=inset_ax,
            color=color,
            show_errors=False,
            **kwargs_violin,
        )
    return ax

# -------------------------------------------------------------------------------------
# VIOLIN PLOT
# -------------------------------------------------------------------------------------

def violin_plot(
    y_true: list,
    y_pred: list,
    ax: object = None,
    ylim: list = [0., +1.5],
    alpha: float = 0.8,
    color: str = "crimson",
    show_errors: bool = True,
) -> object:
    """
    Violin plot of the errors.
    """
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    y_err = np.abs(np.array(y_true)-np.array(y_pred))
    violin = ax.violinplot(
        dataset=[y_err],
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )["bodies"][0]
    violin.set_facecolor(color)
    violin.set_alpha(alpha)
    violin.set_edgecolor("k")
    ax.set_ylabel("Errors [eV]", fontdict={"fontsize": 16})
    ax.get_xaxis().set_visible(False)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=13, width=1.5, length=6, direction="inout")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    if show_errors is True:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        ax.text(
            x=0.85,
            y=0.92*ylim[1],
            s=f"MAE = {mae:6.3f} [eV]\nRMSE = {rmse:6.3f} [eV]",
            fontsize=13,
            ha='center',
            va='center',
            bbox={
                "boxstyle": 'round,pad=0.5',
                "edgecolor": 'black',
                "facecolor": 'white',
                "linewidth": 1.5,
            },
        )
    return ax

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
# END
# -------------------------------------------------------------------------------------