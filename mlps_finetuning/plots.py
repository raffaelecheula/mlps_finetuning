# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import warnings
import numpy as np

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
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.5",
                "edgecolor": "black",
                "facecolor": "white",
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
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.5",
                "edgecolor": "black",
                "facecolor": "white",
                "linewidth": 1.5,
            },
        )
    return ax

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------