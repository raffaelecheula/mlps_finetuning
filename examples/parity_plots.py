# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect

from mlps_finetuning.databases import get_atoms_list_from_db, get_atoms_from_db
from mlps_finetuning.utilities import parity_plot

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Parameters.
    model = "CHGNet" # CHGNet | MACE | OCP | FAIRChem
    finetuning = False
    
    # Results database.
    directory = f"{model.lower()}_adsorbates_relax"
    model_tag = "finetuned" if finetuning is True else "pretrained"
    db_res_name = f"{directory}/ZrO2_{model}_{model_tag}.db"
    # Initialize ase database.
    db_ase = connect(name=db_res_name)
    # Get atoms structures and formation energies from database.
    atoms_list = get_atoms_list_from_db(db_ase)
    e_form_true = []
    e_form_pred = []
    for atoms in atoms_list:
        e_form_true.append(atoms.info["E_form_DFT"])
        e_form_pred.append(atoms.info["E_form_MLP"])
    # Plot formation energy parity plot.
    os.makedirs(f"{directory}/figures", exist_ok=True)
    ax = parity_plot(
        y_true=e_form_true,
        y_pred=e_form_pred,
        alpha=0.4,
        lims=[-4.0, +2.0],
    )
    ax.set_xlabel("E$_{form}$ DFT [eV]")
    ax.set_ylabel("E$_{form}$ MLP [eV]")
    plt.savefig(f"{directory}/figures/parity_E_form_{model}_{model_tag}.png")
    
# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------