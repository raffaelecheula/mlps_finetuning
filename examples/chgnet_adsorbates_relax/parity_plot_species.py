# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect

from mlps_finetuning.databases import get_atoms_list_from_db, get_atoms_from_db

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Results database.
    db_res_name = "ZrO2_chgnet.db"

    # Initialize ase database.
    db_ase = connect(name=db_res_name)

    # Colors for the plot.
    dopants_colors = {
        "Al3+": "gray",
        "Cd2+": "orange",
        "Ga3+": "crimson",
        "In3+": "seagreen",
        "Zn2+": "darkturquoise",
        "Ce4+": "gold",
        "Mg2+": "lightgreen",
        "Ti4+": "purple",
        "Mn2+": "blue",
        "Mn3+": "darkblue",
        "Cr2+": "brown",
        "Cr3+": "maroon",
    }

    # Get atoms structures from database.
    atoms_list = get_atoms_list_from_db(db_ase)
    species_list = list({atoms.info["species"]: None for atoms in atoms_list})
    
    # Plot formation energies.
    # Get atoms structures for species.
    fig, ax = plt.subplots(figsize=(6,6), dpi=100, facecolor="white")
    e_form_all = []
    for species in species_list:
        e_form_list = []
        e_form_DFT_list = []
        for dopant in dopants_colors:
            kwargs_match = {"species": species, "dopant": dopant}
            atoms = get_atoms_from_db(db_ase, none_ok=True, **kwargs_match)
            if atoms is None:
                continue
            e_form = atoms.info["e_form"]
            e_form_DFT = atoms.info["e_form_DFT"]
            e_form_list.append(e_form)
            e_form_DFT_list.append(e_form_DFT)
        ax.plot(e_form_list, e_form_DFT_list, "o", markersize=10, label=species)
        e_form_all += [e_form, e_form_DFT]
    e_min = min(e_form_all) - 0.5
    e_max = max(e_form_all) + 0.5
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(e_min, e_max)
    ax.set_xlabel("E form DFT [eV]")
    ax.set_ylabel("E form MLP [eV]")
    ax.set_aspect("equal")
    ax.set_title(species)
    ax.plot([e_min, e_max], [e_min, e_max], '--', color="black")
    ax.legend()
    # Show plot.
    plt.show()

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------