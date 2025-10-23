# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase.db import connect

from qe_toolkit.io import read_pwo
from mlps_finetuning.databases import (
    get_atoms_from_nested_dirs,
    write_atoms_list_to_db,
)

# -------------------------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------------------------

def main():

    # DFT output and database parameters.
    basedir = "../../doped-ZrO2/"
    index = ":"
    db_dft_name = "ZrO2_dft.db"
    keys_store = ["class", "species", "surface", "dopant", "uid", "index", "relaxed"]

    # Function to read quantum espresso output files.
    read_fun = lambda filepath, index: read_pwo(
        filename=os.path.split(filepath)[1],
        path_head=os.path.split(filepath)[0],
        filepwi="pw.pwi",
        index=index,
    )

    # Get molecules structures.
    atoms_molecules = get_atoms_from_nested_dirs(
        basedir=basedir+"Molecules",
        tree_keys=["species"],
        filename="pw_tot.pwo",
        index=index,
        read_fun=read_fun,
        add_info={"class": "molecules", "dopant": "none"},
    )
    # Get bulks structures.
    atoms_bulks = get_atoms_from_nested_dirs(
        basedir=basedir+"Bulks",
        tree_keys=["species"],
        filename="pw_tot.pwo",
        index=index,
        read_fun=read_fun,
        add_info={"class": "bulks", "dopant": "none"},
    )
    # Get surfaces structures.
    atoms_surfaces = get_atoms_from_nested_dirs(
        basedir=basedir+"Surfaces",
        tree_keys=[None, "dopant"],
        filename="pw_tot.pwo",
        index=index,
        read_fun=read_fun,
        add_info={"class": "surfaces", "species": "clean"},
    )
    # Get hydrogen adsorption structures.
    atoms_hydrogen = get_atoms_from_nested_dirs(
        basedir=basedir+"HydrogenAdsorption",
        tree_keys=["dopant", "species"],
        filename="pw_tot.pwo",
        index=index,
        read_fun=read_fun,
        add_info={"class": "H2-structures"},
    )
    # Get adsorbates and transition states structures.
    atoms_mechanism = get_atoms_from_nested_dirs(
        basedir=basedir+"ReactionPaths",
        tree_keys=["dopant", "class", "species"],
        filename="pw_tot.pwo",
        index=index,
        read_fun=read_fun,
        add_info={},
    )
    for atoms in atoms_mechanism:
        if atoms.info["class"] == "reactions":
            atoms.info["image"] = "TS"
    # Get initial states structures.
    atoms_first = get_atoms_from_nested_dirs(
        basedir=basedir+"ReactionPaths",
        tree_keys=["dopant", "class", "species"],
        filename="first/pw.pwo",
        index=index,
        read_fun=read_fun,
        add_info={"image": "first"},
    )
    # Get final states structures.
    atoms_last = get_atoms_from_nested_dirs(
        basedir=basedir+"ReactionPaths",
        tree_keys=["dopant", "class", "species"],
        filename="last/pw.pwo",
        index=index,
        read_fun=read_fun,
        add_info={"image": "last"},
    )
    # Merge all atoms lists.
    atoms_list = (
        atoms_molecules +
        atoms_bulks +
        atoms_surfaces +
        atoms_hydrogen +
        atoms_mechanism +
        atoms_first +
        atoms_last
    )
    
    # Update information on dopant charges.
    dopant_charges_dict = {
        "Cd": "Cd2+",
        "Ce": "Ce4+",
        "Ga": "Ga3+",
        "In": "In3+",
        "Zn": "Zn2+",
        "Al": "Al3+",
        "Mg": "Mg2+",
        "Zr": "Zr4+",
        "Ti": "Ti4+",
    }
    for atoms in atoms_list:
        if atoms.info["dopant"] in dopant_charges_dict:
            atoms.info["dopant"] = dopant_charges_dict[atoms.info["dopant"]]
        if atoms.info["dopant"] != "none":
            atoms.info["surface"] = "ZrO2(101)+"+atoms.info["dopant"]
        else:
            atoms.info["surface"] = "none"
    
    # Write atoms to ase database.
    db_ase = connect(name=db_dft_name, append=False)
    write_atoms_list_to_db(
        atoms_list=atoms_list,
        db_ase=db_ase,
        keys_store=keys_store,
        keys_match=None,
        fill_stress=False,
        fill_magmom=True,
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------