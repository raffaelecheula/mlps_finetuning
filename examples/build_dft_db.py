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

    # Control.
    get_molecules = True
    get_bulks = False
    get_surfaces = False
    get_hydrogen = False
    get_mechanism = True
    get_first_last = True
    only_active_dopants = True
    exclude_physisorbed = True

    # DFT output and database parameters.
    basedir = "../../doped-ZrO2/"
    index = ":"
    db_DFT_name = "ZrO2_DFT.db"
    keys_store = ["class", "species", "surface", "dopant", "uid", "index", "relaxed"]

    # Function to read quantum espresso output files.
    read_fun = lambda filepath, index: read_pwo(
        filename=os.path.split(filepath)[1],
        path_head=os.path.split(filepath)[0],
        filepwi="pw.pwi",
        initial_magmoms=False,
        index=index,
    )

    # Get atoms structures from DFT output files.
    atoms_list = []
    # Get molecules structures.
    if get_molecules is True:
        atoms_list += get_atoms_from_nested_dirs(
            basedir=basedir + "Molecules",
            tree_keys=["species"],
            filename="pw_tot.pwo",
            index=index,
            read_fun=read_fun,
            add_info={"class": "molecules", "dopant": "none"},
        )
    # Get bulks structures.
    if get_bulks is True:
        atoms_list += get_atoms_from_nested_dirs(
            basedir=basedir + "Bulks",
            tree_keys=["species"],
            filename="pw_tot.pwo",
            index=index,
            read_fun=read_fun,
            add_info={"class": "bulks", "dopant": "none"},
        )
    # Get surfaces structures.
    if get_surfaces is True:
        atoms_list += get_atoms_from_nested_dirs(
            basedir=basedir + "Surfaces",
            tree_keys=[None, "dopant"],
            filename="pw_tot.pwo",
            index=index,
            read_fun=read_fun,
            add_info={"class": "surfaces", "species": "clean"},
        )
    # Get hydrogen adsorption structures.
    if get_hydrogen is True:
        atoms_list += get_atoms_from_nested_dirs(
            basedir=basedir + "HydrogenAdsorption",
            tree_keys=["dopant", "species"],
            filename="pw_tot.pwo",
            index=index,
            read_fun=read_fun,
            add_info={"class": "H2-structures"},
        )
    # Get adsorbates and transition states structures.
    if get_mechanism is True:
        atoms_list += get_atoms_from_nested_dirs(
            basedir=basedir + "ReactionPaths",
            tree_keys=["dopant", "class", "species"],
            filename="pw_tot.pwo",
            index=index,
            read_fun=read_fun,
            add_info={},
        )
    # Get initial and final states structures.
    if get_first_last is True:
        atoms_list += get_atoms_from_nested_dirs(
            basedir=basedir + "ReactionPaths",
            tree_keys=["dopant", "class", "species"],
            filename="first/pw.pwo",
            index=index,
            read_fun=read_fun,
            add_info={"image": "IS"},
        )
        atoms_list += get_atoms_from_nested_dirs(
            basedir=basedir + "ReactionPaths",
            tree_keys=["dopant", "class", "species"],
            filename="last/pw.pwo",
            index=index,
            read_fun=read_fun,
            add_info={"image": "FS"},
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
            atoms.info["surface"] = "ZrO2(101)+" + atoms.info["dopant"]
        else:
            atoms.info["surface"] = "none"
        if atoms.info["class"] == "reactions" and "image" not in atoms.info:
            atoms.info["image"] = "TS"
    
    # Exclude structures of non-active dopants.
    if only_active_dopants is True:
        active_dopants = ["none", "Al3+", "Cd2+", "Ga3+", "In3+", "Zn2+"]
        atoms_list = [
            atoms for atoms in atoms_list if atoms.info["dopant"] in active_dopants
        ]
    # Exclude physisorbed adsorbates.
    if exclude_physisorbed is True:
        excluded = ["17_CH2O+OH+H", "19_CH2O+H2O", "23_CH2O+2H", "18_CH2O+OH+H_2"]
        atoms_list = [
            atoms for atoms in atoms_list if atoms.info["species"] not in excluded
        ]

    # Write atoms to ase database.
    db_ase = connect(name=db_DFT_name, append=False)
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