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
    filename = "pw_tot.pwo"
    index = ":"
    db_dft_name = "ZrO2_dft.db"
    keys_store = ["class", "dopant", "name", "index", "relaxed"]

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
        tree_keys=["name"],
        filename=filename,
        index=index,
        read_fun=read_fun,
        add_info={"class": "molecules", "dopant": "none"},
    )
    # Get bulks structures.
    atoms_bulks = get_atoms_from_nested_dirs(
        basedir=basedir+"Bulks",
        tree_keys=["name"],
        filename=filename,
        index=index,
        read_fun=read_fun,
        add_info={"class": "bulks", "dopant": "none"},
    )
    # Get surfaces structures.
    atoms_surfaces = get_atoms_from_nested_dirs(
        basedir=basedir+"Surfaces",
        tree_keys=[None, "dopant"],
        filename=filename,
        index=index,
        read_fun=read_fun,
        add_info={"class": "surfaces", "name": "clean"},
    )
    # Get hydrogen adsorption structures.
    atoms_hydrogen = get_atoms_from_nested_dirs(
        basedir=basedir+"HydrogenAdsorption",
        tree_keys=["dopant", "name"],
        filename=filename,
        index=index,
        read_fun=read_fun,
        add_info={"class": "H2-structures", "name": "clean"},
    )
    # Get reaction paths structures.
    atoms_reactions = get_atoms_from_nested_dirs(
        basedir=basedir+"ReactionPaths",
        tree_keys=["dopant", "class", "name"],
        filename=filename,
        index=index,
        read_fun=read_fun,
        add_info={},
    )
    # Merge all atoms lists.
    atoms_list = atoms_bulks+atoms_surfaces+atoms_hydrogen+atoms_reactions
    
    # Write atoms to ase database.
    db_ase = connect(name=db_dft_name, append=False)
    write_atoms_list_to_db(
        atoms_list=atoms_list,
        db_ase=db_ase,
        keys_store=keys_store,
        keys_match=None,
        fill_stress=False,
        fill_magmom=False,
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------