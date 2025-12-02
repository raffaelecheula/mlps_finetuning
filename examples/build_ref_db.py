# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase.io import read
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
    index = -1
    db_ref_name = "ZrO2_ref.db"
    keys_store = []

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
        add_info={"class": "molecules", "dopant": None},
    )
    # Get bulks structures.
    atoms_bulks = get_atoms_from_nested_dirs(
        basedir=basedir+"Bulks",
        tree_keys=["name"],
        filename=filename,
        index=index,
        read_fun=read_fun,
        add_info={"class": "bulks", "dopant": None},
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
    # Merge all atoms lists.
    atoms_list = atoms_molecules+atoms_bulks+atoms_surfaces
    
    # Write atoms to ase database.
    db_ase = connect(name=db_ref_name, append=False)
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