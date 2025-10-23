# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase.db import connect

from qe_toolkit.io import read_pwo
from qe_toolkit.neb import read_neb_path
from mlps_finetuning.databases import (
    get_atoms_from_nested_dirs,
    write_atoms_list_to_db,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # DFT output and database parameters.
    basedir = "../../doped-ZrO2/"
    index = ":"
    db_dft_name = "ZrO2_dft_nebs.db"
    keys_store = ["class", "species", "surface", "dopant", "uid", "index", "relaxed"]

    # Function to read quantum espresso output files.
    def read_fun(filepath, index):
        # Read NEB path.
        path_head, filename = os.path.split(filepath)
        filename_path = os.path.join(path_head, "path", "pwscf.path")
        if not os.path.isfile(filename_path):
            return None
        # Read TS atoms.
        atoms_TS = read_pwo(
            filename=filename,
            path_head=path_head,
            filepwi="first/pw.pwi",
            index=-1,
        )
        images = [atoms_TS.copy() for ii in range(10)]
        images = read_neb_path(images=images, filename=filename_path)
        return images

    # Get NEB structures.
    atoms_list = get_atoms_from_nested_dirs(
        basedir=basedir+"ReactionPaths",
        tree_keys=["dopant", "class", "species"],
        filename="pw.pwo",
        index=index,
        read_fun=read_fun,
        add_info={},
    )
    for atoms in atoms_list:
        atoms.info["relaxed"] = True

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
            atoms.info["surface"] = "ZrO2-101+"+atoms.info["dopant"]
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