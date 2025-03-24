# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import shutil
import numpy as np
from ase.db import connect
from ase.gui.gui import GUI

from mlps_finetuning.databases import get_atoms_list_from_db, write_atoms_to_db
from mlps_finetuning.utilities import optimal_reorder_indices, reorder_atoms

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Show atoms.
    show_atoms = False

    # Copy database.
    shutil.copy2("ZrO2_dft.db", "ZrO2_dft_ordered.db")

    # Initialize ase database.
    db_ase = connect(name="ZrO2_dft_ordered.db")

    # Get list of atoms structures from database.
    selection = "class=adsorbates,index=0"
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, selection=selection)
    selection = "class=reactions,index=0"
    atoms_list += get_atoms_list_from_db(db_ase=db_ase, selection=selection)
    atoms_list_all = []

    # Order atoms.
    for ii, atoms in enumerate(atoms_list):
        uid = atoms.info["uid"]
        surface = atoms.info["surface"]
        selection = f"species=00_clean,relaxed=True,surface={surface}"
        atoms_ref = get_atoms_list_from_db(db_ase=db_ase, selection=selection)[0]
        atoms_copy = atoms.copy()[:len(atoms_ref)]
        indices = optimal_reorder_indices(atoms=atoms_copy, atoms_ref=atoms_ref)
        selection = f"uid={uid}"
        atoms_list_ii = get_atoms_list_from_db(db_ase=db_ase, selection=selection)
        for jj, atoms_ii in enumerate(atoms_list_ii):
            reorder_atoms(atoms=atoms_list_ii[jj], indices=indices)
        atoms_list_all += atoms_list_ii
    
    # Show atoms.
    if show_atoms:
        gui = GUI(atoms_list_all)
        gui.run()
    
    # Write atoms to database.
    keys_store = ["class", "species", "surface", "dopant", "uid", "index", "relaxed"]
    keys_match = ["uid", "index"]
    for atoms in atoms_list_all:
        write_atoms_to_db(
            atoms=atoms,
            db_ase=db_ase,
            keys_store=keys_store,
            keys_match=keys_match,
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