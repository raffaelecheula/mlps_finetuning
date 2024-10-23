# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

from ase.db import connect
from ase.gui.gui import GUI

from mlps_finetuning.databases import get_atoms_list_from_db

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    selection = "class=surfaces"

    # Initialize ase database.
    db_ase = connect(name="ZrO2_dft.db")

    # Print number of selected atoms.
    selected = list(db_ase.select(selection=selection))

    print(f"number of calculations:", len(selected))

    atoms_list = get_atoms_list_from_db(db_ase=db_ase, selection=selection)

    dopants_list = set()
    for atoms in atoms_list:
        dopants_list.add(atoms.info["dopant"][:-2])
    print(list(dopants_list))

    # TODO: group dopants with different charges.
    # TODO: get indices of train test val to do out of domain tasks.

    #gui = GUI(atoms_list)
    #gui.run()

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------