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

    # possible selection keys:
    # "class": it can be: molecules, bulks, surfaces, adsorbates, reactions.
    # "dopant": the name of the dopant (including the charge).
    # "name": the name of the structure (different meaning for each class).
    # "index": the index of relaxation images. Initial structures have index=0.
    # "relaxed": if True, it returns only final (relaxed) structures.

    # Magnetic moments are available only for class=surfaces.

    selection = "class=surfaces,relaxed=True"

    # Initialize ase database.
    db_ase = connect(name="ZrO2_dft.db")

    # Print number of selected atoms.
    selected = list(db_ase.select(selection=selection))
    print(f"number of calculations:", len(selected))

    # Get list of atoms structures from database.
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, selection=selection)

    # Get list of groups identified by a specific key.
    group_key = "dopant"
    group_list = []
    for atoms in atoms_list:
        if atoms.info[group_key] not in group_list:
            group_list.append(atoms.info[group_key])
    print(group_list)

    show_atoms = True
    write_atoms = False

    if show_atoms:
        gui = GUI(atoms_list)
        gui.run()
    
    if write_atoms:
        from ase.io import Trajectory
        traj = Trajectory("atoms.traj", "w")
        for atoms in atoms_list:
            traj.write(atoms, **atoms.calc.results)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------