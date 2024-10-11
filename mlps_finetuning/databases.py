# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase.io import read
from ase.db import connect

# -------------------------------------------------------------------------------------
# DFT TO ASE DB 
# -------------------------------------------------------------------------------------

def dft_to_ase_db(
    basedirs,
    filename,
    db_dft_name="dft.db",
    index=":",
    fill_stress=False,
    fill_magmom=False,
):

    # Read dft output files.
    atoms_list = []
    for basedir in basedirs:
        for path, folders, files in os.walk(basedir):
            if filename in files:
                atoms = read(os.path.join(path, filename), index=index)
                atoms_list += atoms if isinstance(atoms, list) else [atoms]

    # Write atoms data to ase database.
    db_dft = connect(name=db_dft_name, append=False)
    for atoms in atoms_list:
        if fill_stress and "stress" not in atoms.calc.results:
            atoms.calc.results["stress"] = np.zeros(6)
        if fill_magmom and "magmoms" not in atoms.calc.results:
            atoms.calc.results["magmoms"] = np.zeros(len(atoms))
        db_dft.write(atoms=atoms, data=atoms.info)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------