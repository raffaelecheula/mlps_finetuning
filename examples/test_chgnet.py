# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import ase
from ase.build import bulk

from chgnet.model.dynamics import CHGNetCalculator

# -------------------------------------------------------------------------------------
# TEST CHGNET
# -------------------------------------------------------------------------------------

atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)

path = "finetune/bestE_epoch1_e3_f163_sNA_m14.pth.tar"
atoms.calc = CHGNetCalculator.from_file(path=path)
print(atoms.get_potential_energy())

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------