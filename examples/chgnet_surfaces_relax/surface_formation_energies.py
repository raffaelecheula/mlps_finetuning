# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.formula import Formula
from ase.constraints import FixAtoms
from ase.db import connect

from mlps_finetuning.databases import get_atoms_from_db

# -------------------------------------------------------------------------------------
# DATA
# -------------------------------------------------------------------------------------

# Ase database.
db_ase_name = "../ZrO2_dft.db"
db_ase = connect(name=db_ase_name)

# Reference species for calculating the formation energies.
reference = 'H2O+H2'

# List of charges.
charge_list = ['2+', '3+', '4+', '5+', '6+']

# Dictionary of folders containing the bulk structures.
bulk_ref_dict = {
    'Ca': 'CaO',
    'Cd': 'CdO',
    'Mg': 'MgO',
    'Zn': 'ZnO',
    'Mn': 'Mn2O3',
    'Cr': 'Cr2O3',
    'Al': 'Al2O3',
    'Ga': 'Ga2O3',
    'In': 'In2O3',
    'Sc': 'Sc2O3',
    'Y': 'Y2O3',
    'La': 'La2O3',
    'Ce': 'Ce2O3',
    'Ti': 'Ti2O3',
    'V': 'V2O3_1',
    'Mo': 'MoO2',
    'Zr': 'ZrO2_monoclinic',
}

# Correction energy for O2.
energy_corr_O2 = 0.443938 # [eV]

# Data of reference species.
energy_ZP_O2 = 0.098
energy_ZP_H2 = 0.274
energy_ZP_H2O = 0.569
energy_ZP_O_surf = 0.065

deltaG_H2 = -0.589
deltaG_H2O = -1.113
deltaG_O_surf = -0.080

# Function to get the energy of atoms structures.
def get_energy_atoms(atoms):
    energy = atoms.get_potential_energy()
    return energy

# -------------------------------------------------------------------------------------
# DATA
# -------------------------------------------------------------------------------------

metal_list = list(bulk_ref_dict.keys())
dopant_list = list([metal for metal in metal_list if metal != 'Zr'])

# Get energy of O2, H2O, and H2.
kwargs_match = {"class": "molecules", "relaxed": True}
atoms_O2 = get_atoms_from_db(db_ase, species='O2', **kwargs_match)
energy_O2 = get_energy_atoms(atoms=atoms_O2)
atoms_H2O = get_atoms_from_db(db_ase, species='H2O', **kwargs_match)
energy_H2O = get_energy_atoms(atoms=atoms_H2O)
atoms_H2 = get_atoms_from_db(db_ase, species='H2', **kwargs_match)
energy_H2 = get_energy_atoms(atoms=atoms_H2)

# Get energy of reference O.
if reference == 'O2':
    # We do not use the Gibbs free energy because we do not know the concentration
    # of O2 under reaction conditions.
    energy_O_ref = (energy_O2+energy_corr_O2+energy_ZP_O2)/2.-energy_ZP_O_surf
else:
    energy_O_ref = energy_H2O-energy_H2
    # The energy of O species is calculated from the Gibbs free energy of H2O and
    # H2 at the reaction conditions (thermodynamic equilibrium concentrations).
    energy_O_ref = (
        +energy_H2O+energy_ZP_H2O+deltaG_H2O
        -energy_H2-energy_ZP_H2-deltaG_H2
        -energy_ZP_O_surf-deltaG_O_surf
    )

# Get energies of bulk structures.
energy_bulk_dict = {}
x_bulk_dict = {}
for metal in metal_list:
    # Read bulk structure.
    bulk = bulk_ref_dict[metal]
    kwargs_match = {"class": "bulks", "species": bulk, "relaxed": True}
    atoms_bulk = get_atoms_from_db(db_ase, **kwargs_match)
    
    # Analyze bulk formula.
    formula_dict_bulk = Formula(atoms_bulk.get_chemical_formula()).count()
    x_bulk = formula_dict_bulk['O']/formula_dict_bulk[metal]
    
    # Read energy of bulk structure.
    energy_bulk_tot = get_energy_atoms(atoms=atoms_bulk)
    energy_bulk = energy_bulk_tot/formula_dict_bulk[metal]

    energy_bulk_dict[metal] = energy_bulk
    x_bulk_dict[metal] = x_bulk

# Get energies of relaxed and fixed ZrO2(101) surface.
kwargs_match = {"class": "surfaces", "dopant": "Zr4+"}
atoms_ZrO2surf_fix = get_atoms_from_db(db_ase, index=0, **kwargs_match)
atoms_ZrO2surf_relax = get_atoms_from_db(db_ase, relaxed=True, **kwargs_match)
atoms_ZrO2surf_fix.constraints = FixAtoms(indices=range(len(atoms_ZrO2surf_fix)))
energy_ZrO2surf_fix = get_energy_atoms(atoms=atoms_ZrO2surf_fix)
energy_ZrO2surf_relax = get_energy_atoms(atoms=atoms_ZrO2surf_relax)

# Get formation energies of relaxed and fixed ZrO2(101) surface.
formula_dict_ZrO2surf = Formula(atoms_ZrO2surf_relax.get_chemical_formula()).count()
energy_form_ZrO2surf_fix = (
    energy_ZrO2surf_fix-formula_dict_ZrO2surf['Zr']*energy_bulk_dict['Zr']
)
energy_form_ZrO2surf_relax = (
    energy_ZrO2surf_relax-formula_dict_ZrO2surf['Zr']*energy_bulk_dict['Zr']
)

# Get area of ZrO2(101) surface.
area_ZrO2surf = np.cross(*atoms_ZrO2surf_fix.cell[:2,:])[2]

# Get energies of surface structures.
energy_form_dict = {}
for charge in charge_list:
    energy_form_dict[charge] = []
    for dopant in dopant_list:
        # Read surface structure.
        kwargs_match = {
            "class": "surfaces", "relaxed": True, "dopant": f"{dopant}{charge}"
        }
        atoms_surf = get_atoms_from_db(db_ase, **kwargs_match)
        formula_dict = Formula(atoms_surf.get_chemical_formula()).count()

        # Read energy of surface structure.
        energy = get_energy_atoms(atoms=atoms_surf)
        
        # E_form = E_slab-N_Zr*E_ZrO2_bulk-N_M*E_MOx_bulk-(N_O-2*N_Zr-x*N_M)*E_O_ref
        energy_form = (
            +energy
            -formula_dict['Zr']*energy_bulk_dict['Zr']
            -formula_dict[dopant]*energy_bulk_dict[dopant]
            -(
                +formula_dict['O']
                -x_bulk_dict['Zr']*formula_dict['Zr']
                -x_bulk_dict[dopant]*formula_dict[dopant]
            )*energy_O_ref
        )
        
        # Calculate formation energy relative to 1 dopant atom.
        # E_form_per_M = [E_form+E_form_ZrO2surf*(N_M-1)]/N_M
        energy_form_spec = (
            energy_form+energy_form_ZrO2surf_relax*(formula_dict[dopant]-1)
        )/formula_dict[dopant]
        
        # Subtract the formation energy of bottom surface.
        energy_form_spec_top = energy_form_spec-energy_form_ZrO2surf_fix/2.
        
        # Convert into meV/Å².
        energy_form_final = energy_form_spec_top*1000/area_ZrO2surf # [meV/Å²]
        energy_form_dict[charge].append(energy_form_final)

# -------------------------------------------------------------------------------------
# PLOT
# -------------------------------------------------------------------------------------

# Dictionary of colours.
colours_dict = {
    '2+': 'darkturquoise',
    '3+': 'orange',
    '4+': 'darkorchid',
    '5+': 'seagreen',
    '6+': 'orangered',
}

# Initialize the figure.
plt.figure(figsize=(14, 6))
plt.rc('font', size=13) # Controls default text sizes.
plt.rc('axes', titlesize=20) # Fontsize of the axes title.
plt.rc('axes', labelsize=15) # Fontsize of the x and y labels.
plt.rc('xtick', labelsize=15) # Fontsize of the x tick labels.
plt.rc('ytick', labelsize=15) # Fontsize of the y tick labels.
plt.rc('legend', fontsize=13) # Legend fontsize.
plt.rc('figure', titlesize=13) # Fontsize of the figure title.

# Plot data in grouped manner of bar type.
width = 0.17
x_plot = np.arange(len(dopant_list))
plt.bar(x_plot-2*width, energy_form_dict['2+'], width, color=colours_dict['2+'])
plt.bar(x_plot-1*width, energy_form_dict['3+'], width, color=colours_dict['3+'])
plt.bar(x_plot+0*width, energy_form_dict['4+'], width, color=colours_dict['4+'])
plt.bar(x_plot+1*width, energy_form_dict['5+'], width, color=colours_dict['5+'])
plt.bar(x_plot+2*width, energy_form_dict['6+'], width, color=colours_dict['6+'])

# Customize figure.
ax = plt.gca()
ax.get_xaxis().set_visible(False)
plt.ylabel("Formation energy / Area [meV/Å$^2$]")
#plt.legend(charge_list, loc='upper left', facecolor='white', framealpha=1)
plt.legend(charge_list, loc='upper right', facecolor='white', framealpha=1)
plt.xlim(-0.5, len(dopant_list)-0.5)
#plt.ylim(0, 145)
plt.ylim(0, 165)

# Add vertical lines to separate dopants bars.
for ii in range(len(dopant_list)-1):
    plt.axvline(x=ii+0.5, color='lightgrey', linewidth=1)

# Set colours to the cells of the table.
cell_colours_1 = []
cell_colours_2 = []
for ii in range(len(dopant_list)):
    aa, bb = np.argsort([energy_form_dict[jj][ii] for jj in charge_list])[:2]
    charge_a, charge_b = charge_list[aa], charge_list[bb]
    cell_colours_1.append(colours_dict[charge_a])
    if energy_form_dict[charge_b][ii]-energy_form_dict[charge_a][ii] < 2:
        cell_colours_2.append(colours_dict[charge_b])
    else:
        cell_colours_2.append(colours_dict[charge_a])

# Shape used to divide cells of two colors.
separation = "triangles"

# Add table cells with the colors.
if separation == "triangles":
    table_colours_1 = plt.table(
        loc='bottom',
        cellLoc='center',
        cellColours=[cell_colours_1],
    )
    table_colours_2 = plt.table(
        loc='bottom',
        cellLoc='center',
        cellColours=[cell_colours_2],
        edges='BR',
    )
    table_colours_1.scale(1, 1.8)
    table_colours_2.scale(1, 1.8)

elif separation == "rectangles":
    table_colours = plt.table(
        loc='bottom',
        cellLoc='center',
        cellColours=np.vstack([cell_colours_1, cell_colours_2]),
    )
    table_colours.scale(1, 0.9)
    for key, cell in table_colours.get_celld().items():
        cell.set_edgecolor(cell_colours_1[key[1]])

else:
    table_colours = plt.table(
        loc='bottom',
        cellLoc='center',
        cellColours=np.vstack([cell_colours_1]),
    )
    table_colours.scale(1, 1.8)

# Add a table with the names of the dopants.
table = plt.table(
    cellText=[dopant_list],
    loc='bottom',
    cellLoc='center',
)
table.scale(1, 1.8)
table.set_fontsize(13)
for key, cell in table.get_celld().items():
    cell.set_fill(False)

# Show the plot.
plt.rcParams.update({'mathtext.default': 'regular'})
plt.show()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------