from ase.io import read, write
import numpy as np

# Define input and output filenames
input_file = 'input.data'
output_file = 'water.lammps-data'

# 1. Read the LAMMPS data file
# We specify style='atomic' to match the format in your snippet
atoms = read(input_file, format='lammps-data', style='atomic')

# 2. Sort the atoms by atomic number (H = 1, O = 8)
# This will naturally place all Hydrogen atoms before all Oxygen atoms
sorted_indices = np.argsort(atoms.numbers)
sorted_atoms = atoms[sorted_indices]

# 3. Write the sorted atoms back to a new LAMMPS data file
# Using specorder ensures that H is mapped to atom type 1 and O to atom type 2
write(output_file, sorted_atoms, format='lammps-data', atom_style='atomic', specorder=['H', 'O'])

print(f"Successfully sorted species and saved to {output_file}")
