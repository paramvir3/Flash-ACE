import os
from ase import units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
import numpy as np

# -------------------------------------------------------------
# 1. SETUP ATOMS
# -------------------------------------------------------------
# Load your starting structure
atoms = read('start.xyz') 

# -------------------------------------------------------------
# 2. CRITICAL FIX: ATTACH YOUR CALCULATOR
# -------------------------------------------------------------
# You must attach the generic calculator here.
# Since you are using FlashACE, it likely looks something like this:
# (Uncomment and adjust the lines below to match your specific calculator setup)

# from ace import ACEMcalculator  # <--- REPLACE WITH YOUR ACTUAL IMPORT
# calc = ACEMcalculator(model_path="path/to/your/potential.yace")
# atoms.calc = calc 

# IF YOU DO NOT ATTACH A CALCULATOR HERE, MD CANNOT RUN.
# -------------------------------------------------------------

# Set initial temperature (e.g. 300K)
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# -------------------------------------------------------------
# 3. DEFINE THE OUTPUT FUNCTION (CLEAN OUTPUT)
# -------------------------------------------------------------
xyz_filename = "md_trajectory.xyz"

def save_xyz():
    """
    Saves ONLY coordinates and energy.
    Removes forces and stress from the output to keep files small.
    """
    # 1. Get the current energy so it is cached
    current_energy = atoms.get_potential_energy()
    
    # 2. Create a temporary copy of atoms to write (so we don't break the running MD)
    # We copy so we can delete the calculator and forces without stopping the sim
    atoms_to_write = atoms.copy()
    
    # 3. Manually set the info to ONLY contain energy
    atoms_to_write.info = {'energy': current_energy}
    
    # 4. Remove the calculator from the copy so ASE doesn't try to write calculator results (forces/stress)
    atoms_to_write.calc = None
    
    # 5. Write the clean atoms object
    write(xyz_filename, atoms_to_write, append=True)

# -------------------------------------------------------------
# 4. SETUP & RUN MD
# -------------------------------------------------------------
# Time step in femtoseconds
dt = 1.0 * units.fs

# Initialize Dynamics
dyn = VelocityVerlet(atoms, dt)

# Attach the saver (every 10 steps)
dyn.attach(save_xyz, interval=10)

# Print progress to console
def print_status():
    epot = atoms.get_potential_energy()
    print(f"Step: {dyn.nsteps} | Epot: {epot:.3f}")

dyn.attach(print_status, interval=10)

print(f"Starting MD... Saving to {xyz_filename}")

# Run
dyn.run(steps=10000)

print("MD finished.")
