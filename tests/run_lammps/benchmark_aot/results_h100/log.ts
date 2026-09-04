LAMMPS (22 Jul 2025 - Update 6)
OMP_NUM_THREADS environment is not set. Defaulting to 1 thread.
  using 1 OpenMP thread(s) per MPI task
# Paired benchmark: transformers_ace (TorchScript) vs transformers_ace/aot.
#
# Both runs start from the same equilibrated configuration with the same seed
# and integrate the same number of steps, so the comparison is like for like in
# both wall time and physics. Select the style with -var style and the model
# file with -var model, e.g.
#
#   lmp -in in.bench -var style aot -var model ../../../trace_h100.pt2
#   lmp -in in.bench -var style ts  -var model ../model.transformers_ace.pt

units           metal
atom_style      atomic
dimension       3
newton          on
boundary        p p p

read_data       ../test_lammps_cspbi3_mpi/data.NVT
Reading data file ...
  triclinic box = (0 0 0) to (83.4736 76.648 71.044) with tilt (0 0 0)
  1 by 1 by 1 MPI processor grid
  reading atoms ...
  10240 atoms
  reading velocities ...
  10240 velocities
  read_data CPU = 0.047 seconds

mass            1 132.905   # Cs
mass            2 204.199   # Pb
mass            3 126.900   # I

if "${style} == aot" then   "pair_style  transformers_ace/aot"   "pair_coeff  * * ${model} Cs Pb I" else   "pair_style  transformers_ace"   "pair_coeff  * * ${model} Cs Pb I"
pair_style  transformers_ace
pair_coeff  * * ${model} Cs Pb I
pair_coeff  * * /home/user/TRACE/trace_h100.transformers_ace.pt Cs Pb I

neighbor        1.0 bin
neigh_modify    delay 5 every 1

velocity        all create 400.0 1428 dist gaussian
fix             1 all nve
timestep        0.002

# Warm-up is excluded from the timing: it absorbs the first neighbour build,
# AOTI's lazy runner setup and any one-off allocator growth.
thermo          50
thermo_style    custom step temp pe ke etotal press
run             50
Neighbor list info ...
  update: every = 1 steps, delay = 5 steps, check = yes
  max neighbors/atom: 2000, page size: 100000
  master list distance cutoff = 7
  ghost atom cutoff = 7
  binsize = 3.5, bins = 24 22 21
  1 neighbor lists, perpetual/occasional/extra = 1 0 0
  (1) pair transformers_ace, perpetual
      attributes: full, newton on, ghost
      pair build: full/bin/ghost
      stencil: full/ghost/bin/3d
      bin: standard
Per MPI rank memory allocation (min/avg/max) = 7.473 | 7.473 | 7.473 Mbytes
   Step          Temp          PotEng         KinEng         TotEng         Press     
         0   400           -326446.84      529.39785     -325917.45      6004.8001    
        50   399.50934     -326446.22      528.74847     -325917.47      7289.849     
Loop time of 6.96782 on 1 procs for 50 steps with 10240 atoms

Performance: 1.240 ns/day, 19.355 hours/ns, 7.176 timesteps/s, 73.481 katom-step/s
92.2% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 6.9297     | 6.9297     | 6.9297     |   0.0 | 99.45
Neigh   | 0.02908    | 0.02908    | 0.02908    |   0.0 |  0.42
Comm    | 0.0025631  | 0.0025631  | 0.0025631  |   0.0 |  0.04
Output  | 6.3015e-05 | 6.3015e-05 | 6.3015e-05 |   0.0 |  0.00
Modify  | 0.0047346  | 0.0047346  | 0.0047346  |   0.0 |  0.07
Other   |            | 0.001699   |            |       |  0.02

Nlocal:          10240 ave       10240 max       10240 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           6742 ave        6742 max        6742 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:       292602 ave      292602 max      292602 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 292602
Ave neighs/atom = 28.574414
Neighbor list builds = 1
Dangerous builds = 0

reset_timestep  0
run             200
Per MPI rank memory allocation (min/avg/max) = 7.473 | 7.473 | 7.473 Mbytes
   Step          Temp          PotEng         KinEng         TotEng         Press     
         0   399.50934     -326446.22      528.74847     -325917.47      7289.8484    
        50   389.28606     -326432.66      515.21801     -325917.44      7701.8919    
       100   409.25229     -326459.09      541.6432      -325917.45      5882.794     
       150   416.22399     -326468.31      550.87022     -325917.44      7328.7133    
       200   403.18944     -326451.06      533.61905     -325917.44      7967.3249    
Loop time of 9.71449 on 1 procs for 200 steps with 10240 atoms

Performance: 3.558 ns/day, 6.746 hours/ns, 20.588 timesteps/s, 210.819 katom-step/s
99.4% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 9.5311     | 9.5311     | 9.5311     |   0.0 | 98.11
Neigh   | 0.14852    | 0.14852    | 0.14852    |   0.0 |  1.53
Comm    | 0.010272   | 0.010272   | 0.010272   |   0.0 |  0.11
Output  | 0.00029935 | 0.00029935 | 0.00029935 |   0.0 |  0.00
Modify  | 0.018018   | 0.018018   | 0.018018   |   0.0 |  0.19
Other   |            | 0.006276   |            |       |  0.06

Nlocal:          10240 ave       10240 max       10240 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           6699 ave        6699 max        6699 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:       290714 ave      290714 max      290714 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 290714
Ave neighs/atom = 28.390039
Neighbor list builds = 5
Dangerous builds = 0
Total wall time: 0:00:18
