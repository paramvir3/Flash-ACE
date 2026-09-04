# Delta--cubic CsPbI3 free energies

This directory implements a classical, constant-pressure free-energy calculation
for 240-atom delta and cubic CsPbI3 cells using the exact TRACE checkpoint at
`training/model.pt`. It is a LAMMPS-driven Frenkel--Ladd calculation followed by
constant-pressure Gibbs--Helmholtz integration from 300 to 650 K. The target
pressure is 1.01325 bar.

The calculation is intentionally not a copy of the earlier `240_NVT_ASE`
script. That script labels a fixed-volume Helmholtz estimate as a Gibbs free
energy, omits masses and Planck's constant from its analytical Einstein
reference, and has no center-of-mass correction or statistical uncertainty.
Those approximations are not used here.

## Thermodynamic definition

For a phase with equilibrium 1-atm volume at the anchor temperature
\(T_0=450\) K, LAMMPS switches between TRACE and a species-resolved Einstein
crystal,

\[
U_\lambda=(1-\lambda)U_{\mathrm{TRACE}}+\lambda U_{\mathrm{Ein}},
\qquad
U_{\mathrm{Ein}}=\frac{1}{2}\sum_i k_{s_i}
\lvert\mathbf r_i-\mathbf r_i^0\rvert^2 .
\]

The spring for species \(s\) is initialized from its mean-square displacement,
\(k_s=3k_\mathrm{B}T_0/\langle\lvert\mathbf r_i-\mathbf r_i^0\rvert^2\rangle_s\).
Changing a well-conditioned reference spring does not change the converged
free energy; it changes the variance and dissipation of the switching path.

Because LAMMPS switches from the physical solid at \(\lambda=0\) to the
Einstein crystal at \(\lambda=1\), the physical Helmholtz free energy is

\[
F_{\mathrm{TRACE}}(T_0,V_0)=F_{\mathrm{Ein}}(T_0,V_0)
+\int_0^1\!d\lambda\,
\left\langle U_{\mathrm{TRACE}}-U_{\mathrm{Ein}}\right\rangle_\lambda .
\]

For finite-time switching, the code averages the forward integral
\(I_\mathrm{f}\) and the oppositely oriented reverse integral
\(I_\mathrm{b}\) as \(I=(I_\mathrm{f}-I_\mathrm{b})/2\). The residual
\((I_\mathrm{f}+I_\mathrm{b})/2\) is reported as switching hysteresis.

For masses \(m_i\), springs \(k_i\), \(\beta=(k_\mathrm{B}T)^{-1}\), total
volume \(V\), and mass fractions \(\mu_i=m_i/\sum_jm_j\), the analytical
Einstein free energy per atom is

\[
\frac{F_{\mathrm{Ein}}}{N}=\frac{k_\mathrm{B}T}{N}
\left[
\sum_i\ln\!\left(\frac{\beta^2 k_i h^2}{4\pi^2m_i}\right)^{3/2}
-\ln\!\left{
V\left(\frac{\beta}{2\pi\sum_i\mu_i^2/k_i}\right)^{3/2}
\right\}
\right].
\]

The second logarithm is the finite-size center-of-mass correction. All factors
inside the logarithms are dimensionless after conversion to SI units.

At the anchor point,

\[
G(T_0,P)\simeq F(T_0,V_0)+PV_0,
\]

where \(V_0\) is the mean NPT volume. Along each phase branch at fixed pressure,
the code evaluates

\[
\frac{G(T,P)}{T}=\frac{G(T_0,P)}{T_0}
-\int_{T_0}^{T}\frac{H(T',P)}{T'^2}\,dT'.
\]

Here the sampled enthalpy is constructed as
\(H=U_{\mathrm{TRACE}}+K+P_{\mathrm{ext}}V\). In particular, the analysis does
not substitute the instantaneous internal-pressure product
\(P_{\mathrm{int}}V\) printed by LAMMPS's generic `enthalpy` keyword for the
external-pressure work required by the isothermal--isobaric ensemble.

The reported stability is
\(\Delta G_{\delta-\alpha}=G_\delta-G_\alpha\) in kJ mol\(^{-1}\) per CsPbI3
formula unit. Negative values favor delta; positive values favor cubic.

## Cells and ensembles

The relaxed 20-atom structures are repeated as follows:

| phase | repeat | atoms | pressure coupling |
|---|---:|---:|---|
| edge-sharing delta | 2 x 6 x 1 | 240 | orthorhombic `aniso` |
| cubic alpha | 2 x 3 x 2 | 240 | symmetry-preserving `iso` |

The cubic branch retains cubic metric ratios while its volume fluctuates. The
delta branch permits independent orthorhombic lattice fluctuations. Both use a
Nosé--Hoover chain NPT ensemble for enthalpies. The fixed-cell switching paths
use NVE integration plus a zero-net-force Langevin thermostat, as required by
LAMMPS `fix ti/spring`; all three species fixes use the same coupling schedule.

## Run the calculation

Activate the Transformers-ACE environment and point the workflow to the
LAMMPS binary that contains `pair_style transformers_ace` and `fix ti/spring`:

```bash
cd /path/to/Transformers-ACE
source /path/to/transformers-ace-venv/bin/activate
export LAMMPS_COMMAND=/path/to/lammps/build/lmp
```

First run the short syntax and data-flow pilot:

```bash
python tests/cspbi3/free_energy_ti/run_ti.py all \
  --profile pilot \
  --overwrite
```

The pilot contains only tens of MD steps. It must never be interpreted as a
free-energy result.

Run the 300--650 K production workflow in resumable stages:

```bash
python tests/cspbi3/free_energy_ti/run_ti.py prepare --profile production
python tests/cspbi3/free_energy_ti/run_ti.py npt --profile production
python tests/cspbi3/free_energy_ti/run_ti.py anchors --profile production
python tests/cspbi3/free_energy_ti/run_ti.py msd --profile production
python tests/cspbi3/free_energy_ti/run_ti.py ti --profile production
python tests/cspbi3/free_energy_ti/run_ti.py analyze --profile production
```

On a 10-core Mac, a practical resumable launch is:

```bash
caffeinate -dimsu python tests/cspbi3/free_energy_ti/run_ti.py all \
  --profile production \
  --workers 4 \
  --threads-per-job 2
```

The validated pilot sustained approximately 9--11 MD steps s\(^{-1}\) per
240-atom CPU process on the current Mac. The configured production calculation
contains about 1.54 million aggregate MD steps, corresponding to roughly
40--48 h serial or, depending on memory bandwidth and process contention,
approximately 12--20 h with four workers.

Completed jobs are skipped. `--phase`, `--temperature`, and `--replica` select
one independent task. `--workers N` runs independent LAMMPS jobs concurrently;
choose `--threads-per-job` so that workers times threads does not exceed the
available physical CPU cores. Use `--overwrite` only when the selected generated
output should be replaced.

The final files are written below
`tests/cspbi3/free_energy_ti/runs/production/results/`:

- `phase_diagram.csv`, `phase_diagram.png`, and `phase_diagram.pdf`;
- `npt_summary.csv` with enthalpy, pressure, temperature, and block statistics;
- `anchor_switching.csv` with every forward/reverse work and its hysteresis;
- `structural_diagnostics.csv` with Pb--I coordination and phase fingerprints;
- `report.json` with the crossing temperature and bootstrap interval.

`report.json` also contains `scientifically_interpretable` and a list of failed
quality checks. A diagnostic plot is visibly marked when temperature, pressure,
block count, switching hysteresis, coordination, or phase-fingerprint checks do
not pass.

## Required convergence checks

The configured production profile is an initial 240-atom estimate, not an
automatic publication-quality result. Before reporting a transition
temperature:

1. Confirm that mean temperature and pressure agree with their targets and that
   every NPT trajectory remains on its intended phase branch.
2. Double NPT equilibration and sampling times; require the phase free-energy
   difference to agree within its block-bootstrap uncertainty.
3. Double the switching duration and increase the number of independent paths;
   require forward/reverse hysteresis to be small relative to the desired
   uncertainty in \(\Delta G\).
4. Repeat at 480 and at least 960 atoms. A 240-atom Frenkel--Ladd result contains
   finite-size errors, including residual volume and long-wavelength phonon
   contributions.
5. Add temperatures near the observed crossing and repeat the integration. A
   50 K grid is insufficient for a high-precision transition temperature.
6. Verify timestep convergence and the stability of both metastable branches.

The result is a classical-nuclei free energy of the trained TRACE Hamiltonian.
It does not include nuclear quantum effects or correct deficiencies in the
r2SCAN+rVV10 training data, phase coverage, or model equation of state.

## Method references

- R. Freitas, M. Asta, and M. de Koning, [Nonequilibrium free-energy
  calculation of solids using LAMMPS](https://doi.org/10.1016/j.commatsci.2015.10.050),
  *Comput. Mater. Sci.* **112**, 333 (2016).
- S. Menon *et al.*, [Automated free-energy calculation from atomistic
  simulations](https://doi.org/10.1103/PhysRevMaterials.5.103801), *Phys. Rev.
  Materials* **5**, 103801 (2021).
- [LAMMPS `fix ti/spring` documentation](https://docs.lammps.org/fix_ti_spring.html).
