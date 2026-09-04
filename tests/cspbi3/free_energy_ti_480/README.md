# CsPbI3 delta/cubic free energies: 480 atoms

This is the 480-atom finite-size repeat of the tested Frenkel--Ladd and
Gibbs--Helmholtz workflow in `tests/cspbi3/free_energy_ti`. It uses the same
TRACE checkpoint, equations, temperatures, trajectory lengths, switching
protocol, replica count, and analysis criteria as the 240-atom production run.
Its outputs are isolated below `free_energy_ti_480/runs/`.

The 20-atom relaxed cells are repeated as follows:

| phase | repeat | atoms | cell treatment |
|---|---:|---:|---|
| edge-sharing delta | 2 x 6 x 2 | 480 | orthorhombic `aniso` |
| cubic alpha | 3 x 4 x 2 | 480 | symmetry-preserving `iso` |

From the repository root, activate the Transformers-ACE environment and define
the LAMMPS executable:

```bash
export LAMMPS_COMMAND=/path/to/lammps/build/lmp
```

Run the short pilot first:

```bash
caffeinate -dimsu python tests/cspbi3/free_energy_ti_480/run_ti.py all \
  --profile pilot \
  --workers 2 \
  --threads-per-job 2
```

After the pilot succeeds, launch the resumable production calculation:

```bash
caffeinate -dimsu python tests/cspbi3/free_energy_ti_480/run_ti.py all \
  --profile production \
  --workers 4 \
  --threads-per-job 2
```

Completed jobs are skipped automatically. Do not add `--overwrite` when
resuming. Final tables and phase-diagram figures are written to
`tests/cspbi3/free_energy_ti_480/runs/production/results/`.

## Replot the phase diagram

The standalone plotting script reads the generated `phase_diagram.csv`:

```bash
python tests/cspbi3/free_energy_ti_480/plot_phase_diagram.py
```

This writes `phase_diagram_custom.png` and `phase_diagram_custom.pdf` beside
the CSV. To select another table or output name:

```bash
python tests/cspbi3/free_energy_ti_480/plot_phase_diagram.py \
  --data tests/cspbi3/free_energy_ti_480/runs/production/results/phase_diagram.csv \
  --output-prefix tests/cspbi3/free_energy_ti_480/runs/production/results/my_phase_diagram
```

Useful editing options include `--xmin`, `--xmax`, `--xtick-step`,
`--experimental-temperature`, `--font-size`, `--label-size`, `--title-size`,
`--tick-size`, `--width`, `--height`, the four color options, `--hide-ci`,
`--hide-experimental`, and `--hide-transition-label`. Run
`python tests/cspbi3/free_energy_ti_480/plot_phase_diagram.py --help` for the
complete list.
