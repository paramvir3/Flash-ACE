# R2 methyl-migration free energy

This example applies TRACE to the gas-phase rearrangement of
2,2-dimethylisoindene to 1,2-dimethylindene (`C11H12`). Reference energies and
forces are PBE0-D3BJ/def2-SVP values from
[Vitartas *et al.*](https://doi.org/10.1039/D5DD00261C).

The collective variable is

\[
s(\mathbf R)=r_1-r_2
=\lVert\mathbf r_{14}-\mathbf r_{11}\rVert
-\lVert\mathbf r_{14}-\mathbf r_{10}\rVert ,
\]

with zero-based atom indices. The corresponding PLUMED pairs are `15,12` and
`15,11`. Distances use `NOPBC` because the molecule is nonperiodic.

## Data and training

Prepare the source data from the repository root:

```bash
python training_examples/active_learning_metadynamics/prepare_data.py
cd training_examples/active_learning_metadynamics/r2_methyl_shift
```

Available configurations are:

| Configuration | Training data | Output checkpoint |
|---|---|---|
| `config.yaml` | 192 WTMetaD-IB frames | `models/r2_trace_v2_wtmetad.pt` |
| `config_downhill.yaml` | 131 downhill frames | `models/r2_trace_v2_downhill.pt` |
| `config_combined.yaml` | 323 combined frames | `models/r2_trace_v2_combined.pt` |

Train one model with:

```bash
python ../../../train.py --config config_combined.yaml
```

Evaluate it only after training:

```bash
python ../evaluate_checkpoint.py \
  --model models/r2_trace_v2_combined.pt \
  --data data/us_test.xyz \
  --output evaluation/us_combined \
  --transition-min -0.20 \
  --transition-max 0.00
```

The 326-frame `us_test.xyz` set is independent and must not be used for fitting
or model selection.

## PLUMED environment

Use the same Python environment for TRACE and the PLUMED Python module:

```bash
source /path/to/transformers-ace-venv/bin/activate
source /path/to/plumed2/sourceme.sh
python -c "import plumed; print('PLUMED Python interface is available')"
```

If the import fails, install or rebuild the PLUMED Python extension for the
active interpreter before running dynamics.

## Umbrella sampling

Run a short pilot first:

```bash
python run_umbrella_window.py \
  --model models/r2_trace_v2_combined.pt \
  --replica 0 \
  --window 14 \
  --duration-ps 1 \
  --equilibration-ps 0.2 \
  --output results/pilot/replica_00/window_14 \
  --device cpu
```

For the refined ten-replica protocol used in the current comparison:

```bash
caffeinate -dimsu python run_rapid12h.py \
  --preset full3 \
  --methods umbrella \
  --model models/r2_trace_v2_combined_last.pt \
  --replicas 10 \
  --window-centers-file window_centers_refined_ts.txt \
  --output-directory combined_last_refined_10rep \
  --workers 6 \
  --device cpu
```

Completed windows are skipped on restart. Use `--overwrite` only when an
incomplete output should be replaced.

Analyze each completed replica:

```bash
for replica in $(seq 0 9); do
  tag=$(printf "%02d" "$replica")
  python analyze_umbrella.py \
    --replica-root "results/combined_last_refined_10rep/umbrella/replica_${tag}"
done
```

Plot the free-energy profile:

```bash
python plot_figure4b.py \
  --umbrella 'results/combined_last_refined_10rep/umbrella/replica_*/fes_wham.csv' \
  --allow-missing-methods \
  --ci-method student-t \
  --output results/combined_last_refined_10rep/figure4b_umbrella_trace.png
```

Audit molecular integrity, sampling, and window overlap before interpreting the
barrier:

```bash
python audit_umbrella.py \
  --umbrella-root results/combined_last_refined_10rep/umbrella \
  --output results/combined_last_refined_10rep/audit
```

Generated checkpoints, data, results, and trajectories remain local and are
excluded from Git.

## References

- Vitartas *et al.*, *Digital Discovery* **5**, 108-122 (2026),
  [doi:10.1039/D5DD00261C](https://doi.org/10.1039/D5DD00261C).
- Source data, [doi:10.6084/m9.figshare.28631591.v4](https://doi.org/10.6084/m9.figshare.28631591.v4).
