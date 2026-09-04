# R2 umbrella-sampling audit and recovery

## Diagnosis of the completed `full3` calculation

The completed calculation contains 90 nominally complete trajectories: three
replicas of 30 windows, 40 ps per window, with the first 10 ps discarded. The
temperature, timestep, harmonic force constant, Langevin friction, and window
centers match the protocol reported by Vitartas *et al.* The problem is not an
interrupted run.

The result must nevertheless be rejected for quantitative thermodynamics:

1. The launcher used `models/r2_trace_v2_wtmetad.pt`. The combined checkpoints
   were trained later and therefore were not used in this calculation.
2. The independent 326-frame DFT test gives a force RMSE of 0.168 eV/A over
   the full set and 0.263 eV/A in the transition interval. Pointwise errors on
   this on-manifold set do not reveal the off-manifold instability below.
3. Nineteen windows visit a collapsed geometry in which both reactive C-C
   distances are shorter than 1.70 A. The smallest observed value of
   `r1 + r2` is about 2.62 A, whereas it is at least 3.34 A in the released DFT
   training data, 3.64 A in the independent DFT umbrella set, and 3.77 A along
   the DFT IRC.
4. One central trajectory breaks a methyl C-H bond. Other central windows
   form short contacts between methyl hydrogens and framework carbons.
5. The slow central windows have only about 3-11 effective samples after the
   nominal 30 ps production segment. Some neighboring histograms have zero or
   very weak overlap, and the product free energy changes by several
   kcal/mol between time blocks.
6. The original WHAM grid clipped endpoint samples and the barrier search was
   truncated at -0.25 A. Retaining every sample and locating the actual
   profile maximum gives individual barriers of 21.3-24.0 kcal/mol (mean
   22.7 kcal/mol) and a pooled raw-histogram estimate near 23.6 kcal/mol, with
   maxima around -0.28 to -0.33 A. This correction does not rescue the result:
   the published MLIP
   result is 28.2 kcal/mol with its transition region near -0.10 A, and the
   corrected TRACE estimate is built from the wrong chemical pathway.

Run the reproducible audit with:

```bash
python audit_umbrella.py \
  --umbrella-root results/full3/umbrella \
  --output results/full3/audit
```

The audit writes window-level metrics, a diagnostic plot, and a diverse set of
unlabelled off-manifold structures. `ood_candidates_unlabeled.xyz` does not
contain DFT energies or forces and must never be passed directly to training.

## Umbrella-only recovery strategy

"Umbrella only" describes the free-energy estimator. It does not remove the
need for a potential that is accurate and stable on every configuration
visited by the umbrellas. More sampling with the current checkpoint would
converge the free energy of its spurious collapsed pathway.

### 1. Validate the combined final checkpoint

The requested production model is the final epoch-2000 checkpoint
`models/r2_trace_v2_combined_last.pt` (SHA-256
`da4a24b3b9874871b1e333d2369f888d78ccaf159b61d08895527acfe99cbb54`).
Evaluate it without changing the independent DFT test set:

```bash
python ../evaluate_checkpoint.py \
  --model models/r2_trace_v2_combined_last.pt \
  --data data/us_test.xyz \
  --output evaluation/us_combined_last \
  --transition-min -0.20 \
  --transition-max 0.00
```

This checkpoint gives energy/force RMSEs of 1.35 meV/atom and 0.113 eV/A over
the 326 independent structures. In the ten structures with
`-0.20 <= s <= 0.00 A`, the corresponding values are 0.62 meV/atom and
0.135 eV/A. A static extrapolation screen also places all 32 collapsed
structures recovered from the failed run above the predicted IRC transition
state; the old checkpoint placed 31 of 32 below it. This is encouraging but
does not establish the accuracy of those unlabelled configurations.

The combined released set contains no collapsed dual-C-C structures, and
TRACE currently has no explicit short-range repulsive prior. A short dynamics
gate therefore remains mandatory. If that gate fails, label a small, diverse
subset of `results/full3/audit/ood_candidates_unlabeled.xyz` with the same
PBE0-D3BJ/def2-SVP oracle and retrain. Never infer those labels from TRACE.

Before production, verify a two-dimensional `(r1, r2)` scan and short
restrained trajectories. The repaired potential must not contain a low-energy
basin at `r1 < 1.7 A` and `r2 < 1.7 A`, and it must preserve all methyl C-H
bonds. The source reaction path has `r1 + r2` close to 3.8 A around the
transition structure.

### 2. Run a refined central-window chemistry gate

Use a new output directory and first run only the difficult windows. The
runtime chemistry guard aborts when `r1 + r2 < 3.20 A` or a methyl C-H bond
exceeds 1.35 A; it is a failure detector, not an added bias.

Activate the same TRACE environment used for training, source PLUMED, and
retain the PyTorch compatibility setting required by the installed e3nn:

```bash
source ../../../../work/transformers-ace-venv/bin/activate
source /path/to/plumed2/sourceme.sh
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -c "import torch, plumed, transformers_ace; print(torch.__version__)"
```

```bash
caffeinate -dimsu python run_rapid12h.py \
  --preset full3 \
  --methods umbrella \
  --model models/r2_trace_v2_combined_last.pt \
  --replicas 3 \
  --window-centers-file window_centers_refined_ts.txt \
  --windows 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 \
  --umbrella-duration-ps 10 \
  --umbrella-equilibration-ps 5 \
  --output-directory combined_last_refined_gate \
  --workers 6 \
  --device cpu

python audit_umbrella.py \
  --umbrella-root results/combined_last_refined_gate/umbrella \
  --output results/combined_last_refined_gate/audit \
  --chemistry-only
```

Do not proceed if any chemistry guard fires, a persistent bond changes, or
the central windows settle into a second basin that is absent from the DFT
data.

### 3. Produce the umbrella trajectories

After the chemistry gate passes, run the full-length trajectories in a fresh
directory. `window_centers_refined_ts.txt` preserves all 30 published centers
and adds nine midpoint centers from `s=-0.418` to `0.202 A`. The refined
spacing is 0.0344 A in this interval and includes a center at `-0.1078 A`,
close to the DFT transition structure at `-0.1033 A`. Ten independent
replicas reproduce the statistical design used for the source uncertainty.
At 365.6 K and `kappa=20 eV/A^2`, the isolated harmonic width
`sqrt(k_B T/kappa)` is 0.0397 A. The refined spacing is therefore about one
harmonic width, compared with 0.0689 A between the original centers.

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

This calculation contains 39 windows x 10 replicas x 40 ps = 15.6 ns of
sampling, or 31.2 million 0.5-fs integration steps. It is expected to take
about 21-22 hours on the M2 Pro for which the earlier 3 x 30-window run took
about five hours. Rerunning the command safely skips completed windows. If an
interruption leaves nonempty partial windows, the launcher stops before
submitting new jobs; add `--overwrite` to restart only those incomplete
windows. Completed windows whose checkpoint and schedule hashes match are
still preserved. A disk-space preflight estimates the remaining trajectory
storage and retains a 1-GiB safety margin.

If adjacent overlap remains below 0.15, add windows between the failing
centers or use replica-exchange umbrella sampling. Replica exchange remains an
umbrella-sampling method and is preferable to arbitrarily smoothing a broken
profile. Extend only the windows that fail block convergence or have fewer
than 100 effective samples.

### 4. Accept convergence before plotting

A production profile should satisfy all of the following before comparison
with experiment or DFT:

- no collapsed dual-C-C structures, persistent-bond changes, or hydrogen
  transfers;
- minimum neighboring-window overlap at least 0.15, preferably 0.20;
- at least 100 effective samples in every window;
- first-half and second-half mean CV values differing by less than 0.02 A;
- barrier changes below 0.3 kcal/mol and profile RMS changes below
  0.5 kcal/mol between the last two production blocks;
- product-minus-reactant free energy stable within 0.5 kcal/mol;
- independent replicas place the maximum in the same transition region.

Analyze and audit the completed replicas before plotting:

```bash
for replica in $(seq 0 9); do
  tag=$(printf "%02d" "$replica")
  python analyze_umbrella.py \
    --replica-root "results/combined_last_refined_10rep/umbrella/replica_${tag}"
done

python audit_umbrella.py \
  --umbrella-root results/combined_last_refined_10rep/umbrella \
  --output results/combined_last_refined_10rep/audit

python plot_figure4b.py \
  --umbrella 'results/combined_last_refined_10rep/umbrella/replica_*/fes_wham.csv' \
  --allow-missing-methods \
  --ci-method normal \
  --output results/combined_last_refined_10rep/figure4b_umbrella.png
```

Pool raw histograms only after every replica passes these gates. Estimate
uncertainty by resampling independent replicas or trajectory blocks. A smooth
curve should emerge from overlap, adequate effective sampling, and ensemble
averaging; visual filtering must not be used to hide poor convergence.
