# Reactive free-energy example

This directory contains the completed TRACE workflow for the gas-phase methyl
migration (R2) studied by Vitartas *et al.*, *Digital Discovery* **5**, 108-122
(2026), [doi:10.1039/D5DD00261C](https://doi.org/10.1039/D5DD00261C). The source
data are distributed under CC BY 4.0 at
[doi:10.6084/m9.figshare.28631591.v4](https://doi.org/10.6084/m9.figshare.28631591.v4).

## Prepare the data

From the repository root:

```bash
python training_examples/active_learning_metadynamics/prepare_data.py
```

The script verifies the source checksums, converts the labels to nonperiodic
ExtXYZ, creates deterministic training and validation splits, and keeps the
independent US/AIMD set separate. It also downloads the IRC coordinates needed
to initialize umbrella windows. Generated data and trajectories are not stored
in Git.

One fixed reference energy is subtracted from every label to preserve relative
energies in float32:

\[
E_{\mathrm{label}}(\mathbf R)
=E_{\mathrm{DFT}}(\mathbf R)-E_{\mathrm{ref}}.
\]

This shift does not change forces or free-energy differences. The reference is
recorded in `r2_methyl_shift/data_manifest.json`.

## Train

```bash
cd training_examples/active_learning_metadynamics/r2_methyl_shift
python ../../../train.py --config config.yaml
```

`config.yaml` uses the 192-frame WTMetaD-IB set. For the 323-frame union of the
WTMetaD-IB and downhill active-learning sets, use:

```bash
python ../../../train.py --config config_combined.yaml
```

The independent test-set evaluation, sampling commands, and Figure 4 analysis
are in [`r2_methyl_shift/README.md`](r2_methyl_shift/README.md).

## Scope

The workflow reproduces the published data partition, collective variable,
thermodynamic state, and free-energy estimators with TRACE as the potential. It
does not reproduce the historical active-learning selection sequence or the
electronic-structure calculations.
