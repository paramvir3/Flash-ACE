# Flash-ACE
This repository is an attempt to make use attention mechanism on Atomic Cluster Expansion (Drautz, Phys. Rev. B 99, 2019) for making precise and scalable machine learning interatomic potentials

Please DO NOT USE as this is purely for research purposes

## Running training

`train.py` accepts `--config / -c` to point at any YAML file. If you omit the
flag, it will search for `config.yaml` in the repository root and then fall
back to `training/config.yaml`.

Example:

```bash
python train.py --config training/config.yaml
```

## Why deeper attention stacks need extra care

Flash-ACE relies on dense self-attention over equivariant descriptors rather than any message-passing layers, but deeper attention stacks are not guaranteed to outperform shallower ones out of the box. In practice, energy errors can rise when simply increasing `num_layers` for several reasons:

- **Optimization sensitivity.** Attention layers introduce more parameters and sharper loss landscapes, so learning rates that were stable for 2 layers (e.g., `1e-2`) can become too aggressive at 4 layers. Start with a smaller LR (`1e-3`–`5e-4`), enable gradient clipping, and let the LR scheduler react faster.
- **Loss balance.** With forces weighted 10× over energies, the model can overfit forces when capacity increases, hurting energy RMSE. Temporarily raise `energy_weight` or lower `forces_weight` while deeper attention trains, then restore the balance once energies improve.
- **Basis/field coverage.** Reducing radial or angular basis while adding attention (e.g., `num_radial` 8 vs. 12) can bottleneck expressiveness, so the extra depth does not translate into better energy modeling. Maintain or increase `num_radial`/`l_max` when adding layers.
- **Batch noise.** Smaller batches (e.g., 4) increase gradient variance; with deeper attention this amplifies oscillations. Use a larger batch or gradient accumulation if memory allows.

Attention helps when its optimization and capacity needs are matched to the dataset. Treat layer count, basis size, LR schedule, and loss weights as coupled knobs rather than increasing depth alone.

## Throughput levers for faster, more scalable training

- **Mixed precision on CUDA.** Set `use_amp: true` (default) to enable `torch.cuda.amp` with `amp_dtype` (`float16` or `bfloat16`). This typically halves memory traffic and speeds up matmuls/attention without hurting accuracy.
- **Gradient accumulation.** When GPU memory is tight, raise `grad_accum_steps` to simulate larger batches while keeping per-step memory smaller. Losses are normalized automatically so learning dynamics stay stable.
- **Cached neighbor lists.** Enable `precompute_neighbors: true` to build ASE neighbor lists once per structure and reuse them every epoch. This reduces Python/CPU overhead on large datasets where the geometry does not change during training.

## Checkpointing and flexible resumes

- **Save as frequently as you like.** Set `checkpoint_interval: N` in `config.yaml` to dump checkpoints every `N` epochs into `checkpoint_dir` (default `checkpoints/`). A full checkpoint includes model weights plus optimizer, scheduler, and AMP scaler states for exact restarts.
- **Resume while changing loss weights.** Point `resume_from` to a saved checkpoint. By default, only the model weights and the stored energy normalization shift are restored, so you can freely tweak `energy_weight`, `forces_weight`, `stress_weight`, or other training hyperparameters between runs. Flip on `resume_load_optimizer`, `resume_load_scheduler`, or `resume_load_scaler` if you want to continue with the same optimizer/scheduler/AMP state instead.
- **Energy normalization consistency.** You can remove composition trends either with a single `energy_shift_per_atom`, with explicit per-species references via `atomic_energies` (e.g., `{H: -0.5, O: -17.0}`), or by letting the trainer solve for `solve_atomic_energies: true` using a NequIP/MACE-style least-squares fit. `use_checkpoint_energy_shift: true` keeps whichever normalization you used in the checkpoint for consistent resumes; set it to `false` to recompute for a new dataset split.

## Rotational augmentation with Wigner matrices

If your dataset is small or lacks diverse orientations, you can enable per-item SO(3) rotations during training (`random_rotation: true` in `config.yaml`). The loader samples a random Wigner rotation, applies it to atomic positions, and consistently rotates forces and stresses. Energies stay invariant, so this augmentation teaches the network the expected equivariant responses without changing the underlying physics. Disable the flag for validation/test splits to measure accuracy on unaugmented geometries.

## Descriptor tweaks inspired by GRACE and other ML potentials

You can now switch the ACE radial basis between two families to probe descriptor bias/variance trade-offs without rewriting the model:

- **Bessel (default).** Matches MACE/ACE with a smooth polynomial cutoff. Set `radial_trainable: true` to learn the Bessel frequencies, similar to adaptive grids explored in GRACE.
- **Gaussian.** Smooth, localized Gaussians akin to PaiNN/SpookyNet descriptors. Use `radial_basis_type: gaussian` and tune `gaussian_width` to widen or narrow shells; combine with `radial_trainable: true` to let centers/widths shift toward chemically relevant distances.

Additional knobs: `envelope_exponent` controls how sharply the polynomial cutoff decays near `r_max` (higher exponents emulate the steep envelopes used in some tensored ACE variants). These levers let you test whether your system benefits more from oscillatory (Bessel) or localized (Gaussian) radial support without touching the architecture.


## Stress computation hygiene

Stresses are now derived from a symmetric small-strain parameterization (six unique components) and normalized by the *deformed* cell volume. This mirrors the Cauchy stress definition used in ACE/MACE and avoids antisymmetric rotational artifacts or inflated stresses under volumetric deformation.


