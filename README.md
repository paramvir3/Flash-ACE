# Flash-ACE
This repository is an attempt to make use attention mechanism on Atomic Cluster Expansion (Drautz, Phys. Rev. B 99, 2019) for making precise and scalable machine learning interatomic potentials

Please DO NOT USE as this is purely for research purposes

## Why attention may not always beat message passing

The Flash-ACE blocks add self-attention on top of an ACE-style message passing backbone, but deeper attention stacks are not guaranteed to outperform shallower message passing networks out of the box. In practice, energy errors can rise when simply increasing `num_layers` for several reasons:

- **Optimization sensitivity.** Attention layers introduce more parameters and sharper loss landscapes, so learning rates that were stable for 2 layers (e.g., `1e-2`) can become too aggressive at 4 layers. Start with a smaller LR (`1e-3`–`5e-4`), enable gradient clipping, and let the LR scheduler react faster.
- **Loss balance.** With forces weighted 10× over energies, the model can overfit forces when capacity increases, hurting energy RMSE. Temporarily raise `energy_weight` or lower `forces_weight` while deeper attention trains, then restore the balance once energies improve.
- **Basis/field coverage.** Reducing radial or angular basis while adding attention (e.g., `num_radial` 8 vs. 12) can bottleneck expressiveness, so the extra depth does not translate into better energy modeling. Maintain or increase `num_radial`/`l_max` when adding layers.
- **Batch noise.** Smaller batches (e.g., 4) increase gradient variance; with deeper attention this amplifies oscillations. Use a larger batch or gradient accumulation if memory allows.

Attention helps when its optimization and capacity needs are matched to the dataset. Treat layer count, basis size, LR schedule, and loss weights as coupled knobs rather than increasing depth alone.

## Rotational augmentation with Wigner matrices

If your dataset is small or lacks diverse orientations, you can enable per-item SO(3) rotations during training (`random_rotation: true` in `config.yaml`). The loader samples a random Wigner rotation, applies it to atomic positions, and consistently rotates forces and stresses. Energies stay invariant, so this augmentation teaches the network the expected equivariant responses without changing the underlying physics. Disable the flag for validation/test splits to measure accuracy on unaugmented geometries.


