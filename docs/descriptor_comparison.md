# Descriptor comparison checklist

The Flash-ACE descriptor is intended to match the Atomic Cluster Expansion (ACE)
construction used by [MACE](https://github.com/ACEsuit/mace). To validate this in
practice, you can run side-by-side descriptor evaluations on public datasets and check
numerical agreement. This guide suggests datasets and gives a short example script for
quantitative comparison.

## Suggested datasets

| Dataset | Why use it | How to fetch |
| --- | --- | --- |
| QM9 (small molecules) | Tiny systems with diverse chemistry; fast to load; good for verifying chemical sensitivity. | `torch_geometric.datasets.QM9` (CPU friendly) |
| rMD17/MD17 (molecular dynamics) | Tests descriptor stability for distorted geometries; used by MACE benchmarks. | `mace.tools.data.RMD17` (requires `mace-torch`) |
| OC20-S2EF (surface adsorption) | Larger systems with variable coordination; stresses memory scaling. | Download small validation splits from [OpenCatalystProject/ocp](https://github.com/OpenCatalystProject/ocp) CLI |

Start with QM9 or rMD17 because they are quick to download and do not require GPUs.

## Example: compare descriptors on a QM9 mini-batch

The snippet below shows how to compute descriptors for one mini-batch of QM9 using both
Flash-ACE and MACE. It computes cosine similarity and relative error between the two
outputs so you can see how closely they match.

```python
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_cluster import radius_graph

from flashace.physics import ACE_Descriptor
# AtomicClusterExpansion is exposed from mace.modules in the mace-torch package
from mace.modules import AtomicClusterExpansion

# 1) Load a tiny subset
qm9 = QM9(root="./data/QM9", pre_transform=None)
loader = DataLoader(qm9[:8], batch_size=4)
batch = next(iter(loader)).to(torch.device("cpu"))

# 2) Build a simple neighbor graph (use same cutoff for both models)
cutoff = 5.0
edge_index = radius_graph(batch.pos, r=cutoff, batch=batch.batch)
rel_vec = batch.pos[edge_index[0]] - batch.pos[edge_index[1]]
edge_len = rel_vec.norm(dim=1)
rel_vec = rel_vec / edge_len.view(-1, 1)

# 3) Instantiate both descriptors with matching hyperparameters
params = dict(r_max=cutoff, l_max=2, num_radial=6, hidden_dim=64)
flash_desc = ACE_Descriptor(**params).eval()
mace_desc = AtomicClusterExpansion(**params).eval()  # provided by mace-torch

with torch.no_grad():
    flash_out = flash_desc(batch.x.float(), edge_index, rel_vec, edge_len)
    mace_out = mace_desc(batch.x.float(), edge_index, rel_vec, edge_len)

# 4) Measure similarity
cos_sim = torch.nn.functional.cosine_similarity(
    flash_out.reshape(len(batch.x), -1),
    mace_out.reshape(len(batch.x), -1),
    dim=1,
).mean()
rel_err = (flash_out - mace_out).norm() / mace_out.norm()
print(f"Mean cosine similarity: {cos_sim.item():.4f}")
print(f"Relative Frobenius error: {rel_err.item():.4f}")
```

**Expected outcome:** when the descriptors are aligned, cosine similarity stays close to
1.0 and the relative error is small. Differences point to implementation gaps, mismatched
hyperparameters, or differing normalization conventions. You can tighten the cutoff,
increase `l_max`, or run across more batches for a stricter check.

## Tips for reliable comparisons

- Ensure both models share identical cutoff, `l_max`, and radial basis size.
- Use the same neighbor graph; switching between radius and k-nearest builds will change
  the descriptor.
- Keep everything on CPU for deterministic results. Enable `torch.use_deterministic_algorithms(True)`
  if you want strict reproducibility across runs.
- Save the raw descriptor tensors to disk so you can plot histograms or per-channel
  scatter plots when debugging differences.
