# Does attention help beyond the ACE descriptor?

TRACE couples an ACE density-correlation descriptor to an equivariant
cross-attention block. The descriptor alone is already a strong body-ordered
representation, so a fair reader will ask whether the reported accuracy comes
from ACE with a neural readout, and attention is decoration. This experiment is
designed to answer that, including the answer "attention does not help", which
is a result worth reporting either way.

## What the attention block actually does

For each edge `e` into receiver `i`, with `H` heads and key width `K`:

```
  l_eh = (q_{r(e)} · k_e)/sqrt(K)  +  b_h(r_e)  -  softplus(s_h) · r_e
  alpha = softmax over the neighbour set of i, weighted by the cutoff envelope
  update_i = out( (1/H) sum_h sum_{e->i} alpha_eh · V_h(edge_e) )
```

Three separable ingredients are bundled together:

1. `(q·k)/sqrt(K)` -- a **content-dependent** weight. Two neighbours at the same
   distance can be weighted differently according to their features.
2. `b_h(r) - softplus(s_h) r` -- a **learned radial filter**. This is what plain
   ACE and MACE already provide.
3. `V_h`, `out` -- equivariant aggregation of edge features onto nodes.

Only ingredient 1 is specific to attention. An ablation that removes the whole
block confounds all three and cannot answer the question.

## Arms

| arm | logits | parameters | vs full | isolates |
|---|---|---:|---:|---|
| `full` | `(q·k)/√K + b(r) − s·r` | 131,877 | — | baseline |
| `no_qk` | `b(r) − s·r` | 126,725 | −3.91% | content-dependence |
| `no_qk_matched` | `b(r) − s·r`, wider FFN | 131,858 | **−0.01%** | content-dependence at fixed capacity |
| `uniform` | `0` | 126,481 | −4.09% | all learned weighting |
| `none` | update skipped | 118,305 | −10.29% | the entire block |

**`full` vs `no_qk_matched` is the decisive comparison.** Both keep the learned
radial filter and both have the same parameter count to within 0.01%, so a
difference cannot be attributed to capacity. `uniform` and `none` bound the
other end: what ACE plus a neural readout delivers on its own.

Everything else is held fixed -- data, blocked split, optimiser, learning-rate
and temperature schedules, loss weights, epochs. Only the arm and the seed vary.

## The mechanism differs, verified

Two neighbours placed at an identical distance (2.5 Å) from the centre but of
different species. Only a content-dependent mechanism can distinguish them:

| arm | α(Pb) | α(I) | difference |
|---|---:|---:|---:|
| `full` | 0.396387 | 0.346986 | **4.9e-02** |
| `no_qk` | 0.023373 | 0.023373 | 0 |
| `uniform` | 0.235407 | 0.235407 | 0 |

So the controls are not merely re-parameterised versions of the same function;
they are structurally unable to do what `full` does.

## All arms remain physically valid

Checked with randomised (not near-initialisation) parameters, in float64:

| arm | rotation ΔE | rotation ΔF | permutation ΔE | force vs finite difference |
|---|---:|---:|---:|---:|
| full | 3.6e-12 | 8.2e-12 | 6.5e-09 | 2.4e-06 |
| no_qk | 1.3e-12 | 9.8e-12 | 9.1e-11 | 7.6e-06 |
| uniform | 1.2e-12 | 9.4e-12 | 4.3e-10 | 1.0e-05 |
| none | 4.6e-13 | 3.9e-12 | 0 | 1.2e-06 |

Equivariance, permutation invariance and force conservativeness hold in every
arm, so no arm is handicapped by a broken symmetry.

## Running it

```bash
cd experiments/attention_ablation
bash run_ablation.sh 5          # 20 runs, 5 concurrent
```

Then `analyze_ablation.py` writes `ablation.png` -- validation total loss,
energy RMSE, force RMSE and stress RMSE against epoch, one curve per arm with a
band across seeds -- and prints a summary table with the final metrics as
mean ± s.d. and the change relative to `full`.

## Reading the result honestly

- The gap between arms must exceed the **seed scatter**. The analyzer flags any
  difference smaller than the pooled standard deviation as within noise.
- A null result is informative. It would mean that on this dataset the ACE
  descriptor carries the signal, and the honest claim becomes about the
  descriptor and the deployment path rather than about attention.
- **Caveat on scope.** The dataset is 979 frames of 40-atom CsPbI3. If attention
  helps mainly by resolving diverse local environments, a small single-phase
  dataset may not reveal it. A null result here bounds the claim to this regime;
  it does not establish that attention never helps. Extending to a
  multi-phase or multi-composition set would be the follow-up.
- Learning curves matter as much as final numbers: if attention improves
  sample-efficiency or optimisation without changing the converged error, that
  shows as a gap early in training that closes later.
