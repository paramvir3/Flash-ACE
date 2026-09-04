# H100 Deployment Benchmark

NVIDIA H100 80GB (driver 550.163.01), torch 2.12.1+cu129, LAMMPS 30 Mar 2026,
single GPU, no MPI. CsPbI3 from `data.NVT`, NVE, 2 fs timestep, float32.
Both pair styles run the **same checkpoint** (`training/h100_run/model_h100.pt`)
so the comparison is runtime, not weights.

## Scaling

| atoms | AOT ms/step | TorchScript ms/step | speedup | AOT us/atom/step | AOT ns/day |
|---:|---:|---:|---:|---:|---:|
| 10,240 | 29.25 | 49.63 | **1.70x** | 2.856 | 5.91 |
| 20,480 | 51.37 | 81.20 | **1.58x** | 2.508 | 3.36 |
| 40,960 | 100.19 | 148.66 | **1.48x** | 2.446 | 1.72 |
| 102,400 | 235.84 | 349.96 | **1.48x** | 2.303 | 0.73 |

Scaling is sub-linear throughout -- 2.00x atoms costs 1.76x then 1.95x time,
and 2.50x atoms costs 2.35x -- so cost per atom *falls* monotonically with size
(2.856 -> 2.303 us). Fixed overhead amortises rather than compounding, which is
the property that matters for large-cell production runs.

The speedup settles at 1.48x from 40k atoms upward. That is the expected
signature: AOTI removes kernel-launch overhead, which dominates when kernels are
small, so its relative advantage is largest at 10k and plateaus once real
compute dominates.

### Size ceiling on one 80 GB H100

100k atoms exports and runs. 150k and 200k do **not** export:

| target | edges | export peak | outcome |
|---:|---:|---:|---|
| 102,400 | 2.10M | ~47 GiB | exported and benchmarked |
| 153,600 | 3.15M | 88.7 GiB | OOM (67.7 GiB held, 21 GiB request) |
| 204,800 | 4.19M | 96.7 GiB | OOM (68.7 GiB held, 28 GiB request) |
| 491,520 | 9.70M | -- | OOM |

The limit is **export-time**, not run-time, and the distinction matters.
Inference alone costs a measured 7.55 GiB per 1M edges:

| max_edges | inference peak |
|---:|---:|
| 217,088 | 1.67 GiB |
| 425,984 | 3.23 GiB |
| 851,968 | 6.44 GiB |

So a 200k-atom system needs only ~32 GiB to *run* and would fit comfortably --
it cannot be *built*, because export holds the eager reference, the make_fx
trace and the Inductor compile simultaneously, roughly 3x inference. Moving the
eager reference off the device (or compiling on a larger-memory GPU) should lift
the ceiling to ~200k without touching the model. That is the single highest-value
follow-up for large-cell work.

## Physics agreement

Two independently integrated 200-step trajectories at 10,240 atoms:

| step | AOT TotEng (eV) | TorchScript TotEng (eV) |
|---:|---:|---:|
| 0 | -325917.44 | -325917.47 |
| 100 | -325917.42 | -325917.45 |
| 200 | -325917.41 | -325917.44 |

A *constant* 0.03 eV offset on 325,917 eV (9e-8 relative) -- float32
accumulation order, not divergence. Temperature agrees to six figures
(403.18943 vs 403.18944).

## Energy conservation (AOT)

| atoms | drift over 100 steps | relative |
|---:|---:|---:|
| 10,240 | 0.000 eV | 0 |
| 20,480 | 0.040 eV | 6.1e-08 |
| 40,960 | 0.100 eV | 7.7e-08 |

## Where the time goes (10,240 atoms)

| | ms/step |
|---|---:|
| model compute (AOTI) | 23.65 |
| host-to-device transfer | 0.39 |
| LAMMPS neighbour build, D2H, force accumulation | ~5.2 |

Transfer is 1.6% after moving the two structurally constant inputs
(`edge_shift`, `strain`) onto the device permanently -- previously 41% of the
per-step transfer was a re-sent buffer of zeros. Making the remaining five
buffers persistent would also enable CUDA-graph capture, but is worth at most
1.6% and was not attempted.

## Export verification (CUDA)

```
verify energy  max |delta| = 0.000e+00 eV
verify forces  max |delta| = 2.384e-06 eV/A
verify virial  max |delta| = 6.676e-06 eV
```

## Bug found during this run

The TorchScript export ignored `--device` entirely: it forced `model.cpu()` and
built CPU example tensors, so `torch.jit.trace` baked a CPU constant into
`torch.eye(3, device=pos.device)`. Every TorchScript artifact this repository
had ever produced was CPU-only, and the pair style aborted on GPU with

```
RuntimeError: mat2 is on cpu, different from other tensors on cuda:0
```

meaning all prior LAMMPS benchmarking was CPU-only. Fixed by tracing on the
target device; the TorchScript column above exists only because of that fix.
