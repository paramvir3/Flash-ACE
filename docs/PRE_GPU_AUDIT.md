# Pre-GPU Deployment Audit

Review of the AOTI export -> LAMMPS deployment path before committing paid GPU
time. Scope: everything on the critical path from a training checkpoint to
forces inside an MD step. The training path is excluded, having already run to
completion on an H100.

Method: read the code, then verify each claim by execution or by mutation
(reintroduce the defect and confirm a test fails). Findings are separated into
defects fixed, properties confirmed correct, and residual risk.

---

## A. Defects found and fixed

### A1. Per-atom energy silently reported as zero  (correctness, both pair styles)

`ev_init()` allocates and zeroes `eatom`/`vatom` when the caller requests
per-atom output. Neither pair style ever writes `eatom`. `compute pe/atom`
therefore returned an array of genuine-looking zeros rather than failing.
`vflag_atom` was guarded, `eflag_atom` was not, so the two per-atom paths
behaved inconsistently.

The model returns one extensive energy and a global virial; no per-atom
decomposition exists to report. Both are now refused explicitly. The guards were
also moved ahead of the model call -- the `vflag_atom` check previously ran
*after* a full evaluation, so it burned a forward pass before erroring.

### A2. 41% of per-step host-to-device traffic was a buffer of zeros  (performance)

Two of the seven model inputs are structurally constant for an entire run:

- `edge_shift` -- LAMMPS ghost atoms already carry unwrapped coordinates, so
  `r_ij = r_j - r_i` is the true minimum-image vector and the lattice shift is
  identically zero.
- `strain` -- the derivative is baked into the compiled program and evaluated
  at zero strain.

Both were re-zeroed on the host and re-copied to the device every timestep. At
`max_edges = 131072` the shift buffer alone is 1.57 MB:

| input | bytes/step | changes? |
|---|---|---|
| edge_index | 2.10 MB | yes |
| edge_shift | 1.57 MB | **never** |
| pos | 0.15 MB | yes |
| z | 0.10 MB | yes |
| mask | 0.05 MB | rarely |
| cell, strain | ~60 B | NPT only / **never** |

1.62 MB of 3.97 MB per step, ~0.32 ms at pageable H2D bandwidth. Both now live
on the device, written once at `pair_coeff` time. The host-side `std::fill` of
393k floats per step is gone with them.

### A3. Metadata mislabelled every production export  (correctness of the contract)

`architecture=trace_v3_or_v4` was hardcoded from when AOT supported only v3/v4.
v2 is the production architecture, so every real export was mislabelled. Now
derived from the checkpoint's `architecture_version`.

### A4. The format contract was written but never checked

`aot_deploy` emits `format=transformers_ace_aoti_v1`; the pair style never read
it. That string is the carrier for units, the padding scheme and the
force/virial sign conventions -- a future revision that changed any of them
would have produced plausible but wrong dynamics. The pair style now refuses
anything it does not recognise.

### A5. Export could not be validated without a GPU

`compile_lammps_aot_model` raised on any non-CUDA device. That block is what
prevented the entire pipeline from being exercised offline -- including the
packaging defect below, which had been open for the whole session. Relaxed to a
warning; production artifacts must still be built on the target GPU.

### A6. Exported packages could not be loaded at all  (blocking)

`aot_compile` was called without `aot_inductor.package: True`, so it returned a
bare shared library and `package_aoti` produced an archive with no model entry.
Every load failed with `File not found: _metadata.json`.

Two hypotheses were wrong and are recorded so they are not revisited:

- Freezing the layer scales is **not** the cause of the CUDA codegen failure.
  193 nodes carry the `(*, 4, 5)` shape with or without the freeze; it is
  intrinsic to e3nn's l=2 block layout.
- Passing a dict to `package_aoti` fixed nothing. Reverting it leaves every
  test green.

Also established: `torch.export` still cannot capture the `torch.func.grad`
program (`Cannot access data pointer of Tensor that doesn't have storage`), so
`aoti_compile_and_package` is unusable and the `make_fx` route must stay. A
consequence worth knowing is that the Python `aoti_load_package` helper cannot
read these archives either -- it wants pytree specs that only an
`ExportedProgram` writes. Deployment is unaffected, because the C++ loader takes
a flat tensor vector. `load_aot_force_program()` now binds that same C++ loader
so the Python tests and LAMMPS exercise one path.

### A7. Export shipped unverified

Nothing compared the compiled program against eager. A silently wrong package
would have reached LAMMPS with no way to detect it. Export now verifies and
raises on mismatch:

```
verify energy  max |delta| = 0.000e+00 eV    [ok]
verify forces  max |delta| = 2.384e-07 eV/A  [ok]
verify virial  max |delta| = 4.768e-07 eV    [ok]
```

### A8. Session-script defects that would have failed on the clock

- Capacities were set to 12288 atoms / 131072 edges. The benchmark system needs
  ~16.9k and ~197k. The run would have aborted on the first step.
- `CMAKE_PREFIX_PATH` pointed at the torch package root rather than
  `torch.utils.cmake_prefix_path`.
- The benchmark had no TorchScript baseline built from the same checkpoint;
  comparing against different weights would have measured weights, not runtime,
  and made the energy cross-check meaningless.
- `*.pt` does not match `*.pt2`, so the produced package would have been
  committed -- 3 MB of single-architecture build output.

---

## B. Properties confirmed correct

These were checked because they are the places where a mistake is silent.

### B1. Receptive field equals `r_max` exactly

The pair style builds edges only into local atoms and requests ghosts to
`r_max`. If the true receptive field were larger, forces near subdomain
boundaries would be wrong. Measured by displacing a probe atom:

| probe distance to nearest local atom | change in local forces |
|---|---|
| 4.5 A (inside `r_max`) | 7.2e-03 eV/A |
| 6.0 A (at `r_max`) | **0.00e+00** |
| 9.0 A | **0.00e+00** |

Exactly zero, not merely small. The ghost cutoff and the local-receiver edge
construction are both correct.

### B2. The Python/C++ ABI matches

AOTI takes a flat tensor vector, so a reordered or retyped argument is not a
load error -- it is silently wrong dynamics. All seven arguments agree in order
and dtype, and the output accessors match what the program returns
(`energy` scalar, `forces` `(N,3)`, `virial` `(6,)`, all float32, read as
`item<double>`, `accessor<float,2>`, `accessor<float,1>`). Pinned by a test
that fails when two arguments are swapped.

### B3. Both pair styles compile

Verified against real LAMMPS (30 Mar 2026) and torch headers, exit 0, no
diagnostics. The C++ calls also match torch's `model_package_loader.h`.

### B4. Capacity overflow aborts rather than truncating

Both the atom and edge guards call `error->one()` with the exact remedy. A
silently truncated neighbour list would be wrong physics, not a crash.

### B5. The `scatter_reduce` zero-initialisation is deliberate

`_cutoff_softmax` initialises `max_per_node` to zeros rather than `-inf`. This
looks like a defect next to `_segment_softmax` but is load-bearing: the zero
*is* the null token's logit, so `amax` yields `max(0, max_k l_k)` and
`null_weight = exp(-max)` is consistent. Correct as written.

### B6. Export cost is flat in capacity

50 s at 512/8192 versus 57 s at 4096/65536, package size constant at 2.9 MB.
Inductor compiles graph structure, not tensor extents, so benchmark-scale
export will not stall the session.

---

## C. Residual risk

**One genuine unknown remains:** whether the derivative graph survives CUDA
Inductor codegen. `max_autotune` is now off by default, which is the leading
suspect for the earlier assertion

```
failed to set ranges [4, 1] ([N, 4, 5], [])
```

If it recurs with autotune already off, the fault is in plain lowering and the
next probe is `TORCH_LOGS=+inductor` on the failing node. Everything downstream
of the export is verified.

Lower-order, accepted:

- `.pt2` artifacts are architecture- and toolkit-specific; the CPU package
  proves the pipeline, not the binary.
- torch must match the driver. cu130 failed against driver 550 previously, so
  Stage 0 prints both before anything expensive runs.
- The five per-step `.to(device_)` calls that remain allocate from the caching
  allocator each step. Making them persistent would also make the input
  addresses static, which is a precondition for CUDA-graph capture. Deferred:
  it is a real refactor and the buffers genuinely change, so the win is
  allocator churn rather than transfer volume.
- `local_rank()` falls back to the global MPI rank when no launcher variable is
  set, which maps GPUs incorrectly for multi-node runs without
  `OMPI_COMM_WORLD_LOCAL_RANK` or equivalent.

---

## D. Test coverage

102 tests pass. The new AOT tests are mutation-tested rather than merely green:

| mutation | result |
|---|---|
| `aot_inductor.package: False` | 4 tests fail |
| swap two C++ input arguments | ABI test fails |
| `package_aoti(list)` instead of `{name: list}` | **all pass** -- so that change was never the fix |

The last row is why the comment in `aot.py` no longer claims it was.
