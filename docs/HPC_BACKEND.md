# TRACE GPU Production Backend

## Current status

`pair_style transformers_ace` is a correctness-oriented LibTorch interface. It
uses LAMMPS full neighbor lists and evaluates local owned-atom energies with
ghost atoms. It correctly requires `newton on` for MPI runs, differentiates one
scalar energy to obtain conservative forces and virial, and maps one MPI rank
to one visible GPU. It is **not** a million-atom H100 implementation: every
force evaluation presently allocates CPU tensors, copies positions/types/edges
to CUDA, builds a dynamic autograd graph, and copies forces and strain gradients
back to CPU.

This distinction is deliberate. Strict physical locality is necessary for
linear weak scaling, but it is insufficient without a device-resident execution
path.

## Performance premise and comparison standard

Allegro is not a plain neural-network baseline. It is a strictly local,
pair-indexed equivariant tensor-product potential whose locality makes the
required MPI halo independent of model depth. TRACE shares this locality
property; it does not obtain a communication advantage merely by using
attention. The relevant references are the original
[Allegro paper](https://doi.org/10.1038/s41467-023-36329-y) and its recent
[AOT and custom-tensor-product performance study](https://doi.org/10.1039/D5DD00423C).

Attention is not intrinsically faster than learned tensor products. In the
current TRACE v2 implementation, it adds per-edge key projections, per-head
value projections, exponentials, segment reductions, and scattered updates.
The present CUDA LAMMPS path adds a still larger cost through host-device
copies and dynamic autograd. Therefore no current TRACE implementation should
be described as faster or more scalable than Allegro.

TRACE has one potential algorithmic advantage worth testing. Its high-order ACE
correlations are formed once per center atom, while fixed directed edge tensors
are used only for the subsequent local query. If this factorization reaches a
chosen energy, force, and stress error with fewer expensive edge-local tensor
products than an ordered-pair representation, it can improve the
accuracy-per-local-FLOP tradeoff. This is a research hypothesis, not a result.
All speed comparisons must use matched data, cutoff, neighbor count, precision,
physical error, and hardware.

## AOT force program

TRACE v3/v4 have an export-safe deployment adapter in
`transformers_ace/aot.py`. It replaces the eager e3nn helper modules by the
same frozen linear maps, Wigner-3j tensor products, squared tensor norms, and
real spherical harmonics. The adapter is tested against the e3nn model for
energy, position derivatives, and strain derivatives. It also materializes the
complete energy/force/virial graph before invoking AOTInductor; no finite
differences or separately learned force head are introduced.

Compile this artifact only on the target CUDA software stack, using a fixed
local-plus-ghost atom and directed-edge capacity for one MPI rank:

```bash
python -m transformers_ace.aot_deploy \
  --checkpoint model.pt \
  --output trace_v3_h100.pt2 \
  --type-map Cs Pb I \
  --example-structure tests/cspbi3/structures/cubic_alpha_phase.vasp \
  --max-atoms 125000 \
  --max-edges 10000000 \
  --device cuda
```

The output library and its adjacent metadata file define an exact fixed-shape
contract for the planned Kokkos pair style. Padding atoms carry zero
local-energy mask and padding edges lie outside the compact cutoff, so padding
does not alter energy, forces, or virial. The native Kokkos pair style is still
the remaining integration step; do not use this AOT library with the current
`pair_style transformers_ace`, which expects TorchScript.

## Required production design

A production `pair_style transformers_ace/kk` backend must satisfy all of the
following conditions.

1. Use LAMMPS Kokkos CUDA device views for positions, types, neighbor lists,
   forces, and virial. No CPU tensor construction or host-device transfer is
   permitted in the per-timestep force path.
2. Build a directed CSR edge list on the GPU from the local-plus-ghost full
   neighbor list. Reuse capacity-managed device buffers rather than allocating
   edge tensors on every step.
3. Execute radial functions, real spherical harmonics, ACE-density reductions,
   Clebsch-Gordan products, segment softmax, tensorial attention, atomic-energy
   reduction, and the position/strain backward pass on the GPU. These kernels
   must preserve the same O(3) conventions as the Python reference.
4. Keep force and virial tensors device resident. Use GPU-aware MPI for halo
   exchange and reverse communication with one MPI rank per GPU. A rank may
   communicate only its cutoff halo; attention depth must not enlarge that halo.
5. Export a static inference-and-derivative program ahead of time. The runtime
   must not invoke Python, TorchScript tracing, or dynamic autograd graph
   construction on each molecular-dynamics step.

The most practical implementation route is a Kokkos pair style coupled to an
ahead-of-time exported TRACE program, with fused or specialized GPU kernels for
the sparse tensor products and segment reductions. A direct port of the current
generic LibTorch operations would be correct but is unlikely to be competitive
on H100 hardware.

## TRACE-Fast architecture research roadmap

The following alternatives preserve the defining TRACE constraint: a center may
query fixed descriptors of its initial cutoff environment, but no attention
layer may consume an updated sender state. Each proposal must be tested against
the existing model at matched force and stress error before it is adopted.

### 1. Shared-value multi-query local attention

TRACE v2 currently performs one equivariant value projection for each head. A
fast variant should retain multiple scalar queries and keys but use one shared
equivariant edge value,

```text
v_ij = W_V a_ij,
c_ij = H^{-1} sum_p alpha_ijp g_ip,
u_i = W_O sum_j c_ij v_ij,
```

where `g_ip` is an invariant scalar gate and `alpha_ijp` is an invariant
attention coefficient. The scalar coefficient cannot change the O(3)
transformation law of `v_ij`. This removes the v2 head-by-head value-projection
loop and retains fixed-environment cross-attention. Head-specific irrep-copy
gates are a possible low-rank extension when one shared value is too restrictive.

TRACE v4 already shares a center-token tensor-product value across heads. That
does not automatically make it a fast MD architecture: a single
center-dependent tensor product can be more costly than v2's edge-linear value.
The two choices must be profiled at matched accuracy rather than selected from
parameter count alone.

### 2. Cross-layer key/value cache

Fixed edge tokens permit a cache that evolving graph states do not. Tie or
factorize the key/value projections across attention blocks, compute the static
edge keys and values once after neighbor-list construction, and retain only
layer-specific queries and output maps. The cache trades device memory for
compute, so it is most useful for multiple attention blocks and should be
stored in BF16 only after force and virial validation.

### 3. Smooth compressed ACE tokens

The strongest algorithmic candidate is to replace attention over every neighbor
edge by attention over a small, fixed number of smooth equivariant ACE tokens,

```text
T_im = sum_{j in N(i)} phi_m(Z_j, r_ij) a_ij.
```

Here `phi_m` is an invariant, differentiable radial/species partition of unity
or a learned separable radial-species function. Each `T_im` transforms like
`a_ij`, because it is a sum of equivariant tensors multiplied by scalars. The
TRACE edge tensor `a_ij` already contains the compact cutoff through its radial
basis, so `phi_m` must remain finite and differentiable rather than applying a
second cutoff envelope.
Queries then attend to `M` fixed tokens per atom, with `M` typically much
smaller than the neighbor count. The edge pass remains linear in the number of
neighbors, but the expensive multi-head score/value attention becomes dense and
fixed-shape, scaling as `O(N M)` rather than `O(E)`. Hard radial bins and hard
top-k selection are prohibited because they introduce discontinuous forces.

This is not the same as merely adding the v3/v4 ACE moment tokens while keeping
every edge token. A speed-oriented experiment must remove or strongly compress
the fine edge-token attention path and establish the accuracy loss, if any.

### 4. Exact fused local softmax

Dense FlashAttention is not a suitable default because molecular neighbor sets
are sparse, short, and irregular. Instead, receiver-sorted CSR edges should be
processed with one warp or cooperative block per center. A stable online
log-sum-exp reduction can accumulate the null-normalized attention numerator
and the weighted equivariant value in registers or shared memory, without
materializing `alpha[E, H]` or invoking global `scatter_reduce` and
`index_add_` operations. The fused kernel must reproduce the current cutoff
normalization exactly, including the unit null contribution.

### 5. Normalization-free local attention as an ablation

A separate fast model can replace softmax by a smooth invariant gate,

```text
g_ijp = sigmoid(s_ijp),
u_i = sum_{j,p} f_c(r_ij) g_ijp v_ijp.
```

It retains strict locality, differentiability, equivariance, and conservative
forces when derived from the scalar energy, while removing the softmax maximum
and denominator reductions. It is an architectural ablation, not a drop-in
replacement for the current checkpoint, and must be evaluated for loss of
accuracy and stable MD behavior.

### 6. Structured equivariant compression

Optimize physical error per FLOP using only hardware-friendly structure:

- prune whole irrep copies and Clebsch-Gordan paths with group penalties, then
  generate a new fixed kernel for the retained paths;
- use low-rank factorizations of scalar key/value maps and shared radial bases;
- tune `ell_max`, correlation order, channel multiplicities, and head count on
  an error-throughput Pareto frontier; and
- specialize fixed type maps and, where profitable, type-pair cutoffs without
  changing the trained model specification. Type-pair cutoffs must be included
  during training and saved with the checkpoint; they must never be altered
  independently at deployment.

Unstructured sparse weights, dynamic mixture-of-experts routing, and
data-dependent hard neighbor selection should not be expected to improve
H100/B100 MD throughput.

## Kernel and runtime implementation priorities

The architecture research above must be accompanied by the following
device-level work. These are implementation requirements, not optional
optimizations.

1. **Receiver-major GPU execution.** Retain the LAMMPS full neighbor list in
   receiver-sorted CSR form. Assign a warp or block to a center environment,
   reduce ACE densities and attention updates locally, and write each center
   result once. This avoids the current scattered `index_add_` accumulation.
2. **Fused edge geometry.** Fuse displacement, periodic shift, cutoff, radial
   basis, real spherical harmonics, species embedding lookup, and initial edge
   tensor construction. For each physical pair, reuse the distance, cutoff,
   radial basis, and the parity relation
   `Y_lm(-r_hat) = (-1)^l Y_lm(r_hat)` to form both directed edge contributions
   where the neighbor-list convention permits it.
3. **Specialized tensor products.** Generate packed, sparse Wigner-3j kernels
   for the exact retained irreps and paths. Use tensor-core-friendly dense
   subblocks where possible. Generic eager e3nn operations are a reference
   implementation, not the target MD execution path.
4. **Compiled conservative derivatives.** Export and compile the full scalar
   energy and its reverse-mode position/strain derivatives. Implement fused
   adjoint kernels for radial functions, harmonics, ACE contractions, and local
   attention. Never introduce finite differences or a separately learned force
   head for speed.
5. **Force reduction strategy.** Profile atomics against edge-force buffering
   followed by a receiver/sender segmented reduction. Use the lower-cost method
   on the target GPU while preserving Newton's third-law bookkeeping and MPI
   reverse communication.
6. **Mixed precision policy.** Use BF16 or TF32 for validated tensor-core
   projections and tensor products; retain geometry, cutoff, reductions, energy
   accumulation, force, and virial in FP32. FP8 is excluded until energy,
   force, stress, and long-NVE tests support it.
7. **Persistent execution.** Reuse fixed-capacity atom and edge buffers,
   capture stable timestep sequences with CUDA Graphs where LAMMPS permits, and
   eliminate per-step allocations, Python calls, TorchScript tracing, and host
   synchronization.
8. **Communication overlap and load balance.** With one GPU-aware MPI rank per
   GPU, begin cutoff-halo exchange on a dedicated stream, compute interior atoms
   concurrently, then compute boundary atoms and perform GPU-aware reverse
   communication. Balance domains using edge count and tensor cost, not atom
   count alone, for inhomogeneous systems.

## Benchmark protocol and decision gates

Architecture and backend improvements must be reported separately. A smaller
model is not faster in a scientifically useful sense unless it reaches the same
target accuracy. A faster kernel is not a scalable production result unless it
evaluates the same conservative energy, forces, and virial.

For every candidate, record:

- energy, force, and stress errors with identical train/validation/test splits;
- molecular-dynamics throughput, atoms per GPU, neighbor count, cutoff,
  precision, peak memory, and energy/force/virial agreement;
- Nsight Systems evidence of zero per-step host-device transfer and zero
  per-step allocation in the production path;
- single-GPU H100/B100 results before multi-GPU results;
- weak scaling at fixed atoms per GPU and strong scaling at fixed total system
  size, including communication time; and
- NVE energy drift, NPT pressure/equation-of-state behavior, and one-GPU versus
  multi-GPU trajectory agreement.

Only a matched error-throughput Pareto curve can establish whether TRACE beats
Allegro for a specified chemistry and hardware target. A million-atom claim
additionally requires the production backend to satisfy every verification gate
below at the intended system size.

## Verification gates

The backend is not allowed to claim production H100 scaling until it passes:

- energy, force, and six-component virial agreement against the Python
  reference for periodic cells, including triclinic cells and cutoff-boundary
  configurations;
- agreement between one GPU and multi-GPU domain decomposition at `run 0` and
  over a short NVE trajectory;
- no host-device copies or per-step allocations in an Nsight Systems trace;
- strong-scaling and weak-scaling measurements with fixed density/cutoff,
  reported with atoms per GPU, neighbor count, precision, and throughput;
- a long NVE energy-drift test and NPT equation-of-state/pressure test.

Until these gates pass, the native pair style should be used for correctness,
small-to-medium simulations, and LAMMPS/PLUMED workflow validation rather than
as evidence of million-atom performance. The phrase "million-atom H100/B100
ready" is reserved for a backend that passes every gate above at the intended
system size, not for an exported model alone.
