"""Export-safe tensor algebra for compiling TRACE v3 with AOTInductor.

The training model uses e3nn modules, whose generated TorchScript helpers are
excellent for eager execution but are not consistently compatible with
``torch.export`` plus functional differentiation.  This module freezes the
same learned maps into ordinary PyTorch tensor operations.  It is used only for
deployment; training continues to use e3nn.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
from e3nn import o3


def _irrep_blocks(irreps: o3.Irreps):
    return tuple((int(mul), int(irrep.l), int(irrep.dim)) for mul, irrep in irreps)


class ExportableLinear(nn.Module):
    """Exact e3nn ``o3.Linear`` evaluation without generated helper modules."""

    def __init__(self, source: o3.Linear) -> None:
        super().__init__()
        self.input_blocks = _irrep_blocks(source.irreps_in)
        self.output_blocks = _irrep_blocks(source.irreps_out)
        self.input_slices = tuple((part.start, part.stop) for part in source.irreps_in.slices())
        self.output_slices = tuple((part.start, part.stop) for part in source.irreps_out.slices())
        self.instructions = tuple(
            (
                int(instruction.i_in),
                int(instruction.i_out),
                float(instruction.path_weight),
                tuple(int(value) for value in instruction.path_shape),
            )
            for instruction in source.instructions
        )
        offsets = []
        offset = 0
        for instruction in source.instructions:
            width = math.prod(instruction.path_shape)
            offsets.append((offset, offset + width))
            offset += width
        self.weight_slices = tuple(offsets)
        self.register_buffer("weight", source.weight.detach().clone())
        self.register_buffer("bias", source.bias.detach().clone())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        outputs = [
            features.new_zeros((batch, multiplicity, irrep_dim))
            for multiplicity, _, irrep_dim in self.output_blocks
        ]
        for instruction_index, (input_index, output_index, path_weight, path_shape) in enumerate(
            self.instructions
        ):
            input_start, input_stop = self.input_slices[input_index]
            input_mul, _, input_dim = self.input_blocks[input_index]
            output_mul, _, output_dim = self.output_blocks[output_index]
            weight_start, weight_stop = self.weight_slices[instruction_index]
            value = features[:, input_start:input_stop].reshape(batch, input_mul, input_dim)
            weight = self.weight[weight_start:weight_stop].reshape(path_shape)
            update = torch.einsum("bui,uw->bwi", value, weight) * path_weight
            outputs[output_index] = outputs[output_index] + update.reshape(
                batch, output_mul, output_dim
            )
        return torch.cat([block.reshape(batch, -1) for block in outputs], dim=-1)


class ExportableFullyConnectedTensorProduct(nn.Module):
    """Exact frozen ``uvw`` e3nn tensor product using stored Wigner-3j maps."""

    def __init__(self, source: o3.FullyConnectedTensorProduct) -> None:
        super().__init__()
        unsupported = {instruction.connection_mode for instruction in source.instructions} - {"uvw"}
        if unsupported:
            raise ValueError(f"AOT TRACE supports only uvw tensor products, got {unsupported}")
        self.input1_blocks = _irrep_blocks(source.irreps_in1)
        self.input2_blocks = _irrep_blocks(source.irreps_in2)
        self.output_blocks = _irrep_blocks(source.irreps_out)
        self.input1_slices = tuple((part.start, part.stop) for part in source.irreps_in1.slices())
        self.input2_slices = tuple((part.start, part.stop) for part in source.irreps_in2.slices())
        self.instructions = tuple(
            (
                int(instruction.i_in1),
                int(instruction.i_in2),
                int(instruction.i_out),
                bool(instruction.has_weight),
                float(instruction.path_weight),
                tuple(int(value) for value in instruction.path_shape),
            )
            for instruction in source.instructions
        )
        offsets = []
        offset = 0
        for instruction_index, instruction in enumerate(source.instructions):
            width = math.prod(instruction.path_shape) if instruction.has_weight else 0
            offsets.append((offset, offset + width))
            offset += width
            l1 = source.irreps_in1[instruction.i_in1].ir.l
            l2 = source.irreps_in2[instruction.i_in2].ir.l
            l3 = source.irreps_out[instruction.i_out].ir.l
            self.register_buffer(
                f"wigner_{instruction_index}",
                o3.wigner_3j(l1, l2, l3, dtype=source.weight.dtype),
            )
        self.weight_slices = tuple(offsets)
        self.shared_weights = bool(source.shared_weights)
        self.internal_weights = bool(source.internal_weights)
        if self.internal_weights:
            self.register_buffer("weight", source.weight.detach().clone())

    def forward(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.internal_weights and weights is None:
            raise ValueError("External tensor-product weights are required")
        batch = first.shape[0]
        outputs = [
            first.new_zeros((batch, multiplicity, irrep_dim))
            for multiplicity, _, irrep_dim in self.output_blocks
        ]
        for instruction_index, (
            input1_index,
            input2_index,
            output_index,
            has_weight,
            path_weight,
            path_shape,
        ) in enumerate(self.instructions):
            input1_start, input1_stop = self.input1_slices[input1_index]
            input2_start, input2_stop = self.input2_slices[input2_index]
            input1_mul, _, input1_dim = self.input1_blocks[input1_index]
            input2_mul, _, input2_dim = self.input2_blocks[input2_index]
            output_mul, _, output_dim = self.output_blocks[output_index]
            first_block = first[:, input1_start:input1_stop].reshape(batch, input1_mul, input1_dim)
            second_block = second[:, input2_start:input2_stop].reshape(batch, input2_mul, input2_dim)
            wigner = getattr(self, f"wigner_{instruction_index}").to(dtype=first.dtype)
            if has_weight:
                start, stop = self.weight_slices[instruction_index]
                flat_weight = self.weight[start:stop] if self.internal_weights else weights[:, start:stop]
                if self.shared_weights:
                    coupling_weight = flat_weight.reshape(1, *path_shape)
                else:
                    coupling_weight = flat_weight.reshape(batch, *path_shape)
                update = torch.einsum(
                    "bui,bvj,ijk,buvw->bwk",
                    first_block,
                    second_block,
                    wigner,
                    coupling_weight,
                )
            else:
                update = torch.einsum("bui,bvj,ijk->buvk", first_block, second_block, wigner)
            outputs[output_index] = outputs[output_index] + (
                update * path_weight
            ).reshape(batch, output_mul, output_dim)
        return torch.cat([block.reshape(batch, -1) for block in outputs], dim=-1)


class ExportableSquaredNorm(nn.Module):
    """Squared O(3)-invariant norm for a direct sum of non-scalar irreps."""

    def __init__(self, source: o3.Norm) -> None:
        super().__init__()
        self.blocks = _irrep_blocks(source.irreps_in)
        self.slices = tuple((part.start, part.stop) for part in source.irreps_in.slices())
        if not source.squared:
            raise ValueError("TRACE AOT export requires squared tensor norms")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        if not self.blocks:
            return features.new_empty((batch, 0))
        values = []
        for (multiplicity, _, irrep_dim), (start, stop) in zip(self.blocks, self.slices):
            block = features[:, start:stop].reshape(batch, multiplicity, irrep_dim)
            values.append(block.square().sum(dim=-1))
        return torch.cat(values, dim=-1)


class ExportableSphericalHarmonics(nn.Module):
    """Real e3nn component-normalized spherical harmonics through l=3."""

    def __init__(self, source: o3.SphericalHarmonics) -> None:
        super().__init__()
        if not source.normalize or source.normalization != "component" or not source._is_range_lmax:
            raise ValueError("TRACE AOT export requires normalized component spherical harmonics")
        self.l_max = int(source._lmax)
        if self.l_max > 3:
            raise ValueError("TRACE AOT export currently supports l_max <= 3")

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        vector = torch.nn.functional.normalize(vector, dim=-1)
        x, y, z = vector.unbind(dim=-1)
        values = [torch.ones_like(x)]
        if self.l_max == 0:
            return torch.stack(values, dim=-1)
        root3 = math.sqrt(3.0)
        values.extend((root3 * x, root3 * y, root3 * z))
        if self.l_max == 1:
            return torch.stack(values, dim=-1)
        root15 = math.sqrt(15.0)
        y2 = y.square()
        x2z2 = x.square() + z.square()
        h20 = root15 * x * z
        h24 = 0.5 * root15 * (z.square() - x.square())
        values.extend((h20, root15 * x * y, math.sqrt(5.0) * (y2 - 0.5 * x2z2), root15 * y * z, h24))
        if self.l_max == 2:
            return torch.stack(values, dim=-1)
        values.extend(
            (
                math.sqrt(42.0) * (h20 * z + h24 * x) / 6.0,
                math.sqrt(7.0) * h20 * y,
                math.sqrt(168.0) * (4.0 * y2 - x2z2) * x / 8.0,
                math.sqrt(7.0) * y * (2.0 * y2 - 3.0 * x2z2) / 2.0,
                math.sqrt(168.0) * z * (4.0 * y2 - x2z2) / 8.0,
                math.sqrt(7.0) * h24 * y,
                math.sqrt(42.0) * (h24 * z - h20 * x) / 6.0,
            )
        )
        return torch.stack(values, dim=-1)


def make_aot_compatible(model: nn.Module) -> nn.Module:
    """Return a frozen TRACE copy with export-safe equivariant primitives."""

    # The residual layer scale is a compile-time constant at inference. Baking it
    # removes a (mul, 2l+1) expand-and-concatenate from the exported graph, which
    # the CUDA Triton backend of Inductor could not lower ("failed to set ranges
    # [4, 1] ([N, 4, 5], [])"), and saves the broadcast at every MD step.
    from flashace.model import freeze_layer_scales

    freeze_layer_scales(model)
    converted = copy.deepcopy(model).eval()

    def replace(module: nn.Module) -> None:
        for name, child in tuple(module.named_children()):
            if isinstance(child, o3.FullyConnectedTensorProduct):
                setattr(module, name, ExportableFullyConnectedTensorProduct(child))
            elif isinstance(child, o3.Linear):
                setattr(module, name, ExportableLinear(child))
            elif isinstance(child, o3.Norm):
                setattr(module, name, ExportableSquaredNorm(child))
            elif isinstance(child, o3.SphericalHarmonics):
                setattr(module, name, ExportableSphericalHarmonics(child))
            else:
                replace(child)

    replace(converted)
    for parameter in converted.parameters():
        parameter.requires_grad_(False)
    return converted


def pad_lammps_inputs(
    inputs: tuple[torch.Tensor, ...],
    max_atoms: int,
    max_edges: int,
    r_max: float,
) -> tuple[torch.Tensor, ...]:
    """Pad one LAMMPS graph with physically null atoms and edges.

    Ahead-of-time model containers use fixed tensor extents.  Padded atoms have
    zero local-energy mask; padded edges join two padding atoms at a separation
    larger than the compact cutoff.  They therefore contribute exactly zero to
    the ACE density, energy, forces, and strain derivative.
    """
    z, pos, cell, edge_index, edge_shift, strain, local_mask = inputs
    n_atoms = int(z.shape[0])
    n_edges = int(edge_index.shape[1])
    if max_atoms < n_atoms + 2:
        raise ValueError("max_atoms must leave room for two null padding atoms")
    if max_edges < n_edges:
        raise ValueError("max_edges is smaller than the example edge count")

    padded_z = torch.ones(max_atoms, dtype=z.dtype, device=z.device)
    padded_pos = torch.zeros((max_atoms, 3), dtype=pos.dtype, device=pos.device)
    padded_mask = torch.zeros(max_atoms, dtype=local_mask.dtype, device=local_mask.device)
    padded_z[:n_atoms] = z
    padded_pos[:n_atoms] = pos
    padded_mask[:n_atoms] = local_mask
    padded_pos[-1, 0] = 2.0 * float(r_max)

    padded_edges = torch.empty((2, max_edges), dtype=edge_index.dtype, device=edge_index.device)
    padded_edges[0].fill_(max_atoms - 1)
    padded_edges[1].fill_(max_atoms - 2)
    padded_edges[:, :n_edges] = edge_index
    padded_shifts = torch.zeros((max_edges, 3), dtype=edge_shift.dtype, device=edge_shift.device)
    padded_shifts[:n_edges] = edge_shift
    return padded_z, padded_pos, cell, padded_edges, padded_shifts, strain, padded_mask


AOTI_MODEL_NAME = "model"


def compile_aot_force_program(
    program: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: Path,
    max_autotune: bool = False,
) -> Path:
    """Compile a fixed-shape energy/force/virial program into an AOTI package.

    ``make_fx`` materializes the functional derivative graph before invoking
    Inductor.  This avoids the current ``torch.export`` limitation for
    ``torch.func.grad`` and produces one callable ``.pt2`` model package.
    The documented C++ ``AOTIModelPackageLoader`` consumes this package rather
    than the raw shared library returned by ``aot_compile``.
    Compilation must be run on the target CUDA software stack, since CUDA AOT
    kernels are architecture- and toolkit-specific.
    """
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._inductor import aot_compile
    from torch._inductor.package import package_aoti

    graph = make_fx(program)(*inputs)
    # ``make_fx`` records FakeTensor instances from more than one tracing mode;
    # AOTInductor expects to construct its own mode from real example tensors.
    for node in graph.graph.nodes:
        node.meta.clear()
    output = output.expanduser().resolve()
    if output.suffix != ".pt2":
        raise ValueError("AOT LAMMPS deployment output must end in .pt2")
    output.parent.mkdir(parents=True, exist_ok=True)
    # ``torch.export`` cannot capture this functional derivative graph on all
    # supported PyTorch releases. Package the already-static FX graph directly.
    # PyTorch 2.12 returns a single artifact while newer versions return a list.
    # ``max_autotune`` is off by default. It is a poor trade for a one-shot AOT
    # export: it benchmarks tens of thousands of Triton variants (minutes of pure
    # CPU time) for a marginal gain, and on the CUDA backend it drove the
    # derivative graph into a codegen assertion,
    #   "failed to set ranges [4, 1] ([N, 4, 5], [])",
    # where (4, 5) is the l=2 irrep block that e3nn materialises throughout the
    # graph (193 such nodes; the shape is intrinsic to the layout, not removable).
    # Plain Inductor lowering compiles the same graph.
    # ``aot_inductor.package`` makes aot_compile emit the full artifact set
    # (wrapper, kernels and their ``*_metadata.json`` entries) instead of a bare
    # shared library. Without it the archive carries only a ``.so`` and the
    # loader fails with "File not found: _metadata.json".
    options = {"max_autotune": bool(max_autotune), "aot_inductor.package": True}
    artifacts = aot_compile(graph, inputs, options=options)
    if isinstance(artifacts, (str, Path)):
        artifacts = [str(artifacts)]
    # Package under an explicit model name, which must match the one the LAMMPS
    # pair style asks for (``"model"``).
    #
    # Note on loaders: the Python ``torch._inductor.aoti_load_package`` helper
    # cannot read this archive. It calls ``treespec_loads`` to recover the
    # input/output pytree specs, and those are only written when packaging an
    # ``ExportedProgram`` -- which is exactly the path ``torch.func.grad``
    # blocks. That limitation does not affect deployment: the C++
    # ``AOTIModelPackageLoader`` used by the pair style passes a flat tensor
    # vector and never consults a pytree spec. ``load_aot_force_program`` below
    # binds to that same C++ loader, so the Python and LAMMPS paths agree.
    package_aoti(str(output), {AOTI_MODEL_NAME: artifacts})
    return output


def load_aot_force_program(package: Path | str, device_index: int = -1):
    """Load a ``.pt2`` produced by :func:`compile_aot_force_program`.

    Binds the C++ ``AOTIModelPackageLoader`` -- the same loader the LAMMPS pair
    style uses -- so a package validated here is validated for deployment.
    Returns a callable mapping the padded LAMMPS inputs to
    ``(energy, forces, virial)``.
    """
    loader = torch._C._aoti.AOTIModelPackageLoader(
        str(Path(package).expanduser().resolve()),
        AOTI_MODEL_NAME,
        False,  # run_single_threaded
        1,  # num_runners
        device_index,
    )

    def run(*inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(loader.boxed_run(list(inputs)))

    run.loader = loader  # keep the container alive for the caller
    return run
