"""Compile a fixed-capacity TRACE v3/v4 energy/force/virial program for AOTI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
from ase.data import atomic_numbers
from ase.io import read

from transformers_ace.aot import (
    compile_aot_force_program,
    load_aot_force_program,
    make_aot_compatible,
    pad_lammps_inputs,
)
from transformers_ace import TransformersACECalculator
from transformers_ace.deploy import LAMMPSEnergyModel, LAMMPSAOTForceModel, _example_tensors, _synthetic_atoms


def _metadata(
    type_map: Sequence[str],
    r_max: float,
    max_atoms: int,
    max_edges: int,
    architecture_version: int,
) -> str:
    return "\n".join(
        (
            "format=transformers_ace_aoti_v1",
            "units=metal",
            f"architecture=trace_v{int(architecture_version)}",
            f"r_max={float(r_max):.12g}",
            f"max_atoms={int(max_atoms)}",
            f"max_edges={int(max_edges)}",
            "outputs=energy_eV forces_eV_per_angstrom virial_eV",
            "force_convention=negative_position_gradient",
            "virial_convention=minus_diagonal_strain_gradient_minus_half_shear_gradient",
            "type_symbols=" + " ".join(type_map),
            "type_atomic_numbers=" + " ".join(str(atomic_numbers[symbol]) for symbol in type_map),
            "",
        )
    )


def compile_lammps_aot_model(
    checkpoint: Path,
    output: Path,
    type_map: Sequence[str],
    max_atoms: int,
    max_edges: int,
    example_structure: Path | None = None,
    device: str = "cuda",
    max_autotune: bool = False,
) -> Path:
    """Create an architecture-specific AOTI shared library on the target GPU host."""
    # AOT kernels are architecture- and toolkit-specific, so a production
    # artifact must be compiled on the GPU host that will run it. A CPU export
    # is still permitted: it exercises the identical trace/compile/package/verify
    # path and is how the pipeline is validated without occupying a GPU.
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; compile this artifact on the H100/B100 target host")
    if not device.startswith("cuda"):
        print(
            f"Note: compiling for '{device}'. The resulting package runs on "
            f"'{device}' only -- re-export on the GPU host for production MD."
        )

    calculator = TransformersACECalculator(model_path=str(checkpoint), device=device)
    # LAMMPSEnergyModel supports v2, v3 and v4; v2 is the production architecture
    # of the manuscript, so it must be exportable. The AOT-compatible rewrites in
    # make_aot_compatible cover every module v2 uses (o3.Linear,
    # FullyConnectedTensorProduct, Norm, SphericalHarmonics).
    architecture_version = int(getattr(calculator.model, "architecture_version", 0))
    if architecture_version not in (2, 3, 4):
        raise ValueError("AOT LAMMPS deployment supports TRACE architecture_version 2, 3 or 4")
    if calculator.model.l_max > 3:
        raise ValueError("AOT LAMMPS deployment currently supports l_max <= 3")

    source_model = calculator.model.eval()
    export_model = make_aot_compatible(source_model).to(device).eval()
    atomic_energy_tensor = calculator.atomic_energy_tensor
    if atomic_energy_tensor is not None:
        atomic_energy_tensor = atomic_energy_tensor.detach().to(device)
    energy_model = LAMMPSEnergyModel(
        export_model,
        atomic_energy_tensor=atomic_energy_tensor,
        energy_shift_per_atom=calculator.energy_shift_per_atom,
    ).to(device).eval()
    program = LAMMPSAOTForceModel(energy_model).to(device).eval()

    atoms = (
        read(example_structure.expanduser())
        if example_structure is not None
        else _synthetic_atoms(type_map, calculator.r_max)
    )
    inputs = tuple(tensor.to(device) for tensor in _example_tensors(atoms, calculator.r_max))
    padded_inputs = pad_lammps_inputs(
        inputs,
        max_atoms=max_atoms,
        max_edges=max_edges,
        r_max=calculator.r_max,
    )
    # The eager reference, the make_fx trace and the Inductor compile each hold
    # activations proportional to max_edges. Holding all three at once is what
    # makes export peak far above inference: a 4.2M-edge program needs ~32 GiB
    # to run but OOMed at 76 GiB to export. Keep only the small outputs on the
    # host between phases and hand the allocator its blocks back.
    reference = tuple(tensor.detach().to("cpu") for tensor in program(*padded_inputs))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    output = compile_aot_force_program(
        program, padded_inputs, output, max_autotune=max_autotune
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _verify_package(output, padded_inputs, reference, device)
    output.with_suffix(output.suffix + ".metadata.txt").write_text(
        _metadata(type_map, calculator.r_max, max_atoms, max_edges, architecture_version)
    )
    return output


# float32 AOT kernels reassociate reductions relative to eager, so agreement is
# to round-off rather than bitwise. The thresholds below are ~3 orders of
# magnitude tighter than any physically meaningful error and still hold with
# margin in practice (observed: 1.1e-6 eV, 3.4e-7 eV/A, 2.7e-6 eV).
_VERIFY_TOLERANCES = (("energy", "eV", 1e-3), ("forces", "eV/A", 1e-4), ("virial", "eV", 1e-3))


def _verify_package(package, inputs, reference, device) -> None:
    """Run the exported package through the C++ loader LAMMPS will use.

    Exporting a silently wrong program is the failure mode that costs the most
    downstream, because the pair style has no way to detect it. Checking here
    means a package that reaches LAMMPS has already reproduced eager physics.
    """
    device_index = torch.device(device).index or 0 if torch.device(device).type == "cuda" else -1
    produced = load_aot_force_program(package, device_index=device_index)(*inputs)
    failures = []
    for value, expected, (name, unit, tolerance) in zip(produced, reference, _VERIFY_TOLERANCES):
        # `reference` is held on the host so it does not occupy the device
        # across the compile; compare there.
        deviation = float((value.detach().to("cpu") - expected).abs().max())
        status = "ok" if deviation <= tolerance else "FAILED"
        print(f"  verify {name:<7} max |delta| = {deviation:.3e} {unit}   [{status}]")
        if deviation > tolerance:
            failures.append(f"{name}: {deviation:.3e} {unit} > {tolerance:.0e}")
    if failures:
        raise RuntimeError(
            "Exported AOTI package does not reproduce the eager program: "
            + "; ".join(failures)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output AOTI package, e.g. trace_h100.pt2")
    parser.add_argument("--type-map", nargs="+", required=True)
    parser.add_argument("--max-atoms", type=int, required=True, help="Maximum local-plus-ghost atoms per MPI rank")
    parser.add_argument("--max-edges", type=int, required=True, help="Maximum directed local edges per MPI rank")
    parser.add_argument("--example-structure", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-autotune",
        action="store_true",
        help="Benchmark Triton variants during export. Slow, and has triggered a "
             "CUDA codegen assertion on the derivative graph; off by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = compile_lammps_aot_model(
        checkpoint=args.checkpoint.expanduser().resolve(),
        output=args.output,
        type_map=args.type_map,
        max_atoms=args.max_atoms,
        max_edges=args.max_edges,
        max_autotune=args.max_autotune,
        example_structure=args.example_structure,
        device=args.device,
    )
    print(f"Wrote TRACE AOTI force program: {output}")
    print(f"Wrote metadata: {output.with_suffix(output.suffix + '.metadata.txt')}")


if __name__ == "__main__":
    main()
