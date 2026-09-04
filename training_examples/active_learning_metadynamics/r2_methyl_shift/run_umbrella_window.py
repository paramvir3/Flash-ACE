#!/usr/bin/env python3
"""Run one source-matched R2 umbrella-sampling window with TRACE and ASE."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from ase import units
from ase.constraints import FixCom
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
)


EXAMPLE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLE_ROOT.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from transformers_ace import TransformersACECalculator  # noqa: E402


WINDOW_COUNT = 30
GENERATED_FILES = (
    "COLVAR",
    "md.log",
    "plumed.dat",
    "plumed.log",
    "run_metadata.json",
    "run_status.json",
    "trajectory.traj",
)


def reaction_coordinate(atoms) -> float:
    if len(atoms) != 23 or any(atoms.numbers[index] != 6 for index in (10, 11, 14)):
        raise ValueError("The R2 atom ordering no longer matches the published CV")
    r_1 = np.linalg.norm(atoms.positions[14] - atoms.positions[11])
    r_2 = np.linalg.norm(atoms.positions[14] - atoms.positions[10])
    return float(r_1 - r_2)


def irc_structures_and_centers(
    path: Path, centers_path: Path | None = None
) -> tuple[list, np.ndarray]:
    frames = read(path, index=":")
    if not frames:
        raise ValueError(f"No IRC structures were read from {path}")
    coordinates = np.asarray([reaction_coordinate(atoms) for atoms in frames])
    if not coordinates[-1] > coordinates[0]:
        raise ValueError("The R2 IRC endpoints must be ordered from reactant to product")
    if centers_path is None:
        centers = np.linspace(coordinates[0], coordinates[-1], WINDOW_COUNT)
    else:
        centers = np.atleast_1d(np.loadtxt(centers_path, comments="#", dtype=float))
        if centers.ndim != 1 or len(centers) < 2:
            raise ValueError("The window-center file must contain at least two values")
        if not np.isfinite(centers).all() or not np.all(np.diff(centers) > 0.0):
            raise ValueError("Umbrella centers must be finite and strictly increasing")
        if not np.isclose(centers[0], coordinates[0], atol=1.0e-8) or not np.isclose(
            centers[-1], coordinates[-1], atol=1.0e-8
        ):
            raise ValueError(
                "A custom center schedule must retain the two IRC endpoint centers"
            )
    return frames, centers


def nearest_irc_structure(frames: list, center: float):
    coordinates = np.asarray([reaction_coordinate(atoms) for atoms in frames])
    return frames[int(np.argmin(np.abs(coordinates - center)))].copy()


def plumed_lines(
    colvar: Path,
    center: float,
    kappa: float,
    print_stride: int,
) -> list[str]:
    return [
        "UNITS LENGTH=A TIME=ps ENERGY=eV",
        "r_1: DISTANCE ATOMS=15,12 NOPBC",
        "r_2: DISTANCE ATOMS=15,11 NOPBC",
        "diff_r: COMBINE ARG=r_1,r_2 COEFFICIENTS=1,-1 PERIODIC=NO",
        f"umbrella: RESTRAINT ARG=diff_r AT={center:.17g} KAPPA={kappa:.17g}",
        (
            "PRINT ARG=r_1,r_2,diff_r,umbrella.bias "
            f"STRIDE={print_stride} FILE={colvar}"
        ),
    ]


def minimum_pair_distance(atoms) -> float:
    displacement = atoms.positions[:, None, :] - atoms.positions[None, :, :]
    distances = np.linalg.norm(displacement, axis=-1)
    distances[np.diag_indices_from(distances)] = np.inf
    return float(np.min(distances))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--replica", type=int, default=0)
    parser.add_argument(
        "--model",
        type=Path,
        default=EXAMPLE_ROOT / "models/r2_trace_v2_wtmetad.pt",
    )
    parser.add_argument(
        "--irc", type=Path, default=EXAMPLE_ROOT / "structures/r2_irc.xyz"
    )
    parser.add_argument("--window-centers-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--temperature", type=float, default=365.6)
    parser.add_argument("--time-step-fs", type=float, default=0.5)
    parser.add_argument("--duration-ps", type=float, default=40.0)
    parser.add_argument("--equilibration-ps", type=float, default=10.0)
    parser.add_argument("--sample-fs", type=float, default=10.0)
    parser.add_argument("--kappa", type=float, default=20.0)
    parser.add_argument("--minimum-reactive-sum-A", type=float, default=3.20)
    parser.add_argument("--maximum-methyl-CH-A", type=float, default=1.35)
    parser.add_argument("--disable-chemistry-guard", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.replica < 0:
        raise ValueError("Replica index must be non-negative")
    if (
        args.temperature <= 0.0
        or args.time_step_fs <= 0.0
        or args.duration_ps <= 0.0
        or args.kappa <= 0.0
        or args.minimum_reactive_sum_A < 0.0
        or args.maximum_methyl_CH_A <= 0.0
    ):
        raise ValueError("Temperature, timestep, duration, and kappa must be positive")
    if not 0.0 <= args.equilibration_ps < args.duration_ps:
        raise ValueError("Equilibration must satisfy 0 <= equilibration < duration")

    sample_stride = int(round(args.sample_fs / args.time_step_fs))
    if not np.isclose(sample_stride * args.time_step_fs, args.sample_fs):
        raise ValueError("The timestep must exactly represent the output interval")
    steps = int(round(1000.0 * args.duration_ps / args.time_step_fs))
    centers_path = (
        args.window_centers_file.resolve()
        if args.window_centers_file is not None
        else None
    )
    if centers_path is not None and not centers_path.is_file():
        raise FileNotFoundError(f"Window-center file not found: {centers_path}")
    irc_frames, window_centers = irc_structures_and_centers(
        args.irc.resolve(), centers_path
    )
    if not 0 <= args.window < len(window_centers):
        raise ValueError(
            f"Window must lie between 0 and {len(window_centers) - 1}"
        )
    center = float(window_centers[args.window])

    output = args.output
    if output is None:
        output = (
            EXAMPLE_ROOT
            / "results/umbrella"
            / f"replica_{args.replica:02d}"
            / f"window_{args.window:02d}"
        )
    output = output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output}. Pass --overwrite or "
            "choose another --output directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for filename in GENERATED_FILES:
            path = output / filename
            if path.is_file():
                path.unlink()

    atoms = nearest_irc_structure(irc_frames, center)
    atoms.pbc = False
    atoms.set_cell(np.zeros((3, 3)))
    initial_coordinate = reaction_coordinate(atoms)
    colvar = output / "COLVAR"
    input_lines = plumed_lines(colvar, center, args.kappa, sample_stride)
    (output / "plumed.dat").write_text("\n".join(input_lines) + "\n")

    random_seed = 303000 + 100 * args.replica + args.window
    model_path = args.model.resolve()
    metadata = {
        "source_protocol": "Vitartas et al., Digital Discovery 5, 108-122 (2026)",
        "reaction": "R2 2,2-dimethylisoindene methyl shift",
        "method": "umbrella sampling",
        "model": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "irc": str(args.irc.resolve()),
        "replica": args.replica,
        "window_index": args.window,
        "window_count": len(window_centers),
        "window_centers_file": str(centers_path) if centers_path is not None else None,
        "window_centers_sha256": (
            hashlib.sha256(centers_path.read_bytes()).hexdigest()
            if centers_path is not None
            else None
        ),
        "first_window_center_A": float(window_centers[0]),
        "last_window_center_A": float(window_centers[-1]),
        "window_center_A": center,
        "initial_coordinate_A": initial_coordinate,
        "random_seed": random_seed,
        "temperature_K": args.temperature,
        "time_step_fs": args.time_step_fs,
        "duration_ps": args.duration_ps,
        "equilibration_discard_ps": args.equilibration_ps,
        "sample_interval_fs": args.sample_fs,
        "steps": steps,
        "harmonic_kappa_eV_per_A2": args.kappa,
        "harmonic_convention": "0.5*kappa*(s-center)^2",
        "chemistry_guard_enabled": not args.disable_chemistry_guard,
        "minimum_reactive_distance_sum_A": args.minimum_reactive_sum_A,
        "maximum_methyl_CH_distance_A": args.maximum_methyl_CH_A,
        "langevin_friction_inverse_ASE_time": 0.02,
        "center_of_mass_constraint": "ase.constraints.FixCom",
        "collective_variable": "r1(14,11) - r2(14,10), zero-based atom indices",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.write_only:
        print(f"Wrote {output / 'plumed.dat'}")
        return

    try:
        import plumed  # noqa: F401
        from ase.calculators.plumed import Plumed
    except ImportError as exc:
        raise ImportError(
            "The PLUMED Python module is unavailable. Activate the TRACE "
            "environment and source PLUMED's sourceme.sh."
        ) from exc

    class ScalarEnergyPlumed(Plumed):
        """Normalize the PLUMED bias energy for ASE with NumPy 2."""

        def compute_energy_and_forces(self, positions, step):
            energy, forces = super().compute_energy_and_forces(positions, step)
            return float(np.asarray(energy).reshape(-1)[0]), forces

    base_calculator = TransformersACECalculator(
        model_path=str(args.model.resolve()), device=args.device
    )
    atoms.calc = ScalarEnergyPlumed(
        calc=base_calculator,
        input=input_lines,
        timestep=args.time_step_fs * units.fs,
        atoms=atoms,
        kT=units.kB * args.temperature,
        log=str(output / "plumed.log"),
    )

    rng = np.random.default_rng(random_seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)
    atoms.set_constraint(FixCom())
    dynamics = Langevin(
        atoms,
        args.time_step_fs * units.fs,
        temperature_K=args.temperature,
        friction=0.02,
        rng=rng,
        fixcm=False,
    )
    status = {"state": "running", "completed_steps": 0, "reason": None}

    def check_stability() -> None:
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces())
        maximum_force = float(np.linalg.norm(forces, axis=1).max())
        minimum_distance = minimum_pair_distance(atoms)
        r1 = float(np.linalg.norm(atoms.positions[14] - atoms.positions[11]))
        r2 = float(np.linalg.norm(atoms.positions[14] - atoms.positions[10]))
        reactive_sum = r1 + r2
        maximum_methyl_ch = max(
            float(np.linalg.norm(atoms.positions[14] - atoms.positions[index]))
            for index in (20, 21, 22)
        )
        chemistry_failure = (
            not args.disable_chemistry_guard
            and (
                reactive_sum < args.minimum_reactive_sum_A
                or maximum_methyl_ch > args.maximum_methyl_CH_A
            )
        )
        finite = (
            np.isfinite(atoms.positions).all()
            and np.isfinite(energy)
            and np.isfinite(forces).all()
        )
        if (
            not finite
            or minimum_distance < 0.55
            or maximum_force > 100.0
            or chemistry_failure
        ):
            status.update(
                state="aborted_unphysical",
                completed_steps=int(dynamics.nsteps),
                reason=(
                    f"finite={finite}, minimum_pair_distance_A={minimum_distance:.6f}, "
                    f"maximum_force_eV_per_A={maximum_force:.6f}, "
                    f"r1_A={r1:.6f}, r2_A={r2:.6f}, "
                    f"reactive_distance_sum_A={reactive_sum:.6f}, "
                    f"maximum_methyl_CH_distance_A={maximum_methyl_ch:.6f}"
                ),
            )
            (output / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
            raise RuntimeError(status["reason"])

    trajectory = Trajectory(output / "trajectory.traj", "w", atoms)
    try:
        dynamics.attach(check_stability, interval=sample_stride)
        dynamics.attach(trajectory.write, interval=sample_stride)
        dynamics.attach(
            MDLogger(
                dynamics,
                atoms,
                output / "md.log",
                header=True,
                stress=False,
                peratom=False,
                mode="w",
            ),
            interval=sample_stride,
        )
        dynamics.run(steps)
        status.update(state="complete", completed_steps=int(dynamics.nsteps))
        (output / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
    except Exception as exc:
        if status["state"] == "running":
            status.update(
                state="failed",
                completed_steps=int(dynamics.nsteps),
                reason=f"{type(exc).__name__}: {exc}",
            )
            (output / "run_status.json").write_text(
                json.dumps(status, indent=2) + "\n"
            )
        raise
    finally:
        trajectory.close()
        if hasattr(atoms.calc, "plumed"):
            atoms.calc.plumed.finalize()


if __name__ == "__main__":
    main()
