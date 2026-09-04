#!/usr/bin/env python3
"""Run one source-matched R2 WTMetaD replica with TRACE and ASE/PLUMED."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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


GENERATED_FILES = (
    "COLVAR",
    "HILLS",
    "md.log",
    "plumed.dat",
    "plumed.log",
    "run_metadata.json",
    "run_status.json",
    "trajectory.traj",
)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_r2_structure(atoms) -> None:
    if len(atoms) != 23 or any(atoms.numbers[index] != 6 for index in (10, 11, 14)):
        raise ValueError(
            "Expected the released 23-atom R2 ordering with carbon atoms at "
            "zero-based indices 10, 11, and 14"
        )
    if atoms.pbc.any():
        raise ValueError("R2 is a finite gas-phase molecule and must be nonperiodic")


def plumed_lines(
    hills: Path,
    colvar: Path,
    temperature: float,
    pace: int,
    print_stride: int,
    mode: str,
) -> list[str]:
    bias_factor = 50 if mode == "standard" else 80
    restart = " RESTART=YES" if mode == "inherited" else ""
    return [
        "UNITS LENGTH=A TIME=ps ENERGY=eV",
        "r_1: DISTANCE ATOMS=15,12 NOPBC",
        "r_2: DISTANCE ATOMS=15,11 NOPBC",
        "diff_r: COMBINE ARG=r_1,r_2 COEFFICIENTS=1,-1 PERIODIC=NO",
        (
            "metad: METAD ARG=diff_r SIGMA=0.07 HEIGHT=0.0158 "
            f"PACE={pace} BIASFACTOR={bias_factor} TEMP={temperature:.8g}"
            f"{restart} FILE={hills}"
        ),
        (
            "PRINT ARG=r_1,r_2,diff_r,metad.bias "
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
    parser.add_argument(
        "--model",
        type=Path,
        default=EXAMPLE_ROOT / "models/r2_trace_v2_wtmetad.pt",
    )
    parser.add_argument(
        "--structure", type=Path, default=EXAMPLE_ROOT / "structures/r_r2.xyz"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--replica", type=int, default=0)
    parser.add_argument("--mode", choices=("standard", "inherited"), default="standard")
    parser.add_argument(
        "--inherited-bias",
        type=Path,
        default=EXAMPLE_ROOT / "inherited_bias/bias_after_iter_15.dat",
    )
    parser.add_argument("--temperature", type=float, default=365.6)
    parser.add_argument("--time-step-fs", type=float, default=0.5)
    parser.add_argument("--duration-ps", type=float, default=None)
    parser.add_argument("--sample-fs", type=float, default=10.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.replica < 0:
        raise ValueError("Replica index must be non-negative")
    duration_ps = args.duration_ps
    if duration_ps is None:
        duration_ps = 500.0 if args.mode == "standard" else 250.0
    if args.temperature <= 0.0 or args.time_step_fs <= 0.0 or duration_ps <= 0.0:
        raise ValueError("Temperature, timestep, and duration must be positive")
    if args.sample_fs <= 0.0:
        raise ValueError("The output interval must be positive")

    pace = int(round(100.0 / args.time_step_fs))
    if not np.isclose(pace * args.time_step_fs, 100.0):
        raise ValueError("The timestep must exactly represent the published 100 fs hill pace")
    sample_stride = int(round(args.sample_fs / args.time_step_fs))
    if not np.isclose(sample_stride * args.time_step_fs, args.sample_fs):
        raise ValueError("The timestep must exactly represent the output interval")
    steps = int(round(1000.0 * duration_ps / args.time_step_fs))

    output = args.output
    if output is None:
        output = EXAMPLE_ROOT / "results" / args.mode / f"replica_{args.replica:02d}"
    output = output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output}. Pass --overwrite to "
            "replace this run, or choose a new directory with --output."
        )
    output.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for filename in GENERATED_FILES:
            path = output / filename
            if path.is_file():
                path.unlink()

    atoms = read(args.structure, index=0)
    atoms.pbc = False
    atoms.set_cell(np.zeros((3, 3)))
    validate_r2_structure(atoms)

    hills = output / "HILLS"
    inherited_bias_sha256 = None
    if args.mode == "inherited":
        source_bias = args.inherited_bias.resolve()
        if not source_bias.is_file():
            raise FileNotFoundError(f"Published inherited bias not found: {source_bias}")
        shutil.copyfile(source_bias, hills)
        inherited_bias_sha256 = sha256sum(source_bias)

    colvar = output / "COLVAR"
    input_lines = plumed_lines(
        hills, colvar, args.temperature, pace, sample_stride, args.mode
    )
    (output / "plumed.dat").write_text("\n".join(input_lines) + "\n")
    metadata = {
        "source_protocol": "Vitartas et al., Digital Discovery 5, 108-122 (2026)",
        "reaction": "R2 2,2-dimethylisoindene methyl shift",
        "model": str(args.model.resolve()),
        "structure": str(args.structure.resolve()),
        "mode": args.mode,
        "replica": args.replica,
        "random_seed": 202600 + args.replica,
        "temperature_K": args.temperature,
        "time_step_fs": args.time_step_fs,
        "duration_ps": duration_ps,
        "steps": steps,
        "sample_interval_fs": args.sample_fs,
        "langevin_friction_inverse_ASE_time": 0.02,
        "center_of_mass_constraint": "ase.constraints.FixCom",
        "collective_variable": "r1(14,11) - r2(14,10), zero-based atom indices",
        "hill_sigma_A": 0.07,
        "hill_height_eV": 0.0158,
        "hill_pace_fs": 100.0,
        "bias_factor": 50 if args.mode == "standard" else 80,
        "inherited_bias_source": (
            str(args.inherited_bias.resolve()) if args.mode == "inherited" else None
        ),
        "inherited_bias_sha256": inherited_bias_sha256,
        "stability_abort_criteria": {
            "minimum_pair_distance_A": 0.55,
            "maximum_atomic_force_eV_per_A": 100.0,
            "all_positions_energies_forces_finite": True,
        },
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
            "environment, source PLUMED's sourceme.sh, and verify that "
            "`python -c \"import plumed\"` succeeds."
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
        restart=False,
    )

    rng = np.random.default_rng(metadata["random_seed"])
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
        finite = (
            np.isfinite(atoms.positions).all()
            and np.isfinite(energy)
            and np.isfinite(forces).all()
        )
        if not finite or minimum_distance < 0.55 or maximum_force > 100.0:
            status.update(
                state="aborted_unphysical",
                completed_steps=int(dynamics.nsteps),
                reason=(
                    f"finite={finite}, minimum_pair_distance_A={minimum_distance:.6f}, "
                    f"maximum_force_eV_per_A={maximum_force:.6f}"
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
    finally:
        trajectory.close()
        if hasattr(atoms.calc, "plumed"):
            atoms.calc.plumed.finalize()


if __name__ == "__main__":
    main()
