#!/usr/bin/env python3
"""Reconstruct one standard or inherited-bias R2 WTMetaD profile."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase.io import read, write


EXAMPLE_ROOT = Path(__file__).resolve().parent
KB_EV_PER_K = 8.617333262145e-5
EV_TO_KCAL_MOL = 23.06054783061903


def plumed_executable(requested: str | None) -> str:
    if requested is not None:
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PLUMED executable not found: {path}")
        return str(path)
    path = shutil.which("plumed")
    if path is None:
        raise FileNotFoundError(
            "PLUMED was not found on PATH. Source PLUMED's sourceme.sh or "
            "pass --plumed /absolute/path/to/plumed."
        )
    return path


def completed_run(replica_root: Path, allow_incomplete: bool) -> dict:
    metadata_path = replica_root / "run_metadata.json"
    status_path = replica_root / "run_status.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if not allow_incomplete:
        if not status_path.is_file():
            raise FileNotFoundError(f"Run status not found: {status_path}")
        status = json.loads(status_path.read_text())
        if status.get("state") != "complete":
            raise RuntimeError(f"The replica is not complete: {status}")
    return metadata


def run_command(command: list[str], cwd: Path) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"PLUMED failed with exit code {result.returncode}:\n{result.stdout}"
        )


def standard_profile(
    executable: str,
    hills: Path,
    grid_min: float,
    grid_max: float,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="trace-r2-sum-hills-") as directory:
        temporary = Path(directory)
        output = temporary / "fes.dat"
        run_command(
            [
                executable,
                "sum_hills",
                "--hills",
                str(hills),
                "--outfile",
                str(output),
                "--min",
                str(grid_min),
                "--max",
                str(grid_max),
                "--bin",
                str(bins - 1),
                "--mintozero",
            ],
            cwd=temporary,
        )
        data = np.loadtxt(output, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("PLUMED sum_hills did not produce a one-dimensional FES")
    return np.asarray(data[:, 0]), np.asarray(data[:, 1])


def inherited_profile(
    executable: str,
    replica_root: Path,
    metadata: dict,
    grid_min: float,
    grid_max: float,
    bins: int,
    bandwidth: float,
    discard_ps: float,
) -> tuple[np.ndarray, np.ndarray]:
    trajectory_path = replica_root / "trajectory.traj"
    hills_path = replica_root / "HILLS"
    if not trajectory_path.is_file() or not hills_path.is_file():
        raise FileNotFoundError("Inherited-bias reweighting requires trajectory.traj and HILLS")

    frames = read(trajectory_path, index=":")
    sample_fs = float(metadata["sample_interval_fs"])
    discard_frames = int(round(1000.0 * discard_ps / sample_fs))
    if discard_frames >= len(frames):
        raise ValueError("The requested discard time removes every trajectory frame")
    frames = frames[discard_frames:]
    if len(frames) < 100:
        raise ValueError("At least 100 saved frames are required for reweighting")

    temperature = float(metadata["temperature_K"])
    bias_factor = int(metadata["bias_factor"])
    if bias_factor != 80:
        raise ValueError("The released inherited-bias production protocol uses BIASFACTOR=80")

    with tempfile.TemporaryDirectory(prefix="trace-r2-reweight-") as directory:
        temporary = Path(directory)
        trajectory_xyz = temporary / "trajectory.xyz"
        copied_hills = temporary / "HILLS"
        histogram = temporary / "hist.dat"
        input_path = temporary / "reweight.dat"
        write(trajectory_xyz, frames, format="xyz")
        shutil.copyfile(hills_path, copied_hills)

        input_lines = [
            "UNITS LENGTH=A ENERGY=eV TIME=ps",
            "r_1: DISTANCE ATOMS=15,12 NOPBC",
            "r_2: DISTANCE ATOMS=15,11 NOPBC",
            "diff_r: COMBINE ARG=r_1,r_2 COEFFICIENTS=1,-1 PERIODIC=NO",
            (
                "metad: METAD ARG=diff_r SIGMA=0.07 HEIGHT=0.0158 "
                f"PACE=100000000 BIASFACTOR={bias_factor} TEMP={temperature:.8g} "
                "RESTART=YES UPDATE_FROM=1e30 FILE=HILLS"
            ),
            f"weights: REWEIGHT_BIAS TEMP={temperature:.8g} ARG=metad.bias",
            (
                "hist: HISTOGRAM ARG=diff_r STRIDE=1 "
                f"GRID_MIN={grid_min:.17g} GRID_MAX={grid_max:.17g} "
                f"GRID_BIN={bins - 1} BANDWIDTH={bandwidth:.17g} "
                "LOGWEIGHTS=weights"
            ),
            "DUMPGRID GRID=hist FILE=hist.dat",
        ]
        input_path.write_text("\n".join(input_lines) + "\n")
        run_command(
            [
                executable,
                "driver",
                "--ixyz",
                str(trajectory_xyz),
                "--plumed",
                str(input_path),
                "--length-units",
                "A",
                "--box",
                "100,100,100",
                "--timestep",
                f"{sample_fs / 1000.0:.17g}",
                "--trajectory-stride",
                "1",
            ],
            cwd=temporary,
        )
        data = np.loadtxt(histogram, comments="#")

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("PLUMED reweighting did not produce a one-dimensional histogram")
    coordinate = np.asarray(data[:, 0])
    probability = np.asarray(data[:, 1])
    free_energy = np.full_like(probability, np.nan, dtype=float)
    positive = np.isfinite(probability) & (probability > 0.0)
    free_energy[positive] = -KB_EV_PER_K * temperature * np.log(probability[positive])
    return coordinate, free_energy


def align_and_measure(
    coordinate: np.ndarray, free_energy: np.ndarray
) -> tuple[np.ndarray, dict]:
    finite = np.isfinite(coordinate) & np.isfinite(free_energy)
    reactant = finite & (coordinate >= -1.20) & (coordinate <= -0.60)
    transition = finite & (coordinate >= -0.25) & (coordinate <= 0.05)
    product = finite & (coordinate >= 0.60) & (coordinate <= 1.20)
    if not reactant.any() or not transition.any() or not product.any():
        raise ValueError("The profile does not cover the reactant, TS, and product regions")
    reference = float(np.min(free_energy[reactant]))
    relative = free_energy - reference
    transition_indices = np.flatnonzero(transition)
    transition_index = transition_indices[int(np.nanargmax(relative[transition]))]
    product_indices = np.flatnonzero(product)
    product_index = product_indices[int(np.nanargmin(relative[product]))]
    result = {
        "transition_coordinate_A": float(coordinate[transition_index]),
        "activation_free_energy_eV": float(relative[transition_index]),
        "activation_free_energy_kcal_per_mol": float(
            relative[transition_index] * EV_TO_KCAL_MOL
        ),
        "product_coordinate_A": float(coordinate[product_index]),
        "product_minus_reactant_eV": float(relative[product_index]),
        "product_minus_reactant_kcal_per_mol": float(
            relative[product_index] * EV_TO_KCAL_MOL
        ),
    }
    return relative, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("standard", "inherited"), default=None)
    parser.add_argument("--plumed", default=None)
    parser.add_argument("--grid-min", type=float, default=-1.30)
    parser.add_argument("--grid-max", type=float, default=1.30)
    parser.add_argument("--bins", type=int, default=500)
    parser.add_argument("--bandwidth", type=float, default=0.02)
    parser.add_argument("--discard-ps", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    replica_root = args.replica_root.resolve()
    metadata = completed_run(replica_root, args.allow_incomplete)
    mode = args.mode or metadata.get("mode")
    if mode not in ("standard", "inherited"):
        raise ValueError("Could not infer standard or inherited mode from metadata")
    if metadata.get("mode") != mode:
        raise ValueError(f"Requested mode {mode!r} disagrees with run metadata")
    if args.bins < 50 or args.grid_min >= args.grid_max:
        raise ValueError("Use at least 50 bins and an increasing grid interval")
    if args.bandwidth <= 0.0 or args.discard_ps < 0.0:
        raise ValueError("Bandwidth must be positive and discard time non-negative")

    executable = plumed_executable(args.plumed)
    if mode == "standard":
        coordinate, free_energy = standard_profile(
            executable,
            replica_root / "HILLS",
            args.grid_min,
            args.grid_max,
            args.bins,
        )
        estimator = "PLUMED sum_hills well-tempered estimator"
    else:
        coordinate, free_energy = inherited_profile(
            executable,
            replica_root,
            metadata,
            args.grid_min,
            args.grid_max,
            args.bins,
            args.bandwidth,
            args.discard_ps,
        )
        estimator = "final-bias reweighting with a continuous kernel histogram"

    relative, result = align_and_measure(coordinate, free_energy)
    result.update(
        mode=mode,
        replica=int(metadata["replica"]),
        estimator=estimator,
        temperature_K=float(metadata["temperature_K"]),
        grid_min_A=args.grid_min,
        grid_max_A=args.grid_max,
        grid_points=len(coordinate),
        reweighting_bandwidth_A=args.bandwidth if mode == "inherited" else None,
        discarded_trajectory_ps=args.discard_ps if mode == "inherited" else None,
        source_replica_root=str(replica_root),
    )

    output = args.output
    if output is None:
        output = replica_root / f"fes_{mode}.csv"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["cv_r1_minus_r2_A", "free_energy_eV", "free_energy_kcal_per_mol"]
        )
        for value, energy in zip(coordinate, relative):
            writer.writerow(
                [
                    value,
                    energy,
                    energy * EV_TO_KCAL_MOL if np.isfinite(energy) else np.nan,
                ]
            )
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"FES: {output}")


if __name__ == "__main__":
    main()
