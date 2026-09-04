#!/usr/bin/env python3
"""Evaluate a TRACE checkpoint on an untouched reaction dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from transformers_ace import TransformersACECalculator  # noqa: E402


def coordinate_name(atoms) -> str | None:
    names = sorted(key for key in atoms.info if key.startswith("cv_"))
    if len(names) > 1:
        raise ValueError(f"Expected at most one CV field, found {names}")
    return names[0] if names else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--transition-min", type=float, default=-0.20)
    parser.add_argument("--transition-max", type=float, default=0.00)
    args = parser.parse_args()

    frames = read(args.data, index=":")
    if not frames:
        raise ValueError(f"No structures found in {args.data}")
    if any(atoms.pbc.any() for atoms in frames):
        raise ValueError("Reaction test data must be nonperiodic")
    atom_counts = {len(atoms) for atoms in frames}
    if len(atom_counts) != 1:
        raise ValueError("This evaluator expects one fixed molecular composition")

    cv_name = coordinate_name(frames[0])
    calculator = TransformersACECalculator(
        model_path=str(args.model.resolve()), device=args.device
    )
    records = []
    energy_errors_per_atom = []
    force_errors_by_frame = []

    for index, atoms in enumerate(frames):
        reference_energy = float(atoms.get_potential_energy())
        reference_forces = np.asarray(atoms.get_forces(), dtype=float)
        source = str(atoms.info.get("source", "unknown"))
        cv = float(atoms.info[cv_name]) if cv_name else float("nan")

        predicted = atoms.copy()
        predicted.calc = calculator
        predicted_energy = float(predicted.get_potential_energy())
        predicted_forces = np.asarray(predicted.get_forces(), dtype=float)

        energy_error_per_atom = (predicted_energy - reference_energy) / len(atoms)
        force_error = predicted_forces - reference_forces
        energy_errors_per_atom.append(energy_error_per_atom)
        force_errors_by_frame.append(force_error.reshape(-1))
        records.append(
            {
                "frame": index,
                "source": source,
                "cv_A": cv,
                "reference_energy_eV": reference_energy,
                "predicted_energy_eV": predicted_energy,
                "energy_error_meV_per_atom": 1000.0 * energy_error_per_atom,
                "force_rmse_meV_per_A": 1000.0 * float(np.sqrt(np.mean(force_error**2))),
                "force_mae_meV_per_A": 1000.0 * float(np.mean(np.abs(force_error))),
                "maximum_force_error_meV_per_A": 1000.0
                * float(np.linalg.norm(force_error, axis=1).max()),
            }
        )

    energy_errors_per_atom = np.asarray(energy_errors_per_atom)
    force_errors = np.concatenate(force_errors_by_frame)
    metrics = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "frames": len(frames),
        "atoms_per_frame": next(iter(atom_counts)),
        "energy_mae_meV_per_atom": 1000.0
        * float(np.mean(np.abs(energy_errors_per_atom))),
        "energy_rmse_meV_per_atom": 1000.0
        * float(np.sqrt(np.mean(energy_errors_per_atom**2))),
        "force_mae_meV_per_A": 1000.0 * float(np.mean(np.abs(force_errors))),
        "force_rmse_meV_per_A": 1000.0
        * float(np.sqrt(np.mean(force_errors**2))),
    }
    if cv_name is not None:
        coordinates = np.asarray([record["cv_A"] for record in records])
        transition = (
            (coordinates >= args.transition_min)
            & (coordinates <= args.transition_max)
        )
        if transition.any():
            transition_energy = energy_errors_per_atom[transition]
            transition_forces = np.concatenate(
                [
                    errors
                    for selected, errors in zip(transition, force_errors_by_frame)
                    if selected
                ]
            )
            metrics["transition_region"] = {
                "cv_name": cv_name,
                "cv_min_A": args.transition_min,
                "cv_max_A": args.transition_max,
                "frames": int(transition.sum()),
                "energy_mae_meV_per_atom": 1000.0
                * float(np.mean(np.abs(transition_energy))),
                "energy_rmse_meV_per_atom": 1000.0
                * float(np.sqrt(np.mean(transition_energy**2))),
                "force_mae_meV_per_A": 1000.0
                * float(np.mean(np.abs(transition_forces))),
                "force_rmse_meV_per_A": 1000.0
                * float(np.sqrt(np.mean(transition_forces**2))),
            }
        else:
            metrics["transition_region"] = {
                "cv_name": cv_name,
                "cv_min_A": args.transition_min,
                "cv_max_A": args.transition_max,
                "frames": 0,
            }

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "predictions.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    try:
        import matplotlib.pyplot as plt

        x = np.arange(len(records)) if cv_name is None else np.asarray(
            [record["cv_A"] for record in records]
        )
        order = np.argsort(x)
        reference = np.asarray([record["reference_energy_eV"] for record in records])
        predicted = np.asarray([record["predicted_energy_eV"] for record in records])
        reference -= reference.min()
        predicted -= predicted.min()

        figure, axes = plt.subplots(2, 1, figsize=(7.0, 6.5), sharex=True)
        axes[0].plot(x[order], reference[order], color="#222222", label="Reference")
        axes[0].plot(x[order], predicted[order], color="#0072B2", label="TRACE")
        axes[0].set_ylabel("Relative energy (eV)")
        axes[0].legend(frameon=False)
        axes[1].scatter(
            x,
            1000.0 * energy_errors_per_atom,
            s=12,
            color="#D55E00",
            alpha=0.75,
        )
        axes[1].axhline(0.0, color="#222222", linewidth=0.8)
        axes[1].set_ylabel("Error (meV/atom)")
        axes[1].set_xlabel(cv_name.replace("_", " ") if cv_name else "Frame")
        figure.tight_layout()
        figure.savefig(args.output / "energy_profile.png", dpi=220)
        plt.close(figure)
    except ImportError:
        pass

    print(json.dumps(metrics, indent=2))
    print(f"Per-frame predictions: {csv_path}")


if __name__ == "__main__":
    main()
