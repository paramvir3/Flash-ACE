#!/usr/bin/env python3
"""Audit R2 umbrella trajectories before accepting a free-energy profile."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ase.data import covalent_radii
from ase.io import read, write
from ase.io.trajectory import Trajectory


EXAMPLE_ROOT = Path(__file__).resolve().parent
REACTIVE_PAIRS = {(10, 14), (11, 14)}


def integrated_autocorrelation_time(values: np.ndarray) -> float:
    """Estimate tau_int with Geyer's initial-positive paired sequence."""
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    count = len(centered)
    if count < 4 or not np.any(centered):
        return 0.5
    fft_size = 1 << (2 * count - 1).bit_length()
    transform = np.fft.rfft(centered, fft_size)
    autocovariance = np.fft.irfft(transform * np.conjugate(transform), fft_size)[:count]
    autocovariance /= np.arange(count, 0, -1)
    autocorrelation = autocovariance / autocovariance[0]
    positive_sum = 0.0
    for index in range(1, count - 1, 2):
        pair_sum = autocorrelation[index] + autocorrelation[index + 1]
        if pair_sum <= 0.0:
            break
        positive_sum += float(pair_sum)
    return max(0.5, 0.5 + positive_sum)


def endpoint_bonds(reactant, product) -> set[tuple[int, int]]:
    def bonds(atoms) -> set[tuple[int, int]]:
        result = set()
        for first in range(len(atoms)):
            for second in range(first):
                distance = np.linalg.norm(
                    atoms.positions[first] - atoms.positions[second]
                )
                cutoff = 1.25 * (
                    covalent_radii[atoms.numbers[first]]
                    + covalent_radii[atoms.numbers[second]]
                )
                if distance < cutoff:
                    result.add((second, first))
        return result

    return (bonds(reactant) & bonds(product)) - REACTIVE_PAIRS


def reactive_coordinates(atoms) -> tuple[float, float]:
    r1 = float(np.linalg.norm(atoms.positions[14] - atoms.positions[11]))
    r2 = float(np.linalg.norm(atoms.positions[14] - atoms.positions[10]))
    return r1, r2


def load_colvar(window: Path, metadata: dict) -> np.ndarray:
    data = np.loadtxt(window / "COLVAR", comments="#")
    if data.ndim == 1:
        data = data[None, :]
    discard = int(
        round(
            1000.0
            * float(metadata["equilibration_discard_ps"])
            / float(metadata["sample_interval_fs"])
        )
    )
    return data[discard:]


def load_temperature(window: Path, equilibration_ps: float) -> np.ndarray:
    data = np.loadtxt(window / "md.log", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return data[data[:, 0] >= equilibration_ps - 1.0e-12, 4]


def farthest_point_candidates(candidates, count: int):
    if len(candidates) <= count:
        return [item[0] for item in candidates]
    descriptors = []
    for atoms, _ in candidates:
        distances = np.linalg.norm(
            atoms.positions[:, None, :] - atoms.positions[None, :, :], axis=2
        )
        descriptors.append(distances[np.triu_indices(len(atoms), 1)])
    descriptors = np.asarray(descriptors)
    descriptors /= np.maximum(np.std(descriptors, axis=0), 0.05)
    sums = np.asarray([item[1]["reactive_distance_sum_A"] for item in candidates])
    selected = [int(np.argmin(sums))]
    minimum_distance = np.full(len(candidates), np.inf)
    while len(selected) < count:
        latest = descriptors[selected[-1]]
        distance = np.sqrt(np.mean((descriptors - latest) ** 2, axis=1))
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -np.inf
        selected.append(int(np.argmax(minimum_distance)))
    return [candidates[index][0] for index in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--umbrella-root",
        type=Path,
        default=EXAMPLE_ROOT / "results/full3/umbrella",
    )
    parser.add_argument(
        "--output", type=Path, default=EXAMPLE_ROOT / "results/full3/audit"
    )
    parser.add_argument("--trajectory-stride", type=int, default=10)
    parser.add_argument("--minimum-overlap", type=float, default=0.15)
    parser.add_argument("--minimum-effective-samples", type=float, default=100.0)
    parser.add_argument("--maximum-half-drift-A", type=float, default=0.02)
    parser.add_argument("--minimum-reactive-sum-A", type=float, default=3.20)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument(
        "--chemistry-only",
        action="store_true",
        help="Treat this as a short stability gate, not a free-energy audit.",
    )
    args = parser.parse_args()

    if args.trajectory_stride < 1 or args.candidate_count < 1:
        raise ValueError("Trajectory stride and candidate count must be positive")
    umbrella_root = args.umbrella_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    reactant = read(EXAMPLE_ROOT / "structures/r_r2.xyz")
    product = read(EXAMPLE_ROOT / "structures/p_r2.xyz")
    fixed_bonds = endpoint_bonds(reactant, product)
    symbols = reactant.get_chemical_symbols()
    window_paths = sorted(umbrella_root.glob("replica_*/window_*"))
    if not window_paths:
        raise FileNotFoundError(f"No umbrella windows found below {umbrella_root}")

    rows = []
    series = {}
    candidates = []
    unexpected_pairs = Counter()
    broken_pairs = Counter()
    total_audited_frames = 0

    for window in window_paths:
        metadata = json.loads((window / "run_metadata.json").read_text())
        status = json.loads((window / "run_status.json").read_text())
        replica = int(metadata["replica"])
        window_index = int(metadata["window_index"])
        colvar = load_colvar(window, metadata)
        coordinate = np.asarray(colvar[:, 3], dtype=float)
        midpoint = len(coordinate) // 2
        tau = integrated_autocorrelation_time(coordinate)
        temperature = load_temperature(
            window, float(metadata["equilibration_discard_ps"])
        )
        series[(replica, window_index)] = coordinate

        trajectory = Trajectory(str(window / "trajectory.traj"), "r")
        first_frame = int(
            round(
                1000.0
                * float(metadata["equilibration_discard_ps"])
                / float(metadata["sample_interval_fs"])
            )
        )
        audited = collapsed = fixed_breaks = unexpected = 0
        minimum_r1 = minimum_r2 = minimum_sum = np.inf
        maximum_diameter = 0.0
        for frame_index in range(first_frame, len(trajectory), args.trajectory_stride):
            atoms = trajectory[frame_index]
            distances = np.linalg.norm(
                atoms.positions[:, None, :] - atoms.positions[None, :, :], axis=2
            )
            r1, r2 = reactive_coordinates(atoms)
            reactive_sum = r1 + r2
            minimum_r1 = min(minimum_r1, r1)
            minimum_r2 = min(minimum_r2, r2)
            minimum_sum = min(minimum_sum, reactive_sum)
            maximum_diameter = max(maximum_diameter, float(np.max(distances)))
            audited += 1

            collapsed_frame = r1 < 1.70 and r2 < 1.70
            broken_frame = False
            for first, second in fixed_bonds:
                limit = 1.35 if "H" in (symbols[first], symbols[second]) else 1.90
                if distances[first, second] > limit:
                    broken_frame = True
                    broken_pairs[(first, second)] += 1

            unexpected_frame = False
            for first in range(len(atoms)):
                for second in range(first):
                    pair = (second, first)
                    if pair in fixed_bonds or pair in REACTIVE_PAIRS:
                        continue
                    cutoff = 1.12 * (
                        covalent_radii[atoms.numbers[first]]
                        + covalent_radii[atoms.numbers[second]]
                    )
                    if distances[first, second] < cutoff:
                        unexpected_frame = True
                        unexpected_pairs[pair] += 1

            collapsed += int(collapsed_frame)
            fixed_breaks += int(broken_frame)
            unexpected += int(unexpected_frame)
            if collapsed_frame or broken_frame or unexpected_frame:
                saved = atoms.copy()
                saved.calc = None
                saved.info.update(
                    source_replica=replica,
                    source_window=window_index,
                    source_frame=frame_index,
                    r1_A=r1,
                    r2_A=r2,
                    cv_r1_minus_r2_A=r1 - r2,
                    reactive_distance_sum_A=reactive_sum,
                    collapsed_dual_CC=collapsed_frame,
                    persistent_bond_break=broken_frame,
                    unexpected_close_contact=unexpected_frame,
                    dft_labels_present=False,
                )
                candidates.append((saved, dict(saved.info)))

        total_audited_frames += audited
        rows.append(
            {
                "replica": replica,
                "window": window_index,
                "state": status.get("state"),
                "center_A": float(metadata["window_center_A"]),
                "production_samples": len(coordinate),
                "mean_cv_A": float(np.mean(coordinate)),
                "std_cv_A": float(np.std(coordinate, ddof=1)),
                "half_drift_A": float(
                    np.mean(coordinate[midpoint:]) - np.mean(coordinate[:midpoint])
                ),
                "tau_int_samples": tau,
                "effective_samples": len(coordinate) / (2.0 * tau),
                "mean_temperature_K": float(np.mean(temperature)),
                "audited_frames": audited,
                "collapsed_fraction": collapsed / audited,
                "persistent_bond_break_fraction": fixed_breaks / audited,
                "unexpected_contact_fraction": unexpected / audited,
                "minimum_r1_A": minimum_r1,
                "minimum_r2_A": minimum_r2,
                "minimum_reactive_sum_A": minimum_sum,
                "maximum_molecular_diameter_A": maximum_diameter,
                "overlap_with_next": np.nan,
            }
        )

    row_lookup = {(row["replica"], row["window"]): row for row in rows}
    replica_indices = sorted({row["replica"] for row in rows})
    overlaps = []
    for replica in replica_indices:
        replica_keys = sorted(key for key in series if key[0] == replica)
        values = [series[key] for key in replica_keys]
        low = min(float(item.min()) for item in values)
        high = max(float(item.max()) for item in values)
        edges = np.linspace(low, high, 501)
        histograms = []
        for item in values:
            histogram = np.histogram(item, bins=edges)[0].astype(float)
            histograms.append(histogram / histogram.sum())
        for index in range(len(histograms) - 1):
            overlap = float(np.minimum(histograms[index], histograms[index + 1]).sum())
            row_lookup[replica_keys[index]]["overlap_with_next"] = overlap
            overlaps.append(overlap)

    metric_path = output / "window_metrics.csv"
    with metric_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected_candidates = farthest_point_candidates(candidates, args.candidate_count)
    candidate_path = output / "ood_candidates_unlabeled.xyz"
    if selected_candidates:
        write(candidate_path, selected_candidates, format="extxyz")

    minimum_overlap = min(overlaps) if overlaps else 0.0
    model_hashes = {
        json.loads((window / "run_metadata.json").read_text()).get("model_sha256")
        for window in window_paths
    }
    schedule_hashes = {
        json.loads((window / "run_metadata.json").read_text()).get(
            "window_centers_sha256"
        )
        for window in window_paths
    }
    if len(model_hashes) != 1 or len(schedule_hashes) != 1:
        raise ValueError("Audit input mixes checkpoints or umbrella schedules")
    chemistry_valid = not any(
        row["collapsed_fraction"] > 0.0
        or row["persistent_bond_break_fraction"] > 0.0
        or row["unexpected_contact_fraction"] > 0.0
        or row["minimum_reactive_sum_A"] < args.minimum_reactive_sum_A
        for row in rows
    )
    sampling_valid = (
        minimum_overlap >= args.minimum_overlap
        and min(row["effective_samples"] for row in rows)
        >= args.minimum_effective_samples
        and max(abs(row["half_drift_A"]) for row in rows)
        <= args.maximum_half_drift_A
    )
    report = {
        "umbrella_root": str(umbrella_root),
        "audit_mode": "chemistry-only" if args.chemistry_only else "production",
        "model_sha256": model_hashes.pop(),
        "window_centers_sha256": schedule_hashes.pop(),
        "window_count": len(rows),
        "replica_count": len(replica_indices),
        "audited_frames": total_audited_frames,
        "chemistry_valid": chemistry_valid,
        "sampling_valid": sampling_valid,
        "chemistry_gate_passed": chemistry_valid,
        "free_energy_profile_acceptable": (
            None if args.chemistry_only else chemistry_valid and sampling_valid
        ),
        "windows_visiting_collapsed_dual_CC": sum(
            row["collapsed_fraction"] > 0.0 for row in rows
        ),
        "windows_with_persistent_bond_breaks": sum(
            row["persistent_bond_break_fraction"] > 0.0 for row in rows
        ),
        "windows_with_unexpected_close_contacts": sum(
            row["unexpected_contact_fraction"] > 0.0 for row in rows
        ),
        "minimum_reactive_distance_sum_A": min(
            row["minimum_reactive_sum_A"] for row in rows
        ),
        "minimum_adjacent_overlap": minimum_overlap,
        "mean_adjacent_overlap": float(np.mean(overlaps)) if overlaps else None,
        "minimum_effective_samples": min(row["effective_samples"] for row in rows),
        "maximum_absolute_half_drift_A": max(
            abs(row["half_drift_A"]) for row in rows
        ),
        "mean_temperature_K": float(np.mean([row["mean_temperature_K"] for row in rows])),
        "unlabeled_ood_candidate_count": len(selected_candidates),
        "unlabeled_ood_candidates": str(candidate_path) if selected_candidates else None,
        "unexpected_close_pairs": [
            {
                "atoms_zero_based": list(pair),
                "symbols": symbols[pair[0]] + symbols[pair[1]],
                "observations": count,
            }
            for pair, count in unexpected_pairs.most_common()
        ],
        "broken_persistent_pairs": [
            {
                "atoms_zero_based": list(pair),
                "symbols": symbols[pair[0]] + symbols[pair[1]],
                "observations": count,
            }
            for pair, count in broken_pairs.most_common()
        ],
        "acceptance_thresholds": {
            "minimum_adjacent_overlap": args.minimum_overlap,
            "minimum_effective_samples_per_window": args.minimum_effective_samples,
            "maximum_absolute_half_drift_A": args.maximum_half_drift_A,
            "minimum_reactive_distance_sum_A": args.minimum_reactive_sum_A,
        },
        "conclusion": (
            (
                "The short chemistry gate passed; this does not establish "
                "free-energy convergence."
                if chemistry_valid
                else "The short chemistry gate failed."
            )
            if args.chemistry_only
            else (
                "Reject this trajectory set for quantitative thermodynamics."
                if not (chemistry_valid and sampling_valid)
                else "The automated chemistry and sampling gates passed."
            )
        ),
    }
    (output / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
    training = read(EXAMPLE_ROOT / "data/wtmetad_train.xyz", index=":")
    testing = read(EXAMPLE_ROOT / "data/us_test.xyz", index=":")
    irc = read(EXAMPLE_ROOT / "structures/r2_irc.xyz", index=":")
    for frames, label, color, marker, size in (
        (training, "WTMetaD-IB training", "#0072B2", "o", 13),
        (testing, "Independent DFT test", "#777777", ".", 8),
    ):
        coordinates = np.asarray([reactive_coordinates(atoms) for atoms in frames])
        axes[0, 0].scatter(
            coordinates[:, 0], coordinates[:, 1], s=size, marker=marker,
            color=color, alpha=0.55, edgecolors="none", label=label,
        )
    irc_coordinates = np.asarray([reactive_coordinates(atoms) for atoms in irc])
    axes[0, 0].plot(irc_coordinates[:, 0], irc_coordinates[:, 1], color="#000000", lw=1.4, label="DFT IRC")
    if candidates:
        candidate_coordinates = np.asarray(
            [[item[1]["r1_A"], item[1]["r2_A"]] for item in candidates]
        )
        axes[0, 0].scatter(
            candidate_coordinates[:, 0], candidate_coordinates[:, 1], s=7,
            color="#D55E00", alpha=0.25, edgecolors="none", label="Invalid MD frames",
        )
    axes[0, 0].set_xlabel(r"$r_1$ ($\AA$)")
    axes[0, 0].set_ylabel(r"$r_2$ ($\AA$)")
    axes[0, 0].legend(fontsize=8)

    for replica in replica_indices:
        replica_rows = sorted(
            (row for row in rows if row["replica"] == replica),
            key=lambda row: row["window"],
        )
        axes[0, 1].plot(
            [row["center_A"] for row in replica_rows],
            [row["mean_cv_A"] for row in replica_rows],
            marker="o", ms=2.8, lw=0.9, label=f"Replica {replica}",
        )
    limits = [min(row["center_A"] for row in rows), max(row["center_A"] for row in rows)]
    axes[0, 1].plot(limits, limits, color="#555555", ls="--", lw=0.9)
    axes[0, 1].set_xlabel("Umbrella center (A)")
    axes[0, 1].set_ylabel("Sampled mean CV (A)")
    axes[0, 1].legend(fontsize=8)

    for replica in replica_indices:
        replica_rows = sorted(
            (row for row in rows if row["replica"] == replica),
            key=lambda row: row["window"],
        )
        axes[1, 0].plot(
            [row["window"] for row in replica_rows[:-1]],
            [row["overlap_with_next"] for row in replica_rows[:-1]],
            marker="o", ms=2.8, lw=0.9, label=f"Replica {replica}",
        )
    axes[1, 0].axhline(args.minimum_overlap, color="#D55E00", ls="--", lw=1.0)
    axes[1, 0].set_xlabel("Window i (overlap with i+1)")
    axes[1, 0].set_ylabel("Histogram overlap")

    for replica in replica_indices:
        replica_rows = sorted(
            (row for row in rows if row["replica"] == replica),
            key=lambda row: row["window"],
        )
        axes[1, 1].plot(
            [row["window"] for row in replica_rows],
            [row["collapsed_fraction"] for row in replica_rows],
            marker="o", ms=2.8, lw=0.9, label=f"Replica {replica}",
        )
    axes[1, 1].set_xlabel("Window")
    axes[1, 1].set_ylabel("Collapsed dual-C-C fraction")
    axes[1, 1].set_ylim(bottom=0.0)
    figure.tight_layout()
    figure.savefig(output / "sampling_audit.png", dpi=250)
    plt.close(figure)

    print(json.dumps(report, indent=2))
    print(f"Window metrics: {metric_path}")
    print(f"Audit figure: {output / 'sampling_audit.png'}")


if __name__ == "__main__":
    main()
