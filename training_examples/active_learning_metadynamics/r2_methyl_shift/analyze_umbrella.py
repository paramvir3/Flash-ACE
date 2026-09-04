#!/usr/bin/env python3
"""Reconstruct one R2 umbrella-sampling profile with discrete WHAM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EXAMPLE_ROOT = Path(__file__).resolve().parent
KB_EV_PER_K = 8.617333262145e-5
EV_TO_KCAL_MOL = 23.06054783061903


def logsumexp(values: np.ndarray, axis=None) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    finite_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    summed = np.sum(np.exp(values - finite_maximum), axis=axis, keepdims=True)
    result = finite_maximum + np.log(summed)
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    return result


def read_window(path: Path) -> tuple[dict, np.ndarray]:
    metadata_path = path / "run_metadata.json"
    status_path = path / "run_status.json"
    colvar_path = path / "COLVAR"
    if (
        not metadata_path.is_file()
        or not status_path.is_file()
        or not colvar_path.is_file()
    ):
        raise FileNotFoundError(f"Incomplete umbrella window: {path}")
    metadata = json.loads(metadata_path.read_text())
    status = json.loads(status_path.read_text())
    if status.get("state") != "complete":
        raise RuntimeError(f"Umbrella window is not complete: {path}: {status}")
    data = np.loadtxt(colvar_path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 4:
        raise ValueError(f"Expected time, r1, r2, and diff_r in {colvar_path}")
    samples = np.asarray(data[:, 3], dtype=float)
    discard = int(
        round(
            1000.0
            * float(metadata["equilibration_discard_ps"])
            / float(metadata["sample_interval_fs"])
        )
    )
    if discard >= len(samples):
        raise ValueError(f"Equilibration discard removes every sample in {path}")
    samples = samples[discard:]
    if not np.isfinite(samples).all():
        raise ValueError(f"Non-finite collective-variable samples in {path}")
    return metadata, samples


def wham(
    histograms: np.ndarray,
    sample_counts: np.ndarray,
    bias_energies: np.ndarray,
    beta: float,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    window_count, bin_count = histograms.shape
    free_offsets = np.zeros(window_count)
    total_counts = histograms.sum(axis=0)
    populated = total_counts > 0
    if populated.sum() < bin_count // 2:
        raise ValueError("Umbrella histograms leave more than half of the WHAM grid empty")

    log_sample_counts = np.log(sample_counts)
    log_probability = np.full(bin_count, -np.inf)
    for iteration in range(1, maximum_iterations + 1):
        denominator_terms = (
            log_sample_counts[:, None]
            + beta * free_offsets[:, None]
            - beta * bias_energies
        )
        log_denominator = logsumexp(denominator_terms, axis=0)
        log_probability[:] = -np.inf
        log_probability[populated] = (
            np.log(total_counts[populated]) - log_denominator[populated]
        )
        log_probability[populated] -= logsumexp(log_probability[populated])

        new_offsets = -(1.0 / beta) * logsumexp(
            log_probability[None, :] - beta * bias_energies, axis=1
        )
        new_offsets -= new_offsets[0]
        if np.max(np.abs(new_offsets - free_offsets)) < tolerance:
            free_offsets = new_offsets
            break
        free_offsets = new_offsets
    else:
        raise RuntimeError(f"WHAM did not converge in {maximum_iterations} iterations")

    probability = np.zeros(bin_count)
    probability[populated] = np.exp(log_probability[populated])
    return probability, free_offsets, iteration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replica-root",
        type=Path,
        default=EXAMPLE_ROOT / "results/umbrella/replica_00",
    )
    parser.add_argument("--bins", type=int, default=500)
    parser.add_argument("--grid-min", type=float, default=None)
    parser.add_argument("--grid-max", type=float, default=None)
    parser.add_argument("--transition-min", type=float, default=-0.60)
    parser.add_argument("--transition-max", type=float, default=0.50)
    parser.add_argument("--tolerance-eV", type=float, default=1.0e-10)
    parser.add_argument("--maximum-iterations", type=int, default=100000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    window_paths = sorted(args.replica_root.glob("window_*"))
    if not window_paths:
        raise FileNotFoundError(f"No umbrella windows found in {args.replica_root}")
    metadata_and_samples = [read_window(path) for path in window_paths]
    metadata = [item[0] for item in metadata_and_samples]
    samples = [item[1] for item in metadata_and_samples]
    expected_counts = {int(item["window_count"]) for item in metadata}
    model_hashes = {item.get("model_sha256") for item in metadata}
    schedule_hashes = {item.get("window_centers_sha256") for item in metadata}
    window_indices = [int(item["window_index"]) for item in metadata]
    if len(expected_counts) != 1:
        raise ValueError("Umbrella metadata mixes different window schedules")
    if len(model_hashes) != 1:
        raise ValueError("Umbrella windows were generated with different checkpoints")
    if len(schedule_hashes) != 1:
        raise ValueError("Umbrella windows use different center schedules")
    expected_count = expected_counts.pop()
    if len(window_paths) != expected_count or window_indices != list(
        range(expected_count)
    ):
        raise ValueError(
            f"Expected complete windows 0..{expected_count - 1}, found "
            f"{window_indices}"
        )

    centers = np.asarray([item["window_center_A"] for item in metadata], dtype=float)
    kappas = np.asarray(
        [item["harmonic_kappa_eV_per_A2"] for item in metadata], dtype=float
    )
    temperatures = {float(item["temperature_K"]) for item in metadata}
    replica_indices = {int(item["replica"]) for item in metadata}
    if len(temperatures) != 1 or len(replica_indices) != 1:
        raise ValueError("Umbrella windows mix temperatures or replica indices")
    if not np.all(np.diff(centers) > 0.0):
        raise ValueError("Umbrella window centers must be strictly increasing")

    sample_minimum = min(float(values.min()) for values in samples)
    sample_maximum = max(float(values.max()) for values in samples)
    grid_minimum = sample_minimum if args.grid_min is None else args.grid_min
    grid_maximum = sample_maximum if args.grid_max is None else args.grid_max
    if grid_minimum >= grid_maximum:
        raise ValueError("The WHAM grid minimum must be below its maximum")
    if grid_minimum > sample_minimum or grid_maximum < sample_maximum:
        raise ValueError(
            "The requested WHAM grid clips production samples; expand --grid-min "
            "and --grid-max instead of silently discarding data"
        )
    grid = np.linspace(grid_minimum, grid_maximum, args.bins)
    width = grid[1] - grid[0]
    edges = np.linspace(grid[0] - 0.5 * width, grid[-1] + 0.5 * width, args.bins + 1)
    histograms = np.asarray([np.histogram(values, bins=edges)[0] for values in samples])
    sample_counts = histograms.sum(axis=1).astype(float)
    if np.any(sample_counts == 0):
        raise ValueError("At least one umbrella window has no samples inside the WHAM grid")
    expected_sample_counts = np.asarray(
        [len(values) for values in samples], dtype=float
    )
    if not np.array_equal(sample_counts, expected_sample_counts):
        raise RuntimeError("The WHAM grid unexpectedly discarded production samples")
    bias_energies = 0.5 * kappas[:, None] * (grid[None, :] - centers[:, None]) ** 2
    temperature = temperatures.pop()
    beta = 1.0 / (KB_EV_PER_K * temperature)
    probability, offsets, iterations = wham(
        histograms,
        sample_counts,
        bias_energies,
        beta,
        args.tolerance_eV,
        args.maximum_iterations,
    )

    free_energy = np.full_like(probability, np.nan, dtype=float)
    populated = probability > 0.0
    free_energy[populated] = -(1.0 / beta) * np.log(probability[populated])
    reactant = populated & (grid >= -0.97) & (grid <= -0.60)
    transition = (
        populated
        & (grid >= args.transition_min)
        & (grid <= args.transition_max)
    )
    product = populated & (grid >= 0.60) & (grid <= 1.03)
    if not reactant.any() or not transition.any() or not product.any():
        raise ValueError("The WHAM profile does not cover RS, TS, and PS regions")
    reference = float(np.nanmin(free_energy[reactant]))
    relative = free_energy - reference
    transition_indices = np.flatnonzero(transition)
    transition_index = transition_indices[int(np.nanargmax(relative[transition]))]
    barrier_eV = float(relative[transition_index])
    product_eV = float(np.nanmin(relative[product]))

    normalized_histograms = histograms / sample_counts[:, None]
    adjacent_overlap = np.asarray(
        [
            np.minimum(normalized_histograms[i], normalized_histograms[i + 1]).sum()
            for i in range(len(normalized_histograms) - 1)
        ]
    )
    result = {
        "replica": replica_indices.pop(),
        "temperature_K": temperature,
        "window_count": len(window_paths),
        "model_sha256": model_hashes.pop(),
        "window_centers_sha256": schedule_hashes.pop(),
        "production_samples_per_window": sample_counts.astype(int).tolist(),
        "wham_grid_min_A": float(grid[0]),
        "wham_grid_max_A": float(grid[-1]),
        "transition_search_min_A": args.transition_min,
        "transition_search_max_A": args.transition_max,
        "wham_bins": args.bins,
        "wham_iterations": iterations,
        "minimum_adjacent_histogram_overlap": float(adjacent_overlap.min()),
        "mean_adjacent_histogram_overlap": float(adjacent_overlap.mean()),
        "transition_coordinate_A": float(grid[transition_index]),
        "activation_free_energy_eV": barrier_eV,
        "activation_free_energy_kcal_per_mol": barrier_eV * EV_TO_KCAL_MOL,
        "product_minus_reactant_eV": product_eV,
        "product_minus_reactant_kcal_per_mol": product_eV * EV_TO_KCAL_MOL,
        "window_free_energy_offsets_eV": offsets.tolist(),
    }
    if result["minimum_adjacent_histogram_overlap"] < 0.15:
        result["sampling_warning"] = (
            "Adjacent-window overlap is below 0.15; do not treat this profile "
            "as quantitatively converged without further diagnostics."
        )

    output = args.output
    if output is None:
        output = args.replica_root / "fes_wham.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["cv_r1_minus_r2_A", "free_energy_eV", "free_energy_kcal_per_mol"]
        )
        for coordinate, energy in zip(grid, relative):
            writer.writerow(
                [
                    coordinate,
                    energy,
                    energy * EV_TO_KCAL_MOL if np.isfinite(energy) else np.nan,
                ]
            )
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")

    import matplotlib.pyplot as plt

    figure, (profile_axis, overlap_axis) = plt.subplots(2, 1, figsize=(6.6, 6.0))
    profile_axis.plot(grid, relative * EV_TO_KCAL_MOL, color="#0072B2")
    profile_axis.axvline(-0.10327, color="#333333", linestyle="--", linewidth=0.9)
    profile_axis.set_ylabel(r"$\Delta G$ (kcal mol$^{-1}$)")
    profile_axis.set_xlabel(r"$r_1-r_2$ ($\AA$)")
    for index, values in enumerate(samples):
        overlap_axis.hist(
            values, bins=edges, histtype="step", density=True, linewidth=0.65,
            alpha=0.75,
            label=(
                f"{index:02d}" if index in (0, len(samples) - 1) else None
            ),
        )
    overlap_axis.set_xlabel(r"$r_1-r_2$ ($\AA$)")
    overlap_axis.set_ylabel("Window density")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=250)
    plt.close(figure)

    print(json.dumps(result, indent=2))
    print(f"WHAM profile: {output}")


if __name__ == "__main__":
    main()
