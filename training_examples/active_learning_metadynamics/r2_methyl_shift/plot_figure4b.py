#!/usr/bin/env python3
"""Aggregate R2 free-energy replicas and plot the Figure 4b comparison."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


EXAMPLE_ROOT = Path(__file__).resolve().parent
EV_TO_KCAL_MOL = 23.06054783061903
TRANSITION_SEARCH = (-0.60, 0.50)
METHODS = (
    (
        "umbrella",
        "US/TRACE-MLIP-MD",
        "results/umbrella/replica_*/fes_wham.csv",
        "#4C78A8",
    ),
    (
        "standard",
        "WTMetaD/TRACE-MLIP-MD",
        "results/standard/replica_*/fes_standard.csv",
        "#8E5AA9",
    ),
    (
        "inherited",
        "WTMetaD+IB/TRACE-MLIP-MD",
        "results/inherited/replica_*/fes_inherited.csv",
        "#72A94F",
    ),
)


def expand_patterns(patterns: list[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match).resolve() for match in matches)
        elif Path(pattern).is_file():
            paths.append(Path(pattern).resolve())
    return sorted(dict.fromkeys(paths))


def audit_default_attempts(method: str) -> dict:
    replica_roots = sorted((EXAMPLE_ROOT / "results" / method).glob("replica_*"))
    excluded = []
    for root in replica_roots:
        if method == "umbrella":
            statuses = sorted(root.glob("window_*/run_status.json"))
            complete = len(statuses) == 30 and all(
                json.loads(path.read_text()).get("state") == "complete"
                for path in statuses
            )
            profile_exists = (root / "fes_wham.csv").is_file()
        else:
            status_path = root / "run_status.json"
            complete = status_path.is_file() and (
                json.loads(status_path.read_text()).get("state") == "complete"
            )
            profile_exists = (root / f"fes_{method}.csv").is_file()
        if not complete or not profile_exists:
            excluded.append(str(root.resolve()))
    return {
        "attempted_replica_count": len(replica_roots),
        "excluded_or_unanalyzed_replica_roots": excluded,
    }


def read_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    expected = {"cv_r1_minus_r2_A", "free_energy_kcal_per_mol"}
    if data.dtype.names is None or not expected.issubset(data.dtype.names):
        raise ValueError(f"Unexpected profile columns in {path}")
    coordinate = np.atleast_1d(data["cv_r1_minus_r2_A"]).astype(float)
    free_energy = np.atleast_1d(data["free_energy_kcal_per_mol"]).astype(float)
    finite = np.isfinite(coordinate) & np.isfinite(free_energy)
    coordinate = coordinate[finite]
    free_energy = free_energy[finite]
    order = np.argsort(coordinate)
    coordinate = coordinate[order]
    free_energy = free_energy[order]
    if len(coordinate) < 100 or np.any(np.diff(coordinate) <= 0.0):
        raise ValueError(f"Profile is too short or non-monotonic: {path}")
    reactant = (coordinate >= -0.97) & (coordinate <= -0.60)
    if not reactant.any():
        raise ValueError(f"Profile does not sample the reactant basin: {path}")
    free_energy -= np.min(free_energy[reactant])
    return coordinate, free_energy


def read_references(path: Path) -> list[dict]:
    required = {
        "kind",
        "label",
        "barrier_kcal_per_mol",
        "uncertainty_kcal_per_mol",
        "temperature_K",
        "display_coordinate_A",
        "source",
    }
    if not path.is_file():
        raise FileNotFoundError(f"Figure 4b reference data not found: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected reference columns in {path}")
        references = []
        for row in reader:
            references.append(
                {
                    "kind": row["kind"],
                    "label": row["label"],
                    "barrier_kcal_per_mol": float(row["barrier_kcal_per_mol"]),
                    "uncertainty_kcal_per_mol": float(
                        row["uncertainty_kcal_per_mol"]
                    ),
                    "temperature_K": float(row["temperature_K"]),
                    "display_coordinate_A": float(row["display_coordinate_A"]),
                    "source": row["source"],
                }
            )
    kinds = {item["kind"] for item in references}
    if kinds != {"experiment", "dft"}:
        raise ValueError("Reference data must contain one experiment and one DFT row")
    return references


def barrier(coordinate: np.ndarray, free_energy: np.ndarray) -> dict:
    transition = (
        (coordinate >= TRANSITION_SEARCH[0])
        & (coordinate <= TRANSITION_SEARCH[1])
    )
    product = (coordinate >= 0.60) & (coordinate <= 1.03)
    if not transition.any() or not product.any():
        raise ValueError("Profile does not span the TS and product regions")
    transition_indices = np.flatnonzero(transition)
    transition_index = transition_indices[int(np.argmax(free_energy[transition]))]
    product_indices = np.flatnonzero(product)
    product_index = product_indices[int(np.argmin(free_energy[product]))]
    return {
        "transition_coordinate_A": float(coordinate[transition_index]),
        "activation_free_energy_kcal_per_mol": float(free_energy[transition_index]),
        "product_coordinate_A": float(coordinate[product_index]),
        "product_minus_reactant_kcal_per_mol": float(free_energy[product_index]),
    }


STUDENT_T_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def confidence_multiplier(sample_count: int, method: str) -> float:
    if sample_count < 2:
        return 0.0
    if method == "normal":
        return 1.96
    degrees_of_freedom = sample_count - 1
    return STUDENT_T_95.get(degrees_of_freedom, 1.96)


def aggregate(
    paths: list[Path], grid: np.ndarray, ci_method: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    interpolated = []
    replicas = []
    for path in paths:
        coordinate, free_energy = read_profile(path)
        replicas.append({"file": str(path), **barrier(coordinate, free_energy)})
        values = np.interp(grid, coordinate, free_energy, left=np.nan, right=np.nan)
        interpolated.append(values)
    profiles = np.asarray(interpolated)
    counts = np.sum(np.isfinite(profiles), axis=0)
    mean = np.nanmean(profiles, axis=0)
    if len(profiles) > 1:
        standard_error = np.nanstd(profiles, axis=0, ddof=1) / np.sqrt(counts)
        multipliers = np.asarray(
            [confidence_multiplier(int(count), ci_method) for count in counts]
        )
        confidence = multipliers * standard_error
        confidence[counts < 2] = np.nan
    else:
        confidence = np.zeros_like(mean)
    barriers = np.asarray(
        [item["activation_free_energy_kcal_per_mol"] for item in replicas]
    )
    products = np.asarray(
        [item["product_minus_reactant_kcal_per_mol"] for item in replicas]
    )
    summary = {
        "stable_replica_count": len(replicas),
        "replicas": replicas,
        "mean_activation_free_energy_kcal_per_mol": float(np.mean(barriers)),
        "activation_95_percent_CI_kcal_per_mol": float(
            confidence_multiplier(len(barriers), ci_method)
            * np.std(barriers, ddof=1)
            / np.sqrt(len(barriers))
            if len(barriers) > 1
            else 0.0
        ),
        "mean_product_minus_reactant_kcal_per_mol": float(np.mean(products)),
        "confidence_interval_method": ci_method,
    }
    return mean, confidence, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--umbrella", nargs="*", default=None)
    parser.add_argument("--standard", nargs="*", default=None)
    parser.add_argument("--inherited", nargs="*", default=None)
    parser.add_argument(
        "--references",
        type=Path,
        default=EXAMPLE_ROOT / "references/figure4b_barriers.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=EXAMPLE_ROOT / "results/figure4b_trace.png"
    )
    parser.add_argument(
        "--ci-method",
        choices=("normal", "student-t"),
        default="normal",
        help=(
            "Use normal to reproduce the source paper's CLT interval; use "
            "student-t for the reduced three-replica protocol."
        ),
    )
    parser.add_argument(
        "--diagnostic-label",
        default=None,
        help="Optional figure annotation for preliminary or unconverged profiles.",
    )
    parser.add_argument(
        "--allow-missing-methods",
        action="store_true",
        help="Plot available estimators without requiring all three methods.",
    )
    args = parser.parse_args()
    references = read_references(args.references.resolve())

    supplied = {
        "umbrella": args.umbrella,
        "standard": args.standard,
        "inherited": args.inherited,
    }
    grid = np.linspace(-0.9685435612879105, 1.0283024492805735, 500)
    aggregates = {}
    missing = []
    for key, label, default_pattern, color in METHODS:
        patterns = supplied[key]
        if patterns is None:
            patterns = [str(EXAMPLE_ROOT / default_pattern)]
        paths = expand_patterns(patterns)
        if not paths:
            missing.append(f"{label}: {', '.join(patterns)}")
            continue
        mean, confidence, summary = aggregate(paths, grid, args.ci_method)
        if supplied[key] is None:
            summary.update(audit_default_attempts(key))
        aggregates[key] = {
            "label": f"{label} ($n={summary['stable_replica_count']}$)",
            "color": color,
            "mean": mean,
            "confidence": confidence,
            "summary": summary,
        }
    if missing and not args.allow_missing_methods:
        raise FileNotFoundError(
            "All three Figure 4b estimators require completed profiles. Missing:\n- "
            + "\n- ".join(missing)
        )

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    zoom_axis = axis.inset_axes([0.22, 0.10, 0.47, 0.34])
    method_handles = []
    for key, _, _, _ in METHODS:
        if key not in aggregates:
            continue
        item = aggregates[key]
        mean = item["mean"]
        confidence = item["confidence"]
        line, = axis.plot(
            grid, mean, color=item["color"], linewidth=1.8, label=item["label"]
        )
        method_handles.append(line)
        zoom_axis.plot(grid, mean, color=item["color"], linewidth=1.8)
        if item["summary"]["stable_replica_count"] > 1:
            for target_axis in (axis, zoom_axis):
                target_axis.fill_between(
                    grid,
                    mean - confidence,
                    mean + confidence,
                    color=item["color"],
                    alpha=0.16,
                    linewidth=0,
                )
        transition = (
            (grid >= TRANSITION_SEARCH[0])
            & (grid <= TRANSITION_SEARCH[1])
        )
        transition_indices = np.flatnonzero(transition)
        transition_index = transition_indices[int(np.nanargmax(mean[transition]))]
        zoom_axis.scatter(
            grid[transition_index],
            mean[transition_index],
            s=27,
            color=item["color"],
            edgecolors="white",
            linewidths=0.6,
            zorder=6,
        )

    reference_handles = []
    reference_colors = {"experiment": "#666666", "dft": "#E67E22"}
    for reference in references:
        color = reference_colors[reference["kind"]]
        uncertainty = reference["uncertainty_kcal_per_mol"]
        kwargs = dict(
            x=reference["display_coordinate_A"],
            y=reference["barrier_kcal_per_mol"],
            marker="s",
            markersize=5.0,
            markerfacecolor=color,
            markeredgecolor=color,
            color=color,
            capsize=2.5,
            linewidth=1.0,
            linestyle="none",
            label=reference["label"],
            zorder=7,
        )
        handle = axis.errorbar(
            yerr=uncertainty if uncertainty > 0.0 else None,
            **kwargs,
        )
        reference_handles.append(handle)
        zoom_axis.errorbar(
            yerr=uncertainty if uncertainty > 0.0 else None,
            **{key: value for key, value in kwargs.items() if key != "label"},
        )

    transition_coordinate = -0.10327227367991121
    for target_axis in (axis, zoom_axis):
        target_axis.axvline(
            transition_coordinate,
            color="#555555",
            linestyle=(0, (3, 3)),
            linewidth=0.8,
            zorder=0,
        )
        target_axis.set_ylabel(r"$\Delta G$ (kcal mol$^{-1}$)")
        target_axis.tick_params(direction="in", top=True, right=True)
    axis.set_xlim(grid[0], grid[-1])
    axis.set_ylim(-32.0, 35.0)
    axis.set_xlabel(r"$s=r_1-r_2$ ($\AA$)")
    if args.diagnostic_label:
        axis.text(
            0.02,
            0.97,
            args.diagnostic_label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="#9B1C1C",
        )
    zoom_axis.set_xlim(-0.18, -0.03)
    zoom_values = []
    for reference in references:
        uncertainty = reference["uncertainty_kcal_per_mol"]
        zoom_values.extend(
            [
                reference["barrier_kcal_per_mol"] - uncertainty,
                reference["barrier_kcal_per_mol"] + uncertainty,
            ]
        )
    selected = (grid >= -0.18) & (grid <= -0.03)
    for item in aggregates.values():
        values = item["mean"][selected]
        zoom_values.extend(values[np.isfinite(values)].tolist())
    zoom_axis.set_ylim(min(zoom_values) - 0.6, max(zoom_values) + 0.6)
    zoom_axis.tick_params(labelsize=8)
    axis.legend(
        reference_handles + method_handles,
        [item["label"] for item in references]
        + [
            aggregates[key]["label"]
            for key, _, _, _ in METHODS
            if key in aggregates
        ],
        frameon=True,
        fontsize=8.0,
        loc="upper right",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)

    curve_output = args.output.with_suffix(".csv")
    with curve_output.open("w", newline="") as handle:
        fieldnames = ["cv_r1_minus_r2_A"]
        for key, _, _, _ in METHODS:
            if key not in aggregates:
                continue
            fieldnames.extend(
                [
                    f"{key}_mean_kcal_per_mol",
                    f"{key}_95_percent_CI_kcal_per_mol",
                ]
            )
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for index, coordinate in enumerate(grid):
            row = [coordinate]
            for key, _, _, _ in METHODS:
                if key not in aggregates:
                    continue
                row.extend(
                    [
                        aggregates[key]["mean"][index],
                        aggregates[key]["confidence"][index],
                    ]
                )
            writer.writerow(row)

    summary = {
        key: item["summary"] for key, item in aggregates.items()
    }
    summary["reference_data_file"] = str(args.references.resolve())
    summary["experiment_and_dft_references"] = references
    summary["published_source_MLIP_results_kcal_per_mol"] = {
        "umbrella_sampling": "28.2 +/- 0.1",
        "WTMetaD": "28.1 +/- 0.3",
        "WTMetaD_with_inherited_bias": "28.4 +/- 0.4",
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Figure: {args.output}")
    print(f"Aggregated curves: {curve_output}")


if __name__ == "__main__":
    main()
