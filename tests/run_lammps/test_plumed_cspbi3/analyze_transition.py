#!/usr/bin/env python3
"""Plot the biased order parameter and Pb-I octahedral connectivity."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list


def read_timesteps(path: Path) -> np.ndarray:
    timesteps = []
    with path.open() as handle:
        lines = iter(handle)
        for line in lines:
            if line.strip() == "ITEM: TIMESTEP":
                timesteps.append(int(next(lines).strip()))
    return np.asarray(timesteps, dtype=float)


def connectivity_counts(atoms, cutoff: float) -> tuple[int, int, int]:
    """Count Pb pairs sharing one, two, or at least three iodides."""
    types = atoms.numbers
    centers, neighbors = neighbor_list("ij", atoms, cutoff)
    iodine_to_lead = defaultdict(set)
    for center, neighbor in zip(centers, neighbors):
        if types[center] == 2 and types[neighbor] == 3:
            iodine_to_lead[int(neighbor)].add(int(center))

    shared_iodides = Counter()
    for lead_indices in iodine_to_lead.values():
        lead_indices = sorted(lead_indices)
        for first in range(len(lead_indices)):
            for second in range(first + 1, len(lead_indices)):
                pair = (lead_indices[first], lead_indices[second])
                shared_iodides[pair] += 1

    histogram = Counter(shared_iodides.values())
    corner = histogram[1]
    edge = histogram[2]
    higher = sum(count for shared, count in histogram.items() if shared >= 3)
    return corner, edge, higher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, default=Path("out.1.lammpstrj"))
    parser.add_argument("--colvar", type=Path, default=Path("COLVAR"))
    parser.add_argument("--output", type=Path, default=Path("cspbi3_trace_rare_event.png"))
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--pb-i-cutoff", type=float, default=4.2)
    args = parser.parse_args()

    frames = read(args.trajectory, index=":", format="lammps-dump-text")
    timesteps = read_timesteps(args.trajectory)
    if len(frames) != len(timesteps):
        raise ValueError("Trajectory frame and timestep counts differ")

    frame_time_ps = timesteps * args.timestep_fs / 1000.0
    colvar = np.loadtxt(args.colvar, comments="#")
    colvar_time_ps = colvar[:, 0]
    perovskite_order = colvar[:, 1]
    reported_end = float(colvar_time_ps[-1])

    keep = frame_time_ps <= reported_end + 1.0e-12
    frame_time_ps = frame_time_ps[keep]
    frames = [frame for frame, selected in zip(frames, keep) if selected]
    counts = np.asarray(
        [connectivity_counts(frame, args.pb_i_cutoff) for frame in frames],
        dtype=float,
    )
    totals = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.fontsize": 11.5,
            "axes.linewidth": 1.0,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    axes[0].plot(colvar_time_ps, perovskite_order, color="#285F9E", linewidth=2.2)
    axes[0].set_ylabel(r"Perovskite order $S_{\mathrm{p}}$")
    axes[0].text(
        0.015, 0.90, "(a)", transform=axes[0].transAxes,
        fontweight="bold", fontsize=15,
    )

    labels = ("corner sharing", "edge sharing", "higher sharing")
    colors = ("#168C8C", "#D58B2A", "#6B7280")
    for index, (label, color) in enumerate(zip(labels, colors)):
        axes[1].plot(
            frame_time_ps,
            fractions[:, index],
            color=color,
            linewidth=2.0,
            label=label,
        )
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel("Fraction of linked Pb pairs")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(frameon=False, ncol=3, loc="upper center")
    axes[1].text(
        0.015, 0.88, "(b)", transform=axes[1].transAxes,
        fontweight="bold", fontsize=15,
    )

    for axis in axes:
        axis.set_xlim(0.0, reported_end)
        axis.grid(color="#D7DCE2", linewidth=0.7, alpha=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    initial = counts[0].astype(int)
    final = counts[-1].astype(int)
    print(f"initial corner/edge/higher: {initial.tolist()}")
    print(f"final corner/edge/higher: {final.tolist()}")


if __name__ == "__main__":
    main()
