#!/usr/bin/env python3
"""Plot the published R2 training and independent US/AIMD coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read


EXAMPLE_ROOT = Path(__file__).resolve().parent


def coordinates(atoms) -> tuple[float, float]:
    if len(atoms) != 23 or any(atoms.numbers[index] != 6 for index in (10, 11, 14)):
        raise ValueError("The R2 atom ordering no longer matches the published CV")
    r_1 = float(np.linalg.norm(atoms.positions[14] - atoms.positions[11]))
    r_2 = float(np.linalg.norm(atoms.positions[14] - atoms.positions[10]))
    return r_1, r_2


def coordinate_array(path: Path) -> np.ndarray:
    frames = read(path, index=":")
    if not frames:
        raise ValueError(f"No structures were read from {path}")
    return np.asarray([coordinates(atoms) for atoms in frames], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--downhill", type=Path, default=EXAMPLE_ROOT / "data/downhill_all.xyz"
    )
    parser.add_argument(
        "--wtmetad", type=Path, default=EXAMPLE_ROOT / "data/wtmetad_all.xyz"
    )
    parser.add_argument(
        "--test", type=Path, default=EXAMPLE_ROOT / "data/us_test.xyz"
    )
    parser.add_argument(
        "--output", type=Path, default=EXAMPLE_ROOT / "results/figure4a_trace.png"
    )
    args = parser.parse_args()

    downhill = coordinate_array(args.downhill)
    wtmetad = coordinate_array(args.wtmetad)
    test = coordinate_array(args.test)
    if (len(downhill), len(wtmetad), len(test)) != (131, 192, 326):
        raise ValueError(
            "Expected the published R2 counts (131 downhill, 192 WTMetaD-IB, "
            f"326 US/AIMD), got {(len(downhill), len(wtmetad), len(test))}"
        )

    stationary = {}
    for label, filename in (
        ("RS", "r_r2.xyz"),
        ("TS", "ts_r2.xyz"),
        ("PS", "p_r2.xyz"),
    ):
        stationary[label] = coordinates(read(EXAMPLE_ROOT / "structures" / filename))

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True, sharey=True)
    settings = (
        (downhill, "#3568A8", f"Downhill ($N_{{train}}={len(downhill)}$)"),
        (wtmetad, "#A52A2A", f"WTMetaD-IB ($N_{{train}}={len(wtmetad)}$)"),
    )
    for axis, (training, color, label) in zip(axes, settings):
        axis.scatter(
            test[:, 0], test[:, 1], s=11, color="#8C8C8C", alpha=0.45,
            edgecolors="none", label=f"US/AIMD ($N_{{test}}={len(test)}$)",
            zorder=1,
        )
        axis.scatter(
            training[:, 0], training[:, 1], s=15, color=color, alpha=0.9,
            edgecolors="none", label=label, zorder=2,
        )
        for state, (r_1, r_2) in stationary.items():
            axis.scatter(
                r_1, r_2, marker="x", s=45, linewidths=1.4,
                color="#111111", zorder=4,
            )
            offsets = {"RS": (-0.28, 0.10), "TS": (-0.24, -0.23), "PS": (0.08, -0.25)}
            dx, dy = offsets[state]
            axis.annotate(
                state, (r_1, r_2), xytext=(r_1 + dx, r_2 + dy),
                fontsize=9, fontweight="bold",
            )
        axis.set_xlabel(r"$r_1$ ($\AA$)")
        axis.legend(frameon=False, fontsize=8, loc="upper right")
        axis.set_xlim(0.8, 3.8)
        axis.set_ylim(0.8, 3.8)
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel(r"$r_2$ ($\AA$)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output, dpi=300)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)

    metadata = {
        "downhill_training_frames": len(downhill),
        "wtmetad_ib_training_frames": len(wtmetad),
        "independent_us_aimd_test_frames": len(test),
        "stationary_coordinates_A": {
            key: {"r1": value[0], "r2": value[1], "r1_minus_r2": value[0] - value[1]}
            for key, value in stationary.items()
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Figure: {args.output}")


if __name__ == "__main__":
    main()
