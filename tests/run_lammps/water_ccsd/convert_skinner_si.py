#!/usr/bin/env python3
"""Convert Skinner ambient-water x-ray SI data to an O-O RDF CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def convert(input_path: Path, output_path: Path) -> int:
    rows: list[tuple[float, float]] = []
    for line in input_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("Q") or stripped.startswith("(inv."):
            continue

        parts = stripped.split()
        if len(parts) < 6:
            continue
        try:
            r_angstrom = float(parts[3])
            g_oo = float(parts[4])
        except ValueError:
            continue
        rows.append((r_angstrom, g_oo))

    if not rows:
        raise RuntimeError(f"No numeric r, g_OO rows found in {input_path}")

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["r_A", "g_OO"])
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Skinner SI text file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experimental_goo_skinner_295K.csv"),
        help="Output CSV with r_A,g_OO columns.",
    )
    args = parser.parse_args()

    n_rows = convert(args.input, args.output)
    print(f"Wrote {args.output} with {n_rows} rows")


if __name__ == "__main__":
    main()
