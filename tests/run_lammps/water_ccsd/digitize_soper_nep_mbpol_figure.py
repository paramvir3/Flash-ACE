#!/usr/bin/env python3
"""Digitize visible Soper O--H and H--H markers from a NEP-MB-pol RDF figure.

The input is a raster export of the three-panel NEP-MB-pol RDF figure supplied
with this test.  The script extracts only markers visible inside the plotted
range.  It is not a replacement for an original tabulated Soper data release.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


# Plot rectangles and axes calibrated from the supplied 2075 x 1005 pixel image.
# Each tuple is (x_left, x_right, y_top, y_bottom, r_left, r_right).
PANELS = {
    "oh": (68, 723, 345, 588, 0.5, 4.0),
    "hh": (68, 723, 671, 914, 1.0, 4.0),
}
DISPLAY_G_MAX = 4.0


def digitize_panel(image: np.ndarray, panel: str) -> np.ndarray:
    """Return visible black cross-marker centers as ``r_A, g`` pairs."""
    x_left, x_right, y_top, y_bottom, r_left, r_right = PANELS[panel]
    dark = np.all(image < 80, axis=2)
    values = []

    # Exclude the frame and ticks.  A small window averages the arms of each x.
    for x in range(x_left + 3, x_right - 3):
        window = dark[y_top + 15 : y_bottom - 20, x - 2 : x + 3]
        y_pixels = np.where(window)[0]
        if not len(y_pixels):
            continue
        y = y_top + 15 + float(np.median(y_pixels))
        r = r_left + (x - x_left) * (r_right - r_left) / (x_right - x_left)
        g = (y_bottom - y) * DISPLAY_G_MAX / (y_bottom - y_top)
        values.append((r, g, x))

    values_array = np.asarray(values)
    sampled = []
    for r_start in np.arange(r_left, r_right, 0.04):
        rows = values_array[
            (values_array[:, 0] >= r_start) & (values_array[:, 0] < r_start + 0.04)
        ]
        if len(rows) < 2:
            continue
        r, g, x = np.median(rows, axis=0)

        # Remove top-frame/tick artifacts and the parts hidden by the clipped peak.
        if not 0.06 < g < 3.65:
            continue
        if any(abs(x - tick) <= 6 for tick in _major_tick_positions(panel)):
            continue
        sampled.append((r, g))

    data = np.asarray(sampled)
    if panel == "oh":
        data = data[data[:, 0] >= 0.70]
    else:
        data = data[data[:, 0] >= 1.25]
    return data


def _major_tick_positions(panel: str) -> list[float]:
    x_left, x_right, _, _, r_left, r_right = PANELS[panel]
    first_integer = int(np.ceil(r_left))
    return [
        x_left + (value - r_left) * (x_right - x_left) / (r_right - r_left)
        for value in range(first_integer, int(r_right) + 1)
    ]


def write_csv(path: Path, data: np.ndarray, pair: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "Digitized from the user-supplied NEP-MB-pol RDF figure (Xu et al., 2025),\n"
        "which marks the Soper 2000 298 K neutron-diffraction/EPSR reference.\n"
        "Figure-coordinate digitization only: values near the clipped intramolecular peak are omitted.\n"
        f"r_A,g_{pair}"
    )
    np.savetxt(path, data, delimiter=",", header=header, comments="# ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("experimental_digitized"))
    args = parser.parse_args()

    image = np.asarray(Image.open(args.figure).convert("RGB"))
    if image.shape[:2] != (1005, 2075):
        raise ValueError(
            "This calibrated digitizer expects the supplied 2075 x 1005 pixel figure. "
            "Use original Soper tabulations for a different image."
        )

    oh_path = args.output_dir / "experimental_goh_soper_298K_digitized.csv"
    hh_path = args.output_dir / "experimental_ghh_soper_298K_digitized.csv"
    write_csv(oh_path, digitize_panel(image, "oh"), "OH")
    write_csv(hh_path, digitize_panel(image, "hh"), "HH")
    print(f"Wrote: {oh_path}")
    print(f"Wrote: {hh_path}")


if __name__ == "__main__":
    main()
