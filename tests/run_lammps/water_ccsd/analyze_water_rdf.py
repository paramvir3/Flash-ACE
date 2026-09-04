#!/usr/bin/env python3
"""Analyze LAMMPS water RDF output and compare O-O RDF to experiment.

The LAMMPS input in this folder uses

    compute myRDF all rdf 100 1 1 1 2 2 2

with atom type 1 = H and atom type 2 = O.  Therefore the RDF columns are

    r, g_HH, N_HH, g_HO, N_HO, g_OO, N_OO

after the leading row-index column written by ``fix ave/time``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROW_COL = 0
R_COL = 1
G_HH_COL = 2
N_HH_COL = 3
G_HO_COL = 4
N_HO_COL = 5
G_OO_COL = 6
N_OO_COL = 7


@dataclass
class DumpSummary:
    n_frames: int
    n_atoms: int | None
    first_step: int | None
    last_step: int | None
    type_counts: dict[int, int]


def parse_lammps_rdf(path: Path) -> list[tuple[int, np.ndarray]]:
    """Parse complete RDF blocks written by LAMMPS ``fix ave/time``."""
    blocks: list[tuple[int, np.ndarray]] = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) != 2:
            i += 1
            continue

        try:
            timestep = int(float(parts[0]))
            nrows = int(float(parts[1]))
        except ValueError:
            i += 1
            continue

        row_lines = lines[i + 1 : i + 1 + nrows]
        rows = []
        for row_line in row_lines:
            row_line = row_line.strip()
            if not row_line or row_line.startswith("#"):
                continue
            try:
                rows.append([float(x) for x in row_line.split()])
            except ValueError:
                break

        if len(rows) == nrows:
            block = np.asarray(rows, dtype=float)
            if block.ndim == 2 and block.shape[1] >= 8:
                blocks.append((timestep, block))

        i += 1 + nrows

    if not blocks:
        raise RuntimeError(f"No complete RDF blocks found in {path}")
    return blocks


def parse_dump_summary(path: Path) -> DumpSummary:
    """Parse a lightweight summary from a LAMMPS custom dump."""
    if not path.exists():
        return DumpSummary(0, None, None, None, {})

    n_frames = 0
    n_atoms = None
    first_step = None
    last_step = None
    type_counts: dict[int, int] = {}

    with path.open() as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            step_line = handle.readline()
            if not step_line:
                break
            step = int(step_line.strip())
            if first_step is None:
                first_step = step
            last_step = step
            n_frames += 1

            number_header = handle.readline()
            if not number_header.startswith("ITEM: NUMBER OF ATOMS"):
                break
            natoms_line = handle.readline()
            if not natoms_line:
                break
            natoms = int(natoms_line.strip())
            if n_atoms is None:
                n_atoms = natoms

            box_header = handle.readline()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                break
            for _ in range(3):
                handle.readline()

            atom_header = handle.readline()
            if not atom_header.startswith("ITEM: ATOMS"):
                break
            fields = atom_header.split()[2:]
            try:
                type_index = fields.index("type")
            except ValueError:
                type_index = None

            frame_type_counts: dict[int, int] = {}
            for _ in range(natoms):
                atom_line = handle.readline()
                if not atom_line:
                    break
                if n_frames == 1 and type_index is not None:
                    values = atom_line.split()
                    atom_type = int(values[type_index])
                    frame_type_counts[atom_type] = frame_type_counts.get(atom_type, 0) + 1

            if n_frames == 1:
                type_counts = frame_type_counts

    return DumpSummary(n_frames, n_atoms, first_step, last_step, type_counts)


def load_experimental_rdf(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a two-column experimental partial RDF file.

    Accepted format: CSV or whitespace separated columns with r in angstrom and
    g_OO(r). Lines beginning with ``#`` are ignored.
    """
    if not path.exists():
        return None

    try:
        data = np.genfromtxt(path, comments="#", delimiter=",", names=True)
        if data.dtype.names and len(data.dtype.names) >= 2:
            names = data.dtype.names
            r = np.asarray(data[names[0]], dtype=float)
            g = np.asarray(data[names[1]], dtype=float)
            return r, g
    except Exception:
        pass

    try:
        data = np.loadtxt(path, comments="#", delimiter=",")
    except ValueError:
        data = np.loadtxt(path, comments="#", delimiter=None)
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(f"Experimental RDF file must have at least two columns: {path}")
    return data[:, 0], data[:, 1]


def peak_position(r: np.ndarray, g: np.ndarray, window: tuple[float, float]) -> tuple[float, float]:
    mask = (r >= window[0]) & (r <= window[1])
    if not np.any(mask):
        return float("nan"), float("nan")
    idx_local = int(np.argmax(g[mask]))
    r_window = r[mask]
    g_window = g[mask]
    return float(r_window[idx_local]), float(g_window[idx_local])


def rmse_against_experiment(
    r_model: np.ndarray,
    g_model: np.ndarray,
    r_exp: np.ndarray,
    g_exp: np.ndarray,
    r_min: float = 2.2,
    r_max: float = 6.0,
) -> float:
    lo = max(r_min, float(np.min(r_model)), float(np.min(r_exp)))
    hi = min(r_max, float(np.max(r_model)), float(np.max(r_exp)))
    mask = (r_exp >= lo) & (r_exp <= hi)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    g_interp = np.interp(r_exp[mask], r_model, g_model)
    return float(np.sqrt(np.mean((g_interp - g_exp[mask]) ** 2)))


def write_csv(path: Path, final_block: np.ndarray) -> None:
    header = "r_A,g_HH,N_HH,g_HO,N_HO,g_OO,N_OO"
    cols = final_block[:, [R_COL, G_HH_COL, N_HH_COL, G_HO_COL, N_HO_COL, G_OO_COL, N_OO_COL]]
    np.savetxt(path, cols, delimiter=",", header=header, comments="")


def make_plots(
    output_png: Path,
    final_block: np.ndarray,
    exp_data: tuple[np.ndarray, np.ndarray] | None,
    title_suffix: str,
) -> None:
    mpl_cache = output_png.parent / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache.resolve()))

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "legend.fontsize": 10.5,
            "axes.linewidth": 0.9,
        }
    )

    r = final_block[:, R_COL]
    g_hh = final_block[:, G_HH_COL]
    g_ho = final_block[:, G_HO_COL]
    g_oo = final_block[:, G_OO_COL]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    axes[0].plot(r, g_oo, lw=2.2, label="TRACE/LAMMPS O-O")
    if exp_data is not None:
        r_exp, g_exp = exp_data
        axes[0].plot(
            r_exp,
            g_exp,
            linestyle="None",
            marker="x",
            color="black",
            markersize=3.5,
            markevery=8,
            label="Experiment (Skinner, 295 K)",
        )
    else:
        axes[0].text(
            0.03,
            0.95,
            "Add experimental_goo_skinner_295K.csv\nfor direct O-O comparison",
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
    axes[0].set_xlabel(r"$r$ ($\mathrm{\AA}$)")
    axes[0].set_ylabel(r"$g_{\mathrm{OO}}(r)$")
    axes[0].set_title(f"O-O RDF{title_suffix}")
    axes[0].set_xlim(1.8, 6.0)
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(r, g_oo, lw=2.0, label="O-O")
    axes[1].plot(r, g_ho, lw=2.0, label="O-H")
    axes[1].plot(r, g_hh, lw=2.0, label="H-H")
    axes[1].set_xlabel(r"$r$ ($\mathrm{\AA}$)")
    axes[1].set_ylabel(r"$g(r)$")
    axes[1].set_title("All partial RDFs")
    axes[1].set_xlim(0.0, 6.0)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def make_manuscript_figure(
    output_png: Path,
    final_block: np.ndarray,
    exp_oo: tuple[np.ndarray, np.ndarray],
    exp_oh: tuple[np.ndarray, np.ndarray] | None = None,
    exp_hh: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Write manuscript-style O--O, O--H, and H--H partial RDF panels."""
    mpl_cache = output_png.parent / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache.resolve()))

    import matplotlib.pyplot as plt

    r = final_block[:, R_COL]
    g_hh = final_block[:, G_HH_COL]
    g_ho = final_block[:, G_HO_COL]
    g_oo = final_block[:, G_OO_COL]
    r_exp, g_exp = exp_oo
    fig, axes = plt.subplots(3, 1, figsize=(5.4, 7.7), constrained_layout=True)

    oo_axis, oh_axis, hh_axis = axes
    oo_axis.plot(r, g_oo, color="#0072B2", lw=2.2, label="TRACE, classical NPT")
    oo_axis.plot(
        r_exp,
        g_exp,
        linestyle="None",
        marker="x",
        color="black",
        markersize=3.6,
        markeredgewidth=0.8,
        markevery=8,
        label="X-ray experiment, 295 K",
    )
    oo_axis.set_ylabel(r"$g_{\mathrm{OO}}(r)$")
    oo_axis.set_xlim(2.0, 6.0)
    oo_axis.set_ylim(0.0, 2.75)
    oo_axis.legend(frameon=False, loc="upper right")

    oh_axis.plot(r, g_ho, color="#0072B2", lw=2.2, label="TRACE, classical NPT")
    if exp_oh is not None:
        oh_axis.plot(
            exp_oh[0],
            exp_oh[1],
            linestyle="None",
            marker="x",
            color="black",
            markersize=3.6,
            markeredgewidth=0.8,
            label="Soper neutron/EPSR, 298 K",
        )
    oh_axis.set_ylabel(r"$g_{\mathrm{OH}}(r)$")
    oh_axis.set_xlim(0.5, 4.0)
    # The covalent O--H peak is much larger than the intermolecular features.
    # Matching the customary water-RDF presentation exposes the latter clearly.
    oh_axis.set_ylim(0.0, 4.0)
    oh_axis.legend(frameon=False, loc="upper right", fontsize=10.5)

    hh_axis.plot(r, g_hh, color="#0072B2", lw=2.2, label="TRACE, classical NPT")
    if exp_hh is not None:
        hh_axis.plot(
            exp_hh[0],
            exp_hh[1],
            linestyle="None",
            marker="x",
            color="black",
            markersize=3.6,
            markeredgewidth=0.8,
            label="Soper neutron/EPSR, 298 K",
        )
    hh_axis.set_xlabel(r"$r$ ($\mathrm{\AA}$)")
    hh_axis.set_ylabel(r"$g_{\mathrm{HH}}(r)$")
    hh_axis.set_xlim(1.0, 4.0)
    hh_axis.set_ylim(0.0, 4.0)
    hh_axis.legend(frameon=False, loc="upper right", fontsize=10.5)

    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(
            -0.14,
            1.02,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize=15,
        )
        axis.grid(alpha=0.18)

    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def analyze(
    rdf_path: Path,
    dump_path: Path,
    experimental_path: Path,
    timestep_ps: float,
    output_dir: Path,
    manuscript_figure: Path | None = None,
    experimental_oh_path: Path | None = None,
    experimental_hh_path: Path | None = None,
) -> dict[str, object]:
    blocks = parse_lammps_rdf(rdf_path)
    final_step, final_block = blocks[-1]
    dump_summary = parse_dump_summary(dump_path)
    exp_data = load_experimental_rdf(experimental_path)
    exp_oh = load_experimental_rdf(experimental_oh_path) if experimental_oh_path else None
    exp_hh = load_experimental_rdf(experimental_hh_path) if experimental_hh_path else None

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rdf_trace_final.csv"
    png_path = output_dir / "rdf_oo_trace_vs_experiment.png"
    summary_path = output_dir / "rdf_summary.txt"

    write_csv(csv_path, final_block)

    r = final_block[:, R_COL]
    g_oo = final_block[:, G_OO_COL]
    oo_peak = peak_position(r, g_oo, (2.3, 3.2))

    exp_peak = None
    exp_rmse = None
    if exp_data is not None:
        exp_peak = peak_position(exp_data[0], exp_data[1], (2.3, 3.2))
        exp_rmse = rmse_against_experiment(r, g_oo, exp_data[0], exp_data[1])

    title_suffix = f" ({len(blocks)} RDF blocks, {final_step * timestep_ps:.2f} ps)"
    make_plots(png_path, final_block, exp_data, title_suffix)
    if manuscript_figure is not None:
        if exp_data is None:
            raise RuntimeError("A manuscript RDF figure requires experimental O-O data")
        manuscript_figure.parent.mkdir(parents=True, exist_ok=True)
        make_manuscript_figure(manuscript_figure, final_block, exp_data, exp_oh, exp_hh)

    n_oxygen = dump_summary.type_counts.get(2)
    n_hydrogen = dump_summary.type_counts.get(1)
    n_waters = n_oxygen if n_oxygen is not None and n_hydrogen == 2 * n_oxygen else None

    lines = [
        "Water RDF analysis",
        "==================",
        f"RDF file: {rdf_path}",
        f"Complete RDF blocks parsed: {len(blocks)}",
        f"First RDF timestep: {blocks[0][0]}",
        f"Final RDF timestep: {final_step}",
        f"Time step: {timestep_ps:g} ps",
        f"Final sampled time: {final_step * timestep_ps:.6g} ps",
        "",
        "Trajectory summary",
        f"Dump file: {dump_path if dump_path.exists() else 'not found'}",
        f"Frames parsed: {dump_summary.n_frames}",
        f"Atoms: {dump_summary.n_atoms}",
        f"Type counts from first frame: {dump_summary.type_counts}",
        f"Water molecules inferred from type 1=H, type 2=O: {n_waters}",
        "",
        "RDF column mapping",
        "type 1 = H, type 2 = O",
        "g_HH = c_myRDF[2], g_HO = c_myRDF[4], g_OO = c_myRDF[6]",
        "Because the LAMMPS input uses 'ave running', the final RDF block is the cumulative average.",
        "",
        "O-O RDF diagnostics",
        f"TRACE/LAMMPS first O-O peak: r = {oo_peak[0]:.4f} A, g = {oo_peak[1]:.4f}",
    ]
    if exp_data is None:
        lines.extend(
            [
                f"Experimental file not found: {experimental_path}",
                "Add a two-column CSV named experimental_goo_skinner_295K.csv with r_A,g_OO columns.",
                "Recommended O-O reference: Skinner et al., J. Chem. Phys. 138, 074506 (2013).",
            ]
        )
    else:
        lines.extend(
            [
                f"Experimental file: {experimental_path}",
                f"Experimental first O-O peak: r = {exp_peak[0]:.4f} A, g = {exp_peak[1]:.4f}",
                f"O-O RDF RMSE over common 2.2-6.0 A interval: {exp_rmse:.6f}",
            ]
        )
    lines.extend(["", f"Wrote: {csv_path}", f"Wrote: {png_path}"])
    if experimental_oh_path is not None:
        lines.append(f"O-H reference file: {experimental_oh_path}")
    if experimental_hh_path is not None:
        lines.append(f"H-H reference file: {experimental_hh_path}")
    summary_path.write_text("\n".join(lines) + "\n")

    return {
        "n_blocks": len(blocks),
        "final_step": final_step,
        "time_ps": final_step * timestep_ps,
        "dump_summary": dump_summary,
        "n_waters": n_waters,
        "oo_peak": oo_peak,
        "exp_peak": exp_peak,
        "exp_rmse": exp_rmse,
        "csv_path": csv_path,
        "png_path": png_path,
        "summary_path": summary_path,
        "experimental_found": exp_data is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdf", type=Path, default=Path("h2o.rdf"))
    parser.add_argument("--dump", type=Path, default=Path("h2o.xyz"))
    parser.add_argument("--experimental", type=Path, default=Path("experimental_goo_skinner_295K.csv"))
    parser.add_argument("--timestep-ps", type=float, default=0.0005)
    parser.add_argument("--output-dir", type=Path, default=Path("rdf_analysis"))
    parser.add_argument("--experimental-oh", type=Path, default=None)
    parser.add_argument("--experimental-hh", type=Path, default=None)
    parser.add_argument(
        "--manuscript-figure",
        type=Path,
        default=None,
        help="Optional three-panel O-O, O-H, and H-H RDF figure with Skinner O-O markers.",
    )
    args = parser.parse_args()

    result = analyze(
        args.rdf,
        args.dump,
        args.experimental,
        args.timestep_ps,
        args.output_dir,
        args.manuscript_figure,
        args.experimental_oh,
        args.experimental_hh,
    )
    print(Path(result["summary_path"]).read_text())


if __name__ == "__main__":
    main()
