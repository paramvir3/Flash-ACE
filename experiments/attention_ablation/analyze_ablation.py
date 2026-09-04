"""Collect and plot the attention ablation.

Produces one figure with four panels -- total loss, energy RMSE, force RMSE and
stress RMSE -- each showing the validation metric against epoch for every arm,
with the band spanning the seeds. A summary table reports the final metric as
mean +/- s.d. over seeds, and the effect size relative to the `full` arm.

The comparison that answers the question is `full` vs `no_qk`: both keep a
learned radial filter, and they differ only by the content-dependent query-key
term. `uniform` and `none` bound the other end -- what the ACE descriptor alone
delivers.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ARMS = ["full", "no_qk", "no_qk_matched", "uniform", "none"]
LABEL = {
    "full": "full attention  (q·k + radial)",
    "no_qk": "no query-key  (radial only)",
    "no_qk_matched": "no query-key, parameter-matched",
    "uniform": "uniform weights",
    "none": "no attention block  (ACE + FFN)",
}
COLOR = {"full": "#1b6ca8", "no_qk": "#d1495b", "no_qk_matched": "#e08a00",
         "uniform": "#8d6a9f", "none": "#3f8f5e"}
PANELS = [
    ("val_loss", "Total validation loss", "loss"),
    ("val_energy_rmse_offset_free", "Energy RMSE (offset-free)", "meV/atom"),
    ("val_force_rmse", "Force RMSE", "eV/Å"),
    ("val_stress_rmse", "Stress RMSE", "eV/Å³"),
]


def load(run: pathlib.Path):
    csv = run / "plots" / "training_history.csv"
    if not csv.exists():
        return None
    rows = csv.read_text().strip().splitlines()
    header = rows[0].split(",")
    data = np.array([[float(x) for x in r.split(",")] for r in rows[1:]])
    return {name: data[:, i] for i, name in enumerate(header)}


def collect():
    out = {}
    for arm in ARMS:
        runs = sorted(pathlib.Path("runs").glob(f"{arm}_seed*"))
        series = [h for h in (load(r) for r in runs) if h is not None]
        if series:
            out[arm] = series
    return out


def main() -> int:
    history = collect()
    if not history:
        print("No completed runs found under runs/. Run run_ablation.sh first.")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    for ax, (key, title, unit) in zip(axes.ravel(), PANELS):
        for arm, series in history.items():
            usable = [s[key] for s in series if key in s]
            if not usable:
                continue
            n = min(len(u) for u in usable)
            stack = np.stack([u[:n] for u in usable])
            epochs = np.arange(1, n + 1)
            mean = stack.mean(axis=0)
            ax.plot(epochs, mean, color=COLOR[arm], lw=1.8, label=LABEL[arm])
            if stack.shape[0] > 1:
                ax.fill_between(epochs, stack.min(axis=0), stack.max(axis=0),
                                color=COLOR[arm], alpha=0.16, lw=0)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{title}  [{unit}]" if unit != "loss" else title)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, which="both", lw=0.5)
    axes[0, 0].legend(fontsize=8.5, framealpha=0.95)
    fig.suptitle(
        "Does attention help beyond the ACE descriptor?  CsPbI$_3$, "
        f"{max(len(v) for v in history.values())} seeds per arm",
        fontsize=12.5,
    )
    fig.tight_layout()
    fig.savefig("ablation.png", dpi=200)
    print("  wrote ablation.png")

    # ---- summary table: final-epoch metric, mean +/- s.d. over seeds ----
    print()
    print(f"{'arm':<10} {'seeds':>5} " + " ".join(f"{t:>22}" for _, t, _ in PANELS))
    print("-" * 104)
    finals = {}
    for arm in ARMS:
        if arm not in history:
            continue
        row, cells = [], []
        for key, _, _ in PANELS:
            vals = np.array([s[key][-1] for s in history[arm] if key in s])
            row.append(vals)
            cells.append(f"{vals.mean():>10.4g} +/- {vals.std(ddof=1) if len(vals)>1 else 0:<8.2g}"
                         if len(vals) else f"{'-':>22}")
        finals[arm] = row
        print(f"{arm:<10} {len(history[arm]):>5} " + " ".join(cells))

    if "full" in finals:
        print()
        print("Change relative to full attention (positive = worse without it):")
        for arm in ARMS:
            if arm == "full" or arm not in finals:
                continue
            parts = []
            for (key, title, _), base, other in zip(PANELS, finals["full"], finals[arm]):
                if not len(base) or not len(other):
                    continue
                pct = 100 * (other.mean() - base.mean()) / base.mean()
                # Is the gap larger than the seed scatter it sits in?
                pooled = np.sqrt((base.var(ddof=1) + other.var(ddof=1)) / 2) if len(base) > 1 else 0.0
                sig = "" if pooled == 0 else (
                    "  (within seed scatter)" if abs(other.mean() - base.mean()) < pooled else "")
                parts.append(f"    {title:<28} {pct:+7.1f}%{sig}")
            print(f"  {arm}:")
            print("\n".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
