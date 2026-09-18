"""IEEE-style figures for the TIM manuscript (single column 3.5 in, 8 pt fonts, line/marker coded).

fig_goldilocks.pdf : k_max median-window MAE vs m for MCalib, Lyon RFM5, Lyon SV (log-log), m* marked.
fig_budget.pdf     : the six pre-registered equal-budget cells as paired bars (3-view vs other).
Lyon values are DEVELOPMENT values until the confirmation run replaces experiments paths below.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cross_rig_analysis import load_mcalib, load_lyon, mae_table, kmax_curve  # noqa: E402

OUT = ROOT / "paper/tim"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "font.family": "serif", "lines.linewidth": 1.0, "lines.markersize": 3.5, "axes.linewidth": 0.6})
STYLES = {2: ("s", "--"), 3: ("o", "-"), 4: ("^", ":"), 5: ("v", "-."), 7: ("D", "--"), 9: ("*", "-")}


def median_curve(rows, k):
    """median over windows of the error, for the median subset at each m (k_max: the single all-camera subset)."""
    per = {}
    for m in sorted({r["m"] for r in rows if r["k"] == k}):
        subs = {}
        for r in rows:
            if r["k"] == k and r["m"] == m and r["err"] is not None:
                subs.setdefault(r["cameras"], []).append(abs(r["err"]) * 1000)
        vals = [np.median(v) for v in subs.values() if len(v) >= 3]
        if vals:
            per[m] = float(np.median(vals))
    return per


def fig_goldilocks(data):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3))  # double-column width
    for ax, (label, rows) in zip(axes, data.items()):
        ks = sorted({r["k"] for r in rows})
        kmax = max(ks)
        for k in ks:
            c = median_curve(rows, k)
            mk, ls = STYLES.get(k, ("x", "-"))
            ax.plot(list(c), list(c.values()), marker=mk, ls=ls, label=f"$k={k}$" + (" (all)" if k == kmax else ""), color="k" if k == kmax else None)
        cm = median_curve(rows, kmax)
        ax.axvline(min(cm, key=cm.get), color="0.5", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("instants per 4 s window, $m$")
        ax.set_title(label, fontsize=8)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
        ax.legend(frameon=False, ncol=2, handlelength=1.8)
    axes[0].set_ylabel("median |error| of path length (mm)")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_goldilocks.pdf")
    fig.savefig(OUT / "fig_goldilocks.png", dpi=200)


def fig_budget(data):
    cells = []  # (rig/target, budget, label3, mae3, labelo, maeo)
    for label, rows in data.items():
        med, _ = mae_table(rows)
        if label.startswith("MCalib"):
            # rig-A budget cells come from the 2026-09-16 sensitivity analysis (rule triple vs 7 views,
            # spatial arm, uniform grid, evaluation records 15-18): m=56/24 are not on the confirm grid
            V = json.loads((ROOT / "experiments/allocation_revision_v104_2026-09-16/summary.json").read_text())["paired"]
            for B, a, b in [(84, (3, 28), (7, 12)), (168, (3, 56), (7, 24))]:
                u = next(p for p in V if p["split"] == "evaluation" and p["budget"] == B and p["arm"] == "spatial" and p["grid"] == "uniform")
                cells.append((label, B, f"{a[0]}x{a[1]}", u["mae3_mm"], f"{b[0]}x{b[1]}", u["mae7_mm"]))
            continue
        spec = [(240, (3, 80), (5, 48)), (360, (3, 120), (9, 40))]
        for B, a, b in spec:
            if a in med and b in med:
                cells.append((label, B, f"{a[0]}x{a[1]}", med[a], f"{b[0]}x{b[1]}", med[b]))
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    x = np.arange(len(cells))
    ax.bar(x - 0.18, [c[3] for c in cells], 0.36, color="0.25", label="3 views")
    ax.bar(x + 0.18, [c[5] for c in cells], 0.36, color="0.7", label="more views")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c[0].split(' ')[0]}\n{c[1]} fr\n{c[2]} | {c[4]}" for c in cells], fontsize=6)
    ax.set_ylabel("MAE (mm), median subset")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", which="both", lw=0.3, alpha=0.4)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_budget.pdf")
    fig.savefig(OUT / "fig_budget.png", dpi=200)
    return cells


def main():
    mc = load_mcalib()
    ly = load_lyon()
    data = {"MCalib, hand-held marker": mc, "Lyon, foot marker RFM5 [dev]": [r for r in ly if r["target"] == "RFM5"],
            "Lyon, head marker SV [dev]": [r for r in ly if r["target"] == "SV"]}
    fig_goldilocks(data)
    cells = fig_budget(data)
    (OUT / "figure_values.json").write_text(json.dumps(cells, indent=1))
    print("wrote", OUT / "fig_goldilocks.pdf", OUT / "fig_budget.pdf", "cells:", len(cells))


if __name__ == "__main__":
    main()
