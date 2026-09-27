"""Cross-rig mechanism figure: the Goldilocks density is the zero crossing of the
signed sampling bias, under whichever statistic defines it.

Panel (a): signed bias as a percentage of the measurand, against the number of
instants per window, for rig A (MCalib, k=7), rig B (DPJAIT real, k=4, eight
records pooled) and the simulated 8-camera twin (three records, k=8).

Panel (b): the m at which the signed bias crosses zero (log-linear interpolation
on the grid) against the m that minimises |error|, for every rig and statistic.
Points on the diagonal mean the zero crossing predicts the Goldilocks density.

    python src/make_mechanism_figure.py

Sources: experiments/allocation_confirm_v1_2026-09-15/scored.json (rig A, sealed
confirmation) and experiments/dpjait_prereg_v0.1/scored/*.json (rig B, sealed
confirmation).  No new measurement is made here; both are re-readings of scored
predictions.
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
from cross_rig_analysis import load_lyon, load_mcalib  # noqa: E402

OUT = ROOT / "paper/tim"
RUN = ROOT / "experiments/dpjait_prereg_v0.1"
GRID_B = [4, 6, 8, 10, 13, 17, 20, 25, 34, 50, 100]
REAL = ["R05_D1", "R06_D1", "R07_D1", "R08_D1", "R09_D1", "R10_D1", "R11_D1", "R16_D1_A"]
SIM = ["S08_D8", "S09_D6", "S10_D6"]
MK = ["o", "s", "v", "^", "D", "P", "X"]
MCALIB_REF_M = 6.625   # mean measurand of the rig-A confirmation set
LYON_REF_M = {"RFM5": 6.0, "SV": 0.64}   # nominal per-window path length, rig C (development)

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif",
                     "lines.linewidth": 1.0, "lines.markersize": 3.5, "axes.linewidth": 0.6})


def dpjait_rows(name):
    return json.loads((RUN / "scored" / (name + ".json")).read_text())["rows"]


def series_mcalib(k=7):
    rows = [r for r in load_mcalib() if r["k"] == k and r["err"] is not None]
    per = {}
    for m in sorted({r["m"] for r in rows}):
        per[m] = np.asarray([r["err"] for r in rows if r["m"] == m]) * 1000.0
    return per, MCALIB_REF_M


def series_lyon(target):
    """Rig C, development slice only (five windows): all nine cameras."""
    rows = [r for r in load_lyon() if r["target"] == target and r["err"] is not None]
    kmax = max(r["k"] for r in rows)
    per = {}
    for m in sorted({r["m"] for r in rows if r["k"] == kmax}):
        per[m] = np.asarray([r["err"] for r in rows if r["k"] == kmax and r["m"] == m]) * 1000.0
    return per, LYON_REF_M[target]


def series_dpjait(names, k):
    rows = [r for n in names for r in dpjait_rows(n)
            if r["chain"] == "annotated" and r["k"] == k and not r.get("failed")
            and np.isfinite(r.get("reference_m", np.nan))]
    per, refs = {}, []
    for m in GRID_B:
        v = [1000.0 * (r["est_polygon"] - r["reference_m"]) for r in rows if r["m"] == m]
        if v:
            per[m] = np.asarray(v)
    refs = [r["reference_m"] for r in rows]
    return per, float(np.median(refs))


def summarise(per, stat):
    """Signed bias and |error| per m under the chosen statistic."""
    f = np.mean if stat == "mean" else np.median
    bias = {m: float(f(v)) for m, v in per.items()}
    err = {m: float(f(np.abs(v))) for m, v in per.items()}
    return bias, err


def zero_crossing(bias):
    ms = sorted(bias)
    for a, b in zip(ms, ms[1:]):
        if bias[a] < 0 <= bias[b]:
            # interpolate in log m, which is how the grid is spaced
            w = -bias[a] / (bias[b] - bias[a])
            return float(np.exp(np.log(a) + w * (np.log(b) - np.log(a))))
    return None


def main() -> int:
    sets = [("rig A, MCalib, $k=7$",) + series_mcalib(7),
            ("rig B, DPJAIT real, $k=4$",) + series_dpjait(REAL, 4)]
    sets += [("rig B$'$, %s, $k=8$" % n,) + series_dpjait([n], 8) for n in SIM]
    if "--no-rig-c" not in sys.argv:   # the Measurement version keeps rig C in the supplement only
        sets += [("rig C dev, %s, $k=9$" % t,) + series_lyon(t) for t in ("RFM5", "SV")]
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else OUT
    out.mkdir(parents=True, exist_ok=True)

    single = "--single-column" in sys.argv          # full text width of a one-column journal page
    fs_leg, fs_title = (7.0, 9) if single else (5.8, 8)
    if single:
        plt.rcParams.update({"font.size": 8.5})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0) if single else (7.16, 2.1))
    ax = axes[0]
    ax.axhline(0, color="0.4", lw=0.6)
    values = {}
    for i, (label, per, ref) in enumerate(sets):
        stat = "median"
        bias, err = summarise(per, stat)
        ms = sorted(bias)
        ax.plot(ms, [100.0 * bias[m] / (ref * 1000.0) for m in ms],
                marker=MK[i], ls="--" if "dev" in label else "-", label=label)
        x0 = zero_crossing(bias)
        if x0:
            ax.plot([x0], [0], marker="|", ms=8, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("instants per window, $m$")
    ax.set_ylabel("signed bias (% of the measurand)")
    ax.set_title("(a) sampling bias changes sign at $m^{*}$", fontsize=fs_title)
    ax.set_ylim(-12, 12)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.legend(frameon=False, handlelength=1.4, ncol=2, loc="lower right", fontsize=fs_leg,
              columnspacing=1.0)

    ax = axes[1]
    for i, (label, per, ref) in enumerate(sets):
        for stat, mfc in [("median", "k"), ("mean", "none")]:
            bias, err = summarise(per, stat)
            x0 = zero_crossing(bias)
            mstar = min(err, key=err.get)
            key = "%s|%s" % (label, stat)
            values[key] = dict(zero_crossing=x0, m_star=mstar,
                               censored=bool(x0 is None))
            if x0 is None:
                # the bias never crosses on the grid: both quantities are censored
                # at the frame rate, so the point is drawn at the last grid value
                # with an arrow away from the origin
                last = max(err)
                ax.annotate("", xy=(last * 1.9, mstar * 1.9), xytext=(last, mstar),
                            arrowprops=dict(arrowstyle="->", lw=0.6, color="0.4"))
                ax.plot([last], [mstar], marker=MK[i], mfc=mfc, mec="k", ls="none",
                        label=label + " (censored)" if stat == "median" else None)
            else:
                ax.plot([x0], [mstar], marker=MK[i], mfc=mfc, mec="k", ls="none",
                        label=label if stat == "median" else None)
    lim = [3, 300]
    ax.plot(lim, lim, color="0.5", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("$m$ where the signed bias crosses zero")
    ax.set_ylabel("$m^{*}$, minimiser of |error|")
    ax.set_title("(b) the crossing predicts $m^{*}$ (filled: median, open: mean)", fontsize=fs_title)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.legend(frameon=False, handlelength=1.0, loc="upper left", fontsize=fs_leg if single else None)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(out / ("fig_mechanism.%s" % ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    (out / "figure_values_mechanism.json").write_text(json.dumps(values, indent=1), encoding="utf-8")
    for k, v in values.items():
        print("%-40s crossing %-8s m* %s" % (k, ("%.1f" % v["zero_crossing"]) if v["zero_crossing"] else "none", v["m_star"]))
    print("wrote %s and %s" % (out / "fig_mechanism.pdf", out / "figure_values_mechanism.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
