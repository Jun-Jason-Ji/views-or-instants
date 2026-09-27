"""Budget figure for the Measurement version: rig B, MAE against instants m and against the
frame budget B = k m, for k = 2, 3, 4 views (published boxes, labelled arm, polygon estimator,
eight four-camera confirmation records, 4 s windows; the k<4 curves average the camera subsets).

Usage: python src/make_budget_figure.py -> paper/measurement/fig_budget.pdf
"""
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RECS = ["R05_D1", "R06_D1", "R07_D1", "R08_D1", "R09_D1", "R10_D1", "R11_D1", "R16_D1_A"]
plt.rcParams.update({"font.size": 8.5})


def main():
    rows = []
    for r in RECS:
        d = json.load(open(ROOT / ("experiments/dpjait_prereg_v0.1/scored/%s.json" % r)))
        rows += [q for q in d["rows"] if q["chain"] == "annotated"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
    vals = {}
    for k, mk, col in ((2, "o", "C0"), (3, "s", "C1"), (4, "^", "C2")):
        by, fail = collections.defaultdict(list), collections.defaultdict(list)
        for q in rows:
            if q["k"] == k:
                fail[q["m"]].append(q["failed"])
                if not q["failed"] and q["err_polygon"] is not None:
                    by[q["m"]].append(q["err_polygon"])
        ms = sorted(by)
        mae = [1000 * np.mean(by[m]) for m in ms]
        vals[k] = dict(m=ms, mae_mm=mae, failure_rate=[float(np.mean(fail[m])) for m in ms])
        ms_star = ms[int(np.argmin(mae))]
        for ax, x in ((axes[0], ms), (axes[1], [k * m for m in ms])):
            ax.plot(x, mae, marker=mk, ms=4, color=col, label="$k=%d$" % k)
        axes[0].plot([ms_star], [min(mae)], marker=mk, ms=9, mfc="none", mec="k")
    for B in (40, 68, 100, 200):
        axes[1].axvline(B, color="0.6", lw=0.7, ls=":")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("instants per 4 s window, $m$"); axes[0].set_ylabel("MAE (mm)")
    axes[0].set_title("(a) $m^{*}=6$ for every $k$ (circled)", fontsize=9)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("frames per window, $B=km$"); axes[1].set_ylabel("MAE (mm)")
    axes[1].set_title("(b) at equal $B$ (dotted), more views win", fontsize=9)
    for ax in axes:
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
        ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(pad=0.4)
    out = ROOT / "paper/measurement"
    fig.savefig(out / "fig_budget.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_budget.png", dpi=200, bbox_inches="tight")
    (out / "figure_values_budget.json").write_text(json.dumps(vals, indent=1))
    print({k: (v["m"][int(np.argmin(v["mae_mm"]))], round(min(v["mae_mm"]), 1), max(v["failure_rate"])) for k, v in vals.items()})


if __name__ == "__main__":
    main()
