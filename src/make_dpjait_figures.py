"""IEEE-style figures for the DPJAIT confirmation run (protocol/DPJAIT_PREREG_v0.1.md).

All values come from experiments/dpjait_prereg_v0.1/scored/*.json, i.e. from sealed
predictions scored after the fact, and from the post-hoc noise diagnostic
experiments/dpjait_prereg_v0.1/sim_obs_noise.json.  The statistic is the one the
protocol pre-registered: MEAN absolute window error of the polygon estimator on the
annotated arm, over completed windows.

fig_dpjait_goldilocks : MAE vs m.  (a) real rig, k=4, one curve per record.
                        (b) simulated rig, k=8, one curve per record.  m* circled.
fig_dpjait_bias       : the mechanism.  (a) signed bias and standard deviation vs m
                        (real, k=4, pooled).  (b) signed bias vs m for the three
                        simulated records, with the zero crossing that locates m*.
fig_dpjait_views      : what views buy.  (a) MAE vs k at fixed m.  (b) window
                        failure rate vs k (independent of m).
fig_dpjait_identity   : cost of the reference-free identity chain, free/annotated
                        MAE ratio and free-arm failure rate vs k.

    python src/make_dpjait_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/dpjait_prereg_v0.1"
OUT = ROOT / "paper/tim"
GRID = [4, 6, 8, 10, 13, 17, 20, 25, 34, 50, 100]
REAL = ["R05_D1", "R06_D1", "R07_D1", "R08_D1", "R09_D1", "R10_D1", "R11_D1", "R16_D1_A"]
SIM = ["S08_D8", "S09_D6", "S10_D6"]
MULTI = "R14_D3"

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif",
                     "lines.linewidth": 1.0, "lines.markersize": 3.5, "axes.linewidth": 0.6})
MARKERS = ["o", "s", "^", "v", "D", "*", "P", "X", "<"]


def load(name):
    return json.loads((RUN / "scored" / (name + ".json")).read_text())["rows"]


def curve(rows, chain, k, field="err_polygon"):
    """Mean |error| in mm over completed windows, per m."""
    out = {}
    for m in GRID:
        v = [r[field] for r in rows
             if r["chain"] == chain and r["k"] == k and r["m"] == m
             and not r.get("failed") and np.isfinite(r.get(field, np.nan))]
        if v:
            out[m] = 1000.0 * float(np.mean(v))
    return out


def signed(rows, k):
    """Signed bias and standard deviation in mm, per m (annotated arm)."""
    bias, sd = {}, {}
    for m in GRID:
        v = [1000.0 * (r["est_polygon"] - r["reference_m"]) for r in rows
             if r["chain"] == "annotated" and r["k"] == k and r["m"] == m
             and not r.get("failed") and np.isfinite(r.get("reference_m", np.nan))]
        if len(v) > 1:
            bias[m], sd[m] = float(np.mean(v)), float(np.std(v, ddof=1))
    return bias, sd


def fail(rows, chain, k):
    sel = [r for r in rows if r["chain"] == chain and r["k"] == k]
    return (sum(1 for r in sel if r.get("failed")) / len(sel)) if sel else np.nan


def crossing(bias):
    """Linear interpolation of the m at which the signed bias changes sign."""
    ms = sorted(bias)
    for a, b in zip(ms, ms[1:]):
        if bias[a] < 0 <= bias[b]:
            w = -bias[a] / (bias[b] - bias[a])
            return a + w * (b - a)
    return None


def label_of(name):
    return name  # plain text: matplotlib is not in usetex mode, so no escaping


def fig_goldilocks(real, sim, values):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2))
    for ax, (names, data, k, title) in zip(axes, [
            (REAL, real, 4, "(a) real rig, $k=4$ (all four cameras)"),
            (SIM, sim, 8, "(b) simulated rig, $k=8$ (all eight cameras)")]):
        for i, n in enumerate(names):
            c = curve(data[n], "annotated", k)
            ax.plot(list(c), list(c.values()), marker=MARKERS[i % len(MARKERS)], ls="-",
                    label=label_of(n))
            mstar = min(c, key=c.get)
            ax.plot([mstar], [c[mstar]], marker="o", ms=7, mfc="none", mec="k", mew=0.8)
            values.setdefault("m_star", {})[n] = mstar
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("instants per 4 s window, $m$")
        ax.set_title(title, fontsize=8)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
        ax.legend(frameon=False, ncol=2, handlelength=1.6)
    axes[0].set_ylabel("mean |error| of path length (mm)")
    fig.tight_layout(pad=0.4)
    save(fig, "fig_dpjait_goldilocks")


def fig_bias(real, sim, values):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2))
    pooled = [r for n in REAL for r in real[n]]
    b, s = signed(pooled, 4)
    ax = axes[0]
    ax.axhline(0, color="0.4", lw=0.6)
    ax.plot(list(b), list(b.values()), marker="o", ls="-", color="k", label="signed bias")
    ax.plot(list(s), list(s.values()), marker="s", ls="--", color="0.45",
            label="standard deviation")
    x0 = crossing(b)
    if x0:
        ax.axvline(x0, color="0.5", ls=":", lw=0.8)
        ax.annotate("bias $=0$ at $m\\approx%.0f$" % x0, xy=(x0, 0), xytext=(x0 * 1.3, 170),
                    fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.5))
    values["real_bias_zero_m"] = x0
    ax.set_xscale("log")
    ax.set_xlabel("instants per 4 s window, $m$")
    ax.set_ylabel("mm")
    ax.set_title("(a) real rig, $k=4$, 8 records pooled", fontsize=8)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.legend(frameon=False, handlelength=1.8)

    ax = axes[1]
    ax.axhline(0, color="0.4", lw=0.6)
    values["sim_bias_zero_m"] = {}
    for i, n in enumerate(SIM):
        b, _ = signed(sim[n], 8)
        ax.plot(list(b), list(b.values()), marker=MARKERS[i], ls="-", label=label_of(n))
        x0 = crossing(b)
        values["sim_bias_zero_m"][n] = x0
        if x0:
            ax.axvline(x0, color="0.6", ls=":", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("instants per 4 s window, $m$")
    ax.set_ylabel("signed bias (mm)")
    ax.set_title("(b) simulated rig, $k=8$: zero crossing locates $m^{*}$", fontsize=8)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.legend(frameon=False, handlelength=1.6)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_dpjait_bias")


def fig_views(real, sim, values):
    """Single column: the failure rate against view count.

    The error-against-$k$ panel this figure used to carry is recorded in
    `values["mae_vs_k"]` and quoted in the text instead; at one column the two
    panels together were unreadable and the page budget does not allow a
    double-column figure here.
    """
    values["mae_vs_k"] = {}
    for n in REAL:
        values["mae_vs_k"][n] = {str(k): curve(real[n], "annotated", k).get(6)
                                 for k in [2, 3, 4]}
    for n in SIM:
        values["mae_vs_k"][n] = {str(k): curve(sim[n], "annotated", k).get(25)
                                 for k in [2, 3, 4, 5, 6, 7, 8]}

    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    values["fail_rate"] = {}
    for i, n in enumerate(SIM):
        ks = [2, 3, 4, 5, 6, 7, 8]
        f = [fail(sim[n], "annotated", k) for k in ks]
        ax.plot(ks, f, marker=MARKERS[i], ls="-", label=label_of(n))
        values["fail_rate"][n] = dict(zip(map(str, ks), f))
    multi = load(MULTI)
    for n, rows, mk in [(REAL[0], real[REAL[0]], "x"), (MULTI, multi, "+")]:
        ks = [2, 3, 4]
        f = [fail(rows, "annotated", k) for k in ks]
        ax.plot(ks, f, marker=mk, ls="--", color="0.55", label=n + " (real)")
        values["fail_rate"][n] = dict(zip(map(str, ks), f))
    ax.set_xlabel("views per instant, $k$")
    ax.set_ylabel("window failure rate")
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.legend(frameon=False, handlelength=1.6, fontsize=6)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_dpjait_views")


def fig_identity(real, values):
    rows_multi = load(MULTI)
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    ks = [2, 3, 4]
    ratios, fails = [], []
    for k in ks:
        ca = curve(rows_multi, "annotated", k)
        mstar = min(ca, key=ca.get)
        cf = curve(rows_multi, "free", k)
        ratios.append(cf[mstar] / ca[mstar] if mstar in cf else np.nan)
        fails.append(fail(rows_multi, "free", k))
    band = []
    for n in REAL:
        rr = []
        for k in ks:
            ca = curve(real[n], "annotated", k)
            if not ca:
                rr.append(np.nan)
                continue
            mstar = min(ca, key=ca.get)
            cf = curve(real[n], "free", k)
            rr.append(cf[mstar] / ca[mstar] if mstar in cf else np.nan)
        band.append(rr)
    band = np.asarray(band, dtype=float)
    ax.fill_between(ks, np.nanmin(band, axis=0), np.nanmax(band, axis=0), color="0.85",
                    label="8 single-drone records (range)")
    ax.plot(ks, ratios, marker="o", ls="-", color="k", label="R14_D3, 3 drones")
    ax.axhline(1.0, color="0.4", lw=0.6)
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xlabel("views per instant, $k$")
    ax.set_ylabel("free / annotated mean |error|")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax2 = ax.twinx()
    ax2.plot(ks, fails, marker="s", ls=":", color="0.45", label="free-arm failure rate")
    ax2.set_ylabel("free-arm failure rate", fontsize=8)
    ax2.tick_params(labelsize=7)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, handlelength=1.8, loc="upper right")
    values["identity_ratio_R14_D3"] = dict(zip(map(str, ks), ratios))
    values["identity_fail_R14_D3"] = dict(zip(map(str, ks), fails))
    values["identity_ratio_single_range"] = {
        str(k): [float(np.nanmin(band[:, i])), float(np.nanmax(band[:, i]))]
        for i, k in enumerate(ks)}
    fig.tight_layout(pad=0.4)
    save(fig, "fig_dpjait_identity")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / ("%s.%s" % (stem, ext)), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s.{pdf,png}" % (OUT / stem))


def main() -> int:
    real = {n: load(n) for n in REAL}
    sim = {n: load(n) for n in SIM}
    values = {}
    fig_goldilocks(real, sim, values)
    fig_bias(real, sim, values)
    fig_views(real, sim, values)
    fig_identity(real, values)
    noise = json.loads((RUN / "sim_obs_noise.json").read_text())
    values["sim_obs_noise"] = {r["record"]: dict(residual_px_median=r["residual_px_median"],
                                                 err3d_mm_median=r["err3d_mm_median"])
                               for r in noise["records"]}
    (OUT / "figure_values_dpjait.json").write_text(json.dumps(values, indent=1), encoding="utf-8")
    print("wrote %s" % (OUT / "figure_values_dpjait.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
