"""Statistics added for the Measurement revision (2026-09-26); post hoc, no verdict changes.

1. Equal-budget cells with uncertainty: per-window paired difference of the
   absolute error (fewer views minus more views), window-bootstrap 95 % interval
   of the MAE difference, and a two-sided sign test over windows.
   Rig A: allocation_revision_v104 (evaluation records 15-18; exploratory, as its
   own protocol file states).  Rig B: DPJAIT_PREREG_v0.1 (pre-registered cells),
   labelled arm, polygon estimator; the fewer-view side is averaged over the
   camera subsets of that size, as in the manuscript table.
2. Benefit of the recommended configuration over the default "all cameras at the
   full frame rate": paired per window, same statistics, plus frames per window.
3. Window-length scaling of m*: on the learned-detector chain at native noise,
   m* of 4 s sub-windows against the 16 s windows of the same records.

Usage: python src/measurement_revision_stats.py  -> experiments/measurement_revision_stats_2026-09-26/stats.json
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "experiments" / "measurement_revision_stats_2026-09-26"
RNG = np.random.default_rng(20260926)
B = 10000
RULE_A = "216f21c1+3e0f8f0+44c4b2e"


def sign_test(d):
    d = [x for x in d if x != 0]
    n, k = len(d), sum(1 for x in d if x > 0)
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return dict(n=n, fewer_views_worse=k, p_two_sided=min(1.0, 2 * tail))


def paired(a, b):
    """a, b: dict window -> absolute error (m).  Difference a - b in mm."""
    keys = sorted(set(a) & set(b))
    x = np.array([a[k] for k in keys]) * 1000
    y = np.array([b[k] for k in keys]) * 1000
    d = x - y
    idx = RNG.integers(0, len(d), (B, len(d)))
    boot = d[idx].mean(axis=1)
    return dict(n_windows=len(d), mae_a_mm=float(x.mean()), mae_b_mm=float(y.mean()),
                diff_mm=float(d.mean()), diff_ci95_mm=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                sign=sign_test(list(d)))


def rig_a_budget():
    rows = []
    for r in (15, 16, 17, 18):
        rows += json.load(open(ROOT / ("experiments/allocation_revision_v104_2026-09-16/record%d.json" % r)))
    out = {}
    for arm in ("dlt", "spatial"):
        for budget, (m3, m7) in ((84, (28, 12)), (168, (56, 24))):
            sel = lambda v, m: {(q["record"], q["start"]): abs(q["error_mm"]) / 1000 for q in rows  # noqa: E731
                                if q["grid"] == "uniform" and q["arm"] == arm and q["views"] == v and q["moments"] == m
                                and q["budget"] == budget and q["error_mm"] is not None
                                and (v == 7 or "+".join(q["cameras"]) == RULE_A)}
            out["A|%s|B%d" % (arm, budget)] = dict(fewer="3x%d" % m3, more="7x%d" % m7, **paired(sel(3, m3), sel(7, m7)))
    return out


def rig_b_rows():
    recs = ["R05_D1", "R06_D1", "R07_D1", "R08_D1", "R09_D1", "R10_D1", "R11_D1", "R16_D1_A"]
    rows = []
    for r in recs:
        d = json.load(open(ROOT / ("experiments/dpjait_prereg_v0.1/scored/%s.json" % r)))
        rows += [q for q in d["rows"] if q["chain"] == "annotated" and not q["failed"] and q["err_polygon"] is not None]
    return rows


def rig_b_budget(rows):
    out = {}
    for budget, (kf, mf) in ((40, (2, 20)), (68, (2, 34)), (100, (3, 34)), (200, (2, 100))):
        few = collections.defaultdict(list)
        more = {}
        for q in rows:
            key = (q["record"], q["window"])
            if q["k"] == kf and q["m"] == mf:
                few[key].append(q["err_polygon"])
            if q["k"] == 4 and q["m"] == budget // 4:
                more[key] = q["err_polygon"]
        few = {k: float(np.mean(v)) for k, v in few.items()}
        out["B|B%d" % budget] = dict(fewer="%dx%d" % (kf, mf), more="4x%d" % (budget // 4), **paired(few, more))
    return out


def benefit(rows_b):
    out = {}
    rows_a = json.load(open(ROOT / "experiments/allocation_confirm_v1_2026-09-15/scored.json"))
    rec = {(q["record"], q["start"]): abs(q["error_m"]) for q in rows_a
           if q["cameras"] == RULE_A and q["moments"] == 33 and q["method"] == "state_mean" and q["error_m"] is not None}
    for meth in ("limited", "polygon"):
        dft = {(q["record"], q["start"]): abs(q["error_m"]) for q in rows_a
               if q["views"] == 7 and q["moments"] == 200 and q["method"] == meth and q["error_m"] is not None}
        out["A|default_7x200_%s" % meth] = dict(recommended="3x33 (99 frames)", default="7x200 (1400 frames)", **paired(rec, dft))
    recb = {(q["record"], q["window"]): q["err_polygon"] for q in rows_b if q["k"] == 4 and q["m"] == 6}
    dftb = {(q["record"], q["window"]): q["err_polygon"] for q in rows_b if q["k"] == 4 and q["m"] == 100}
    out["B|default_4x100"] = dict(recommended="4x6 (24 frames)", default="4x100 (400 frames)", **paired(recb, dftb))
    return out


def t_scaling():
    """m* on 4 s sub-windows vs 16 s windows, learned-detector chain, native noise, rig B real."""
    import dpjait_noise_runner as v1
    exp = ROOT / "experiments/dpjait_noise_prereg_v0.2"
    grid4 = [4, 5, 6, 8, 10, 12, 14, 17, 20, 24, 28, 34, 40, 50, 60, 80, 100]
    err4 = collections.defaultdict(list)
    for name in v1.REAL:
        rec, _ = v1.open_record(name)
        sc = json.loads((exp / "scored" / (name + ".json")).read_text())
        P = np.load(exp / "predictions" / (name + "_dense.npz"))["A|0|0|drone"].astype(float)
        for w in sorted({q["window"] for q in sc["rows"]}):
            for s in range(4):
                f0 = w * 400 + s * 100
                if (f0 + 100) * 4 >= len(rec.reference["drone"]):
                    continue
                ref = rec.reference_path_length("drone", f0, f0 + 100)
                if not np.isfinite(ref):
                    continue
                for m in grid4:
                    pts = P[v1.instants(f0, 100, m)]
                    ok = ~np.isnan(pts).any(axis=1)
                    if ok[0] and ok[-1] and ok.sum() >= 2:
                        err4[m].append(abs(v1.polyline(pts[ok]) - ref))
    mae4 = {m: float(np.mean(v)) for m, v in err4.items()}
    m4 = min(mae4, key=mae4.get)
    dec = json.loads((exp / "decision.json").read_text())["result"]["families"]["real"]["levels"][0]
    return dict(m_star_4s=m4, m_star_16s=dec["m_star"], ratio=dec["m_star"] / m4, expected_ratio=4.0,
                mae_4s_mm={m: round(v * 1000, 1) for m, v in mae4.items()},
                n_subwindows=len(err4[grid4[0]]))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows_b = rig_b_rows()
    res = dict(rig_a_budget=rig_a_budget(), rig_b_budget=rig_b_budget(rows_b), benefit=benefit(rows_b),
               t_scaling=t_scaling(), bootstrap_resamples=B, seed=20260926)
    (OUT / "stats.json").write_text(json.dumps(res, indent=1))
    for sect in ("rig_a_budget", "rig_b_budget", "benefit"):
        for k, v in res[sect].items():
            print("%-26s n=%3d  a=%7.1f  b=%7.1f  diff=%7.1f [%7.1f, %7.1f]  sign %d/%d p=%.2g" % (
                k, v["n_windows"], v["mae_a_mm"], v["mae_b_mm"], v["diff_mm"], *v["diff_ci95_mm"],
                v["sign"]["fewer_views_worse"], v["sign"]["n"], v["sign"]["p_two_sided"]))
    t = res["t_scaling"]
    print("T scaling: m*(4 s) = %d, m*(16 s) = %d, ratio %.2f (expected 4); %d sub-windows" % (
        t["m_star_4s"], t["m_star_16s"], t["ratio"], t["n_subwindows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
