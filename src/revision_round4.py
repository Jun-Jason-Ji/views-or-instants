"""Analyses for the fourth review round of the Measurement version (2026-09-26); post hoc.

a1  record-level statistics for rig B: per-record budget cells and savings, a record-cluster
    bootstrap (records resampled, windows kept together), and leave-one-record-out winners.
a2  per-record noise-ladder slopes (log m* against log sigma_a) on both observation chains.
a3  m* as a function of the number of views k (rig B, published boxes, 4 s windows; rig A).
a4  dense-estimator baselines: keep every frame and smooth, against sampling at m*.
    Learned-detector chain, native noise, rig B, 16 s windows; smoother settings chosen by
    leave-one-record-out on the error (so each record is scored with settings tuned on the others).
a5  budget figure data: MAE(k, m) on rig B with iso-budget lines.

Usage: python src/revision_round4.py -> experiments/revision_round4_2026-09-26/results.json
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "experiments" / "revision_round4_2026-09-26"
RNG = np.random.default_rng(20260927)
RECS = ["R05_D1", "R06_D1", "R07_D1", "R08_D1", "R09_D1", "R10_D1", "R11_D1", "R16_D1_A"]


# ---------------------------------------------------------------- a1
def rig_b_rows():
    rows = []
    for r in RECS:
        d = json.load(open(ROOT / ("experiments/dpjait_prereg_v0.1/scored/%s.json" % r)))
        rows += [q for q in d["rows"] if q["chain"] == "annotated" and not q["failed"] and q["err_polygon"] is not None]
    return rows


def per_window(rows, k, m, subset_mean=True):
    by = collections.defaultdict(list)
    for q in rows:
        if q["k"] == k and q["m"] == m:
            by[(q["record"], q["window"])].append(q["err_polygon"])
    return {w: float(np.mean(v)) for w, v in by.items()}


def cluster_stats(a, b):
    """a, b: dict (record, window) -> abs error.  Record-level MAE difference a-b (mm)."""
    keys = sorted(set(a) & set(b))
    recs = sorted({k[0] for k in keys})
    per = {r: 1000 * np.mean([a[k] - b[k] for k in keys if k[0] == r]) for r in recs}
    d = np.array([per[r] for r in recs])
    boot = []
    for _ in range(10000):
        pick = RNG.integers(0, len(recs), len(recs))
        # windows of a resampled record are kept together; weight records by their window count
        num = sum(sum(1000 * (a[k] - b[k]) for k in keys if k[0] == recs[i]) for i in pick)
        den = sum(sum(1 for k in keys if k[0] == recs[i]) for i in pick)
        boot.append(num / den)
    pooled = 1000 * np.mean([a[k] - b[k] for k in keys])
    return dict(n_records=len(recs), pooled_diff_mm=float(pooled),
                record_cluster_ci95_mm=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                per_record_diff_mm={r: round(float(v), 1) for r, v in per.items()},
                records_favouring_a=int(np.sum(d < 0)), records_favouring_b=int(np.sum(d > 0)),
                leave_one_record_out_sign_stable=bool(all(
                    np.sign(np.mean([per[r] for r in recs if r != x])) == np.sign(pooled) for x in recs)))


def a1(rows):
    out = {}
    for budget, (kf, mf) in ((40, (2, 20)), (68, (2, 34)), (100, (3, 34)), (200, (2, 100))):
        out["budget_B%d" % budget] = cluster_stats(per_window(rows, kf, mf), per_window(rows, 4, budget // 4))
    for m in (6, 8, 10, 13):
        out["saving_4x%d_vs_4x100" % m] = cluster_stats(per_window(rows, 4, m), per_window(rows, 4, 100))
    return out


# ---------------------------------------------------------------- a2
def a2():
    out = {}
    for chain, exp in (("blob", "dpjait_noise_prereg_v0.1"), ("learned", "dpjait_noise_prereg_v0.2")):
        slopes = {}
        for r in RECS:
            sc = json.load(open(ROOT / "experiments" / exp / "scored" / (r + ".json")))
            rows = [q for q in sc["rows"] if q.get("T", 400) == 400 and q.get("ladder", "A") == "A" and q["err"] is not None]
            sig = sc["sigma_a_m"]
            xs, ys = [], []
            for li in range(6):
                by = collections.defaultdict(list)
                for q in rows:
                    if q["level"] == li:
                        by[q["m"]].append(abs(q["err"]))
                if not by:
                    continue
                mae = {m: np.mean(v) for m, v in by.items()}
                ms = min(mae, key=mae.get)
                key = ("A|%d" % li) if ("A|%d" % li) in sig else str(li)
                if key in sig and ms not in (min(mae), max(mae)):
                    xs.append(np.log(sig[key])); ys.append(np.log(ms))
            slopes[r] = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 4 else None
        v = [s for s in slopes.values() if s is not None]
        out[chain] = dict(per_record=slopes, median=float(np.median(v)), min=float(min(v)), max=float(max(v)),
                          in_band=sum(-0.7 <= s <= -0.3 for s in v), n=len(v))
    return out


# ---------------------------------------------------------------- a3
def a3(rows):
    out = {"rig_B_4s_published_boxes": {}}
    for k in (2, 3, 4):
        by = collections.defaultdict(list)
        for q in rows:
            if q["k"] == k:
                by[q["m"]].append(q["err_polygon"])
        mae = {m: 1000 * np.mean(v) for m, v in by.items()}
        out["rig_B_4s_published_boxes"]["k=%d" % k] = dict(m_star=min(mae, key=mae.get),
                                                           mae_mm={m: round(v, 1) for m, v in sorted(mae.items())})
    ra = json.load(open(ROOT / "experiments/allocation_confirm_v1_2026-09-15/scored.json"))
    out["rig_A_state_mean"] = {}
    for views in (2, 3, 7):
        by = collections.defaultdict(list)
        for q in ra:
            if q["views"] == views and q["method"] == "state_mean" and q["error_m"] is not None:
                by[q["moments"]].append(abs(q["error_m"]))
        mae = {m: 1000 * np.mean(v) for m, v in by.items()}
        out["rig_A_state_mean"]["k=%d" % views] = dict(m_star=min(mae, key=mae.get),
                                                       mae_mm={m: round(v, 1) for m, v in sorted(mae.items())})
    return out


# ---------------------------------------------------------------- a4
def rts_cv(P, dt, q, r):
    """Constant-velocity Kalman filter + RTS smoother, per axis. P: (n,3) with NaN gaps."""
    F = np.array([[1, dt], [0, 1]])
    Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    H = np.array([[1.0, 0.0]])
    out = np.full_like(P, np.nan)
    for ax in range(3):
        z = P[:, ax]
        n = len(z)
        x = np.array([z[~np.isnan(z)][0], 0.0]); C = np.diag([r, 1.0])
        xs, Cs, xp_, Cp_ = [], [], [], []
        for i in range(n):
            xp = F @ x; Cp = F @ C @ F.T + Q
            if np.isfinite(z[i]):
                S = H @ Cp @ H.T + r; K = Cp @ H.T / S
                x = xp + (K * (z[i] - H @ xp)).ravel(); C = (np.eye(2) - K @ H) @ Cp
            else:
                x, C = xp, Cp
            xs.append(x); Cs.append(C); xp_.append(xp); Cp_.append(Cp)
        xsm = xs[-1]; res = [xsm]
        for i in range(n - 2, -1, -1):
            G = Cs[i] @ F.T @ np.linalg.inv(Cp_[i + 1])
            xsm = xs[i] + G @ (xsm - xp_[i + 1]); res.append(xsm)
        out[:, ax] = np.array(res[::-1])[:, 0]
    return out


def a4():
    import dpjait_noise_runner as v1
    exp = ROOT / "experiments/dpjait_noise_prereg_v0.2"
    T = 400
    grid = [4, 5, 6, 8, 10, 12, 14, 17, 20, 24, 28, 34, 40, 50, 60, 80, 100, 134, 200, 400]
    sg_w = [5, 9, 15, 25, 41, 61, 81, 101, 151]
    rts_q = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    data = []   # per window: dict with errors for every candidate
    for name in RECS:
        rec, _ = v1.open_record(name)
        sc = json.loads((exp / "scored" / (name + ".json")).read_text())
        P = np.load(exp / "predictions" / (name + "_dense.npz"))["A|0|0|drone"].astype(float)
        refs = {q["window"]: q["reference_m"] for q in sc["rows"]}
        for w, ref in sorted(refs.items()):
            f0 = w * T
            if f0 + T + 1 > len(P) or not np.isfinite(ref):
                continue
            Pw = P[f0:f0 + T + 1]
            ok = ~np.isnan(Pw).any(axis=1)
            if ok.mean() < 0.9:
                continue
            idx = np.arange(len(Pw))
            Pf = np.stack([np.interp(idx, idx[ok], Pw[ok, i]) for i in range(3)], axis=1)
            e = {"poly_m%d" % m: abs(v1.polyline(Pf[v1.instants(0, T, m)]) - ref) for m in grid}
            for wl in sg_w:
                e["sg_%d" % wl] = abs(v1.polyline(savgol_filter(Pf, wl, 2, axis=0)) - ref)
            d2 = Pf[2:] - 2 * Pf[1:-1] + Pf[:-2]
            r = float(np.mean(np.var(d2, axis=0)) / 6.0)
            for q in rts_q:
                e["rts_%g" % q] = abs(v1.polyline(rts_cv(Pw.copy(), 1 / 25.0, q, max(r, 1e-6))) - ref)
            data.append((name, e))
    keys = list(data[0][1])
    def mae(sel, key):
        return 1000 * np.mean([e[key] for n, e in data if sel(n)])
    res = {"n_windows": len(data), "fixed": {k: round(mae(lambda n: True, k), 1) for k in keys}}
    # leave-one-record-out choice of the smoother setting and of m for the polyline
    loro = collections.defaultdict(list)
    for fam, cand in (("polyline_sparse", [k for k in keys if k.startswith("poly_m") and k != "poly_m400"]),
                      ("savitzky_golay_full_rate", [k for k in keys if k.startswith("sg_")]),
                      ("rts_full_rate", [k for k in keys if k.startswith("rts_")])):
        errs, picks = [], []
        for held in RECS:
            best = min(cand, key=lambda k: mae(lambda n: n != held, k))
            picks.append(best)
            errs += [e[best] for n, e in data if n == held]
        loro[fam] = dict(mae_mm=round(1000 * float(np.mean(errs)), 1), chosen=collections.Counter(picks).most_common(3))
    loro["polyline_full_rate"] = dict(mae_mm=res["fixed"]["poly_m400"])
    res["leave_one_record_out"] = dict(loro)
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = rig_b_rows()
    prev = OUT / "results.json"
    res = json.loads(prev.read_text()) if prev.exists() and "--a4-only" in sys.argv else dict(a1=a1(rows), a2=a2(), a3=a3(rows))
    res["a4"] = a4()
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print("a2", {c: {k: v for k, v in d.items() if k != "per_record"} for c, d in res["a2"].items()})
    print("a3 B", {k: v["m_star"] for k, v in res["a3"]["rig_B_4s_published_boxes"].items()},
          "A", {k: v["m_star"] for k, v in res["a3"]["rig_A_state_mean"].items()})
    print("a4", json.dumps(res["a4"]["leave_one_record_out"], default=str))
    print("a4 fixed", {k: v for k, v in res["a4"]["fixed"].items() if not k.startswith("poly_m") or k in ("poly_m24", "poly_m400")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
