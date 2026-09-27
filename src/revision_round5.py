"""Fifth-round analyses for the Measurement version (2026-09-26); post hoc, no verdict changes.

b1  Pilot transfer on the two confirmation records, with the cost accounting made explicit:
    - frames are unique camera frames (a frame serves every target seen in it), counted per window,
      not per target;
    - "every frame" means all M = 401 frames of a 16 s window, both for its error (polyline through
      every frame) and for its cost;
    - the baseline measures the N measured windows only; the pilot is an extra cost (its windows
      are not reported as measurements);
    - each cell records success (a density was output) or fallback (no crossing -> every frame).
b2  Same-split comparison of the whole reference-free procedure with fixed smoothers:
    learned-detector chain, native noise, the eight single-target rig-B records.  In each record
    the first third of the windows is the pilot (M3 fitted there, prediction rounded to the grid),
    the rest are test windows.  Smoother settings are those chosen leave-one-record-out in round 4
    (Savitzky-Golay 61 frames, order 2; constant-velocity RTS, q = 1e-4) -- they were chosen with the
    reference of the other records, which the procedure does not use.  All four are scored on the
    same test windows, and per record.

Usage: python src/revision_round5.py -> experiments/revision_round5_2026-09-26/results.json
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
import dpjait_noise_runner as v1  # noqa: E402
import refree_mstar_dev as R  # noqa: E402
from revision_round4 import rts_cv  # noqa: E402

OUT = ROOT / "experiments" / "revision_round5_2026-09-26"
TP, K = 400, 4
GRID = [int(m) for m in R.GRID]


def m3_from(Pws):
    sig = float(np.sqrt(np.mean([R.sigma_white(P) ** 2 for P in Pws])))
    Ls = np.mean([[v1.polyline(P[v1.instants(0, TP, m)]) for m in GRID] for P in Pws], axis=0)
    Is = np.mean([[R.inflation_exact(P, m, sig) for m in GRID] for P in Pws], axis=0)
    return R.m3_predict(Ls, Is, 6, 100)[0]


def windows_of(P, wins, n):
    out = {}
    for w in wins:
        f0 = w * TP
        if f0 + TP + 1 <= min(n, len(P)):
            Pw = R.fill_gaps(P[f0:f0 + TP + 1])
            if Pw is not None:
                out[w] = (Pw, P[f0:f0 + TP + 1])
    return out


def err_at(Praw, m, ref):
    """polyline through the sampled instants that are present (no interpolation of missing ones)."""
    pts = Praw[v1.instants(0, TP, m)] if m < TP + 1 else Praw
    ok = ~np.isnan(pts).any(axis=1)
    return abs(v1.polyline(pts[ok]) - ref) if ok.sum() >= 2 else np.nan


# ---------------------------------------------------------------- b1
def b1():
    exp = ROOT / "experiments" / "refree_mstar_prereg_v0.1"
    cells = []
    for name in ("R04_D2", "R14_D3"):
        sc = json.loads((exp / "scored" / (name + ".json")).read_text())
        dz = np.load(exp / "predictions" / (name + "_dense.npz"))
        wins_by = collections.defaultdict(set)
        refs = {}
        for q in sc["rows"]:
            wins_by[q["drone"]].add(q["window"])
            refs[(q["drone"], q["window"])] = q["reference_m"]
        pilot = {dr: set(sorted(ws)[: max(1, len(ws) // 3)]) for dr, ws in wins_by.items()}
        pilot_frames_windows = set().union(*pilot.values())
        test_windows = set().union(*[wins_by[d] - pilot[d] for d in wins_by])
        for li, px in enumerate([0, 4, 8, 16, 32, 64]):
            Pws, errs = [], collections.defaultdict(list)
            for dr in wins_by:
                for rep in (range(4) if px else [0]):
                    P = dz["A|%d|%d|%s" % (li, rep, dr)].astype(float)
                    for w, (Pw, _) in windows_of(P, pilot[dr], len(P)).items():
                        Pws.append(Pw)
                    for w in wins_by[dr] - pilot[dr]:
                        f0 = w * TP
                        if f0 + TP + 1 > len(P) or not np.isfinite(refs[(dr, w)]):
                            continue
                        Praw = P[f0:f0 + TP + 1]
                        for m in GRID[:-1] + [TP + 1]:
                            e = err_at(Praw, m, refs[(dr, w)])
                            if np.isfinite(e):
                                errs[m].append(e)
            pred = m3_from(Pws)
            status = "success" if np.isfinite(pred) else "fallback"
            m_use = min(GRID[:-1], key=lambda m: abs(np.log(m / pred))) if status == "success" else TP + 1
            mae = {m: 1000 * float(np.mean(v)) for m, v in errs.items()}
            m_best = min(mae, key=mae.get)
            n_p, n_t = len(pilot_frames_windows), len(test_windows)
            base = n_t * K * (TP + 1)
            plan = n_p * K * (TP + 1) + n_t * K * m_use
            cells.append(dict(record=name, noise_px=px, status=status, m_pred=None if status == "fallback" else round(pred, 1),
                              m_used=m_use, m_best=m_best, mae_used_mm=round(mae[m_use], 1), mae_best_mm=round(mae[m_best], 1),
                              mae_every_frame_mm=round(mae[TP + 1], 1), error_ratio_to_best=round(mae[m_use] / mae[m_best], 2),
                              within_1p5_of_best_grid=bool(2 / 3 <= m_use / m_best <= 1.5),
                              pilot_windows=n_p, measured_windows=n_t, frames_baseline=base, frames_plan=plan,
                              cost_ratio=round(base / plan, 2)))
            print(cells[-1])
    return cells


# ---------------------------------------------------------------- b2
def b2():
    exp = ROOT / "experiments" / "dpjait_noise_prereg_v0.2"
    per_rec = {}
    for name in v1.REAL:
        sc = json.loads((exp / "scored" / (name + ".json")).read_text())
        P = np.load(exp / "predictions" / (name + "_dense.npz"))["A|0|0|drone"].astype(float)
        refs = {q["window"]: q["reference_m"] for q in sc["rows"]}
        wins = sorted(w for w in refs if np.isfinite(refs[w]))
        pil = wins[: max(1, len(wins) // 3)]
        test = wins[len(pil):]
        Pws = [a for a, _ in windows_of(P, pil, len(P)).values()]
        pred = m3_from(Pws)
        m_use = min(GRID[:-1], key=lambda m: abs(np.log(m / pred))) if np.isfinite(pred) else TP + 1
        e = collections.defaultdict(list)
        for w, (Pf, Praw) in windows_of(P, test, len(P)).items():
            ref = refs[w]
            e["procedure"].append(err_at(Praw, m_use, ref))
            e["m24_ref_tuned"].append(err_at(Praw, 24, ref))
            e["every_frame"].append(err_at(Praw, TP + 1, ref))
            e["savgol61"].append(abs(v1.polyline(savgol_filter(Pf, 61, 2, axis=0)) - ref))
            d2 = Pf[2:] - 2 * Pf[1:-1] + Pf[:-2]
            r = float(np.mean(np.var(d2, axis=0)) / 6.0)
            e["rts_1e-4"].append(abs(v1.polyline(rts_cv(Praw.copy(), 1 / 25.0, 1e-4, max(r, 1e-6))) - ref))
        per_rec[name] = dict(m_pred=None if not np.isfinite(pred) else round(pred, 1), m_used=m_use, n_test=len(e["procedure"]),
                             **{k: round(1000 * float(np.nanmean(v)), 1) for k, v in e.items()})
        print(name, per_rec[name])
    keys = ["procedure", "m24_ref_tuned", "savgol61", "rts_1e-4", "every_frame"]
    per_rec = {r: v for r, v in per_rec.items() if v["n_test"] > 0 and all(k in v for k in keys)}
    pooled = {k: round(float(np.average([per_rec[r][k] for r in per_rec], weights=[per_rec[r]["n_test"] for r in per_rec])), 1) for k in keys}
    wins = {k: sum(per_rec[r]["procedure"] < per_rec[r][k] for r in per_rec) for k in keys[1:]}
    return dict(per_record=per_rec, pooled_mae_mm=pooled, records_where_procedure_better=wins)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(b1_pilot_transfer=b1(), b2_same_split=b2())
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    c = res["b1_pilot_transfer"]
    print("b1 cost ratio range %.2f-%.2f; success %d/12; within x1.5 %d/12; fallback error ratios %s" % (
        min(x["cost_ratio"] for x in c), max(x["cost_ratio"] for x in c), sum(x["status"] == "success" for x in c),
        sum(x["within_1p5_of_best_grid"] for x in c), [x["error_ratio_to_best"] for x in c if x["status"] == "fallback"]))
    print("b2 pooled", res["b2_same_split"]["pooled_mae_mm"], "procedure better in", res["b2_same_split"]["records_where_procedure_better"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
