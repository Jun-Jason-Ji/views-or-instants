"""Exploratory slow-error diagnostic for the reference-free predictor (post hoc).

D = s(20) / s(1): the split-camera estimate of the error-difference standard
deviation at a lag of 20 frames (0.8 s) over that at one frame.  White error
gives D ~ 1; slowly varying error, which the M3 predictor cannot see, makes D
large.  The threshold D > 3 was set on the development cells (the one M3
failure there has D = 4.6, every other cell D <= 2.2) before this script was
run on the two confirmation records, whose outcomes were already known; the
check is therefore exploratory, not a test.  Rule: if D > 3, use the M4
(split-camera) prediction instead of M3.

Usage: python src/slow_error_diagnostic.py -> experiments/slow_error_diagnostic_2026-09-26/diagnostic.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpjait_noise_runner as v1  # noqa: E402
import refree_mstar_m4 as M4  # noqa: E402
from own_blob_observer import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "experiments" / "refree_mstar_prereg_v0.1"
OUT = ROOT / "experiments" / "slow_error_diagnostic_2026-09-26"
THRESH, LAG = 3.0, 20
PX, SEED = [0, 4, 8, 16, 32, 64], 20460927
RECS = ["R04_D2", "R14_D3"]


def dev_cells():
    rows = json.loads((ROOT / "experiments/refree_mstar_dev_2026-09-26/dev_m4.json").read_text())
    m3 = {(r["chain"], r["family"], r["level"]): r for r in
          json.loads((ROOT / "experiments/refree_mstar_dev_2026-09-26/dev_m3.json").read_text())}
    out = []
    for r in rows:
        k = (r["chain"], r["family"], r["level"])
        out.append(dict(set="dev", cell="%s|%s|L%d" % k, m_star=r["m_star"], M3=m3[k]["M3_6_100"], M4=r["M4"],
                        D=r["s_lag20_mm"] / r["s_lag1_mm"]))
    return out


def conf_cells():
    dec = json.loads((CONF / "decision.json").read_text())["result"]["cells"]
    out = []
    for name in RECS:
        rec, _ = v1.open_record(name)
        meta, obs = load(CONF / "observations" / (name + ".npz"))
        n = min(min(meta["frames_read"].values()), rec.n_video_frames)
        mats = [v1.cam_matrices(rec)[c] for c in rec.cam_ids]
        bound = v1.bind(rec, obs, 1, n, 60.0)
        pred = json.loads((CONF / "predictions" / (name + ".json")).read_text())
        wins = pred["windows"]
        Pn = np.load(CONF / "predictions" / (name + "_dense.npz"))
        for li, px in enumerate(PX):
            uv = bound.copy()
            if px:
                uv = uv + np.random.default_rng([SEED, RECS.index(name), li, 0]).normal(0.0, float(px), uv.shape)
            deltas = []
            for d, dr in enumerate(sorted(rec.reference)):
                frames = np.unique(np.concatenate([np.arange(w * 400, w * 400 + 401) for w in wins]))
                frames = frames[frames < n]
                P0 = Pn["A|0|0|%s" % dr].astype(float)
                Xs = np.concatenate([P0[w * 400: w * 400 + 401: 20] for w in wins])
                for a, b in M4.splits(len(mats)):
                    kf = M4.geometry_factor(mats, Xs, a, b)
                    XA = np.full((n, 3), np.nan); XB = np.full((n, 3), np.nan)
                    XA[frames] = v1.triangulate_dense(uv[frames][:, d][:, list(a)], [mats[i] for i in a])
                    XB[frames] = v1.triangulate_dense(uv[frames][:, d][:, list(b)], [mats[i] for i in b])
                    deltas += [((XA - XB)[w * 400: w * 400 + 401], kf) for w in wins if w * 400 + 401 <= n]
            s2 = M4.s2_at_lags(deltas, [1, LAG])
            c = [x for x in dec if x["record"] == name and x["level"] == li][0]
            out.append(dict(set="confirmation (outcome known)", cell="%s|L%d" % (name, li), m_star=c["m_star"],
                            M3=c["M3"], M4=c["M4"], D=float(np.sqrt(s2[1] / s2[0]))))
            print(out[-1], flush=True)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = dev_cells() + conf_cells()
    band = lambda x, m: bool(np.isfinite(x) and 2 / 3 <= x / m <= 1.5)  # noqa: E731
    for c in cells:
        c["flag"] = c["D"] > THRESH
        c["rule"] = c["M4"] if c["flag"] else c["M3"]
        c["in_M3"], c["in_rule"] = band(c["M3"], c["m_star"]), band(c["rule"], c["m_star"])
    summ = {}
    for s in ("dev", "confirmation (outcome known)"):
        cs = [c for c in cells if c["set"] == s]
        summ[s] = dict(cells=len(cs), flagged=sum(c["flag"] for c in cs), M3_within=sum(c["in_M3"] for c in cs),
                       rule_within=sum(c["in_rule"] for c in cs))
    (OUT / "diagnostic.json").write_text(json.dumps(dict(threshold=THRESH, lag_frames=LAG, summary=summ, cells=cells), indent=1))
    for c in cells:
        if c["flag"] or c["cell"].endswith("L0"):
            print("%-34s D=%.2f flag=%s m*=%3d M3=%6.1f M4=%6.1f -> rule %6.1f" % (
                c["cell"], c["D"], c["flag"], c["m_star"], c["M3"], c["M4"], c["rule"]))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
