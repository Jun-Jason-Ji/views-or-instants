"""M4: reference-free m* predictor with a split-camera noise estimate.

The cameras of a record are split into two disjoint halves (every balanced
split for 4 cameras, three fixed splits otherwise); each half is triangulated
on its own.  The difference of the two half-reconstructions contains only
observation error, white and slowly varying alike, because the true motion is
common to both and the per-camera detection errors are independent.  For the
full reconstruction the error-difference variance at lag l is estimated as
      s^2(l) = k * Var[Delta(t+l) - Delta(t)]   (per axis), k = tr Cov(full)/(tr Cov(A)+tr Cov(B))
      from the calibration Jacobians at the reconstructed positions (a plain /4 split-half
      rule over-estimates 2-3x because two-camera geometry is much weaker),
and the expected inflation of a segment with chord mu is the mean of a
noncentral chi (3 dof) with scale s(l).  The deficit coefficient is then fitted
exactly as in M3.  Only the observed 2-D detections and calibration are used
for the prediction; the reference enters only through identity binding
(inherited from the frozen chain, disclosed) and through the scored targets.

Usage: python src/refree_mstar_m4.py [--chains v0.1_blob v0.2_yolo] [--records ...]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpjait_noise_runner as v1  # noqa: E402
import refree_mstar_dev as R  # noqa: E402
from own_blob_observer import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {"v0.1_blob": 20460925, "v0.2_yolo": 20460926}
OUT = ROOT / "experiments" / "refree_mstar_dev_2026-09-26" / "m4_cache"


def splits(n_cams):
    idx = list(range(n_cams))
    if n_cams == 4:
        return [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    h = n_cams // 2
    out = []
    for comb in itertools.combinations(idx, h):
        rest = tuple(i for i in idx if i not in comb)
        if (rest, comb) not in out:
            out.append((comb, rest))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(out), size=min(3, len(out)), replace=False)
    return [out[i] for i in sorted(pick)]


def process_record(chain, name, levels):
    exp = R.EXPS[chain]
    rec, sim = v1.open_record(name)
    meta, obs = load(exp / "observations" / (name + ".npz"))
    n = min(min(meta["frames_read"].values()), rec.n_video_frames)
    mats = [v1.cam_matrices(rec)[c] for c in rec.cam_ids]
    drones = sorted(rec.reference)
    rec_idx = (v1.REAL + v1.SIM).index(name)
    bound = v1.bind(rec, obs, 1, n, 60.0)
    sc = json.loads((exp / "scored" / (name + ".json")).read_text())
    dz = np.load(exp / "predictions" / (name + "_dense.npz"))
    px_levels = [0, 4, 8, 16, 32, 64]
    res = {}
    for li in levels:
        uv = bound.copy()
        if px_levels[li] > 0:
            uv = uv + np.random.default_rng([SEEDS[chain], rec_idx, li, 0]).normal(0.0, float(px_levels[li]), uv.shape)
        for d, dr in enumerate(drones):
            wins = sorted({r["window"] for r in sc["rows"] if r["drone"] == dr})
            frames = np.unique(np.concatenate([np.arange(w * R.TP, w * R.TP + R.TP + 1) for w in wins]))
            frames = frames[frames < n]
            # fidelity check against the sealed 4-camera reconstruction (first window only)
            f0 = wins[0] * R.TP
            chk = v1.triangulate_dense(uv[f0:f0 + 50, d], mats)
            sealed = dz["A|%d|0|%s" % (li, dr)][f0:f0 + 50].astype(float)
            diff = np.nanmax(np.abs(chk - sealed)) if np.isfinite(chk).any() else np.nan
            halves = []
            for a, b in splits(len(mats)):
                XA = np.full((n, 3), np.nan); XB = np.full((n, 3), np.nan)
                XA[frames] = v1.triangulate_dense(uv[frames][:, d][:, list(a)], [mats[i] for i in a])
                XB[frames] = v1.triangulate_dense(uv[frames][:, d][:, list(b)], [mats[i] for i in b])
                halves.append(XA - XB)
            res["%d|%s" % (li, dr)] = dict(windows=wins, fidelity_max_abs_m=float(diff),
                                           delta=[h[w * R.TP: w * R.TP + R.TP + 1].tolist() for h in halves for w in wins])
    return res


def s2_at_lags(deltas, lags):
    """deltas: list of (TP+1,3) arrays -> per-axis error-difference variance of the
    full reconstruction at each lag (split-half rule), NaN-robust."""
    out = []
    for l in lags:
        acc = []
        for D, kf in deltas:
            if D.ndim != 2 or len(D) <= l + 5 or not np.isfinite(kf):
                continue
            x = D[l:] - D[:-l]
            x = x[~np.isnan(x).any(axis=1)]
            if len(x) > 5:
                acc.append(kf * np.mean(x ** 2))
        out.append(np.mean(acc) if acc else np.nan)
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", nargs="+", default=list(R.EXPS))
    ap.add_argument("--records", nargs="+", default=v1.REAL + v1.SIM)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for chain in a.chains:
        for name in a.records:
            p = OUT / ("%s_%s.json" % (chain, name))
            if p.exists():
                continue
            r = process_record(chain, name, range(6))
            p.write_text(json.dumps(r))
            print(chain, name, "fidelity(max |dX| m):",
                  {k: round(v["fidelity_max_abs_m"], 6) for k, v in r.items()}, flush=True)
    return 0


def _run_eval():
    rows = evaluate()
    (OUT.parent / "dev_m4.json").write_text(json.dumps(rows, indent=1))
    for k in ("eq1_refbased", "M4"):
        r = np.array([q[k] / q["m_star"] for q in rows], float)
        for sub, msk in (("all", np.ones(len(rows), bool)), ("real", np.array([q["family"] == "real" for q in rows])),
                         ("native", np.array([q["level"] == 0 for q in rows]))):
            rr = r[msk & np.isfinite(r)]
            print("  %-13s %-6s %2d/%2d in x1.5  median %.2f  rms-log %.2f" % (
                k, sub, np.sum((rr >= 2 / 3) & (rr <= 1.5)), msk.sum(), np.median(rr), np.sqrt(np.mean(np.log(rr) ** 2))))




# ---------------------------------------------------------------- geometry factor
def cam_jacobian(K, P34, X):
    """d(u,v)/dX for a pinhole camera (distortion ignored), X in world metres."""
    R_, t = P34[:, :3], P34[:, 3]
    xc = R_ @ X + t
    z = xc[2]
    fx, fy = K[0, 0], K[1, 1]
    Jc = np.array([[fx / z, 0, -fx * xc[0] / z ** 2], [0, fy / z, -fy * xc[1] / z ** 2]])
    return Jc @ R_


def geometry_factor(mats, Xs, a, b):
    """k = tr Cov(full) / (tr Cov(A) + tr Cov(B)) for equal isotropic pixel noise,
    averaged over reconstructed positions Xs (reference-free)."""
    ks = []
    for X in Xs:
        if not np.all(np.isfinite(X)):
            continue
        Js = [cam_jacobian(K, P34, X) for (K, _, P34) in mats]
        info = lambda idx: sum(Js[i].T @ Js[i] for i in idx)  # noqa: E731
        try:
            tf = np.trace(np.linalg.inv(info(range(len(mats)))))
            ta = np.trace(np.linalg.inv(info(a)))
            tb = np.trace(np.linalg.inv(info(b)))
        except np.linalg.LinAlgError:
            continue
        ks.append(tf / (ta + tb))
    return float(np.median(ks)) if ks else float("nan")


# ---------------------------------------------------------------- evaluation
def inflation_lagged(P, m, s2_of_lag):
    ii = v1.instants(0, R.TP, int(m))
    x = P[ii]
    d_obs = np.linalg.norm(np.diff(x, axis=0), axis=1)
    s2 = np.array([s2_of_lag(int(l)) for l in np.diff(ii)])
    mu = np.sqrt(np.maximum(d_obs ** 2 - 3 * s2, 0.01 * s2))
    return float(np.sum(R.mean_norm(mu, np.sqrt(s2)) - mu))


def evaluate(fit_lo=6, fit_hi=100):
    rows = []
    for chain, exp in R.EXPS.items():
        cfg = json.loads((exp / "config.json").read_text())
        dec = json.loads((exp / "decision.json").read_text())["result"]["families"]
        for fam, names in R.FAMS.items():
            lv = [l for l in dec[fam]["levels"] if l.get("ladder", "A") == "A"]
            for li, l in enumerate(lv):
                deltas, Pw_all = [], []
                for n in names:
                    cache = json.loads((OUT / ("%s_%s.json" % (chain, n))).read_text())
                    sc = json.loads((exp / "scored" / (n + ".json")).read_text())
                    dz = np.load(exp / "predictions" / (n + "_dense.npz"))
                    rec, _ = v1.open_record(n)
                    mats = [v1.cam_matrices(rec)[c] for c in rec.cam_ids]
                    sp = splits(len(mats))
                    for key, v in cache.items():
                        if int(key.split("|")[0]) != li:
                            continue
                        dr = key.split("|", 1)[1]
                        Pn = dz["A|0|0|%s" % dr].astype(float)
                        Xs = np.concatenate([Pn[w * R.TP: w * R.TP + R.TP + 1: 20] for w in v["windows"]])
                        nw = len(v["windows"])
                        for j, (ca, cb) in enumerate(sp):
                            kf = geometry_factor(mats, Xs, ca, cb)
                            deltas += [(np.asarray(D, float), kf) for D in v["delta"][j * nw:(j + 1) * nw]]
                    for dr in sorted({r["drone"] for r in sc["rows"]}):
                        for rep in (range(cfg["ladder_a_reps"]) if li else [0]):
                            P = dz["A|%d|%d|%s" % (li, rep, dr)].astype(float)
                            for w in sorted({r["window"] for r in sc["rows"] if r["drone"] == dr}):
                                f0 = w * R.TP
                                if f0 + R.TP + 1 > len(P):
                                    continue
                                Pw = R.fill_gaps(P[f0:f0 + R.TP + 1])
                                if Pw is not None:
                                    Pw_all.append(Pw)
                lags = np.arange(1, R.TP + 1)
                s2 = s2_at_lags(deltas, lags)
                s2f = lambda l: s2[min(max(l, 1), R.TP) - 1]  # noqa: E731
                Ls = np.mean([[v1.polyline(Pw[v1.instants(0, R.TP, int(m))]) for m in R.GRID] for Pw in Pw_all], axis=0)
                Is = np.mean([[inflation_lagged(Pw, m, s2f) for m in R.GRID] for Pw in Pw_all], axis=0)
                pred, A = R.m3_predict(Ls, Is, fit_lo, fit_hi)
                eq1 = l.get("pred_uncorrected", l.get("predicted"))
                rows.append(dict(chain=chain, family=fam, level=li, m_star=l["m_star"], crossing=l["crossing"],
                                 eq1_refbased=eq1, M4=pred, A=A, s_lag1_mm=float(np.sqrt(s2[0]) * 1e3),
                                 s_lag20_mm=float(np.sqrt(s2[19]) * 1e3), n_windows=len(Pw_all)))
                print("%-9s %-7s L%d m*=%3d cross=%6.1f eq1=%5.1f | M4=%6.1f | s(1)=%6.1f s(20)=%6.1f mm" % (
                    chain, fam, li, l["m_star"], l["crossing"] or np.nan, eq1, pred, rows[-1]["s_lag1_mm"], rows[-1]["s_lag20_mm"]), flush=True)
    return rows


if __name__ == "__main__":
    if "--evaluate" in sys.argv:
        _run_eval()
    else:
        raise SystemExit(main())
