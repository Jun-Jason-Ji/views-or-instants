"""Runner for protocol REFREE_MSTAR_PREREG_v0.1: prospective test of the
reference-free Goldilocks predictors M3 (primary) and M4 (secondary).

Steps (each refuses to overwrite its outputs):
  freeze  --out DIR                 hash protocol + sources + detector weights
  predict --out DIR --record NAME   dense triangulation, polyline estimates, and
                                    the M3/M4 predictions; sealed.  Never reads a
                                    reference path length or reference position,
                                    except identity binding inherited from the
                                    frozen chain (nearest detection to the projected
                                    reference, disclosed).
  score   --out DIR --record NAME   reads the reference: per-window errors, the
                                    reference-based sigma_a, kappa, v (baseline eq. 1)
  decide  --out DIR                 m*, crossing, H1-H3, regret
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpjait_noise_runner as v1  # noqa: E402
import refree_mstar_dev as R  # noqa: E402
import refree_mstar_m4 as M4  # noqa: E402
from own_blob_observer import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = dict(
    records=["R04_D2", "R14_D3"],
    t_primary=400, video_fps=25.0, gate_px=60.0,
    grid_primary=[int(m) for m in R.GRID],
    ladder_a_px=[0, 4, 8, 16, 32, 64], ladder_a_reps=4, seed=20460927,
    band=[2 / 3, 1.5], m3_fit=[6, 100],
    h1_min_cells=9, h3_min_cells=9,
)
if os.environ.get("REFREE_DRY_RECORDS"):        # dry run on development records only
    CFG["records"] = os.environ["REFREE_DRY_RECORDS"].split(",")
SOURCES = ["protocol/REFREE_MSTAR_PREREG_v0.1.md", "src/refree_prereg_runner.py", "src/refree_mstar_dev.py",
           "src/refree_mstar_m4.py", "src/dpjait_noise_runner.py", "src/learned_det_infer.py",
           "src/own_blob_observer.py", "src/dpjait_record.py", "src/dpjait_prereg_runner.py",
           "experiments/dpjait_learned_det/train/run/weights/best.pt"]


def verify(out: Path) -> None:
    fz = json.loads((out / "freeze.json").read_text())
    for rel, h in fz["hashes"].items():
        if v1.sha(ROOT / rel) != h:
            raise SystemExit("frozen source changed: %s" % rel)


def step_freeze(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    v1.save_new(out / "config.json", dict(CFG, utc=v1.utc()))
    v1.save_new(out / "freeze.json", dict(hashes={s: v1.sha(ROOT / s) for s in SOURCES}, utc=v1.utc()))
    print("frozen", out)


def window_arrays(P, wins, n):
    out = []
    for w in wins:
        f0 = w * R.TP
        if f0 + R.TP + 1 > n:
            continue
        Pw = R.fill_gaps(P[f0:f0 + R.TP + 1])
        if Pw is not None:
            out.append(Pw)
    return out


def predict_m3(Pws):
    sig = float(np.sqrt(np.mean([R.sigma_white(P) ** 2 for P in Pws])))
    Ls = np.mean([[v1.polyline(P[v1.instants(0, R.TP, int(m))]) for m in R.GRID] for P in Pws], axis=0)
    Is = np.mean([[R.inflation_exact(P, m, sig) for m in R.GRID] for P in Pws], axis=0)
    pred, A = R.m3_predict(Ls, Is, *CFG["m3_fit"])
    return dict(M3=pred, M3_A=float(A), sigma_white_mm=sig * 1e3), Ls


def predict_m4(Ls, Pws, deltas):
    lags = np.arange(1, R.TP + 1)
    s2 = M4.s2_at_lags(deltas, lags)
    s2f = lambda l: s2[min(max(l, 1), R.TP) - 1]  # noqa: E731
    Is = np.mean([[M4.inflation_lagged(P, m, s2f) for m in R.GRID] for P in Pws], axis=0)
    pred, A = R.m3_predict(Ls, Is, *CFG["m3_fit"])
    return dict(M4=pred, M4_A=float(A), s_lag1_mm=float(np.sqrt(s2[0]) * 1e3))


def step_predict(out: Path, name: str) -> None:
    verify(out)
    rec, sim = v1.open_record(name)
    meta, obs = load(out / "observations" / (name + ".npz"))
    n = min(min(meta["frames_read"].values()), rec.n_video_frames)
    mats = [v1.cam_matrices(rec)[c] for c in rec.cam_ids]
    drones = sorted(rec.reference)
    rec_idx = CFG["records"].index(name)
    bound = v1.bind(rec, obs, 1, n, CFG["gate_px"])
    wins = [w for w in v1.eligible_windows(rec, R.TP) if (w + 1) * R.TP < n]
    rows, dense, preds = [], {}, {}
    splits = M4.splits(len(mats))
    for li, px in enumerate(CFG["ladder_a_px"]):
        Pws, deltas = [], []
        for rep in range(1 if px == 0 else CFG["ladder_a_reps"]):
            uv = bound.copy()
            if px > 0:
                uv = uv + np.random.default_rng([CFG["seed"], rec_idx, li, rep]).normal(0.0, float(px), uv.shape)
            for d, dr in enumerate(drones):
                P = v1.triangulate_dense(uv[:, d], mats)
                dense["A|%d|%d|%s" % (li, rep, dr)] = P.astype(np.float32)
                for w in wins:
                    for m in CFG["grid_primary"]:
                        pts = P[v1.instants(w * R.TP, R.TP, m)]
                        ok = ~np.isnan(pts).any(axis=1)
                        failed = (not ok[0]) or (not ok[-1]) or ok.sum() < 2
                        rows.append(dict(level=li, value=px, rep=rep, drone=dr, window=w, m=m,
                                         n_used=int(ok.sum()), failed=bool(failed),
                                         est=None if failed else v1.polyline(pts[ok])))
                Pws += window_arrays(P, wins, n)
                if rep == 0:
                    frames = np.unique(np.concatenate([np.arange(w * R.TP, w * R.TP + R.TP + 1) for w in wins]))
                    frames = frames[frames < n]
                    Pn = dense["A|0|0|%s" % dr].astype(float)
                    Xs = np.concatenate([Pn[w * R.TP: w * R.TP + R.TP + 1: 20] for w in wins])
                    for a, b in splits:
                        kf = M4.geometry_factor(mats, Xs, a, b)
                        XA = np.full((n, 3), np.nan); XB = np.full((n, 3), np.nan)
                        XA[frames] = v1.triangulate_dense(uv[frames][:, d][:, list(a)], [mats[i] for i in a])
                        XB[frames] = v1.triangulate_dense(uv[frames][:, d][:, list(b)], [mats[i] for i in b])
                        deltas += [((XA - XB)[w * R.TP: w * R.TP + R.TP + 1], kf) for w in wins
                                   if w * R.TP + R.TP + 1 <= n]
        p3, Ls = predict_m3(Pws)
        p4 = predict_m4(Ls, Pws, deltas)
        preds[str(li)] = dict(n_target_windows=len(Pws), **p3, **p4)
        print("%s level %d: %s" % (name, li, {k: round(v, 2) for k, v in preds[str(li)].items()}), flush=True)
    pred = dict(record=name, drones=drones, cameras=rec.cam_ids, windows=wins, rows=rows,
                predictions=preds, n_frames=n, utc=v1.utc())
    path = out / "predictions" / (name + ".json")
    path.parent.mkdir(exist_ok=True)
    v1.save_new(path, pred)
    dpath = out / "predictions" / (name + "_dense.npz")
    if dpath.exists():
        raise SystemExit("refusing to overwrite %s" % dpath)
    np.savez_compressed(dpath, **dense)
    sp = out / "seal.json"
    seals = json.loads(sp.read_text()) if sp.exists() else {}
    seals[name] = dict(sha256=v1.sha(path), dense_sha256=v1.sha(dpath), utc=v1.utc())
    sp.write_text(json.dumps(seals, indent=1), encoding="utf-8")
    print("%s sealed: %d rows" % (name, len(rows)))


def step_score(out: Path, name: str) -> None:
    rec, sim = v1.open_record(name)
    path = out / "predictions" / (name + ".json")
    dpath = out / "predictions" / (name + "_dense.npz")
    seal = json.loads((out / "seal.json").read_text())[name]
    if v1.sha(path) != seal["sha256"] or v1.sha(dpath) != seal["dense_sha256"]:
        raise SystemExit("prediction files changed after sealing: %s" % name)
    pred = json.loads(path.read_text())
    dense = np.load(dpath)
    per = v1.ref_per_frame(sim)
    dt = 1.0 / (CFG["video_fps"] * per)
    refs, kin = {}, {}
    for r in pred["rows"]:
        key = (r["drone"], r["window"])
        if key not in refs:
            f0, f1 = r["window"] * R.TP, (r["window"] + 1) * R.TP
            refs[key] = rec.reference_path_length(r["drone"], f0, f1)
            kin[key] = v1.curvature_speed(np.asarray(rec.reference[r["drone"]][f0 * per: f1 * per + 1], float), dt)
        r["reference_m"] = refs[key]
        r["err"] = (r["est"] - refs[key]) if (r["est"] is not None and np.isfinite(refs[key])) else None
    sig = {}
    for key in dense.files:
        _, li, rep, dr = key.split("|")
        P = dense[key].astype(float)
        for w in pred["windows"]:
            f0, f1 = w * R.TP, (w + 1) * R.TP
            if f1 * per >= len(rec.reference[dr]):
                continue
            e = P[f0:f1] - np.asarray([rec.reference_at(dr, f) for f in range(f0, f1)])
            ok = ~np.isnan(e).any(axis=1)
            if ok.sum() > 10:
                c = e[ok] - e[ok].mean(axis=0)
                sig.setdefault(li, []).append(float(np.sqrt((c ** 2).mean())))
    pred["sigma_a_ref_m"] = {k: float(np.mean(v)) for k, v in sig.items()}
    pred["kinematics"] = [dict(drone=k[0], window=k[1], kappa=v[0], speed=v[1]) for k, v in kin.items()]
    (out / "scored").mkdir(exist_ok=True)
    v1.save_new(out / "scored" / (name + ".json"), pred)
    print("%s scored" % name)


def step_decide(out: Path) -> None:
    cfg = json.loads((out / "config.json").read_text())
    Ts = cfg["t_primary"] / cfg["video_fps"]
    lo, hi = cfg["band"]
    band = lambda x, m: bool(np.isfinite(x) and lo <= x / m <= hi)  # noqa: E731
    cells = []
    for name in cfg["records"]:
        sc = json.loads((out / "scored" / (name + ".json")).read_text())
        kin = [k for k in sc["kinematics"] if np.isfinite(k["kappa"]) and np.isfinite(k["speed"])]
        kap = float(np.median([k["kappa"] for k in kin])); spd = float(np.median([k["speed"] for k in kin]))
        for li, px in enumerate(cfg["ladder_a_px"]):
            sub = [r for r in sc["rows"] if r["level"] == li and r["err"] is not None]
            a_mean, s_mean = v1._curve(sub, "mean")
            mstar = min(a_mean, key=a_mean.get)
            sigma = sc["sigma_a_ref_m"][str(li)]
            eq1 = Ts * np.sqrt(kap * spd ** 2 / (7 * sigma))
            p = sc["predictions"][str(li)]
            ms = sorted(a_mean)

            def regret(x):
                if not np.isfinite(x):
                    return float("nan")
                near = min(ms, key=lambda m: abs(np.log(m / x)))
                return a_mean[near] / a_mean[mstar] - 1.0
            cells.append(dict(record=name, level=li, px=px, m_star=mstar, crossing=v1._crossing(s_mean),
                              M3=p["M3"], M4=p["M4"], eq1_refbased=float(eq1), sigma_a_ref_mm=sigma * 1e3,
                              sigma_white_mm=p["sigma_white_mm"], fail_rate=float(np.mean([r["failed"] for r in sc["rows"] if r["level"] == li])),
                              regret_M3=regret(p["M3"]), regret_M4=regret(p["M4"]), regret_eq1=regret(eq1),
                              in_M3=band(p["M3"], mstar), in_M4=band(p["M4"], mstar), in_eq1=band(eq1, mstar)))
    n3 = sum(c["in_M3"] for c in cells); n4 = sum(c["in_M4"] for c in cells); ne = sum(c["in_eq1"] for c in cells)
    res = dict(cells=cells,
               H1=dict(within=n3, of=len(cells), pass_=n3 >= cfg["h1_min_cells"]),
               H2=dict(M3_within=n3, eq1_refbased_within=ne, pass_=n3 >= ne),
               H3=dict(within=n4, of=len(cells), pass_=n4 >= cfg["h3_min_cells"]),
               regret_median=dict(M3=float(np.nanmedian([c["regret_M3"] for c in cells])),
                                  M4=float(np.nanmedian([c["regret_M4"] for c in cells])),
                                  eq1=float(np.nanmedian([c["regret_eq1"] for c in cells]))))
    v1.save_new(out / "decision.json", dict(utc=v1.utc(), result=res))
    for c in cells:
        print("%-7s L%d m*=%3d cross=%6.1f | M3=%6.1f M4=%6.1f eq1(ref)=%6.1f | regret M3 %.2f M4 %.2f eq1 %.2f" % (
            c["record"], c["level"], c["m_star"], c["crossing"] or np.nan, c["M3"], c["M4"], c["eq1_refbased"],
            c["regret_M3"], c["regret_M4"], c["regret_eq1"]))
    for h in ("H1", "H2", "H3", "regret_median"):
        print(h, res[h])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["freeze", "predict", "score", "decide"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--record")
    a = ap.parse_args()
    out = ROOT / a.out
    {"freeze": lambda: step_freeze(out), "predict": lambda: step_predict(out, a.record),
     "score": lambda: step_score(out, a.record), "decide": lambda: step_decide(out)}[a.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
