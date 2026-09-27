"""Pre-registration runner: the per-instant noise knob on DPJAIT
(protocol/DPJAIT_NOISE_PREREG_v0.1.md).

Five steps, in order:

    freeze   hash the protocol and the sources, snapshot them
    observe  raw video -> our own blob observations at scales 1, 2, 4, 8
    predict  identity binding, noise ladders, dense triangulation, path-length
             ESTIMATES for every window and m; sealed.  Never reads a reference
             path length.
    score    only now: reference path lengths, per-instant error scale sigma_a,
             reference curvature and speed
    decide   evaluate H1-H5

Two noise ladders act on the same observations:
  A  GUM Supplement 1 style: add isotropic Gaussian noise of known size to every
     bound 2-D observation (full-resolution pixels), 4 seeded replicates;
  B  image downsampling before detection (a real, weaker knob).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_external/dpjait_2026-09-19"
PROTOCOL = "protocol/DPJAIT_NOISE_PREREG_v0.1.md"
SOURCES = [PROTOCOL, "src/dpjait_noise_runner.py", "src/own_blob_observer.py",
           "src/dpjait_record.py", "src/dpjait_sim_record.py",
           "src/dpjait_prereg_runner.py"]

REAL = ["R05_D1", "R06_D1", "R07_D1", "R08_D1", "R09_D1", "R10_D1", "R11_D1", "R16_D1_A"]
SIM = ["S11_D4", "S12_D3", "S13_D3"]
PUBLISHED_MSTAR_4S = {"R05_D1": 6, "R06_D1": 4, "R07_D1": 4, "R08_D1": 6,
                      "R09_D1": 10, "R10_D1": 10, "R11_D1": 8, "R16_D1_A": 6}

CFG = dict(
    video_fps=25.0,
    gate_px=60.0,
    eligibility_proxy_px=75.0,
    t_primary=400,                     # frames = 16 s
    grid_primary=[4, 5, 6, 8, 10, 12, 14, 17, 20, 24, 28, 34, 40, 50, 60, 80, 100, 134, 200, 400],
    t_parity=100,                      # frames = 4 s, the DPJAIT v0.1 window
    grid_parity=[4, 6, 8, 10, 13, 17, 20, 25, 34, 50, 100],
    ladder_a_px=[0, 4, 8, 16, 32, 64],
    ladder_a_reps=4,
    ladder_b_scales=[1, 2, 4, 8],
    seed=20460925,
    band=[2.0 / 3.0, 1.5],             # H2, H3 ratio band
    h1_slope=[-0.70, -0.30],
    h1_min_levels=4,
    h2_min_fraction=0.70,
    h3_min_fraction=0.80,
    h4_tolerance=0.05,
    h5_min_records=6,
)


# ------------------------------------------------------------------ helpers
def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_new(p: Path, v) -> None:
    if p.exists():
        raise SystemExit("refusing to overwrite %s" % p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=1, default=float), encoding="utf-8")


def verify(out: Path) -> None:
    for s, h in json.loads((out / "freeze.json").read_text())["hashes"].items():
        if sha(ROOT / s) != h:
            raise SystemExit("frozen source changed: %s" % s)


def open_record(name: str):
    if name.startswith("S"):
        from dpjait_sim_record import SimRecord
        return SimRecord(DATA / "Simulated_Data" / name), True
    from dpjait_record import Record
    return Record(DATA / "Real_Data" / name), False


def ref_per_frame(sim: bool) -> int:
    return 1 if sim else 4


def instants(f0: int, T: int, m: int) -> np.ndarray:
    return np.unique(f0 + np.round(np.linspace(0, T, m)).astype(int))


def polyline(P: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def cam_matrices(rec):
    out = {}
    for c in rec.cam_ids:
        cam = rec.cameras[c]
        P = np.column_stack([np.asarray(cam["R"], float), np.asarray(cam["t_m"], float).reshape(3)])
        out[c] = (np.asarray(cam["K"], float), np.asarray(cam["D"], float), P)
    return out


def triangulate_dense(uv: np.ndarray, mats: list) -> np.ndarray:
    """uv: (n_frames, n_cams, 2) full-res px with NaN for missing -> (n_frames, 3)."""
    n, k, _ = uv.shape
    norm = np.full_like(uv, np.nan)
    for j, (K, D, _) in enumerate(mats):
        ok = ~np.isnan(uv[:, j, 0])
        if ok.any():
            pts = cv2.undistortPoints(uv[ok, j, :].reshape(-1, 1, 2).astype(np.float64), K, D)
            norm[ok, j, :] = pts.reshape(-1, 2)
    Ps = np.stack([m[2] for m in mats])
    X = np.full((n, 3), np.nan)
    for i in range(n):
        use = np.nonzero(~np.isnan(norm[i, :, 0]))[0]
        if len(use) < 2:
            continue
        rows = []
        for j in use:
            x, y = norm[i, j]
            P = Ps[j]
            rows.append(x * P[2] - P[0])
            rows.append(y * P[2] - P[1])
        _, s, vt = np.linalg.svd(np.asarray(rows))
        h = vt[-1]
        if abs(h[3]) < 1e-12:
            continue
        p = h[:3] / h[3]
        if not np.all(np.isfinite(p)):
            continue
        if any((Ps[j] @ np.r_[p, 1.0])[2] <= 0 for j in use):
            continue
        X[i] = p
    return X


def curvature_speed(seg: np.ndarray, dt: float):
    seg = seg[~np.isnan(seg).any(axis=1)]
    if len(seg) < 7:
        return np.nan, np.nan
    ker = np.ones(5) / 5.0
    S = np.stack([np.convolve(seg[:, i], ker, mode="same") for i in range(3)], axis=1)
    d1 = np.gradient(S, dt, axis=0)
    d2 = np.gradient(d1, dt, axis=0)
    den = np.linalg.norm(d1, axis=1) ** 3
    ok = den > 1e-9
    k = np.linalg.norm(np.cross(d1, d2), axis=1)[ok] / den[ok]
    k = k[2:-2] if len(k) > 4 else k
    L = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
    return float(np.median(k)), L / (dt * (len(seg) - 1))


# -------------------------------------------------------------- eligibility
def eligible_windows(rec, T: int) -> list:
    """Non-overlapping windows of T frames; eligible iff every 4 s sub-window
    passes the reference-free DPJAIT v0.1 motion proxy (>= 75 px)."""
    from dpjait_prereg_runner import window_proxy_px
    sub = 100
    out = []
    w = 0
    while (w + 1) * T < rec.n_video_frames:
        f0 = w * T
        ok = True
        for q in range(T // sub):
            a = f0 + q * sub
            if not (window_proxy_px(rec, a, a + sub) >= CFG["eligibility_proxy_px"]):
                ok = False
                break
        if ok:
            out.append(w)
        w += 1
    return out


# ------------------------------------------------------------------- steps
def step_freeze(out: Path) -> None:
    if out.exists():
        raise SystemExit("refusing to re-freeze an existing run: %s" % out)
    out.mkdir(parents=True)
    save_new(out / "config.json", dict(CFG, real=REAL, sim=SIM, utc=utc()))
    save_new(out / "freeze.json", dict(hashes={s: sha(ROOT / s) for s in SOURCES}, utc=utc()))
    for s in SOURCES:
        dst = out / "source_snapshot" / s
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / s, dst)
    print("frozen %d sources under %s" % (len(SOURCES), out))


def step_observe(out: Path, name: str) -> None:
    verify(out)
    rec, sim = open_record(name)
    dst = out / "observations" / (name + ".npz")
    if dst.exists():
        raise SystemExit("refusing to overwrite %s" % dst)
    from own_blob_observer import observe_record
    meta = observe_record(rec.folder, sim, dst)
    print("%s observed in %.0f s" % (name, meta["seconds"]))


def bind(rec, obs, scale: int, n_frames: int, gate: float) -> np.ndarray:
    """-> (n_frames, n_drones, n_cams, 2) bound blob centroids, NaN if none.
    Identity from the projected reference POSITION (as in DPJAIT v0.1's
    annotated arm); position from our own blob."""
    from dpjait_record import project
    drones = sorted(rec.reference)
    out = np.full((n_frames, len(drones), len(rec.cam_ids), 2), np.nan)
    for d, dr in enumerate(drones):
        for f in range(n_frames):
            X = rec.reference_at(dr, f)
            if np.isnan(X).any():
                continue
            for j, c in enumerate(rec.cam_ids):
                b = obs[c][scale].get(f)
                if b is None or len(b) == 0:
                    continue
                uv, depth = project(rec.cameras[c], X)
                if depth[0] <= 0:
                    continue
                dist = np.linalg.norm(b[:, :2] - uv[0], axis=1)
                i = int(np.argmin(dist))
                if dist[i] <= gate:
                    out[f, d, j] = b[i, :2]
    return out


def runs():
    """(ladder, level_index, value, replicate) in a fixed order."""
    out = []
    for li, px in enumerate(CFG["ladder_a_px"]):
        for r in range(1 if px == 0 else CFG["ladder_a_reps"]):
            out.append(("A", li, px, r))
    for li, s in enumerate(CFG["ladder_b_scales"]):
        if s != 1:                       # scale 1 is ladder A level 0
            out.append(("B", li, s, 0))
    return out


def step_predict(out: Path, name: str) -> None:
    """ESTIMATES only.  Never computes a reference path length."""
    verify(out)
    from own_blob_observer import load
    rec, sim = open_record(name)
    meta, obs = load(out / "observations" / (name + ".npz"))
    n = min(min(meta["frames_read"].values()), rec.n_video_frames)
    mats_d = cam_matrices(rec)
    mats = [mats_d[c] for c in rec.cam_ids]
    drones = sorted(rec.reference)
    rec_idx = (REAL + SIM).index(name)

    bound = {s: bind(rec, obs, s, n, CFG["gate_px"]) for s in CFG["ladder_b_scales"]}
    Tp, Tq = CFG["t_primary"], CFG["t_parity"]
    w16 = [w for w in eligible_windows(rec, Tp) if (w + 1) * Tp < n]
    w4 = [w for w in eligible_windows(rec, Tq) if (w + 1) * Tq < n] if not sim else []

    rows, dense = [], {}
    for ladder, li, val, rep in runs():
        scale = 1 if ladder == "A" else val
        uv = bound[scale].copy()
        if ladder == "A" and val > 0:
            rng = np.random.default_rng([CFG["seed"], rec_idx, li, rep])
            noise = rng.normal(0.0, float(val), size=uv.shape)
            uv = uv + noise                       # NaN stays NaN
        for d, dr in enumerate(drones):
            P = triangulate_dense(uv[:, d], mats)
            dense["%s|%d|%d|%s" % (ladder, li, rep, dr)] = P.astype(np.float32)
            specs = [(Tp, CFG["grid_primary"], w16)]
            if ladder == "A" and val == 0 and not sim:
                specs.append((Tq, CFG["grid_parity"], w4))
            for T, grid, wins in specs:
                for w in wins:
                    f0 = w * T
                    for m in grid:
                        idx = instants(f0, T, m)
                        pts = P[idx]
                        ok = ~np.isnan(pts).any(axis=1)
                        failed = (not ok[0]) or (not ok[-1]) or ok.sum() < 2
                        rows.append(dict(ladder=ladder, level=li, value=val, rep=rep,
                                         drone=dr, T=T, window=w, m=m,
                                         n_used=int(ok.sum()), failed=bool(failed),
                                         est=None if failed else polyline(pts[ok])))
    pred = dict(record=name, sim=sim, n_frames=n, drones=drones, cameras=rec.cam_ids,
                windows_primary=w16, windows_parity=w4, rows=rows, utc=utc())
    path = out / "predictions" / (name + ".json")
    save_new(path, pred)
    dpath = out / "predictions" / (name + "_dense.npz")
    if dpath.exists():
        raise SystemExit("refusing to overwrite %s" % dpath)
    np.savez_compressed(dpath, **dense)
    seal_p = out / "seal.json"
    seals = json.loads(seal_p.read_text()) if seal_p.exists() else {}
    seals[name] = dict(sha256=sha(path), dense_sha256=sha(dpath), utc=utc(), n_rows=len(rows))
    seal_p.write_text(json.dumps(seals, indent=1), encoding="utf-8")
    print("%s: %d rows sealed (%d primary windows, %d parity windows)"
          % (name, len(rows), len(w16), len(w4)))


def step_score(out: Path, name: str) -> None:
    """Only here are reference path lengths and reference positions compared."""
    rec, sim = open_record(name)
    path = out / "predictions" / (name + ".json")
    dpath = out / "predictions" / (name + "_dense.npz")
    seal = json.loads((out / "seal.json").read_text())[name]
    if sha(path) != seal["sha256"] or sha(dpath) != seal["dense_sha256"]:
        raise SystemExit("prediction files changed after sealing: %s" % name)
    pred = json.loads(path.read_text())
    dense = np.load(dpath)
    per = ref_per_frame(sim)
    dt = 1.0 / (CFG["video_fps"] * per)

    refs, kin = {}, {}
    for r in pred["rows"]:
        key = (r["drone"], r["T"], r["window"])
        if key not in refs:
            f0, f1 = r["window"] * r["T"], (r["window"] + 1) * r["T"]
            refs[key] = rec.reference_path_length(r["drone"], f0, f1)
            seg = rec.reference[r["drone"]][f0 * per: f1 * per + 1]
            kin[key] = curvature_speed(np.asarray(seg, float), dt)
        ref = refs[key]
        r["reference_m"] = ref
        r["err"] = (r["est"] - ref) if (r["est"] is not None and np.isfinite(ref)) else None

    # sigma_a per run: per-axis spread of dense 3-D error about its per-window mean
    sig = {}
    Tp = CFG["t_primary"]
    for key in dense.files:
        ladder, li, rep, dr = key.split("|")
        P = dense[key].astype(float)
        res = []
        for w in pred["windows_primary"]:
            f0, f1 = w * Tp, (w + 1) * Tp
            ref = np.asarray([rec.reference_at(dr, f) for f in range(f0, f1)])
            e = P[f0:f1] - ref
            e = e[~np.isnan(e).any(axis=1)]
            if len(e) > 10:
                res.append(e - e.mean(axis=0))
        if res:
            R = np.concatenate(res)
            sig.setdefault("%s|%s" % (ladder, li), []).append(float(np.sqrt((R ** 2).mean())))
    pred["sigma_a_m"] = {k: float(np.mean(v)) for k, v in sig.items()}
    pred["kinematics"] = [dict(drone=k[0], T=k[1], window=k[2], kappa=v[0], speed=v[1])
                          for k, v in kin.items()]
    save_new(out / "scored" / (name + ".json"), pred)
    print("%s scored: sigma_a (mm) %s" % (name, {k: round(v * 1000, 2)
                                                for k, v in sorted(pred["sigma_a_m"].items())}))


# ------------------------------------------------------------------ decide
def _curve(rows, stat):
    f = np.mean if stat == "mean" else np.median
    by = {}
    for r in rows:
        if r["err"] is not None:
            by.setdefault(r["m"], []).append(r["err"])
    abs_ = {m: float(f(np.abs(v))) for m, v in by.items()}
    sgn = {m: float(f(v)) for m, v in by.items()}
    return abs_, sgn


def _crossing(sgn):
    ms = sorted(sgn)
    for a, b in zip(ms, ms[1:]):
        if sgn[a] < 0 <= sgn[b]:
            w = -sgn[a] / (sgn[b] - sgn[a])
            return float(np.exp(np.log(a) + w * (np.log(b) - np.log(a))))
    return None


def step_decide(out: Path) -> None:
    cfg = json.loads((out / "config.json").read_text())
    scored = {p.stem: json.loads(p.read_text()) for p in sorted((out / "scored").glob("*.json"))}
    if not scored:
        raise SystemExit("nothing scored")
    fams = {"real": [n for n in REAL if n in scored]}
    for s in SIM:
        if s in scored:
            fams[s] = [s]
    Tp = cfg["t_primary"]
    Tsec = Tp / cfg["video_fps"]
    grid = cfg["grid_primary"]
    lo_b, hi_b = cfg["band"]
    res = {"families": {}}
    cells = []
    for fam, names in fams.items():
        if not names:
            continue
        rows = [r for n in names for r in scored[n]["rows"] if r["T"] == Tp]
        kin = [k for n in names for k in scored[n]["kinematics"] if k["T"] == Tp
               and np.isfinite(k["kappa"]) and np.isfinite(k["speed"])]
        kap = float(np.median([k["kappa"] for k in kin])) if kin else float("nan")
        spd = float(np.median([k["speed"] for k in kin])) if kin else float("nan")
        levels = []
        for ladder, values in (("A", cfg["ladder_a_px"]), ("B", cfg["ladder_b_scales"])):
            for li, val in enumerate(values):
                if ladder == "B" and val == 1:
                    src_ladder, src_li = "A", 0
                else:
                    src_ladder, src_li = ladder, li
                sub = [r for r in rows if r["ladder"] == src_ladder and r["level"] == src_li]
                if not sub:
                    continue
                s_vals = [scored[n]["sigma_a_m"].get("%s|%d" % (src_ladder, src_li)) for n in names]
                s_vals = [v for v in s_vals if v]
                sigma = float(np.mean(s_vals)) if s_vals else float("nan")
                a_mean, s_mean = _curve(sub, "mean")
                a_med, s_med = _curve(sub, "median")
                mstar = min(a_mean, key=a_mean.get)
                mstar_med = min(a_med, key=a_med.get)
                cross = _crossing(s_mean)
                pred = Tsec * np.sqrt(kap * spd ** 2 / (7.0 * sigma)) if sigma > 0 else float("nan")
                censored = mstar in (grid[0], grid[-1])
                lv = dict(ladder=ladder, value=val, sigma_a_mm=sigma * 1000, m_star=mstar,
                          m_star_median=mstar_med, crossing=cross, predicted=float(pred),
                          censored=bool(censored),
                          fail_rate=float(np.mean([r["failed"] for r in sub])))
                levels.append(lv)
                if ladder == "A":
                    cells.append(dict(family=fam, **lv))
        a_lv = [l for l in levels if l["ladder"] == "A" and not l["censored"]
                and np.isfinite(l["sigma_a_mm"])]
        slope = None
        if len(a_lv) >= cfg["h1_min_levels"]:
            x = np.log([l["sigma_a_mm"] for l in a_lv])
            y = np.log([l["m_star"] for l in a_lv])
            slope = float(np.polyfit(x, y, 1)[0])
        res["families"][fam] = dict(records=names, kappa_median=kap, speed_median=spd,
                                    levels=levels, h1_slope=slope)

    # H1
    s_lo, s_hi = cfg["h1_slope"]
    inr = lambda s: s is not None and s_lo <= s <= s_hi  # noqa: E731
    sims = [f for f in res["families"] if f != "real"]
    res["H1"] = dict(real=res["families"].get("real", {}).get("h1_slope"),
                     sim={f: res["families"][f]["h1_slope"] for f in sims},
                     pass_=bool(inr(res["families"].get("real", {}).get("h1_slope"))
                                and sum(inr(res["families"][f]["h1_slope"]) for f in sims) >= 2))
    # H2, H3
    ok2 = [lo_b <= c["predicted"] / c["m_star"] <= hi_b for c in cells if np.isfinite(c["predicted"])]
    with_cross = [c for c in cells if c["crossing"]]
    ok3 = [lo_b <= c["crossing"] / c["m_star"] <= hi_b for c in with_cross]
    res["H2"] = dict(cells=len(ok2), within=int(sum(ok2)),
                     pass_=bool(ok2) and np.mean(ok2) >= cfg["h2_min_fraction"])
    res["H3"] = dict(cells=len(ok3), within=int(sum(ok3)), no_crossing=len(cells) - len(with_cross),
                     evaluable=len(with_cross) >= len(cells) / 2,
                     pass_=bool(ok3) and len(with_cross) >= len(cells) / 2
                     and np.mean(ok3) >= cfg["h3_min_fraction"])
    # H4 (direction taken from the development record): for this chain, image
    # downsampling does NOT raise sigma_a -- it merges the fragments of a large,
    # thin-armed target and steadies its centroid.
    real_b = sorted([l for l in res["families"].get("real", {}).get("levels", []) if l["ladder"] == "B"],
                    key=lambda l: l["value"])
    sb = [l["sigma_a_mm"] for l in real_b]
    tol = cfg["h4_tolerance"]
    res["H4"] = dict(sigma_a_mm_by_scale=dict(zip([l["value"] for l in real_b], sb)),
                     pass_=bool(len(sb) == len(cfg["ladder_b_scales"]) and sb[-1] <= sb[0] * (1 + tol)))
    # H5: two observation chains on the same eligible 4 s windows.  Ours is the
    # noisier one on the development record, so the mechanism predicts its m*
    # is not above the published-box chain's m* (DPJAIT v0.1 confirmation).
    detail = {}
    for n in fams.get("real", []):
        rows = [r for r in scored[n]["rows"] if r["T"] == cfg["t_parity"] and r["ladder"] == "A"
                and r["level"] == 0]
        if not rows:
            continue
        a_mean, _ = _curve(rows, "mean")
        ours = min(a_mean, key=a_mean.get)
        detail[n] = dict(ours=ours, published=PUBLISHED_MSTAR_4S[n],
                         not_above=ours <= PUBLISHED_MSTAR_4S[n],
                         sigma_a_mm_ours=scored[n]["sigma_a_m"].get("A|0", float("nan")) * 1000)
    n_ok = sum(1 for v in detail.values() if v["not_above"])
    res["H5"] = dict(detail=detail, within=n_ok, pass_=n_ok >= cfg["h5_min_records"])
    save_new(out / "decision.json", dict(utc=utc(), result=res))
    for h in ("H1", "H2", "H3", "H4", "H5"):
        print(h, {k: v for k, v in res[h].items() if k in ("pass_", "real", "sim", "cells", "within",
                                                             "sigma_a_mm_by_scale", "evaluable")})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["freeze", "observe", "predict", "score", "decide"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    if a.step == "freeze":
        step_freeze(out)
    elif a.step == "observe":
        step_observe(out, a.record)
    elif a.step == "predict":
        step_predict(out, a.record)
    elif a.step == "score":
        step_score(out, a.record)
    else:
        step_decide(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
