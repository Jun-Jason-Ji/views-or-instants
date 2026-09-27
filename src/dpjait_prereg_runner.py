"""Preregistration runner for DPJAIT (protocol/DPJAIT_PREREG_v0.1.md).

Four steps, in order, mirroring `lyon_prereg_runner.py`:

    freeze   hash the protocol and the sources, snapshot them
    predict  window eligibility + path-length ESTIMATES, sealed
    score    only now compute reference path lengths and attach errors
    decide   evaluate H1-H5

The separation between `predict` and `score` is the point of the whole file and
is enforced by construction: `predict` never calls `reference_path_length`, so
no estimate can have been influenced by a reference length.  The labelled arm
does read reference *positions*, but only to bind a detection box to a drone
(protocol section 2); that is identity, not the measurand, and it is the very
assumption the free arm exists to price.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dpjait_chain_compare as cc  # noqa: E402
from dpjait_identity import assign_annotated, assign_free, link_tracks, match_tracks_to_drones  # noqa: E402
from dpjait_record import VIDEO_FPS, Record  # noqa: E402
from mcalib_cache_observer import triangulate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = "protocol/DPJAIT_PREREG_v0.1.md"
SOURCES = [PROTOCOL, "src/dpjait_record.py", "src/dpjait_sim_record.py",
           "src/dpjait_identity.py", "src/dpjait_chain_compare.py",
           "src/dpjait_prereg_runner.py", "src/mcalib_cache_observer.py",
           "src/limited_arc_cpu.py"]

CFG = dict(
    protocol="DPJAIT_PREREG_v0.1",
    window_frames=cc.WINDOW_FRAMES,
    m_grid=cc.M_GRID,
    gate_px=cc.GATE_PX,
    consistency_px=cc.CONSISTENCY_PX,
    eligibility_proxy_px=75.0,       # ~0.5 m at the dev-measured 150 px/m
    budgets_real=[40, 52, 68, 80, 100, 136, 200],
    budgets_sim=[40, 52, 68, 80, 100, 136, 200, 272],
    budget_tol=0.12,
    primary_estimator="polygon",
    k_real=[2, 3, 4],
    k_sim=[2, 3, 4, 5, 6, 7, 8],
    max_subsets_sim=10,
    subset_seed=cc.SUBSET_SEED,
    confirm_real_single=["R05_D1", "R06_D1", "R07_D1", "R08_D1", "R09_D1",
                         "R10_D1", "R11_D1", "R16_D1_A", "R18_D1_A"],
    confirm_real_multi=["R14_D3", "R15_D3"],
    confirm_sim=["S08_D8", "S09_D6", "S10_D6"],
)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_new(p: Path, v) -> None:
    if p.exists():
        raise SystemExit("refusing to overwrite %s" % p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=1, default=float))


# ---------------------------------------------------------------- eligibility
def window_proxy_px(rec, f0: int, f1: int) -> float:
    """Reference-free motion proxy: median over cameras of 2-D box travel.

    Single-target records use the one box per frame; multi-target records use
    the whole scene (the centroid of all boxes), which needs no identity.
    """
    vals = []
    for c in rec.cam_ids:
        seq = []
        for f in range(f0, f1 + 1):
            d = rec.detections[c].get(f)
            if d is None or len(d) == 0:
                continue
            seq.append(d[0] if len(d) == 1 else d.mean(axis=0))
        if len(seq) > 2:
            vals.append(float(np.linalg.norm(np.diff(np.asarray(seq), axis=0), axis=1).sum()))
    return float(np.median(vals)) if vals else float("nan")


# ------------------------------------------------------------------- steps
def step_freeze(out: Path) -> None:
    if out.exists():
        raise SystemExit("refusing to re-freeze an existing run: %s "
                         "(a new protocol version needs a new directory)" % out)
    out.mkdir(parents=True)
    save_new(out / "config.json", dict(CFG, utc=utc()))
    save_new(out / "freeze.json",
             dict(hashes={s: sha(ROOT / s) for s in SOURCES}, utc=utc()))
    for s in SOURCES:
        dst = out / "source_snapshot" / s
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / s, dst)
    print("frozen %d sources under %s" % (len(SOURCES), out))


def verify(out: Path) -> None:
    for s, h in json.loads((out / "freeze.json").read_text())["hashes"].items():
        if sha(ROOT / s) != h:
            raise SystemExit("frozen source changed: %s" % s)


def open_record(path: Path, sim: bool):
    if sim:
        from dpjait_sim_record import SimRecord
        return SimRecord(path)
    return Record(path)


def step_predict(out: Path, record: Path, sim: bool, dev: bool, max_windows: int) -> None:
    """Path-length ESTIMATES only.  Never reads a reference path length."""
    if not dev:
        verify(out)
    cfg = json.loads((out / "config.json").read_text())
    rec = open_record(record, sim)
    drones = sorted(rec.reference)
    k_values = [k for k in (cfg["k_sim"] if sim else cfg["k_real"]) if k <= len(rec.cam_ids)]
    max_subsets = cfg["max_subsets_sim"] if sim else 0
    W = cfg["window_frames"]

    n_win = (rec.n_video_frames - 1) // W
    if max_windows:
        n_win = min(n_win, max_windows)

    eligible, rows = [], []
    for w in range(n_win):
        f0, f1 = w * W, (w + 1) * W
        proxy = window_proxy_px(rec, f0, f1)
        ok = np.isfinite(proxy) and proxy >= cfg["eligibility_proxy_px"]
        eligible.append(dict(window=w, proxy_px=proxy, eligible=bool(ok)))
        if not ok:
            continue
        for m in cfg["m_grid"]:
            frames = cc.instants(f0, m)
            times = [f / VIDEO_FPS for f in frames]
            for k in k_values:
                for cams in cc.subsets(rec.cam_ids, k, max_subsets):
                    tag = "+".join(c[-4:] for c in cams)
                    base = dict(record=rec.name, window=w, m=m, k=k, subset=tag)
                    # ---- labelled arm: identity from the reference ----------
                    for d in drones:
                        pts, ts, nv = cc.annotated_points(rec, d, frames, cams)
                        keep = [(p, t) for p, t in zip(pts, ts) if p is not None]
                        failed = (pts[0] is None or pts[-1] is None or len(keep) < 2)
                        r = dict(base, chain="annotated", drone=d,
                                 missing=len(pts) - len(keep), failed=failed,
                                 views_mean=float(np.mean(nv)) if nv else 0.0)
                        if not failed:
                            est = cc.estimate([p for p, _ in keep], [t for _, t in keep])
                            r.update({"est_" + n: v for n, v in est.items()})
                        rows.append(r)
                    # ---- reference-free arm ---------------------------------
                    per_instant, nvf = [], []
                    for f in frames:
                        pts = []
                        for obs in assign_free(rec, f, cams, cfg["consistency_px"]):
                            try:
                                pts.append(triangulate(obs, rec.cameras))
                                nvf.append(len(obs))
                            except ValueError:
                                continue
                        per_instant.append(pts)
                    tracks = link_tracks(per_instant, times)
                    assign = match_tracks_to_drones(tracks, rec, drones, frames)
                    for d in drones:
                        r = dict(base, chain="free", drone=d, n_tracks=len(tracks),
                                 views_mean=float(np.mean(nvf)) if nvf else 0.0)
                        if d not in assign:
                            r.update(failed=True, missing=len(frames))
                            rows.append(r)
                            continue
                        ti, med = assign[d]
                        idx = [i for i, _ in tracks[ti]]
                        pts = [p for _, p in tracks[ti]]
                        covered = (0 in idx) and (len(frames) - 1 in idx)
                        r.update(missing=len(frames) - len(pts),
                                 failed=not covered or len(pts) < 2,
                                 track_median_err_m=med)
                        if not r["failed"]:
                            est = cc.estimate(pts, [times[i] for i in idx])
                            r.update({"est_" + n: v for n, v in est.items()})
                        rows.append(r)

    assert not any("reference_m" in r or "err_polygon" in r for r in rows), \
        "predict must not contain any reference-derived field"

    pred = out / "predictions"
    save_new(pred / (rec.name + ".json"),
             dict(record=rec.name, sim=sim, windows=eligible,
                  n_eligible=sum(1 for e in eligible if e["eligible"]),
                  n_windows=len(eligible), drones=drones,
                  cameras=rec.cam_ids, rows=rows, utc=utc()))
    seal = out / "seal.json"
    seals = json.loads(seal.read_text()) if seal.exists() else {}
    seals[rec.name] = dict(sha256=sha(pred / (rec.name + ".json")), utc=utc(),
                           n_rows=len(rows))
    seal.write_text(json.dumps(seals, indent=1))
    print("%s: %d/%d windows eligible, %d predictions sealed"
          % (rec.name, sum(1 for e in eligible if e["eligible"]), len(eligible), len(rows)))


def _error_structure(out: Path, rec, pred: dict, cfg_m: dict) -> None:
    """H4: is the per-instant error common-mode or independent?

    At the record's own m*, compare the path error that independent errors
    would predict, sqrt(2 * segments) * scatter, against what is observed.
    A ratio near 1 means independent; MCalib's 6.3 means strongly common-mode.
    The constant lever arm is removed first: it is common-mode by construction
    and cannot reach a length.  This runs at score time because it needs
    reference positions.
    """
    W = cfg_m["window_frames"]
    k = max(r["k"] for r in pred["rows"])
    m_star = _best_m(pred["rows"], "annotated", k, cfg_m["m_grid"])[0]
    if not m_star:
        return
    eligible = [e["window"] for e in pred["windows"] if e["eligible"]]
    scat, predicted, actual, lever = [], [], [], []
    for w in eligible:
        f0 = w * W
        frames = cc.instants(f0, m_star)
        P, R = [], []
        for f in frames:
            obs = assign_annotated(rec, pred["drones"][0], f, rec.cam_ids, cfg_m["gate_px"])
            if len(obs) < 2:
                continue
            try:
                p = triangulate(obs, rec.cameras)
            except ValueError:
                continue
            X = rec.reference_at(pred["drones"][0], f)
            if np.isnan(X).any():
                continue
            P.append(p)
            R.append(X)
        if len(P) < 3:
            continue
        P, R = np.asarray(P), np.asarray(R)
        e = P - R
        lever.append(float(np.linalg.norm(e.mean(axis=0))))
        ec = e - e.mean(axis=0)
        s = float(np.sqrt((ec ** 2).sum(axis=1).mean()))
        scat.append(s)
        predicted.append(np.sqrt(2 * (len(P) - 1)) * s)
        actual.append(abs(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
                              - np.linalg.norm(np.diff(R, axis=0), axis=1).sum())))
    if not actual or not np.mean(actual):
        return
    save_new(out / "error_structure" / (rec.name + ".json"), dict(
        record=rec.name, k=k, m_star=m_star, windows=len(actual),
        lever_mm=1e3 * float(np.mean(lever)),
        scatter_mm=1e3 * float(np.mean(scat)),
        predicted_mm=1e3 * float(np.mean(predicted)),
        actual_mm=1e3 * float(np.mean(actual)),
        ratio=float(np.mean(predicted) / np.mean(actual))))


def step_score(out: Path, record: Path, sim: bool) -> None:
    """Only here is a reference path length computed."""
    rec = open_record(record, sim)
    pred_path = out / "predictions" / (rec.name + ".json")
    seals = json.loads((out / "seal.json").read_text())
    if sha(pred_path) != seals[rec.name]["sha256"]:
        raise SystemExit("prediction file changed after sealing: %s" % rec.name)
    pred = json.loads(pred_path.read_text())
    W = json.loads((out / "config.json").read_text())["window_frames"]

    refs = {}
    for r in pred["rows"]:
        key = (r["window"], r["drone"])
        if key not in refs:
            refs[key] = rec.reference_path_length(r["drone"], r["window"] * W,
                                                  (r["window"] + 1) * W)
        ref = refs[key]
        r["reference_m"] = ref
        for name in ("polygon", "limited_arc"):
            if "est_" + name in r and np.isfinite(ref):
                r["err_" + name] = abs(r["est_" + name] - ref)
    save_new(out / "scored" / (rec.name + ".json"), pred)
    _error_structure(out, rec, pred, cfg_m=json.loads((out / "config.json").read_text()))
    done = [r for r in pred["rows"] if not r.get("failed")]
    finite = [v for v in refs.values() if np.isfinite(v)]
    print("%s: scored %d rows (%d completed), reference median %s"
          % (rec.name, len(pred["rows"]), len(done),
             ("%.3f m" % float(np.median(finite))) if finite else "n/a (no eligible window)"))


# -------------------------------------------------------------------- decide
def _mae(rows, chain, k, m, est="polygon"):
    e = [r["err_" + est] for r in rows
         if r["chain"] == chain and r["k"] == k and r["m"] == m
         and not r.get("failed") and np.isfinite(r.get("err_" + est, np.nan))]
    return float(np.mean(e)) if e else None


def _fail(rows, chain, k, m):
    sel = [r for r in rows if r["chain"] == chain and r["k"] == k and r["m"] == m]
    return (sum(1 for r in sel if r.get("failed")) / len(sel)) if sel else None


def _best_m(rows, chain, k, grid, est="polygon"):
    vals = [(m, _mae(rows, chain, k, m, est)) for m in grid]
    vals = [(m, v) for m, v in vals if v is not None]
    return min(vals, key=lambda t: t[1]) if vals else (None, None)


def step_decide(out: Path) -> None:
    cfg = json.loads((out / "config.json").read_text())
    scored = sorted((out / "scored").glob("*.json"))
    if not scored:
        raise SystemExit("nothing scored yet")
    per = {}
    for p in scored:
        d = json.loads(p.read_text())
        per[d["record"]] = d

    real_single = [r for r in cfg["confirm_real_single"] if r in per]
    real_multi = [r for r in cfg["confirm_real_multi"] if r in per]
    sim = [r for r in cfg["confirm_sim"] if r in per]
    grid = cfg["m_grid"]
    res = {}

    # H1: Goldilocks point
    ms_real = [_best_m(per[r]["rows"], "annotated", 4, grid)[0] for r in real_single]
    ms_real = [m for m in ms_real if m]
    ms_sim = [_best_m(per[r]["rows"], "annotated", 8, grid)[0] for r in sim]
    ms_sim = [m for m in ms_sim if m]
    res["H1"] = dict(
        real_m_star=ms_real, real_median=float(np.median(ms_real)) if ms_real else None,
        sim_m_star=ms_sim, sim_median=float(np.median(ms_sim)) if ms_sim else None,
        pass_real=bool(ms_real) and 6 <= float(np.median(ms_real)) <= 13,
        pass_sim=bool(ms_sim) and 20 <= float(np.median(ms_sim)) <= 50)

    # H2a: k=4 wins real budget cells
    cells = []
    for r in real_single:
        rows = per[r]["rows"]
        for B in cfg["budgets_real"]:
            best = {}
            for k in cfg["k_real"]:
                cand = [(m, _mae(rows, "annotated", k, m)) for m in grid
                        if abs(k * m - B) <= cfg["budget_tol"] * B]
                cand = [(m, v) for m, v in cand if v is not None]
                if cand:
                    best[k] = min(cand, key=lambda t: t[1])
            if len(best) >= 2:
                cells.append(dict(record=r, B=B,
                                  winner=int(min(best, key=lambda k: best[k][1])),
                                  values={k: best[k] for k in best}))
    won = sum(1 for c in cells if c["winner"] == 4)
    res["H2a"] = dict(cells=len(cells), k4_wins=won,
                      pass_=bool(cells) and won / len(cells) >= 6 / 7)

    # H2b: simulated winner climbs k, winning m stays near m*
    simcells = []
    for r in sim:
        rows = per[r]["rows"]
        seq = []
        for B in cfg["budgets_sim"]:
            best = {}
            for k in cfg["k_sim"]:
                cand = [(m, _mae(rows, "annotated", k, m)) for m in grid
                        if abs(k * m - B) <= cfg["budget_tol"] * B]
                cand = [(m, v) for m, v in cand if v is not None]
                if cand:
                    best[k] = min(cand, key=lambda t: t[1])
            if len(best) >= 2:
                w = int(min(best, key=lambda k: best[k][1]))
                seq.append(dict(B=B, winner=w, m=best[w][0]))
        if seq:
            ks = [s["winner"] for s in seq]
            mm = [s["m"] for s in seq]
            simcells.append(dict(record=r, sequence=seq,
                                 monotone=all(b >= a for a, b in zip(ks, ks[1:])),
                                 m_in_range=all(17 <= x <= 50 for x in mm)))
    res["H2b"] = dict(records=simcells,
                      pass_=bool(simcells) and all(s["monotone"] and s["m_in_range"]
                                                   for s in simcells))

    # H3: identity cost collapses with k
    h3 = []
    for r in real_multi + real_single:
        rows = per[r]["rows"]
        entry = dict(record=r, group="multi" if r in real_multi else "single")
        for k in cfg["k_real"]:
            m, a = _best_m(rows, "annotated", k, grid)
            f = _mae(rows, "free", k, m) if m else None
            entry["k%d" % k] = dict(m=m, annotated=a, free=f,
                                    ratio=(f / a) if (a and f) else None,
                                    free_fail=_fail(rows, "free", k, m) if m else None)
        h3.append(entry)
    multi = [e for e in h3 if e["group"] == "multi"]

    def _ok(es, k, lo=None, hi=None, fail_lo=None, fail_hi=None):
        vals = [e["k%d" % k] for e in es if e["k%d" % k]["ratio"] is not None]
        if not vals:
            return None
        r_ok = all((lo is None or v["ratio"] > lo) and (hi is None or v["ratio"] < hi) for v in vals)
        f_ok = all((fail_lo is None or v["free_fail"] > fail_lo)
                   and (fail_hi is None or v["free_fail"] < fail_hi) for v in vals)
        return bool(r_ok and f_ok)

    res["H3"] = dict(detail=h3,
                     pass_a=_ok(multi, 2, lo=3.0, fail_lo=0.2),
                     pass_b=(_ok(multi, 3, hi=2.5, fail_hi=0.15)
                             and _ok(multi, 4, hi=2.5, fail_hi=0.15)) if multi else None)

    # H4: errors are not common-mode (written at score time by _error_structure)
    es = []
    for p in sorted((out / "error_structure").glob("*.json")) \
            if (out / "error_structure").is_dir() else []:
        d = json.loads(p.read_text())
        if d["record"] in real_single:
            es.append(d)
    ratios = [d["ratio"] for d in es]
    res["H4"] = dict(detail=es,
                     median_ratio=float(np.median(ratios)) if ratios else None,
                     pass_=bool(ratios) and 0.8 <= float(np.median(ratios)) <= 1.5)

    # H5: no saturation to k=8 on the simulated rig
    h5 = []
    for r in sim:
        rows = per[r]["rows"]
        for m in [x for x in grid if x >= 25]:
            a3, a8 = _mae(rows, "annotated", 3, m), _mae(rows, "annotated", 8, m)
            if a3 and a8:
                h5.append(dict(record=r, m=m, k3=a3, k8=a8, ratio=a3 / a8))
    res["H5"] = dict(detail=h5,
                     pass_=bool(h5) and float(np.median([x["ratio"] for x in h5])) >= 1.5)

    save_new(out / "decision.json", dict(utc=utc(), records=sorted(per), result=res))
    for h in ("H1", "H2a", "H2b", "H3", "H4", "H5"):
        flags = {k: v for k, v in res[h].items() if k.startswith("pass")}
        print("%-4s %s" % (h, flags))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["freeze", "predict", "score", "decide"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", default=None)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--dev", action="store_true", help="skip freeze verification")
    ap.add_argument("--max-windows", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)

    if a.step == "freeze":
        step_freeze(out)
    elif a.step == "predict":
        step_predict(out, Path(a.record), a.sim, a.dev, a.max_windows)
    elif a.step == "score":
        step_score(out, Path(a.record), a.sim)
    else:
        step_decide(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
