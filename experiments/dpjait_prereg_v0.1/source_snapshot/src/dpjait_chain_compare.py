"""What does the labelled-observation assumption cost?

MCalib and Lyon both measure through a *labelled* chain: the reference supplies
each observation's identity.  Every allocation conclusion in the manuscript is
conditioned on that assumption, and on Lyon the reference-free alternative
failed outright (LYON_DETECTOR_FREEZE_2026-09-17).  DPJAIT is the first rig
where both chains run, because the published YOLOv5 detections are complete and
unambiguous on the single-drone records.

This script runs the two chains over the same windows, the same time grids, the
same camera subsets, the same triangulation and the same estimators.  They
differ in one thing only -- how a box becomes a drone -- so the difference
between them is the price of the assumption.

Both chains are scored against the Vicon cross-centroid polyline.  A constant
lever arm does not change a path length, so the 116 mm offset between the box
centre and the cross (see the audit) costs no first-order bias; what enters is
its variation.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dpjait_identity import (assign_annotated, assign_free, link_tracks,  # noqa: E402
                             match_tracks_to_drones)
from dpjait_record import VIDEO_FPS, Record  # noqa: E402
from limited_arc_cpu import limited_arc  # noqa: E402
from mcalib_cache_observer import triangulate  # noqa: E402

WINDOW_FRAMES = 100                      # 4.0 s at 25 fps
M_GRID = [4, 6, 8, 10, 13, 17, 20, 25, 34, 50, 100]
GATE_PX = 60.0                           # annotated chain: >> the 22 px lever arm
# Free chain gate, set by the Lyon rule (3x the rig's own residual) from the
# measured triangulation reprojection residual: median 6.0-6.6 px on four
# independent records -> 20 px.  Derived without any reference trajectory.
CONSISTENCY_PX = 20.0


def instants(f0: int, m: int) -> list:
    idx = np.round(np.linspace(0, WINDOW_FRAMES, m)).astype(int)
    return sorted({f0 + int(i) for i in idx})


def polyline(points) -> float:
    p = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def estimate(points, times) -> dict:
    out = {"polygon": polyline(points)}
    try:
        out["limited_arc"] = limited_arc(points, times)
    except ValueError:
        out["limited_arc"] = float("nan")
    return out


def annotated_points(rec, drone, frames, cams):
    pts, ts, nviews = [], [], []
    for f in frames:
        obs = assign_annotated(rec, drone, f, cams, GATE_PX)
        ts.append(f / VIDEO_FPS)
        if len(obs) < 2:
            pts.append(None)
            nviews.append(len(obs))
            continue
        try:
            pts.append(triangulate(obs, rec.cameras))
            nviews.append(len(obs))
        except ValueError:
            pts.append(None)
            nviews.append(len(obs))
    return pts, ts, nviews


def free_tracks(rec, frames, cams, drones):
    per_instant, nviews = [], []
    for f in frames:
        pts = []
        for obs in assign_free(rec, f, cams, CONSISTENCY_PX):
            try:
                pts.append(triangulate(obs, rec.cameras))
                nviews.append(len(obs))
            except ValueError:
                continue
        per_instant.append(pts)
    times = [f / VIDEO_FPS for f in frames]
    tracks = link_tracks(per_instant, times)
    assignment = match_tracks_to_drones(tracks, rec, drones, frames)
    return tracks, assignment, times, nviews


SUBSET_SEED = 20401


def subsets(cam_ids, k, max_subsets=0):
    """All k-subsets, or a fixed-seed random sample when there are too many.

    Sampling matters on the 8-camera simulated rig, where C(8,4)=70 subsets
    would dominate the run.  The seed is fixed so the choice is reproducible
    and cannot be tuned.
    """
    all_subsets = [list(c) for c in combinations(sorted(cam_ids), k)]
    if not max_subsets or len(all_subsets) <= max_subsets:
        return all_subsets
    rng = np.random.default_rng(SUBSET_SEED + k)
    idx = rng.choice(len(all_subsets), size=max_subsets, replace=False)
    return [all_subsets[i] for i in sorted(idx)]


def run(rec, drones, k_values, max_windows, start_window=0,
        max_subsets=0, chains=("annotated", "free")) -> list:
    rows = []
    # a window spans [f0, f0+WINDOW_FRAMES] inclusive, so the last frame must exist
    n_win = min((rec.n_video_frames - 1) // WINDOW_FRAMES, start_window + max_windows)
    for w in range(start_window, n_win):
        f0 = w * WINDOW_FRAMES
        f1 = f0 + WINDOW_FRAMES
        ref = {d: rec.reference_path_length(d, f0, f1) for d in drones}
        for m in M_GRID:
            frames = instants(f0, m)
            times = [f / VIDEO_FPS for f in frames]
            for k in k_values:
                for cams in subsets(rec.cam_ids, k, max_subsets):
                    tag = "+".join(c[-4:] for c in cams)
                    # ---- labelled chain -------------------------------------
                    for d in drones if "annotated" in chains else []:
                        pts, ts, nv = annotated_points(rec, d, frames, cams)
                        ok = [(p, t) for p, t in zip(pts, ts) if p is not None]
                        missing = len(pts) - len(ok)
                        fail = (pts[0] is None or pts[-1] is None or len(ok) < 2)
                        row = dict(record=rec.name, window=w, m=m, k=k, subset=tag,
                                   chain="annotated", drone=d,
                                   reference_m=ref[d], missing=missing, failed=fail,
                                   views_mean=float(np.mean(nv)) if nv else 0.0)
                        if not fail:
                            est = estimate([p for p, _ in ok], [t for _, t in ok])
                            for name, v in est.items():
                                row["est_" + name] = v
                                row["err_" + name] = abs(v - ref[d]) if np.isfinite(ref[d]) else float("nan")
                        rows.append(row)
                    # ---- reference-free chain -------------------------------
                    if "free" not in chains:
                        continue
                    tracks, assignment, _, nvf = free_tracks(rec, frames, cams, drones)
                    for d in drones:
                        row = dict(record=rec.name, window=w, m=m, k=k, subset=tag,
                                   chain="free", drone=d,
                                   reference_m=ref[d], n_tracks=len(tracks),
                                   views_mean=float(np.mean(nvf)) if nvf else 0.0)
                        if d not in assignment:
                            row.update(failed=True, missing=len(frames))
                            rows.append(row)
                            continue
                        ti, med = assignment[d]
                        tr = tracks[ti]
                        idx = [i for i, _ in tr]
                        pts = [p for _, p in tr]
                        ts = [times[i] for i in idx]
                        covered = (0 in idx) and (len(frames) - 1 in idx)
                        row.update(missing=len(frames) - len(pts),
                                   failed=not covered or len(pts) < 2,
                                   track_median_err_m=med)
                        if not row["failed"]:
                            est = estimate(pts, ts)
                            for name, v in est.items():
                                row["est_" + name] = v
                                row["err_" + name] = abs(v - ref[d]) if np.isfinite(ref[d]) else float("nan")
                        rows.append(row)
    return rows


def summarise(rows, estimator="polygon"):
    key = "err_" + estimator
    agg = {}
    for r in rows:
        kk = (r["record"], r["chain"], r["k"], r["m"])
        a = agg.setdefault(kk, {"errs": [], "fail": 0, "n": 0})
        a["n"] += 1
        if r.get("failed"):
            a["fail"] += 1
        elif np.isfinite(r.get(key, float("nan"))):
            a["errs"].append(r[key])
    out = []
    for (rec, chain, k, m), a in sorted(agg.items()):
        out.append(dict(record=rec, chain=chain, k=k, m=m, n=a["n"],
                        fail_rate=a["fail"] / a["n"] if a["n"] else float("nan"),
                        mae_mm=1e3 * float(np.mean(a["errs"])) if a["errs"] else None,
                        median_mm=1e3 * float(np.median(a["errs"])) if a["errs"] else None))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", default="2,3,4")
    ap.add_argument("--max-windows", type=int, default=8)
    ap.add_argument("--start-window", type=int, default=0)
    ap.add_argument("--drones", default="all")
    ap.add_argument("--tag", default="")
    ap.add_argument("--sim", action="store_true", help="simulated record (8 cameras, 25 Hz reference)")
    ap.add_argument("--max-subsets", type=int, default=0)
    ap.add_argument("--chains", default="annotated,free")
    a = ap.parse_args()

    if a.sim:
        from dpjait_sim_record import SimRecord
        rec = SimRecord(a.record)
    else:
        rec = Record(a.record)
    drones = sorted(rec.reference) if a.drones == "all" else a.drones.split(",")
    k_values = [int(x) for x in a.k.split(",") if int(x) <= len(rec.cam_ids)]

    rows = run(rec, drones, k_values, a.max_windows, a.start_window,
               a.max_subsets, tuple(a.chains.split(",")))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("rows_" + rec.name + a.tag + ".json")).write_text(json.dumps(rows, default=float))
    summary = summarise(rows)
    (out / ("summary_" + rec.name + a.tag + ".json")).write_text(json.dumps(summary, indent=1))

    print("record %s   drones %s   cameras %d   windows %d"
          % (rec.name, ",".join(drones), len(rec.cam_ids),
             min(rec.n_video_frames // WINDOW_FRAMES, a.max_windows)))
    print("%-10s %2s %4s %8s %10s %10s" % ("chain", "k", "m", "fail", "MAE mm", "median mm"))
    for s in summary:
        print("%-10s %2d %4d %8.2f %10s %10s"
              % (s["chain"], s["k"], s["m"], s["fail_rate"],
                 "-" if s["mae_mm"] is None else "%.1f" % s["mae_mm"],
                 "-" if s["median_mm"] is None else "%.1f" % s["median_mm"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
