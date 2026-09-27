"""Identity assignment for the DPJAIT chains -- the subject of the comparison.

Both chains observe exactly the same pixels: the published YOLOv5 box centres.
They differ only in how a box is bound to a drone, which is the one thing the
reference supplies in the labelled chains used on MCalib and Lyon:

* `assign_annotated` -- identity from the reference.  For a given drone the
  reference position is projected into each camera and the nearest box inside
  a gate is taken.  This is the DPJAIT analogue of `lyon_observe.py` and of the
  publisher's centroid cache on MCalib.
* `assign_free` -- identity from multi-view consistency alone.  Boxes are
  grouped across cameras by triangulating every pair and collecting the views
  that agree with it, then groups are chosen greedily and disjointly.  The
  reference is never read.

`link_tracks` joins the free chain's unlabelled 3-D points across the selected
instants with a speed-gated nearest-neighbour rule, which is where the free
chain pays for sparse sampling: the further apart the instants, the more often
the link is wrong.

`match_tracks_to_drones` is scoring only -- it decides which reference a track
should be compared against and never feeds the measurement.
"""
from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np

from dpjait_record import project
from mcalib_cache_observer import triangulate
from robust_triangulation_cpu import reprojection_errors

MAX_SPEED_MPS = 6.0      # drone speed ceiling used for the temporal link gate
LINK_SLACK_M = 0.30      # extra radius absorbing triangulation error


def assign_annotated(rec, drone, frame, cam_ids, gate_px):
    """{cam: uv} for one drone, identity taken from the reference."""
    X = rec.reference_at(drone, frame)
    if np.isnan(X).any():
        return {}
    obs = {}
    for c in cam_ids:
        dets = rec.detections[c].get(frame)
        if dets is None or len(dets) == 0:
            continue
        uv, depth = project(rec.cameras[c], X)
        if depth[0] <= 0:
            continue
        d = np.linalg.norm(dets - uv[0], axis=1)
        j = int(np.argmin(d))
        if d[j] <= gate_px:
            obs[c] = dets[j]
    return obs


def epipolar_px(rec, ca, uva, cb, uvb) -> float:
    """Distance of uvb from the epipolar line of uva, in pixels of camera b.

    Needed because a two-view triangulation reprojects almost exactly whatever
    pair it is given: without this test a pair of boxes belonging to *different*
    drones looks perfectly consistent, and k=2 identity would be scored as if
    the geometry had been checked when it had not.
    """
    A, B = rec.cameras[ca], rec.cameras[cb]
    xa = cv2.undistortPoints(np.asarray(uva, float).reshape(1, 1, 2), A["K"], A["D"]).reshape(2)
    xb = cv2.undistortPoints(np.asarray(uvb, float).reshape(1, 1, 2), B["K"], B["D"]).reshape(2)
    R_rel = B["R"] @ A["R"].T
    t_rel = B["t_m"] - R_rel @ A["t_m"]
    tx = np.array([[0, -t_rel[2], t_rel[1]],
                   [t_rel[2], 0, -t_rel[0]],
                   [-t_rel[1], t_rel[0], 0]])
    E = tx @ R_rel
    line = E @ np.array([xa[0], xa[1], 1.0])
    n = float(np.hypot(line[0], line[1]))
    if n < 1e-12:
        return float("inf")
    d = abs(float(np.array([xb[0], xb[1], 1.0]) @ line)) / n
    return d * float(B["K"][0, 0])


def _support(point, frame, rec, cam_ids, thresh_px):
    """Views whose nearest box agrees with `point`, plus that box."""
    obs, errs = {}, {}
    for c in cam_ids:
        dets = rec.detections[c].get(frame)
        if dets is None or len(dets) == 0:
            continue
        uv, depth = project(rec.cameras[c], point)
        if depth[0] <= 0:
            continue
        d = np.linalg.norm(dets - uv[0], axis=1)
        j = int(np.argmin(d))
        if d[j] <= thresh_px:
            obs[c] = dets[j]
            errs[c] = float(d[j])
    return obs, errs


def assign_free(rec, frame, cam_ids, thresh_px):
    """Group boxes into drones using multi-view consistency only.

    Returns a list of {cam: uv} groups, disjoint in (camera, box), ordered by
    decreasing support.  The reference is never consulted.
    """
    cams = {c: rec.detections[c].get(frame) for c in cam_ids}
    cams = {c: v for c, v in cams.items() if v is not None and len(v)}
    if len(cams) < 2:
        return []

    candidates = []
    for ca, cb in combinations(sorted(cams), 2):
        for ia in range(len(cams[ca])):
            for ib in range(len(cams[cb])):
                if epipolar_px(rec, ca, cams[ca][ia], cb, cams[cb][ib]) > thresh_px:
                    continue
                seed = {ca: cams[ca][ia], cb: cams[cb][ib]}
                try:
                    p = triangulate(seed, rec.cameras)
                except ValueError:
                    continue
                obs, errs = _support(p, frame, rec, cams, thresh_px)
                if len(obs) < 2:
                    continue
                # index the chosen box per camera so groups can be made disjoint
                keys = {}
                for c, uv in obs.items():
                    j = int(np.argmin(np.linalg.norm(cams[c] - uv, axis=1)))
                    keys[c] = j
                candidates.append((len(obs), -float(np.median(list(errs.values()))),
                                   keys, obs))

    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    used, groups = set(), []
    for _, _, keys, obs in candidates:
        if any((c, j) in used for c, j in keys.items()):
            continue
        used.update(keys.items())
        groups.append(obs)
    return groups


def link_tracks(points_by_instant, times):
    """Join unlabelled 3-D points across instants by speed-gated nearest neighbour.

    points_by_instant: list (per instant) of lists of 3-vectors.
    Returns a list of tracks; each track is a list of (instant_index, point).
    """
    tracks = []
    active = []            # (track_index, last_point, last_time)
    for i, (pts, t) in enumerate(zip(points_by_instant, times)):
        taken = set()
        order = []
        for ti, last, tlast in active:
            radius = MAX_SPEED_MPS * abs(t - tlast) + LINK_SLACK_M
            best, bestd = None, radius
            for j, p in enumerate(pts):
                if j in taken:
                    continue
                d = float(np.linalg.norm(p - last))
                if d < bestd:
                    best, bestd = j, d
            order.append((ti, best, bestd))
        # resolve greedily by distance so the closest link wins a contested point
        for ti, best, _ in sorted(order, key=lambda o: o[2]):
            if best is None or best in taken:
                continue
            taken.add(best)
            tracks[ti].append((i, pts[best]))
        new_active = []
        for ti, last, tlast in active:
            if tracks[ti] and tracks[ti][-1][0] == i:
                new_active.append((ti, tracks[ti][-1][1], t))
            else:
                new_active.append((ti, last, tlast))
        for j, p in enumerate(pts):
            if j in taken:
                continue
            tracks.append([(i, p)])
            new_active.append((len(tracks) - 1, p, t))
        active = new_active
    return tracks


def match_tracks_to_drones(tracks, rec, drones, frames):
    """Scoring only: nearest-reference assignment, one track per drone.

    Uses the reference to decide which drone a track should be compared
    against.  It never changes a measured point.
    """
    scores = {}
    for ti, tr in enumerate(tracks):
        for d in drones:
            ds = []
            for i, p in tr:
                X = rec.reference_at(d, frames[i])
                if not np.isnan(X).any():
                    ds.append(float(np.linalg.norm(p - X)))
            if ds:
                scores[(ti, d)] = float(np.median(ds))
    assignment, used_t, used_d = {}, set(), set()
    for (ti, d), s in sorted(scores.items(), key=lambda kv: kv[1]):
        if ti in used_t or d in used_d:
            continue
        assignment[d] = (ti, s)
        used_t.add(ti)
        used_d.add(d)
    return assignment
