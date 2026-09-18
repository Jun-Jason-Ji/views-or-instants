"""Reference-free identification of the skull-vertex (SV) marker from the k
selected Lyon views at one instant, using only those k frames.

Per view : blob candidates = local extrema of EITHER SIGN of a
           difference-of-Gaussians response (the grey passive balls are
           bright against the dark treadmill and dark against bright walls)
           with |response| >= DOG_K * robust noise level (MAD * 1.4826),
           isotropic Hessian (|l_small/l_large| >= ISO_MIN, drops lines),
           centre saturation <= SAT_MAX, at most TOP_N strongest.
           No identification per view.
Per instant, over the k views:
  1. every pair of views: candidates within EPI_PX of each other's epipolar
     line (undistorted coordinates) are triangulated;
  2. 3D points outside the WORKSPACE box are discarded;
  3. support = number of views with a candidate within SUP_PX of the
     reprojection; keep points with support >= min(k, SUP_MIN);
  4. points within MERGE_M of each other are merged (highest support wins);
  5. SV = the surviving point with the largest z (up).  Its supporting
     candidates, refined to intensity-weighted centroids, are the 2D
     observations handed to the measurement chain; unsupported views give None.

WORKSPACE is a fixed box around the treadmill in the QTM frame, declared in
the protocol from the rig layout (not from any trajectory).

Camera dicts follow the project chain: K (3x3), D (dist coeffs), R (3x3),
t_m (3,), plus 'name'.
"""
from __future__ import annotations

import json
import sys
import tomllib
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

PARAMS = dict(
    DOG_S1=1.5, DOG_S2=3.5, DOG_K=6.0, DOG_ABS_MIN=4.0, SAT_MAX=120, TOP_N=600, ISO_MIN=0.4,
    EPI_PX=4.0, SUP_PX=4.0, SUP_MIN=3, MERGE_M=0.02, CENTROID_R=3,
    WORKSPACE=dict(x=(0.8, 2.6), y=(-0.5, 0.9), z=(0.3, 2.0)),
)
VERSION = "lyon_sv_identify_v0.4"


# ---------------------------------------------------------------- calibration
def chain_cameras_from_toml(path: Path) -> dict[str, dict]:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    cams = {}
    for key, val in raw.items():
        if not isinstance(val, dict) or "matrix" not in val:
            continue
        R, _ = cv2.Rodrigues(np.asarray(val["rotation"], dtype=np.float64).reshape(3, 1))
        cams[str(val["name"])] = dict(
            name=str(val["name"]),
            K=np.asarray(val["matrix"], dtype=np.float64).reshape(3, 3).tolist(),
            D=np.asarray(val.get("distortions", [0, 0, 0, 0]), dtype=np.float64).ravel().tolist(),
            R=R.tolist(),
            t_m=np.asarray(val["translation"], dtype=np.float64).ravel().tolist(),
            size=[int(v) for v in val["size"]],
        )
    return cams


def _P(cam):
    return np.asarray(cam["K"]) @ np.hstack([np.asarray(cam["R"]), np.asarray(cam["t_m"]).reshape(3, 1)])


def fundamental(cam_a, cam_b):
    """F such that x_b^T F x_a = 0 for undistorted pixel coordinates."""
    Ra, ta = np.asarray(cam_a["R"]), np.asarray(cam_a["t_m"]).reshape(3)
    Rb, tb = np.asarray(cam_b["R"]), np.asarray(cam_b["t_m"]).reshape(3)
    R = Rb @ Ra.T
    t = tb - R @ ta
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    Ka, Kb = np.asarray(cam_a["K"]), np.asarray(cam_b["K"])
    return np.linalg.inv(Kb).T @ tx @ R @ np.linalg.inv(Ka)


def undistort(cam, pts):
    K, D = np.asarray(cam["K"]), np.asarray(cam["D"])
    if len(pts) == 0:
        return np.zeros((0, 2))
    return cv2.undistortPoints(np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)


def project(cam, X):
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    R, t = np.asarray(cam["R"]), np.asarray(cam["t_m"]).reshape(3, 1)
    img, _ = cv2.projectPoints(X.reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t, np.asarray(cam["K"]), np.asarray(cam["D"]))
    z = (R @ X.T + t)[2]
    return img.reshape(-1, 2), z


# ---------------------------------------------------------------- per view
def blob_candidates(frame_bgr, p=PARAMS):
    """Return (N x 5 array [x, y, signed response, refined u, refined v], gray)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    g = gray.astype(np.float32)
    g1 = cv2.GaussianBlur(g, (0, 0), p["DOG_S1"])
    resp = g1 - cv2.GaussianBlur(g, (0, 0), p["DOG_S2"])
    noise = 1.4826 * float(np.median(np.abs(resp - np.median(resp))))
    thr = max(p["DOG_ABS_MIN"], p["DOG_K"] * noise)
    k3 = np.ones((3, 3), np.uint8)
    pos = (resp >= cv2.dilate(resp, k3)) & (resp >= thr)
    neg = (resp <= cv2.erode(resp, k3)) & (resp <= -thr)
    peak = (pos | neg) & (sat <= p["SAT_MAX"])
    ys, xs = np.nonzero(peak)
    if len(xs) == 0:
        return np.zeros((0, 3)), gray
    # isotropy of the Hessian at the candidate (sign-agnostic)
    gxx = cv2.Sobel(g1, cv2.CV_32F, 2, 0, ksize=3)
    gyy = cv2.Sobel(g1, cv2.CV_32F, 0, 2, ksize=3)
    gxy = cv2.Sobel(g1, cv2.CV_32F, 1, 1, ksize=3)
    a, b, c = gxx[ys, xs], gxy[ys, xs], gyy[ys, xs]
    tr, det = a + c, a * c - b * b
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    l1, l2 = np.abs(tr / 2 + disc), np.abs(tr / 2 - disc)
    lo, hi = np.minimum(l1, l2), np.maximum(l1, l2)
    iso = (det > 0) & (lo >= p["ISO_MIN"] * hi)
    xs, ys = xs[iso], ys[iso]
    if len(xs) == 0:
        return np.zeros((0, 3)), gray
    r = resp[ys, xs]
    order = np.argsort(-np.abs(r))[: p["TOP_N"]]
    xs, ys, r = xs[order], ys[order], r[order]
    ref = np.array([refine(gray, x, y, p["CENTROID_R"]) for x, y in zip(xs, ys)]) if len(xs) else np.zeros((0, 2))
    return np.column_stack([xs, ys, r, ref[:, 0], ref[:, 1]]).astype(np.float64), gray


def cache_candidates(video: Path, p=PARAMS) -> list[np.ndarray]:
    """Per-frame candidate arrays for a whole video (independent of any subset)."""
    cap = cv2.VideoCapture(str(video))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        c, _ = blob_candidates(fr, p)
        out.append(c)
    cap.release()
    return out


def refine(gray, x, y, r):
    h, w = gray.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1, y0, y1 = max(0, xi - r), min(w, xi + r + 1), max(0, yi - r), min(h, yi + r + 1)
    patch = gray[y0:y1, x0:x1].astype(np.float64)
    centre = patch[min(yi, y1 - 1) - y0, min(xi, x1 - 1) - x0]
    wgt = patch - patch.min() if centre >= patch.mean() else patch.max() - patch
    if wgt.sum() <= 0:
        return float(x), float(y)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((wgt * xx).sum() / wgt.sum()), float((wgt * yy).sum() / wgt.sum())


# ---------------------------------------------------------------- per instant
def _tri_pair(cam_a, ua, cam_b, ub):
    Pa, Pb = _P(cam_a), _P(cam_b)
    A = np.stack([ua[:, 0, None] * Pa[2] - Pa[0], ua[:, 1, None] * Pa[2] - Pa[1],
                  ub[:, 0, None] * Pb[2] - Pb[0], ub[:, 1, None] * Pb[2] - Pb[1]], axis=1)  # N x 4 x 4
    _, _, vt = np.linalg.svd(A)
    h = vt[:, -1, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        X = h[:, :3] / h[:, 3:4]
    return X


def identify_sv(frames: dict[str, np.ndarray], cams: dict[str, dict], p=PARAMS, cand=None, gray=None):
    """frames: {cam_name: bgr image} for the k selected views (or None when
    `cand` and `gray` are supplied from cache_candidates).

    Returns dict(point=(3,) or None, uv={cam: (u,v) or None}, support=int,
                 n_candidates={cam: int}, n_points=int, reason=str)."""
    names = sorted(frames if frames is not None else cand)
    k = len(names)
    if cand is None:
        cand, gray = {}, {}
        for n in names:
            c, g = blob_candidates(frames[n], p)
            cand[n], gray[n] = c, g
    full = {n: np.asarray(cand[n]) if len(cand[n]) else np.zeros((0, 5)) for n in names}
    cand = {n: full[n][:, :2] for n in names}
    und = {n: undistort(cams[n], cand[n]) for n in names}
    out = dict(point=None, uv={n: None for n in names}, support=0,
               n_candidates={n: int(len(cand[n])) for n in names}, n_points=0, reason="")
    if k < 2 or any(len(cand[n]) == 0 for n in names):
        out["reason"] = "too_few_views_or_no_candidates"
        return out
    ws = p["WORKSPACE"]
    pts = []
    for a, b in combinations(names, 2):
        F = fundamental(cams[a], cams[b])
        xa = np.hstack([und[a], np.ones((len(und[a]), 1))])
        xb = np.hstack([und[b], np.ones((len(und[b]), 1))])
        lines = xa @ F.T  # lines in b for each a: N_a x 3
        nrm = np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
        d = np.abs(lines @ xb.T) / np.maximum(nrm, 1e-9)  # N_a x N_b
        ia, ib = np.nonzero(d <= p["EPI_PX"])
        if len(ia) == 0:
            continue
        X = _tri_pair(cams[a], und[a][ia], cams[b], und[b][ib])
        ok = np.isfinite(X).all(axis=1)
        ok &= (X[:, 0] >= ws["x"][0]) & (X[:, 0] <= ws["x"][1]) & (X[:, 1] >= ws["y"][0]) & (X[:, 1] <= ws["y"][1]) \
            & (X[:, 2] >= ws["z"][0]) & (X[:, 2] <= ws["z"][1])
        pts.append(X[ok])
    if not pts:
        out["reason"] = "no_epipolar_matches_in_workspace"
        return out
    X = np.vstack(pts)
    if len(X) == 0:
        out["reason"] = "no_epipolar_matches_in_workspace"
        return out
    # support: for each 3D point, views with a candidate within SUP_PX of its reprojection
    sup = np.zeros(len(X), dtype=int)
    match = {}
    for n in names:
        proj, z = project(cams[n], X)
        c = cand[n]
        dd = np.linalg.norm(proj[:, None, :] - c[None, :, :], axis=2)
        j = np.argmin(dd, axis=1)
        okv = (dd[np.arange(len(X)), j] <= p["SUP_PX"]) & (z > 0)
        sup += okv
        match[n] = np.where(okv, j, -1)
    need = min(k, p["SUP_MIN"])
    keep = sup >= need
    if not keep.any():
        out["reason"] = "no_point_with_support"
        out["n_points"] = int(len(X))
        return out
    X, sup = X[keep], sup[keep]
    match = {n: m[keep] for n, m in match.items()}
    # merge near-duplicates: process by (support desc, z desc), suppress neighbours
    order = np.lexsort((-X[:, 2], -sup))
    taken = np.zeros(len(X), dtype=bool)
    reps = []
    for i in order:
        if taken[i]:
            continue
        close = np.linalg.norm(X - X[i], axis=1) <= p["MERGE_M"]
        taken |= close
        reps.append(i)
    reps = np.array(reps)
    out["n_points"] = int(len(reps))
    best = reps[np.argmax(X[reps, 2])]
    out["point"] = X[best].tolist()
    out["support"] = int(sup[best])
    for n in names:
        j = match[n][best]
        if j >= 0:
            out["uv"][n] = (float(full[n][j, 3]), float(full[n][j, 4])) if full[n].shape[1] >= 5 else (float(cand[n][j, 0]), float(cand[n][j, 1]))
    out["reason"] = "ok"
    return out


if __name__ == "__main__":
    root = Path(sys.argv[1])
    cams = chain_cameras_from_toml(root / "calibration/Calib.toml")
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(source=str(root / "calibration/Calib.toml"), units="t_m in metres", cameras=cams), indent=1))
    print("wrote", out, "with", len(cams), "cameras")
