"""Labelled 2D marker observations for the LBMC Lyon videos (data preparation).

This is the Lyon analogue of the publisher's per-camera centroid cache on
MCalib.  Identity comes from the reference: for every camera, frame and
marker the 120 Hz C3D position is projected through Calib.toml and the real
image blob is searched inside a +/- WIN px window around that projection.
What is stored is the blob's intensity-weighted centroid, or NaN when no
blob of sufficient contrast exists in the window.  The projection itself is
never stored and never handed to the measurement chain.

Why oracle labelling: reference-free per-frame identification of the grey
passive balls in visible-light video failed on this rig
(reports/LYON_DETECTOR_FREEZE_2026-09-17.md); the allocation study concerns
reconstruction and sampling under a labelled-observation chain on both rigs.

Audit: per camera and marker the offset between centroid and projection is
summarised, so a reader can verify that the observations are measured, not
copied (offsets have a 1-3 px spread consistent with the checkerboard
calibration residual).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lyon_geometry_selfcheck import load_calib, load_markers  # noqa: E402

PARAMS = dict(WIN=8, DOG_S1=1.5, DOG_S2=3.5, DOG_K=6.0, DOG_ABS_MIN=4.0, ISO_MIN=0.3, CENTROID_R=3, MIN_EDGE=6)
VERSION = "lyon_observe_v0.2"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def response(gray: np.ndarray, p=PARAMS):
    """Returns (resp, thr, peak) where peak marks isotropic local extrema of |resp| above thr."""
    g = gray.astype(np.float32)
    g1 = cv2.GaussianBlur(g, (0, 0), p["DOG_S1"])
    resp = g1 - cv2.GaussianBlur(g, (0, 0), p["DOG_S2"])
    noise = 1.4826 * float(np.median(np.abs(resp - np.median(resp))))
    thr = max(p["DOG_ABS_MIN"], p["DOG_K"] * noise)
    k3 = np.ones((3, 3), np.uint8)
    peak = ((resp >= cv2.dilate(resp, k3)) & (resp >= thr)) | ((resp <= cv2.erode(resp, k3)) & (resp <= -thr))
    gxx = cv2.Sobel(g1, cv2.CV_32F, 2, 0, ksize=3)
    gyy = cv2.Sobel(g1, cv2.CV_32F, 0, 2, ksize=3)
    gxy = cv2.Sobel(g1, cv2.CV_32F, 1, 1, ksize=3)
    tr, det = gxx + gyy, gxx * gyy - gxy * gxy
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    l1, l2 = np.abs(tr / 2 + disc), np.abs(tr / 2 - disc)
    iso = (det > 0) & (np.minimum(l1, l2) >= p["ISO_MIN"] * np.maximum(l1, l2))
    return resp, thr, peak & iso


def centroid(gray, x, y, r, bright):
    h, w = gray.shape
    x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
    patch = gray[y0:y1, x0:x1].astype(np.float64)
    wgt = patch - patch.min() if bright else patch.max() - patch
    if wgt.sum() <= 0:
        return float(x), float(y)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((wgt * xx).sum() / wgt.sum()), float((wgt * yy).sum() / wgt.sum())


def observe_frame(gray, resp, peak, proj, in_front, p=PARAMS):
    """proj: N x 2 projections. Returns N x 2 centroids (NaN when absent).

    Association rule: among isotropic blob peaks inside the +/-WIN window,
    take the one nearest to the projection (nearest-neighbour gating)."""
    h, w = gray.shape
    W = p["WIN"]
    out = np.full((len(proj), 2), np.nan)
    for i, (x, y) in enumerate(proj):
        if not in_front[i] or not np.isfinite(x) or not np.isfinite(y):
            continue
        xi, yi = int(round(x)), int(round(y))
        if xi - W < p["MIN_EDGE"] or yi - W < p["MIN_EDGE"] or xi + W >= w - p["MIN_EDGE"] or yi + W >= h - p["MIN_EDGE"]:
            continue
        pk = peak[yi - W: yi + W + 1, xi - W: xi + W + 1]
        ys, xs = np.nonzero(pk)
        if len(xs) == 0:
            continue
        d = (xs - W + xi - x) ** 2 + (ys - W + yi - y) ** 2
        j = int(np.argmin(d))
        bx, by = xi - W + xs[j], yi - W + ys[j]
        out[i] = centroid(gray, bx, by, p["CENTROID_R"], resp[by, bx] > 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--c3d", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--markers", default="all", help="comma list of marker labels or 'all'")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cams = load_calib(root / "calibration/Calib.toml")
    M, labels, mrate = load_markers(Path(a.c3d))
    idx = list(range(len(labels))) if a.markers == "all" else [labels.index(s) for s in a.markers.split(",")]
    names = [labels[i] for i in idx]
    ratio = 2
    meta = dict(version=VERSION, params=PARAMS, trial=a.trial, c3d_sha256=sha256(Path(a.c3d)), markers=names, cameras={})
    for key, cam in cams.items():
        n = cam["name"]
        video = root / a.trial / "videos" / n / f"{n}.avi"
        cap = cv2.VideoCapture(str(video))
        T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        uv = np.full((T, len(idx), 2), np.nan)
        off = np.full((T, len(idx)), np.nan)
        R, _ = cv2.Rodrigues(cam["rvec"])
        for f in range(T):
            ok, fr = cap.read()
            if not ok:
                break
            mf = ratio * f
            if mf >= M.shape[0]:
                break
            X = M[mf, idx] * 1e-3
            pc = (R @ X.T + cam["tvec"]).T
            proj, _ = cv2.projectPoints(X.reshape(-1, 1, 3), cam["rvec"], cam["tvec"], cam["K"], cam["dist"])
            proj = proj.reshape(-1, 2)
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            resp, thr, peak = response(gray)
            uv[f] = observe_frame(gray, resp, peak, proj, pc[:, 2] > 0)
            off[f] = np.linalg.norm(uv[f] - proj, axis=1)
        cap.release()
        np.save(out / f"uv_{n}.npy", uv)
        det = np.isfinite(uv[:, :, 0])
        meta["cameras"][n] = dict(video=str(video), video_sha256=sha256(video), frames=int(T),
                                  detection_rate_per_marker={names[j]: float(det[:, j].mean()) for j in range(len(idx))},
                                  offset_px_median_per_marker={names[j]: (float(np.nanmedian(off[:, j])) if det[:, j].any() else None) for j in range(len(idx))},
                                  offset_px_overall=dict(median=float(np.nanmedian(off)), p90=float(np.nanpercentile(off, 90)),
                                                         frac_at_window_edge=float(np.nanmean(off >= PARAMS["WIN"] - 1))))
        print(f"cam {n}: detection {det.mean():.3f}, offset median {np.nanmedian(off):.2f} px, p90 {np.nanpercentile(off, 90):.2f}")
    (out / "observe_meta.json").write_text(json.dumps(meta, indent=1))
    # reference amplitude per marker: 4 s window path length (for target choice; development only)
    W = 240 * ratio
    nwin = M.shape[0] // W
    amp = {names[j]: [float(np.linalg.norm(np.diff(M[w * W:(w + 1) * W, idx[j]] * 1e-3, axis=0), axis=1).sum()) for w in range(nwin)] for j in range(len(idx))}
    (out / "reference_window_lengths_m.json").write_text(json.dumps(amp, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
