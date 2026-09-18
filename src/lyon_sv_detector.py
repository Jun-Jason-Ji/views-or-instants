"""Per-frame, reference-free detector for the skull-vertex (SV) marker in the
LBMC Lyon videos.

Rule (identical for every camera and frame, no temporal state, no background
model, no reference):
  1. candidates = local maxima of a difference-of-Gaussians response
     (sigma DOG_S1 / DOG_S2) with response >= DOG_MIN, centre grey >= GREY_MIN
     and centre HSV saturation <= SAT_MAX (drops the green camera LEDs);
  2. blob test: both eigenvalues of the Hessian of the DOG_S1-smoothed image
     are negative and their ratio |l_min/l_max| >= ISO_MIN (drops line-like
     responses such as treadmill rails);
  3. cluster test: a candidate must have >= NB_MIN other candidates within
     NB_RADIUS px (the 48 body markers form a cluster; isolated ceiling or
     floor specks do not);
  4. SV = the surviving candidate with the smallest image row (topmost);
  5. sub-pixel position = intensity-weighted centroid of grey minus its local
     minimum inside a CENTROID_R window around the peak.

Only the CLI's --evaluate flag reads the reference C3D, and only to score the
detector on the public development slice.  The measurement runners never do.

History: v0.1 used a static-background foreground mask built from the
checkerboard videos; it failed (0-19 % hits) because the scene and exposure
differ between those videos and the trials.  v0.2 removes the background
model entirely.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PARAMS = dict(
    DOG_S1=1.8, DOG_S2=4.5, DOG_MIN=25.0, GREY_MIN=110, SAT_MAX=95,
    ISO_MIN=0.4, NB_RADIUS=260, NB_MIN=4, CENTROID_R=4,
)
VERSION = "lyon_sv_detector_v0.2"


def candidates(frame_bgr: np.ndarray, p=PARAMS) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    g = gray.astype(np.float32)
    g1 = cv2.GaussianBlur(g, (0, 0), p["DOG_S1"])
    resp = g1 - cv2.GaussianBlur(g, (0, 0), p["DOG_S2"])
    peak = (resp >= cv2.dilate(resp, np.ones((3, 3), np.uint8))) & (resp >= p["DOG_MIN"])
    peak &= (gray >= p["GREY_MIN"]) & (sat <= p["SAT_MAX"])
    ys, xs = np.nonzero(peak)
    if len(xs) == 0:
        return np.zeros((0, 3))
    # Hessian blob test on the sigma-1 smoothed image
    gxx = cv2.Sobel(g1, cv2.CV_32F, 2, 0, ksize=3)
    gyy = cv2.Sobel(g1, cv2.CV_32F, 0, 2, ksize=3)
    gxy = cv2.Sobel(g1, cv2.CV_32F, 1, 1, ksize=3)
    a, b, c = gxx[ys, xs], gxy[ys, xs], gyy[ys, xs]
    tr, det = a + c, a * c - b * b
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc  # l1 >= l2
    ok = (l1 < 0) & (np.abs(l1) >= p["ISO_MIN"] * np.abs(l2))
    xs, ys = xs[ok], ys[ok]
    if len(xs) == 0:
        return np.zeros((0, 3))
    pts = np.column_stack([xs, ys]).astype(np.float64)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    nb = (d <= p["NB_RADIUS"]).sum(axis=1) - 1
    keep = nb >= p["NB_MIN"]
    return np.column_stack([xs[keep], ys[keep], resp[ys[keep], xs[keep]]])


def refine(gray: np.ndarray, x: int, y: int, r: int) -> tuple[float, float]:
    h, w = gray.shape
    x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
    patch = gray[y0:y1, x0:x1].astype(np.float64)
    wgt = patch - patch.min()
    if wgt.sum() <= 0:
        return float(x), float(y)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((wgt * xx).sum() / wgt.sum()), float((wgt * yy).sum() / wgt.sum())


def detect_sv(frame_bgr: np.ndarray, p=PARAMS):
    """Return (u, v, n_candidates) or (nan, nan, n)."""
    c = candidates(frame_bgr, p)
    if len(c) == 0:
        return float("nan"), float("nan"), 0
    top = c[np.argmin(c[:, 1])]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    u, v = refine(gray, int(top[0]), int(top[1]), p["CENTROID_R"])
    return u, v, int(len(c))


def run_video(video: Path, p=PARAMS) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    uv, nc = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        u, v, n = detect_sv(fr, p)
        uv.append((u, v))
        nc.append(n)
    cap.release()
    return np.array(uv, dtype=np.float64), np.array(nc)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder holding calibration/Calib.toml and <trial>/videos/<cam>/<cam>.avi")
    ap.add_argument("--trial", default="participant_02/gait")
    ap.add_argument("--out", required=True)
    ap.add_argument("--evaluate", default=None, help="C3D reference: score detector on the DEVELOPMENT slice only")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lyon_geometry_selfcheck import load_calib, load_markers, project, depth

    cams = load_calib(root / "calibration/Calib.toml")
    result = {"version": VERSION, "params": PARAMS, "detector_sha256": sha256(Path(__file__)),
              "trial": args.trial, "cameras": {}}
    uv_all = {}
    for key, cam in cams.items():
        n = cam["name"]
        video = root / args.trial / "videos" / n / f"{n}.avi"
        uv, nc = run_video(video)
        uv_all[n] = uv
        np.save(out / f"sv_uv_{n}.npy", uv)
        result["cameras"][n] = {"video": str(video), "video_sha256": sha256(video), "frames": int(len(uv)),
                                "detected": int(np.isfinite(uv[:, 0]).sum()),
                                "median_candidates": float(np.median(nc))}
        print(f"cam {n}: {len(uv)} frames, detected {result['cameras'][n]['detected']}, median candidates {np.median(nc):.0f}")

    if args.evaluate:
        M, labels, _ = load_markers(Path(args.evaluate))
        sv = labels.index("SV")
        ev = {}
        for key, cam in cams.items():
            n = cam["name"]
            uv = uv_all[n]
            T = len(uv)
            ref = M[2 * np.arange(T), sv]
            z = depth(cam, ref, 1e-3)
            proj = project(cam, ref, 1e-3)
            det = np.isfinite(uv[:, 0])
            err = np.linalg.norm(uv - proj, axis=1)
            hit = det & (err <= 6.0) & (z > 0)
            wrong = det & (err > 6.0)
            ev[n] = {"frames": int(T), "detected": int(det.sum()), "hit_le6px": int(hit.sum()), "wrong_gt6px": int(wrong.sum()),
                     "missed": int((~det).sum()), "median_err_hits_px": float(np.median(err[hit])) if hit.any() else None,
                     "p95_err_hits_px": float(np.percentile(err[hit], 95)) if hit.any() else None,
                     "wrong_frames_first20": np.nonzero(wrong)[0][:20].tolist(),
                     "wrong_err_median_px": float(np.median(err[wrong])) if wrong.any() else None}
            print(f"  eval {n}: hit {hit.sum()}/{T}, wrong {wrong.sum()} (median {ev[n]['wrong_err_median_px']}), "
                  f"missed {(~det).sum()}, median err {ev[n]['median_err_hits_px']}")
        result["evaluation_on_development_slice"] = ev
    (out / "detector_run.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
