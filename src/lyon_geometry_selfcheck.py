"""Geometry self-check for the LBMC Lyon example package.

Projects the 120 Hz C3D markers of participant_02/gait through Calib.toml
into each of the 9 rotated 60 Hz videos, and checks that the projections
land on bright marker blobs.  Reports, per camera:

* projection residual (median distance from projected marker to nearest
  bright blob) at the nominal time alignment (marker frame = 2 * video frame);
* the same residual over a small time-offset sweep, to confirm the sync;
* fraction of markers inside the image;
* an annotated PNG for eyeballing.

Nothing here is used for any measurement result; it only validates that the
calibration, the video rotation and the time base are mutually consistent.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np

try:
    import ezc3d
except ImportError:  # pragma: no cover
    ezc3d = None


def load_calib(path: Path) -> dict[str, dict]:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    cams = {}
    for key, val in raw.items():
        if not isinstance(val, dict) or "matrix" not in val:
            continue
        cams[key] = {
            "name": str(val.get("name", key)),
            "size": [int(v) for v in val["size"]],
            "K": np.asarray(val["matrix"], dtype=np.float64).reshape(3, 3),
            "dist": np.asarray(val.get("distortions", [0, 0, 0, 0]), dtype=np.float64).ravel(),
            "rvec": np.asarray(val["rotation"], dtype=np.float64).reshape(3, 1),
            "tvec": np.asarray(val["translation"], dtype=np.float64).reshape(3, 1),
            "fisheye": bool(val.get("fisheye", False)),
        }
    return cams


def load_markers(path: Path) -> tuple[np.ndarray, list[str], float]:
    if ezc3d is None:
        raise SystemExit("pip install ezc3d")
    c = ezc3d.c3d(str(path))
    pts = c["data"]["points"][:3]  # 3 x N x T, mm
    labels = list(c["parameters"]["POINT"]["LABELS"]["value"])
    rate = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    units = c["parameters"]["POINT"].get("UNITS", {}).get("value", ["mm"])[0]
    if units.lower() != "mm":
        raise SystemExit(f"unexpected marker units {units}")
    return np.transpose(pts, (2, 1, 0)).astype(np.float64), labels, rate  # T x N x 3


def project(cam: dict, xyz_mm: np.ndarray, scale: float) -> np.ndarray:
    """xyz_mm: N x 3 in the QTM frame.  scale converts mm to the calib unit."""
    obj = (xyz_mm * scale).reshape(-1, 1, 3)
    img, _ = cv2.projectPoints(obj, cam["rvec"], cam["tvec"], cam["K"], cam["dist"])
    return img.reshape(-1, 2)


def depth(cam: dict, xyz_mm: np.ndarray, scale: float) -> np.ndarray:
    R, _ = cv2.Rodrigues(cam["rvec"])
    pc = (R @ (xyz_mm * scale).T + cam["tvec"]).T
    return pc[:, 2]


def bright_blobs(gray: np.ndarray, thresh: int, min_area: int, max_area: int) -> np.ndarray:
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(bw, 8)
    keep = [i for i in range(1, n) if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]
    return cents[keep] if keep else np.zeros((0, 2))


def residual(proj: np.ndarray, blobs: np.ndarray, size: tuple[int, int]) -> tuple[float, float, int]:
    w, h = size
    inside = (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
    if inside.sum() == 0 or len(blobs) == 0:
        return float("nan"), float("nan"), int(inside.sum())
    p = proj[inside]
    d = np.linalg.norm(p[:, None, :] - blobs[None, :, :], axis=2).min(axis=1)
    return float(np.median(d)), float(np.mean(d < 6.0)), int(inside.sum())


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="extracted Example_Anonymized_Video_Data folder")
    ap.add_argument("--c3d", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="100,400,800", help="video frame indices to test")
    ap.add_argument("--offsets", default="-4,-2,-1,0,1,2,4", help="video-frame time offsets to sweep")
    ap.add_argument("--thresh", type=int, default=200)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    calib_files = list(root.rglob("Calib.toml"))
    if not calib_files:
        raise SystemExit("Calib.toml not found under root")
    cams = load_calib(calib_files[0])
    markers, labels, mrate = load_markers(Path(args.c3d))
    T = markers.shape[0]

    # decide unit of the calibration translation: mm or m
    t_norm = float(np.median([np.linalg.norm(c["tvec"]) for c in cams.values()]))
    scale = 1.0 if t_norm > 100 else 1e-3
    print(f"calibration translation median norm {t_norm:.3f} -> markers scaled by {scale}")

    videos = {}
    for cam_key, cam in cams.items():
        hits = [p for p in root.rglob("*.avi") if cam["name"] in p.name and "heckerboard" not in str(p)]
        hits = [p for p in hits if "Video_Data" in str(p) or "Trial" in str(p) or "gait" in str(p).lower()]
        if hits:
            videos[cam_key] = sorted(hits)[0]
    print(f"{len(videos)} of {len(cams)} cameras matched to a trial video")

    test_frames = [int(v) for v in args.frames.split(",")]
    offsets = [int(v) for v in args.offsets.split(",")]
    report = {"calib": str(calib_files[0]), "c3d": args.c3d, "scale_mm_to_calib": scale,
              "marker_rate_hz": mrate, "marker_frames": T, "cameras": {}}

    for cam_key, vpath in videos.items():
        cam = cams[cam_key]
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS)
        nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ratio = mrate / fps if fps > 0 else 2.0
        entry = {"video": str(vpath), "fps": fps, "video_frames": nfr, "video_size": [w, h],
                 "calib_size": cam["size"], "size_match": [w, h] == cam["size"],
                 "marker_to_video_ratio": ratio, "tests": []}
        for vf in test_frames:
            frame = read_frame(cap, vf)
            if frame is None:
                entry["tests"].append({"video_frame": vf, "error": "read failed"})
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blobs = bright_blobs(gray, args.thresh, 4, 400)
            sweep = {}
            for off in offsets:
                mf = int(round((vf + off) * ratio))
                if not 0 <= mf < T:
                    continue
                xyz = markers[mf]
                z = depth(cam, xyz, scale)
                proj = project(cam, xyz, scale)
                proj = proj[z > 0]
                med, frac, n_in = residual(proj, blobs, (w, h))
                sweep[str(off)] = {"median_px": med, "frac_within_6px": frac, "inside": n_in,
                                   "in_front": int((z > 0).sum())}
            entry["tests"].append({"video_frame": vf, "n_blobs": int(len(blobs)), "sweep": sweep})
            if vf == test_frames[0]:
                mf = int(round(vf * ratio))
                proj = project(cam, markers[mf], scale)
                vis = frame.copy()
                for (x, y) in proj:
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(vis, (int(round(x)), int(round(y))), 8, (0, 0, 255), 2)
                for (x, y) in blobs:
                    cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
                cv2.imwrite(str(out / f"{cam['name']}_frame{vf}.png"), vis)
        cap.release()
        report["cameras"][cam["name"]] = entry
        t0 = entry["tests"][0].get("sweep", {}).get("0", {})
        print(f"cam {cam['name']}: {w}x{h}@{fps:.1f} {nfr} frames, size_match={entry['size_match']}, "
              f"offset0 median {t0.get('median_px', float('nan')):.2f} px, within6px {t0.get('frac_within_6px', float('nan')):.2f}, "
              f"inside {t0.get('inside')}")

    with open(out / "geometry_selfcheck.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", out / "geometry_selfcheck.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
