"""Our own 2-D observation chain for DPJAIT video: static-background blobs.

Replaces the published YOLOv5 box centres with an observation we make ourselves
from the raw video, at several image resolutions, so that the per-instant noise
can be turned as a knob (protocol/DPJAIT_NOISE_PREREG_v0.1.md, section 2).

For each camera: a per-pixel median background over frames sampled across the
whole video (the cameras are static, the targets move, so the targets vanish
from the median); then, per frame and per downsampling factor s, the absolute
difference to the equally downsampled background is thresholded, opened, split
into connected components, and the K largest blobs are kept with their
intensity-weighted centroids mapped back to full-resolution pixels.

Nothing here reads the reference trajectory.  Identity (which blob is which
drone) is decided later, in the runner, exactly as the published-box annotated
arm does it.

    python src/own_blob_observer.py --record <folder> --out <npz> [--sim]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

SCALES = (1, 2, 4, 8)
BG_SAMPLES = 60          # frames spread over the whole video for the median background
THRESH = {1: 25.0, 2: 20.0, 4: 15.0, 8: 12.0}   # grey-level difference, per scale
MIN_AREA = {1: 30, 2: 10, 4: 4, 8: 2}           # pixels at that scale
TOP_K = 16               # blobs kept per frame and scale
# closing kernel per scale: merges the fragments a thin-armed drone breaks into
# after thresholding at full resolution (set on dev records R02_D1, S07_D8)
CLOSE = {1: 9, 2: 5, 4: 3, 8: 0}


def video_path(folder: Path, cam: str) -> Path:
    for name in (cam + ".avi", cam + "i_s.avi"):
        p = folder / name
        if p.exists():
            return p
    raise FileNotFoundError("no video for camera %s in %s" % (cam, folder))


def background(path: Path, n_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n_frames
    idx = np.linspace(0, max(total - 1, 0), BG_SAMPLES).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise RuntimeError("could not read %s" % path)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def blobs(gray: np.ndarray, bg: np.ndarray, s: int):
    """K largest blobs at scale s: (cx, cy, area) with centroids in FULL-res px."""
    if s > 1:
        h, w = gray.shape
        g = cv2.resize(gray, (w // s, h // s), interpolation=cv2.INTER_AREA)
        b = cv2.resize(bg, (w // s, h // s), interpolation=cv2.INTER_AREA)
    else:
        g, b = gray, bg
    diff = cv2.absdiff(g, b)
    mask = (diff > THRESH[s]).astype(np.uint8)
    if s <= 2:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if CLOSE[s]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((CLOSE[s], CLOSE[s]), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    if n <= 1:
        return out
    order = 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA])
    d = diff.astype(np.float32)
    for j in order[:TOP_K]:
        area = int(stats[j, cv2.CC_STAT_AREA])
        if area < MIN_AREA[s]:
            break
        x0, y0 = stats[j, cv2.CC_STAT_LEFT], stats[j, cv2.CC_STAT_TOP]
        ww, hh = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
        sub = (lab[y0:y0 + hh, x0:x0 + ww] == j)
        wts = d[y0:y0 + hh, x0:x0 + ww] * sub
        tot = float(wts.sum())
        if tot <= 0:
            continue
        ys, xs = np.mgrid[y0:y0 + hh, x0:x0 + ww]
        cx = float((xs * wts).sum() / tot)
        cy = float((ys * wts).sum() / tot)
        # pixel (u) at scale s covers full-res [u*s, u*s + s): centre maps to (u+0.5)*s-0.5
        out.append(((cx + 0.5) * s - 0.5, (cy + 0.5) * s - 0.5, area))
    return out


def observe_camera(path: Path, n_frames: int, max_frames: int = 0):
    bg = background(path, n_frames)
    cap = cv2.VideoCapture(str(path))
    rows = []
    f = 0
    limit = min(n_frames, max_frames) if max_frames else n_frames
    while f < limit:
        ok, fr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        for s in SCALES:
            for cx, cy, area in blobs(gray, bg, s):
                rows.append((f, s, cx, cy, area))
        f += 1
    cap.release()
    arr = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5))
    return arr, f


def observe_record(folder: Path, sim: bool, out: Path, max_frames: int = 0) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if sim:
        from dpjait_sim_record import SimRecord
        rec = SimRecord(folder)
    else:
        from dpjait_record import Record
        rec = Record(folder)
    t0 = time.time()
    data, meta = {}, {"record": rec.name, "sim": sim, "cameras": rec.cam_ids,
                      "n_video_frames": int(rec.n_video_frames), "scales": list(SCALES),
                      "thresh": {str(k): v for k, v in THRESH.items()},
                      "min_area": {str(k): v for k, v in MIN_AREA.items()},
                      "bg_samples": BG_SAMPLES, "top_k": TOP_K,
                      "close": {str(k): v for k, v in CLOSE.items()}, "frames_read": {}}
    for cam in rec.cam_ids:
        arr, nread = observe_camera(video_path(folder, cam), rec.n_video_frames, max_frames)
        data[cam] = arr
        meta["frames_read"][cam] = nread
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{"cam_" + c: a for c, a in data.items()},
                        meta=json.dumps(meta))
    meta["seconds"] = round(time.time() - t0, 1)
    return meta


def load(npz: Path):
    """-> (meta, {cam: {scale: {frame: array (n,3) of cx, cy, area}}})"""
    z = np.load(npz, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    obs = {}
    for cam in meta["cameras"]:
        arr = z["cam_" + cam]
        per = {s: {} for s in meta["scales"]}
        for f, s, cx, cy, a in arr:
            per[int(s)].setdefault(int(f), []).append((cx, cy, a))
        obs[cam] = {s: {f: np.asarray(v) for f, v in d.items()} for s, d in per.items()}
    return meta, obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="development only")
    a = ap.parse_args()
    meta = observe_record(Path(a.record), a.sim, Path(a.out), a.max_frames)
    print(json.dumps({k: meta[k] for k in ("record", "frames_read", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
