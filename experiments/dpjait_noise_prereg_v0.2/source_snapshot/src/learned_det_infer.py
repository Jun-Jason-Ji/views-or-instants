"""GPU job 2: run the trained detector on every frame of one record.

Writes an observation cache in exactly the format own_blob_observer.load()
reads (key "cam_<id>" -> rows (frame, scale=1, cx, cy, area) in full-resolution
pixels; "meta" JSON), plus "cam_<id>_conf" with the detection confidences.  It
never reads the reference trajectory.  Resumable: an existing output is skipped.

    .venvs/gpu_det/Scripts/python src/learned_det_infer.py --record R05_D1 \
        --out experiments/dpjait_noise_prereg_v0.2/observations
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data_external/dpjait_2026-09-19"
DONE = ROOT / "experiments/dpjait_learned_det/train_done.json"
IMGSZ, CONF, BATCH = 1280, 0.25, 16


def open_record(name: str):
    if name.startswith("S"):
        from dpjait_sim_record import SimRecord
        return SimRecord(DATA / "Simulated_Data" / name), True
    from dpjait_record import Record
    return Record(DATA / "Real_Data" / name), False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT / a.out / (a.record + ".npz")
    if out.exists():
        print("exists, skipping:", out)
        return 0
    info = json.loads(DONE.read_text())
    from ultralytics import YOLO
    from own_blob_observer import video_path
    model = YOLO(info["best"])
    rec, sim = open_record(a.record)
    t0 = time.time()
    arrays, meta = {}, dict(record=rec.name, sim=sim, cameras=rec.cam_ids, scales=[1],
                            n_video_frames=int(rec.n_video_frames), detector=info["best"],
                            detector_sha256=info["sha256"], imgsz=IMGSZ, conf=CONF, frames_read={})
    for cam in rec.cam_ids:
        cap = cv2.VideoCapture(str(video_path(rec.folder, cam)))
        rows, confs, f, buf, idx = [], [], 0, [], []

        def flush():
            if not buf:
                return
            for fi, r in zip(idx, model.predict(buf, imgsz=IMGSZ, conf=CONF, device=0, half=True,
                                                verbose=False)):
                b = r.boxes
                if b is None or len(b) == 0:
                    continue
                xywh = b.xywh.cpu().numpy()
                for (cx, cy, w, h), c in zip(xywh, b.conf.cpu().numpy()):
                    rows.append((fi, 1, float(cx), float(cy), float(w * h)))
                    confs.append(float(c))
            buf.clear()
            idx.clear()

        while f < rec.n_video_frames:
            ok, fr = cap.read()
            if not ok:
                break
            buf.append(fr)
            idx.append(f)
            if len(buf) == BATCH:
                flush()
            f += 1
        flush()
        cap.release()
        arrays["cam_" + cam] = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5))
        arrays["cam_" + cam + "_conf"] = np.asarray(confs, dtype=np.float32)
        meta["frames_read"][cam] = f
    meta["seconds"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)
    print("%s: %s in %.0f s" % (a.record, meta["frames_read"], meta["seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
