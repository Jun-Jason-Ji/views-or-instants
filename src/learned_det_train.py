"""GPU job 1: fine-tune YOLO11n on the DPJAIT development-record dataset.

Resumable: if train_done.json exists the job returns immediately.  Runs from the
weights directory so that ultralytics' AMP self-check finds yolo11n.pt locally
and needs no network.

    .venvs/gpu_det/Scripts/python src/learned_det_train.py
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/dpjait_learned_det"
WEIGHTS = BASE / "weights"
DATASET = BASE / "dataset"
PROJECT = BASE / "train"
DONE = BASE / "train_done.json"
CFG = dict(model="yolo11n.pt", imgsz=1280, epochs=60, patience=15, batch=8, workers=4,
           seed=20460926, deterministic=True, plots=False, amp=True, device=0, name="run")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if DONE.exists():
        print("already trained:", json.loads(DONE.read_text())["best"])
        return 0
    from ultralytics import YOLO
    os.chdir(WEIGHTS)
    t0 = time.time()
    model = YOLO(str(WEIGHTS / CFG["model"]))
    res = model.train(data=str(DATASET / "data.yaml"), project=str(PROJECT), exist_ok=True,
                      **{k: v for k, v in CFG.items() if k != "model"})
    best = PROJECT / CFG["name"] / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit("training finished without best.pt")
    metrics = {}
    try:
        metrics = {k: float(v) for k, v in res.results_dict.items()}
    except Exception:
        pass
    DONE.write_text(json.dumps(dict(best=str(best), sha256=sha(best), cfg=CFG, metrics=metrics,
                                    seconds=round(time.time() - t0)), indent=1))
    print("trained:", best, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
