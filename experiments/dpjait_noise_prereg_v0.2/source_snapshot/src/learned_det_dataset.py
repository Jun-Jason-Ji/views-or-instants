"""Build a YOLO training set for a drone detector from DPJAIT DEVELOPMENT records only.

Labels are the dataset's published YOLOv5 boxes (dl_data), so the detector we
train learns to reproduce the publishers' detector under an architecture and an
inference pipeline we control; it is not trained on any reference trajectory.
Only records already exposed during development are used:

    real  R01_D2, R02_D1, R03_D1, R12_D4, R13_D3
    sim   S05_D8, S07_D8

Validation split is by record (R03_D1 and S05_D8), never by frame.

    python src/learned_det_dataset.py --out experiments/dpjait_learned_det/dataset
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_external/dpjait_2026-09-19"
TRAIN = {"Real_Data": ["R01_D2", "R02_D1", "R12_D4", "R13_D3"], "Simulated_Data": ["S07_D8"]}
VAL = {"Real_Data": ["R03_D1"], "Simulated_Data": ["S05_D8"]}
STRIDE = 10          # every 10th frame
MIN_CONF = 0.5       # published boxes below this confidence are not used as labels


def read_boxes(csv_path: Path) -> dict:
    """frame -> list of (x, y, w, h) from a dl_data file."""
    out = {}
    with open(csv_path, newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            f = int(float(row[0]))
            vals = [float(v) for v in row[1:] if v.strip() != ""]
            boxes = []
            for i in range(0, len(vals) - 6, 7):
                x, y, w, h, _, _, conf = vals[i:i + 7]
                if conf >= MIN_CONF and w > 1 and h > 1:
                    boxes.append((x, y, w, h))
            out[f] = boxes
    return out


def video_for(folder: Path, stem: str) -> Path | None:
    for name in (stem + ".avi", stem.replace("i_s", "") + ".avi"):
        p = folder / name
        if p.exists():
            return p
    return None


def export(split: str, groups: dict, out: Path, stats: dict) -> None:
    img_dir, lab_dir = out / "images" / split, out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for sub, records in groups.items():
        for rec in records:
            folder = DATA / sub / rec
            for csv_path in sorted((folder / "dl_data").glob("*.csv")):
                stem = csv_path.stem
                vid = video_for(folder, stem)
                if vid is None:
                    stats.setdefault("missing_video", []).append(str(csv_path))
                    continue
                boxes = read_boxes(csv_path)
                cap = cv2.VideoCapture(str(vid))
                f = 0
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    if f % STRIDE == 0 and f in boxes:
                        h, w = fr.shape[:2]
                        name = "%s_%s_%06d" % (rec, stem, f)
                        cv2.imwrite(str(img_dir / (name + ".jpg")), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        with open(lab_dir / (name + ".txt"), "w") as fh:
                            for x, y, bw, bh in boxes[f]:
                                fh.write("0 %.6f %.6f %.6f %.6f\n" % ((x + bw / 2) / w, (y + bh / 2) / h,
                                                                     bw / w, bh / h))
                        stats[split] = stats.get(split, 0) + 1
                        stats[split + "_boxes"] = stats.get(split + "_boxes", 0) + len(boxes[f])
                    f += 1
                cap.release()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT / a.out
    if out.exists():
        raise SystemExit("refusing to overwrite %s" % out)
    stats = {}
    export("train", TRAIN, out, stats)
    export("val", VAL, out, stats)
    (out / "data.yaml").write_text(
        "path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: drone\n" % out.as_posix(),
        encoding="utf-8")
    (out / "dataset_info.json").write_text(json.dumps(dict(train=TRAIN, val=VAL, stride=STRIDE,
                                                           min_conf=MIN_CONF, **stats), indent=1))
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
