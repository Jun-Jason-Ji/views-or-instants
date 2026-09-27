"""Record loader for the DPJAIT simulated sequences (8-camera rig).

Same interface as `dpjait_record.Record`, so the chain and the comparison
runner work unchanged.  Four differences from the real half are handled here:

* calibration is in metres with no distortion coefficients (`load_cameras`
  detects both);
* a camera called `cam1` in the calibration has its detections in `cam1i_s.csv`;
* ground truth is `dronN_pos_25.csv` at 25 Hz -- the *same* rate as the video,
  so REF_PER_VIDEO_FRAME is 1 and the reference path length is itself a 25 Hz
  polyline.  The measurand is therefore "polyline length at the video rate",
  not the 100 Hz measurand of the real half, and the two must not be compared
  as if they were the same quantity;
* row `id=0` of the ground-truth files is a placeholder pose (0, 0, -0) rather
  than the drone's frame-0 position, so it is dropped.

The geometry convention is the one `dpjait_sim_selfcheck.py` established:
Rodrigues rotation, translation taken directly, no axis flip.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from dpjait_record import VIDEO_FPS, load_cameras, load_detections, project  # noqa: F401

REF_PER_VIDEO_FRAME = 1
PLACEHOLDER_ROW = 0


def load_positions(folder: Path) -> dict:
    """drone -> T x 3 (m); the placeholder first row is replaced by NaN."""
    out = {}
    for p in sorted(folder.glob("dron*_pos_25.csv")):
        rows = []
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh, delimiter=";"):
                rows.append([float(row["pos_x[m]"]), float(row["pos_y[m]"]),
                             float(row["pos_z[m]"])])
        arr = np.asarray(rows, dtype=np.float64)
        if len(arr):
            arr[PLACEHOLDER_ROW] = np.nan
        out[p.stem.replace("_pos_25", "")] = arr
    return out


class SimRecord:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.name = self.folder.name
        self.cameras = load_cameras(self.folder / "cameras_calibration.csv")
        self.reference = load_positions(self.folder)
        raw = load_detections(self.folder / "dl_data")
        # re-key detections onto the calibration's camera names
        self.detections = {}
        for cam_id in self.cameras:
            for stem, d in raw.items():
                if stem.startswith(cam_id) and not stem[len(cam_id):].isdigit():
                    self.detections[cam_id] = d
                    break
        self.cam_ids = sorted(set(self.cameras) & set(self.detections))
        n_det = min((max(d) + 1) for d in self.detections.values() if d)
        n_ref = min(v.shape[0] for v in self.reference.values())
        self.n_video_frames = int(min(n_det, n_ref))

    @property
    def n_drones(self) -> int:
        m = re.search(r"_D(\d+)", self.name)
        return int(m.group(1)) if m else len(self.reference)

    def reference_at(self, drone: str, video_frame: int) -> np.ndarray:
        return self.reference[drone][video_frame]

    def reference_path_length(self, drone: str, f0: int, f1: int) -> float:
        seg = self.reference[drone][f0: f1 + 1]
        ok = ~np.isnan(seg).any(axis=1)
        seg = seg[ok]
        if len(seg) < 2:
            return float("nan")
        return float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())

    def summary(self) -> dict:
        return {
            "record": self.name,
            "cameras": self.cam_ids,
            "n_drones_named": self.n_drones,
            "reference_groups": sorted(self.reference),
            "video_frames": self.n_video_frames,
            "duration_s": round(self.n_video_frames / VIDEO_FPS, 2),
            "detections_per_frame_median": {
                c: float(np.median([len(v) for v in self.detections[c].values()]))
                for c in self.cam_ids},
        }
