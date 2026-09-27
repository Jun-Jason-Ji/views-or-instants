"""Loader that puts a DPJAIT record into the frozen measurement chain's format.

Supplies three things, in the units the chain already expects (metres):

* `cameras`  -- {cam_id: {K, D, R, t_m}}, using the convention established in
  reports/DPJAIT_AUDIT_2026-09-19.md: `cam_or_*` is a Rodrigues vector for the
  world->camera rotation and `cam_x/y/z[mm]` is the translation vector t, NOT
  the camera centre.
* `reference` -- per drone, the Vicon cross-marker centroid at 100 Hz (m).
* `detections` -- per camera, the published YOLOv5 box centres per video frame.

Reference and detections keep separate clocks: reference frame = 4 x video
frame (verified in the audit).  Nothing here selects, gates or assigns; the
identity assignment is the experiment's subject and lives in the runner.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import cv2
import ezc3d
import numpy as np

REF_PER_VIDEO_FRAME = 4          # Vicon 100 Hz / video 25 fps
VIDEO_FPS = 25.0


def load_cameras(path: Path) -> dict:
    """cameras[cam_id] = {K, D, R, t_m}; translation in metres."""
    cams = {}
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    for row in rows:
        unit_m = "cam_x[m]" in row
        px = float(row["cam_x[m]"] if unit_m else row["cam_x[mm]"])
        py = float(row["cam_y[m]"] if unit_m else row["cam_y[mm]"])
        pz = float(row["cam_z[m]"] if unit_m else row["cam_z[mm]"])
        scale = 1.0 if unit_m else 1e-3
        rvec = np.array([float(row["cam_or_x[rad]"]), float(row["cam_or_y[rad]"]),
                         float(row["cam_or_z[rad]"])]).reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        K = np.eye(3)
        K[0, 0] = K[1, 1] = float(row["focal_length"])
        K[0, 2] = float(row["cx"])
        K[1, 2] = float(row["cy"])
        D = np.array([float(row.get("coeff_%d" % i, 0.0) or 0.0) for i in (1, 2, 3, 4)])
        cams[row["cam_name"]] = {"K": K, "D": D, "R": R,
                                 "t_m": np.array([px, py, pz]) * scale}
    return cams


def load_reference(path: Path) -> dict:
    """drone label -> T x 3 array (m) of the cross-marker centroid, NaN where absent."""
    c = ezc3d.c3d(str(path))
    pts = c["data"]["points"][:3]
    labels = list(c["parameters"]["POINT"]["LABELS"]["value"])
    rate = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    if abs(rate - 100.0) > 1e-6:
        raise ValueError("unexpected reference rate %r" % rate)
    xyz = np.transpose(pts, (2, 1, 0)).astype(np.float64) * 1e-3   # T x N x 3, metres
    xyz[xyz == 0] = np.nan
    groups = defaultdict(list)
    for i, lab in enumerate(labels):
        # 'Dron4:Dron41' -> 'Dron4';  bare 'Dron1' in single-drone records -> one group
        groups[lab.split(":")[0] if ":" in lab else "drone"].append(i)
    return {g: np.nanmean(xyz[:, idx, :], axis=1) for g, idx in sorted(groups.items())}


def load_detections(folder: Path) -> dict:
    """cam_id -> {video_frame: K x 2 array of box centres}."""
    out = {}
    for p in sorted(folder.glob("*.csv")):
        per_frame = {}
        with open(p, newline="") as fh:
            for row in csv.reader(fh):
                vals = [v for v in row if v != ""]
                if not vals:
                    continue
                rest = vals[1:]
                cent = [(float(rest[i + 4]), float(rest[i + 5]))
                        for i in range(0, len(rest) - 6, 7)]
                if cent:
                    per_frame[int(vals[0])] = np.asarray(cent, dtype=np.float64)
        out[p.stem] = per_frame
    return out


def project(cam: dict, X: np.ndarray) -> tuple:
    """X: N x 3 metres -> (N x 2 pixels, N depths)."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    rvec, _ = cv2.Rodrigues(cam["R"])
    uv, _ = cv2.projectPoints(X.reshape(-1, 1, 3), rvec, cam["t_m"].reshape(3, 1),
                              cam["K"], cam["D"])
    depth = (cam["R"] @ X.T + cam["t_m"].reshape(3, 1)).T[:, 2]
    return uv.reshape(-1, 2), depth


class Record:
    """One DPJAIT record: calibration, per-drone reference, per-camera detections."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.name = self.folder.name
        self.cameras = load_cameras(self.folder / "cameras_calibration.csv")
        c3d = sorted(self.folder.glob("*.c3d"))
        if not c3d:
            raise ValueError("no .c3d in %s" % self.folder)
        self.c3d_path = c3d[0]
        self.reference = load_reference(self.c3d_path)
        self.detections = load_detections(self.folder / "dl_data")
        # keep only cameras that have both calibration and detections
        self.cam_ids = sorted(set(self.cameras) & set(self.detections))
        self.n_video_frames = min(
            (max(d) + 1) for d in self.detections.values() if d)
        ref_len = min(v.shape[0] for v in self.reference.values())
        self.n_video_frames = min(self.n_video_frames,
                                  ref_len // REF_PER_VIDEO_FRAME)

    @property
    def n_drones(self) -> int:
        m = re.search(r"_D(\d+)", self.name)
        return int(m.group(1)) if m else len(self.reference)

    def reference_at(self, drone: str, video_frame: int) -> np.ndarray:
        return self.reference[drone][video_frame * REF_PER_VIDEO_FRAME]

    def reference_path_length(self, drone: str, f0: int, f1: int) -> float:
        """Polyline length (m) of the 100 Hz reference between two video frames."""
        seg = self.reference[drone][f0 * REF_PER_VIDEO_FRAME: f1 * REF_PER_VIDEO_FRAME + 1]
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
            "video_frames": int(self.n_video_frames),
            "duration_s": round(self.n_video_frames / VIDEO_FPS, 2),
            "detections_per_frame_median": {
                c: float(np.median([len(v) for v in self.detections[c].values()]))
                for c in self.cam_ids},
        }
