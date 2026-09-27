"""Post-hoc observation-noise diagnostic for the DPJAIT simulated confirmation records.

NOT part of the frozen preregistration chain (protocol/DPJAIT_PREREG_v0.1.md): it
reads no sealed prediction and changes no decision.  It exists to answer one
question raised by the confirmation result -- why the Goldilocks point m* differs
by a factor of four across three records of the *same* 8-camera geometry.

It measures, per record, the per-instant observation noise (2-D reprojection
residual of the YOLO box centre against the projected reference, and the 3-D error
of the k=8 triangulation), plus the depth and drone speed that could otherwise
explain such a difference.

    python src/dpjait_sim_obs_noise.py [--out experiments/dpjait_prereg_v0.1/sim_obs_noise.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dpjait_identity import assign_annotated  # noqa: E402
from dpjait_record import project  # noqa: E402
from dpjait_sim_record import SimRecord  # noqa: E402
from mcalib_cache_observer import triangulate  # noqa: E402

SIM_DIR = ROOT / "data_external/dpjait_2026-09-19/Simulated_Data"
RECORDS = ["S08_D8", "S09_D6", "S10_D6"]
W = 100          # window length in video frames (protocol section 4)
STRIDE = 10      # every 10th frame: this is a noise estimate, not a measurement
GATE_PX = 60.0   # the annotated arm's identity gate (protocol section 2)


def diagnose(name: str) -> dict:
    rec = SimRecord(SIM_DIR / name)
    cams = rec.cam_ids
    res_px, err3d, depth_m, speed = [], [], [], []
    for w in range(rec.n_video_frames // W):
        for dr in sorted(rec.reference):
            for f in range(w * W, min((w + 1) * W + 1, rec.n_video_frames), STRIDE):
                obs = assign_annotated(rec, dr, f, cams, GATE_PX)
                X = rec.reference_at(dr, f)
                if np.isnan(X).any() or len(obs) < 2:
                    continue
                for c, uv in obs.items():
                    uv_ref, d = project(rec.cameras[c], X)
                    if d[0] > 0:
                        res_px.append(float(np.linalg.norm(np.asarray(uv) - uv_ref[0])))
                        depth_m.append(float(d[0]))
                try:
                    Y = triangulate(obs, rec.cameras)
                except ValueError:
                    continue
                err3d.append(float(np.linalg.norm(np.asarray(Y).reshape(3) - X)) * 1000)
            ref = rec.reference[dr][w * W:(w + 1) * W + 1]
            ref = ref[~np.isnan(ref).any(axis=1)]
            if len(ref) > 1:
                speed.append(float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum()) / (W / 25.0))
    q = lambda v, p: float(np.percentile(v, p))  # noqa: E731
    return dict(record=name, n_residuals=len(res_px), n_instants=len(err3d),
                residual_px_median=q(res_px, 50), residual_px_p90=q(res_px, 90),
                err3d_mm_median=q(err3d, 50), err3d_mm_p90=q(err3d, 90),
                depth_m_median=q(depth_m, 50), speed_ms_median=q(speed, 50))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/dpjait_prereg_v0.1/sim_obs_noise.json")
    a = ap.parse_args()
    rows = [diagnose(n) for n in RECORDS]
    for r in rows:
        print("%s  residual px %.2f / %.2f | 3D err mm %.1f / %.1f | depth %.1f m | speed %.2f m/s"
              % (r["record"], r["residual_px_median"], r["residual_px_p90"],
                 r["err3d_mm_median"], r["err3d_mm_p90"],
                 r["depth_m_median"], r["speed_ms_median"]))
    out = ROOT / a.out
    out.write_text(json.dumps(dict(note="post-hoc diagnostic, not part of the frozen chain",
                                   stride_frames=STRIDE, gate_px=GATE_PX, records=rows),
                              indent=1), encoding="utf-8")
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
