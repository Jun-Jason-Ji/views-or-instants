"""Post-hoc pilot-transfer test of the reference-free predictor (M3), with the pilot's cost counted.

For each confirmation record of REFREE_MSTAR_PREREG_v0.1 (R04_D2, R14_D3) and each noise level:
the first third of the windows (per target, in time order) is the pilot; M3 is fitted on the
pilot's dense reconstructions only and its prediction is rounded to the nearest grid density.
The remaining windows are the measurement windows: their error at the predicted m is compared
with reading every frame (m = 401 frames, the dense polyline) and with the best grid m found
after the fact.  Frames: pilot windows cost every frame of every camera; measurement windows
cost k*m.  Uses the sealed prediction files; the reference is read only for scoring.

Usage: python src/pilot_transfer.py -> experiments/pilot_transfer_2026-09-26/pilot_transfer.json
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import dpjait_noise_runner as v1  # noqa: E402
import refree_mstar_dev as R  # noqa: E402

EXP = ROOT / "experiments" / "refree_mstar_prereg_v0.1"
OUT = ROOT / "experiments" / "pilot_transfer_2026-09-26"
RECS, PX, K, TP = ["R04_D2", "R14_D3"], [0, 4, 8, 16, 32, 64], 4, 400


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = [int(m) for m in R.GRID]
    cells = []
    for name in RECS:
        sc = json.loads((EXP / "scored" / (name + ".json")).read_text())
        dz = np.load(EXP / "predictions" / (name + "_dense.npz"))
        wins_by_drone = collections.defaultdict(set)
        for q in sc["rows"]:
            wins_by_drone[q["drone"]].add(q["window"])
        pilot = {dr: sorted(ws)[: max(1, len(ws) // 3)] for dr, ws in wins_by_drone.items()}
        for li, px in enumerate(PX):
            Pws = []
            for dr, ws in pilot.items():
                for rep in (range(4) if px else [0]):
                    P = dz["A|%d|%d|%s" % (li, rep, dr)].astype(float)
                    for w in ws:
                        f0 = w * TP
                        if f0 + TP + 1 <= len(P):
                            Pw = R.fill_gaps(P[f0:f0 + TP + 1])
                            if Pw is not None:
                                Pws.append(Pw)
            sig = float(np.sqrt(np.mean([R.sigma_white(P) ** 2 for P in Pws])))
            Ls = np.mean([[v1.polyline(P[v1.instants(0, TP, m)]) for m in grid] for P in Pws], axis=0)
            Is = np.mean([[R.inflation_exact(P, m, sig) for m in grid] for P in Pws], axis=0)
            pred, _ = R.m3_predict(Ls, Is, 6, 100)
            m_use = min(grid, key=lambda m: abs(np.log(m / pred))) if np.isfinite(pred) else grid[-1]
            test = [q for q in sc["rows"] if q["level"] == li and q["window"] not in pilot[q["drone"]] and q["err"] is not None]
            mae = collections.defaultdict(list)
            for q in test:
                mae[q["m"]].append(abs(q["err"]))
            mae = {m: 1000 * float(np.mean(v)) for m, v in mae.items()}
            n_pilot = sum(len(ws) for ws in pilot.values())
            n_test = len({(q["drone"], q["window"]) for q in test})
            m_best = min(mae, key=mae.get)
            full = mae[max(mae)]                      # every frame of every camera (m = 400 grid top)
            frames_full = n_test * K * (TP + 1)
            frames_plan = n_pilot * K * (TP + 1) + n_test * K * m_use
            cells.append(dict(record=name, noise_px=px, m_pred=round(pred, 1), m_used=m_use, m_best_after_fact=m_best,
                              mae_at_used_mm=round(mae[m_use], 1), mae_best_mm=round(mae[m_best], 1),
                              mae_every_frame_mm=round(full, 1), pilot_windows=n_pilot, measured_windows=n_test,
                              frames_every_frame=frames_full, frames_pilot_plus_plan=frames_plan,
                              frame_ratio=round(frames_full / frames_plan, 2),
                              breakeven_windows=round(n_pilot * (TP + 1) / ((TP + 1) - m_use), 2)))
            c = cells[-1]
            print("%s %2dpx  m_pred %6.1f -> %3d (best %3d) | MAE used %6.1f  best %6.1f  every-frame %7.1f mm | frames %d vs %d (x%.2f)"
                  % (name, px, pred, m_use, m_best, c["mae_at_used_mm"], c["mae_best_mm"], full, frames_plan, frames_full, c["frame_ratio"]))
    (OUT / "pilot_transfer.json").write_text(json.dumps(cells, indent=1))
    r = np.array([c["mae_at_used_mm"] / c["mae_every_frame_mm"] for c in cells])
    g = np.array([c["mae_at_used_mm"] / c["mae_best_mm"] - 1 for c in cells])
    print("error ratio used/every-frame: median %.2f, range %.2f-%.2f; better in %d of %d cells" % (np.median(r), r.min(), r.max(), (r < 1).sum(), len(r)))
    print("regret vs best: median %.0f%%, max %.0f%%;  frame ratio incl. pilot: %s" % (100 * np.median(g), 100 * g.max(), sorted({c['frame_ratio'] for c in cells})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
