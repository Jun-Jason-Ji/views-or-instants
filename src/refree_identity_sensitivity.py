"""Post-hoc sensitivity: does the reference-projection identity binding matter?

Every noise-knob run binds, per camera and frame, the detection nearest to the
projected reference (within 60 px).  For single-target records identity is
trivial, so the binding can be made reference-free:
  - per camera and frame take the highest-confidence YOLO detection;
  - triangulate with all cameras; if any camera reprojects more than 60 px from
    the triangulated point, drop the worst camera and triangulate once more.
The same injected noise draws as DPJAIT_NOISE_PREREG_v0.2 are used (same seeds),
so the comparison with the reference-bound chain is paired.  Outputs, per noise
level on the eight single-target real records: m* of each chain (from the
reference, for scoring only), and the M3 prediction of each chain (never from the
reference).  Exposed data: a sensitivity analysis, not a test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpjait_noise_runner as v1  # noqa: E402
import refree_mstar_dev as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "dpjait_noise_prereg_v0.2"
OUT = ROOT / "experiments" / "refree_identity_sensitivity_2026-09-26"
SEED, PX, REPS, GATE = 20460926, [0, 4, 8, 16, 32, 64], 4, 60.0


def bind_free(name, cams, n):
    z = np.load(EXP / "observations" / (name + ".npz"))
    uv = np.full((n, 1, len(cams), 2), np.nan)
    for j, c in enumerate(cams):
        arr, conf = z["cam_" + c], z["cam_" + c + "_conf"]
        best = {}
        for (f, _s, cx, cy, _a), q in zip(arr, conf):
            f = int(f)
            if f < n and (f not in best or q > best[f][0]):
                best[f] = (q, cx, cy)
        for f, (_q, cx, cy) in best.items():
            uv[f, 0, j] = (cx, cy)
    return uv


def reproj(mats, X, uvs):
    out = []
    for (K, D, P34), u in zip(mats, uvs):
        rvec, _ = cv2.Rodrigues(P34[:, :3])
        p, _ = cv2.projectPoints(X.reshape(1, 1, 3), rvec, P34[:, 3].reshape(3, 1), K, D)
        out.append(np.linalg.norm(p.ravel() - u))
    return np.array(out)


def triangulate_gated(uv, mats):
    """uv (n, k, 2) -> (n, 3); one reference-free drop of the worst camera over GATE px."""
    P = v1.triangulate_dense(uv, mats)
    for i in np.nonzero(np.isfinite(P).all(axis=1))[0]:
        use = np.nonzero(~np.isnan(uv[i, :, 0]))[0]
        if len(use) < 3:
            continue
        e = reproj([mats[j] for j in use], P[i], uv[i, use])
        if e.max() > GATE:
            keep = use[np.arange(len(use)) != int(np.argmax(e))]
            P[i] = v1.triangulate_dense(uv[i:i + 1, keep], [mats[j] for j in keep])[0]
    return P


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = R.GRID
    per_level = {li: dict(Pw_free=[], Pw_bound=[], err_free={}, err_bound={}) for li in range(len(PX))}
    for name in v1.REAL:
        rec, _ = v1.open_record(name)
        sc = json.loads((EXP / "scored" / (name + ".json")).read_text())
        n = sc["n_frames"]
        mats = [v1.cam_matrices(rec)[c] for c in rec.cam_ids]
        rec_idx = (v1.REAL + v1.SIM).index(name)
        refs = {r["window"]: r["reference_m"] for r in sc["rows"]}
        wins = sorted(refs)
        free = bind_free(name, rec.cam_ids, n)
        dz = np.load(EXP / "predictions" / (name + "_dense.npz"))
        for li, px in enumerate(PX):
            for rep in range(1 if px == 0 else REPS):
                uv = free.copy()
                if px > 0:
                    # same draw as the frozen v0.2 run: shape (n, n_drones=1, n_cams, 2)
                    uv = uv + np.random.default_rng([SEED, rec_idx, li, rep]).normal(0.0, float(px), uv.shape)
                Pf = triangulate_gated(uv[:, 0], mats)
                Pb = dz["A|%d|%d|drone" % (li, rep)].astype(float)
                for tag, P in (("free", Pf), ("bound", Pb)):
                    per_level[li]["Pw_" + tag] += R.fill_gaps_list(P, wins, n) if hasattr(R, "fill_gaps_list") else \
                        [x for x in (R.fill_gaps(P[w * R.TP: w * R.TP + R.TP + 1]) for w in wins
                                     if w * R.TP + R.TP + 1 <= n) if x is not None]
                    for w in wins:
                        for m in grid:
                            pts = P[v1.instants(w * R.TP, R.TP, int(m))]
                            ok = ~np.isnan(pts).any(axis=1)
                            if ok[0] and ok[-1] and ok.sum() >= 2 and np.isfinite(refs[w]):
                                per_level[li]["err_" + tag].setdefault(int(m), []).append(v1.polyline(pts[ok]) - refs[w])
        print(name, "done", flush=True)
    rows = []
    for li, d in per_level.items():
        row = dict(level=li, px=PX[li])
        for tag in ("free", "bound"):
            mae = {m: float(np.mean(np.abs(v))) for m, v in d["err_" + tag].items()}
            mstar = min(mae, key=mae.get)
            sig = float(np.sqrt(np.mean([R.sigma_white(P) ** 2 for P in d["Pw_" + tag]])))
            Ls = np.mean([[v1.polyline(P[v1.instants(0, R.TP, int(m))]) for m in grid] for P in d["Pw_" + tag]], axis=0)
            Is = np.mean([[R.inflation_exact(P, m, sig) for m in grid] for P in d["Pw_" + tag]], axis=0)
            m3, _ = R.m3_predict(Ls, Is, 6, 100)
            row.update({"m_star_" + tag: mstar, "M3_" + tag: m3, "sigma_white_mm_" + tag: sig * 1e3,
                        "n_windows_" + tag: len(d["Pw_" + tag])})
        rows.append(row)
        print("L%d %2dpx | m* bound %3d free %3d | M3 bound %6.1f free %6.1f | ratio free %.2f" % (
            li, row["px"], row["m_star_bound"], row["m_star_free"], row["M3_bound"], row["M3_free"],
            row["M3_free"] / row["m_star_free"]), flush=True)
    (OUT / "identity_sensitivity.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
