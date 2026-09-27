"""Post-hoc diagnosis: why does the closed form m* = T sqrt(kappa v^2/(7 sigma_a))
under-predict the bias crossing on DPJAIT (real family by ~25 %)?

Decomposes the mean signed error of the scored noise-knob runs into
  D(m) = mean reference-only polyline deficit (no noise at all), and
  I(m) = mean signed error - D(m)   (what the observation noise adds),
and compares each with its formula:
  D_formula(m) = T^3 kappa^2 v^3 / (24 m^2)   with the MEDIAN kappa, v used by eq. (1)
  D_jensen(m)  = T^3 <kappa^2 v^3> / (24 m^2) with the time-average inside each window
  I_formula(m) = 2 sigma_a^2 m^2 / L
Reads only sealed scored files and the reference; changes no verdict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpjait_noise_runner as v1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "mstar_underprediction_diag_2026-09-26"
EXPS = {"v0.1_blob": ROOT / "experiments" / "dpjait_noise_prereg_v0.1",
        "v0.2_yolo": ROOT / "experiments" / "dpjait_noise_prereg_v0.2"}
FAMS = {"real": v1.REAL, **{s: [s] for s in v1.SIM}}


def pointwise_kv(seg, dt):
    """Same smoothing/derivatives as v1.curvature_speed, but return the series."""
    seg = seg[~np.isnan(seg).any(axis=1)]
    ker = np.ones(5) / 5.0
    S = np.stack([np.convolve(seg[:, i], ker, mode="same") for i in range(3)], axis=1)
    d1 = np.gradient(S, dt, axis=0)
    d2 = np.gradient(d1, dt, axis=0)
    sp = np.linalg.norm(d1, axis=1)
    ok = sp ** 3 > 1e-9
    k = np.full(len(sp), np.nan)
    k[ok] = np.linalg.norm(np.cross(d1, d2), axis=1)[ok] / sp[ok] ** 3
    return k[2:-2], sp[2:-2]


def crossing(ms, y):
    for i in range(len(ms) - 1):
        if y[i] < 0 <= y[i + 1]:
            w = -y[i] / (y[i + 1] - y[i])
            return float(np.exp(np.log(ms[i]) + w * np.log(ms[i + 1] / ms[i])))
    return np.nan


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((EXPS["v0.2_yolo"] / "config.json").read_text())
    Tp = cfg["t_primary"]; Ts = Tp / cfg["video_fps"]; grid = np.array(cfg["grid_primary"])
    report = {}
    for fam, names in FAMS.items():
        # windows actually scored (same set in both chains)
        wins = []
        for n in names:
            d = json.loads((EXPS["v0.2_yolo"] / "scored" / (n + ".json")).read_text())
            wins += sorted({(n, r["drone"], r["window"]) for r in d["rows"]})
        recs = {n: v1.open_record(n) for n in names}
        Dw, kin = [], []
        for n, dr, w in wins:
            rec, sim = recs[n]
            per = v1.ref_per_frame(sim)
            f0, f1 = w * Tp, (w + 1) * Tp
            L = rec.reference_path_length(dr, f0, f1)
            Dw.append([v1.polyline(np.asarray([rec.reference_at(dr, f) for f in v1.instants(f0, Tp, m)])) - L
                       for m in grid])
            seg = np.asarray(rec.reference[dr][f0 * per: f1 * per + 1], float)
            k, s = pointwise_kv(seg, 1.0 / (cfg["video_fps"] * per))
            ok = np.isfinite(k) & np.isfinite(s)
            kin.append(dict(L=L, k_med=np.median(k[ok]), v=L / Ts,
                            k2v3=np.mean(k[ok] ** 2 * s[ok] ** 3), v_mean=np.mean(s[ok]),
                            k_mean=np.mean(k[ok]), k_p90=np.percentile(k[ok], 90)))
        Dw = np.array(Dw); D = Dw.mean(axis=0)
        kap = np.median([q["k_med"] for q in kin]); spd = np.median([q["v"] for q in kin])
        k2v3 = np.mean([q["k2v3"] for q in kin]); Lm = np.mean([q["L"] for q in kin])
        invL = np.mean([1 / q["L"] for q in kin])
        D_formula = Ts ** 3 * kap ** 2 * spd ** 3 / (24 * grid ** 2)
        D_jensen = Ts ** 3 * k2v3 / (24 * grid ** 2)
        fam_rep = dict(n_windows=len(wins), kappa_median=kap, speed_median=spd,
                       mean_k2v3=k2v3, median_k2_v3=kap ** 2 * spd ** 3,
                       jensen_factor=k2v3 / (kap ** 2 * spd ** 3),
                       kappa_mean_over_median=float(np.mean([q["k_mean"] / q["k_med"] for q in kin])),
                       deficit_ratio_actual_over_formula={int(m): float(-D[i] / D_formula[i]) for i, m in enumerate(grid) if m <= 50},
                       deficit_ratio_actual_over_jensen={int(m): float(-D[i] / D_jensen[i]) for i, m in enumerate(grid) if m <= 50},
                       chains={})
        for ver, exp in EXPS.items():
            dec = json.loads((exp / "decision.json").read_text())["result"]["families"][fam]
            lv = [l for l in dec["levels"] if l.get("ladder", "A") == "A"]
            rows = []
            for n in names:
                d = json.loads((exp / "scored" / (n + ".json")).read_text())
                rows += [r for r in d["rows"] if r.get("T", Tp) == Tp and r.get("ladder", "A") == "A"
                         and r["err"] is not None]
            levels = []
            for li, l in enumerate(lv):
                sig = l["sigma_a_mm"] / 1000
                S = np.array([np.mean([r["err"] for r in rows if r["level"] == li and r["m"] == m]) for m in grid])
                I = S - D
                I_formula = 2 * sig ** 2 * grid ** 2 * invL
                sel = (grid >= 8) & (grid <= 100)
                # effective sigma that the observed inflation implies (least squares on I = c m^2)
                c_obs = np.sum(I[sel] * grid[sel] ** 2) / np.sum(grid[sel] ** 4)
                sig_eff = np.sqrt(max(c_obs, 0) / (2 * invL))
                x_formula = Ts * np.sqrt(kap * spd ** 2 / (7 * sig))
                x_true_D = (Ts ** 3 * k2v3 / (48 * sig ** 2 * invL)) ** 0.25          # Jensen deficit, formula noise
                x_true_I = (Ts ** 3 * kap ** 2 * spd ** 3 / (48 * sig_eff ** 2 * invL)) ** 0.25  # formula deficit, observed noise
                x_both = (Ts ** 3 * k2v3 / (48 * sig_eff ** 2 * invL)) ** 0.25
                levels.append(dict(sigma_a_mm=l["sigma_a_mm"], m_star=l["m_star"], crossing=l["crossing"],
                                   crossing_from_components=crossing(grid, S),
                                   pred_eq1=x_formula, pred_jensen_deficit=x_true_D,
                                   pred_observed_inflation=x_true_I, pred_both=x_both,
                                   sigma_eff_from_inflation_mm=sig_eff * 1000,
                                   inflation_ratio_obs_over_formula=float(c_obs / (2 * sig ** 2 * invL))))
            fam_rep["chains"][ver] = levels
        report[fam] = fam_rep
        print("\n==", fam, "windows", len(wins), "jensen factor <k2v3>/(k_med^2 v_med^3) = %.2f" % fam_rep["jensen_factor"],
              "-> m* factor %.2f" % fam_rep["jensen_factor"] ** 0.25)
        print("   deficit actual/formula(median):", {m: round(v, 2) for m, v in fam_rep["deficit_ratio_actual_over_formula"].items() if m in (6, 10, 17, 28, 50)})
        print("   deficit actual/jensen        :", {m: round(v, 2) for m, v in fam_rep["deficit_ratio_actual_over_jensen"].items() if m in (6, 10, 17, 28, 50)})
        for ver, levels in fam_rep["chains"].items():
            for q in levels:
                print("   %s sig %6.1f cross %5.1f | eq1 %5.1f  jensenD %5.1f  obsI %5.1f  both %5.1f | infl obs/formula %.2f" % (
                    ver, q["sigma_a_mm"], q["crossing"] or np.nan, q["pred_eq1"], q["pred_jensen_deficit"],
                    q["pred_observed_inflation"], q["pred_both"], q["inflation_ratio_obs_over_formula"]))
    (OUT / "diagnosis.json").write_text(json.dumps(report, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
