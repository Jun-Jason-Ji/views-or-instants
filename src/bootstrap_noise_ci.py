"""Post-hoc window-bootstrap confidence intervals for the DPJAIT noise-knob
experiments (v0.1 blob chain, v0.2 learned-detector chain).

Does NOT change any pre-registered verdict: it reads the sealed, scored files
and the frozen decisions, reproduces the point estimates, then resamples
windows.  Resampling unit = (record, drone, window); a resampled window keeps
all its noise levels, replicates and densities (paired design).  sigma_a and
the correlation C are held at their decision values (the scored files store
them aggregated only); curvature and speed medians are resampled with the
windows.

Usage: python src/bootstrap_noise_ci.py [--B 2000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPS = {
    "v0.1_blob": ROOT / "experiments" / "dpjait_noise_prereg_v0.1",
    "v0.2_yolo": ROOT / "experiments" / "dpjait_noise_prereg_v0.2",
}
OUT = ROOT / "experiments" / "noise_bootstrap_ci_2026-09-26"


def crossing(ms, sgn):
    for i in range(len(ms) - 1):
        a, b = sgn[i], sgn[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a < 0 <= b:
            w = -a / (b - a)
            return float(np.exp(np.log(ms[i]) + w * (np.log(ms[i + 1]) - np.log(ms[i]))))
    return np.nan


def interp_min(ms, mae):
    """Continuous m*: parabola in log m through the grid minimum and its neighbours."""
    ok = np.isfinite(mae)
    i = int(np.nanargmin(mae))
    if i == 0 or i == len(ms) - 1 or not (ok[i - 1] and ok[i + 1]):
        return float(ms[i])
    x = np.log(ms[i - 1:i + 2]); y = mae[i - 1:i + 2]
    a, b, _ = np.polyfit(x, y, 2)
    if a <= 0:
        return float(ms[i])
    return float(np.exp(np.clip(-b / (2 * a), x[0], x[2])))


def load_family(exp: Path, names, version):
    cfg = json.loads((exp / "config.json").read_text())
    grid = np.array(cfg["grid_primary"]); nlev = len(cfg["ladder_a_px"]); nrep = cfg["ladder_a_reps"]
    Tp = cfg["t_primary"]
    keys, kin, sig = {}, {}, np.zeros(nlev)
    rows_all = []
    for n in names:
        d = json.loads((exp / "scored" / (n + ".json")).read_text())
        for r in d["rows"]:
            if version == "v0.1_blob" and (r["T"] != Tp or r["ladder"] != "A"):
                continue
            rows_all.append((n, r))
        for k in d["kinematics"]:
            if version == "v0.1_blob" and k["T"] != Tp:
                continue
            kin[(n, k["drone"], k["window"])] = (k["kappa"], k["speed"])
    for n, r in rows_all:
        keys.setdefault((n, r["drone"], r["window"]), len(keys))
    W = len(keys)
    E = np.full((W, nlev, nrep, len(grid)), np.nan)
    mi = {m: j for j, m in enumerate(grid)}
    for n, r in rows_all:
        if r["err"] is not None:
            E[keys[(n, r["drone"], r["window"])], r["level"], r["rep"], mi[r["m"]]] = r["err"]
    K = np.array([kin.get(k, (np.nan, np.nan)) for k in keys])
    return cfg, grid, E, K


def stats(cfg, grid, E, K, sigma_m, C, idx):
    Es = E[idx]
    Tsec = cfg["t_primary"] / cfg["video_fps"]
    with np.errstate(all="ignore"):
        mae = np.nanmean(np.abs(Es), axis=(0, 2))        # (level, m)
        sgn = np.nanmean(Es, axis=(0, 2))
        kk = K[idx]; ok = np.isfinite(kk).all(axis=1)
        kap, spd = np.median(kk[ok, 0]), np.median(kk[ok, 1])
    out = []
    for li in range(E.shape[1]):
        j = int(np.nanargmin(mae[li]))
        ms = float(grid[j])
        pred0 = Tsec * np.sqrt(kap * spd ** 2 / (7 * sigma_m[li]))
        pred1 = Tsec * np.sqrt(kap * spd ** 2 / (7 * sigma_m[li] * np.sqrt(max(0.0, 1 - C[li])))) \
            if C is not None else np.nan
        out.append(dict(m_star=ms, m_star_cont=interp_min(grid, mae[li]),
                        crossing=crossing(grid, sgn[li]), pred0=pred0, pred1=pred1,
                        censored=ms in (grid[0], grid[-1])))
    unc = [l for l in range(len(out)) if not out[l]["censored"]]
    slope = float(np.polyfit(np.log(sigma_m[unc]), np.log([out[l]["m_star"] for l in unc]), 1)[0]) \
        if len(unc) >= cfg["h1_min_levels"] else np.nan
    slope_c = float(np.polyfit(np.log(sigma_m), np.log([o["m_star_cont"] for o in out]), 1)[0])
    return out, slope, slope_c


def pct(a, q=(2.5, 97.5)):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, q[0])), float(np.percentile(a, q[1]))] if len(a) else [None, None]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000)
    B = ap.parse_args().B
    rng = np.random.default_rng(20260926)
    report = {"B": B, "seed": 20260926, "unit": "(record, drone, window)", "experiments": {}}
    for ver, exp in EXPS.items():
        dec = json.loads((exp / "decision.json").read_text())["result"]
        rep = {}
        for fam, fd in dec["families"].items():
            cfg, grid, E, K = load_family(exp, fd["records"], ver)
            lv = [l for l in fd["levels"] if l.get("ladder", "A") == "A"]
            sigma_m = np.array([l["sigma_a_mm"] for l in lv]) / 1000
            C = np.array([l["C"] for l in lv]) if "C" in lv[0] else None
            W = E.shape[0]
            pt, slope, slope_c = stats(cfg, grid, E, K, sigma_m, C, np.arange(W))
            # sanity: must reproduce the frozen decision
            for a, b in zip(pt, lv):
                assert a["m_star"] == b["m_star"], (ver, fam, a["m_star"], b["m_star"])
            if fd["h1_slope"] is not None:
                assert abs(slope - fd["h1_slope"]) < 1e-9, (ver, fam, slope, fd["h1_slope"])
            bs = [stats(cfg, grid, E, K, sigma_m, C, rng.integers(0, W, W)) for _ in range(B)]
            levels = []
            for li, p in enumerate(pt):
                col = lambda k: [b[0][li][k] for b in bs]  # noqa: E731
                r0 = np.array(col("pred0")) / np.array(col("m_star"))
                r1 = np.array(col("pred1")) / np.array(col("m_star"))
                rc = np.array(col("crossing")) / np.array(col("m_star"))
                levels.append(dict(
                    sigma_a_mm=float(sigma_m[li] * 1000), m_star=p["m_star"], m_star_ci=pct(col("m_star")),
                    m_star_cont=p["m_star_cont"], m_star_cont_ci=pct(col("m_star_cont")),
                    crossing=p["crossing"], crossing_ci=pct(col("crossing")),
                    crossing_found_frac=float(np.mean(np.isfinite(col("crossing")))),
                    ratio_uncorrected=p["pred0"] / p["m_star"], ratio_uncorrected_ci=pct(r0),
                    ratio_corrected=(p["pred1"] / p["m_star"]) if C is not None else None,
                    ratio_corrected_ci=pct(r1) if C is not None else None,
                    ratio_crossing_ci=pct(rc),
                    p_in_band_uncorrected=float(np.mean((r0 >= 2 / 3) & (r0 <= 1.5))),
                    p_in_band_corrected=float(np.mean((r1 >= 2 / 3) & (r1 <= 1.5))) if C is not None else None))
            rep[fam] = dict(n_windows=W, slope=slope, slope_ci=pct([b[1] for b in bs]),
                            slope_cont=slope_c, slope_cont_ci=pct([b[2] for b in bs]),
                            p_slope_in_band=float(np.mean([(-0.7 <= b[1] <= -0.3) for b in bs if np.isfinite(b[1])])),
                            levels=levels)
            print(ver, fam, "W=%d" % W, "slope %.2f [%.2f, %.2f]" % (slope, *rep[fam]["slope_ci"]),
                  "cont %.2f [%.2f, %.2f]" % (slope_c, *rep[fam]["slope_cont_ci"]))
        report["experiments"][ver] = rep
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ci.json").write_text(json.dumps(report, indent=1))
    print("wrote", OUT / "ci.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
