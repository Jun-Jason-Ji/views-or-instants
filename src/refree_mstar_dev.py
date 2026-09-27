"""Development of a reference-free predictor of the Goldilocks density m*.

Everything a candidate predictor uses is computed from the observed dense
reconstruction P(t) of the windows themselves -- never from the reference.
The reference enters only through the targets m* and the crossing, read from
the frozen decision files, for scoring the candidates.

Candidates
  M1  fit the mean raw polyline length L(m) = L0 - A m^-a + C m^c (a in [1,3],
      c in [0,3]); predicted crossing (A/C)^(1/(a+c)).
  M1f same with a = c = 2 fixed.
  M2w smooth P with a Savitzky-Golay filter of w frames (cubic); deficit
      estimate D(m) = poly(Ps at m instants) - poly(Ps at all frames);
      inflation estimate I(m) = poly(P at m) - poly(Ps at m); predicted
      crossing where I(m) = -D(m).
Development only (records already analysed); see the report for exposure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpjait_noise_runner as v1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPS = {"v0.1_blob": ROOT / "experiments" / "dpjait_noise_prereg_v0.1",
        "v0.2_yolo": ROOT / "experiments" / "dpjait_noise_prereg_v0.2"}
FAMS = {"real": v1.REAL, **{s: [s] for s in v1.SIM}}
OUT = ROOT / "experiments" / "refree_mstar_dev_2026-09-26"
TP = 400
GRID = np.array([4, 5, 6, 8, 10, 12, 14, 17, 20, 24, 28, 34, 40, 50, 60, 80, 100, 134, 200, 400])
SG_WINDOWS = (9, 25, 51, 101)


def fill_gaps(P):
    ok = ~np.isnan(P).any(axis=1)
    if ok.mean() < 0.9:
        return None
    idx = np.arange(len(P))
    return np.stack([np.interp(idx, idx[ok], P[ok, i]) for i in range(3)], axis=1)


def window_curves(P):
    """P: (TP+1, 3) observed dense positions of one window -> dict of curves over GRID."""
    ii = [v1.instants(0, TP, int(m)) for m in GRID]
    out = {"L": np.array([v1.polyline(P[i]) for i in ii])}
    for w in SG_WINDOWS:
        Ps = savgol_filter(P, w, 3, axis=0, mode="interp")
        Ls_dense = v1.polyline(Ps)
        out["D%d" % w] = np.array([v1.polyline(Ps[i]) for i in ii]) - Ls_dense
        out["I%d" % w] = out["L"] - np.array([v1.polyline(Ps[i]) for i in ii])
    return out


def crossing(ms, y):
    for i in range(len(ms) - 1):
        if y[i] < 0 <= y[i + 1]:
            w = -y[i] / (y[i + 1] - y[i])
            return float(np.exp(np.log(ms[i]) + w * np.log(ms[i + 1] / ms[i])))
    return float("nan")


def fit_M1(L, fixed=False):
    m = GRID.astype(float)
    if fixed:
        f = lambda m, L0, A, C: L0 - A * m ** -2.0 + C * m ** 2.0  # noqa: E731
        p0, lb, ub = [L[-1], 1.0, 1e-6], [0, 0, 0], [np.inf, np.inf, np.inf]
    else:
        f = lambda m, L0, A, a, C, c: L0 - A * m ** -a + C * m ** c  # noqa: E731
        p0, lb, ub = [L[-1], 1.0, 2.0, 1e-4, 1.0], [0, 0, 1, 0, 0], [np.inf, np.inf, 3, np.inf, 3]
    try:
        p, _ = curve_fit(f, m, L, p0=p0, bounds=(lb, ub), maxfev=20000)
    except Exception:
        return float("nan")
    if fixed:
        _, A, C = p; a = c = 2.0
    else:
        _, A, a, C, c = p
    if A <= 0 or C <= 0:
        return float("nan")
    return float((A / C) ** (1.0 / (a + c)))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ver, exp in EXPS.items():
        cfg = json.loads((exp / "config.json").read_text())
        dec = json.loads((exp / "decision.json").read_text())["result"]["families"]
        for fam, names in FAMS.items():
            lv = [l for l in dec[fam]["levels"] if l.get("ladder", "A") == "A"]
            for li, l in enumerate(lv):
                acc = {}
                nwin = 0
                for n in names:
                    rec, sim = v1.open_record(n)
                    sc = json.loads((exp / "scored" / (n + ".json")).read_text())
                    dz = np.load(exp / "predictions" / (n + "_dense.npz"))
                    for dr in sorted({r["drone"] for r in sc["rows"]}):
                        reps = range(cfg["ladder_a_reps"]) if li else [0]
                        for rep in reps:
                            key = "A|%d|%d|%s" % (li, rep, dr)
                            if key not in dz.files:
                                continue
                            P = dz[key].astype(float)
                            for w in sorted({r["window"] for r in sc["rows"] if r["drone"] == dr}):
                                f0 = w * TP
                                if f0 + TP + 1 > len(P):
                                    continue
                                Pw = fill_gaps(P[f0:f0 + TP + 1])
                                if Pw is None:
                                    continue
                                for k, v in window_curves(Pw).items():
                                    acc.setdefault(k, []).append(v)
                                nwin += 1
                mean = {k: np.mean(v, axis=0) for k, v in acc.items()}
                pred = {"M1": fit_M1(mean["L"]), "M1f": fit_M1(mean["L"], fixed=True)}
                for w in SG_WINDOWS:
                    pred["M2_%d" % w] = crossing(GRID, mean["I%d" % w] + mean["D%d" % w])
                eq1 = l.get("pred_uncorrected", l.get("predicted"))
                rows.append(dict(chain=ver, family=fam, level=li, sigma_a_mm=l["sigma_a_mm"], n=nwin,
                                 m_star=l["m_star"], crossing=l["crossing"], eq1_refbased=eq1, **pred))
                print("%-9s %-7s L%d m*=%3d cross=%6.1f eq1=%5.1f | " % (ver, fam, li, l["m_star"], l["crossing"] or np.nan, eq1)
                      + " ".join("%s=%5.1f" % (k, v) for k, v in pred.items()))
    (OUT / "dev_predictions.json").write_text(json.dumps(rows, indent=1))
    # summary
    keys = ["eq1_refbased", "M1", "M1f"] + ["M2_%d" % w for w in SG_WINDOWS]
    print("\nratio to m*: within x1.5 / n, median ratio, RMS log ratio")
    for k in keys:
        r = np.array([q[k] / q["m_star"] for q in rows], float)
        ok = np.isfinite(r)
        print("  %-13s %2d/%2d  median %.2f  rms-log %.2f  (nan %d)" % (
            k, np.sum((r[ok] >= 2 / 3) & (r[ok] <= 1.5)), len(r), np.median(r[ok]), np.sqrt(np.mean(np.log(r[ok]) ** 2)), (~ok).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------- M3
from scipy.special import erf  # noqa: E402


def mean_norm(mu, s):
    """E|X| for X ~ N(mu_vec, s^2 I_3), |mu_vec| = mu (noncentral chi, 3 dof)."""
    mu = np.maximum(mu, 1e-9)
    lam = mu / s
    return s * (np.sqrt(2 / np.pi) * np.exp(-lam ** 2 / 2) + (lam + 1 / lam) * erf(lam / np.sqrt(2)))


def sigma_white(P):
    """Per-axis white-noise sd from lag-1 second differences (Var = 6 sigma^2)."""
    d2 = P[2:] - 2 * P[1:-1] + P[:-2]
    return float(np.sqrt(np.mean(np.var(d2, axis=0)) / 6.0))


def inflation_exact(P, m, sig):
    x = P[v1.instants(0, TP, int(m))]
    d_obs = np.linalg.norm(np.diff(x, axis=0), axis=1)
    mu = np.sqrt(np.maximum(d_obs ** 2 - 6 * sig ** 2, (0.1 * sig) ** 2))   # de-noised chord
    s = np.sqrt(2.0) * sig
    return float(np.sum(mean_norm(mu, s) - mu))


def m3_predict(Ls, Is, fit_lo=10, fit_hi=100):
    """Ls, Is: mean L(m), mean inflation over GRID.  Fit L - I = L0 - A m^-2."""
    sel = (GRID >= fit_lo) & (GRID <= fit_hi)
    X = np.column_stack([np.ones(sel.sum()), -GRID[sel] ** -2.0])
    L0, A = np.linalg.lstsq(X, (Ls - Is)[sel], rcond=None)[0]
    if A <= 0:
        return float("nan"), A
    return crossing(GRID, Is - A * GRID ** -2.0), A
