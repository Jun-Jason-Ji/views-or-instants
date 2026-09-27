"""Pre-registration runner v0.2 (protocol/DPJAIT_NOISE_PREREG_v0.2.md): a second,
learned observation chain, and a prospective test of the correlated-noise
correction to the closed form.

Reuses the frozen v0.1 runner (imported, never modified) for identity binding,
triangulation, windows, estimators and curve statistics.  Differences:

  * observations come from our fine-tuned YOLO11n detector
    (src/learned_det_infer.py), scale 1 only;
  * only ladder A (injected independent noise) -- v0.1 showed downsampling is
    not a noise knob;
  * score also accumulates the autocorrelation of the per-instant error at lags
    1..LAG_MAX, so that decide can apply the correction
        sigma_eff = sigma_a * sqrt(1 - C(lag)),  lag = round(T_frames / m0),
    where m0 is the uncorrected closed-form prediction.

    freeze | predict --record R | score --record R | decide
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dpjait_noise_runner as v1  # noqa: E402  (frozen v0.1, imported read-only)

ROOT = v1.ROOT
PROTOCOL = "protocol/DPJAIT_NOISE_PREREG_v0.2.md"
SOURCES = [PROTOCOL, "src/dpjait_noise_runner_v02.py", "src/dpjait_noise_runner.py",
           "src/own_blob_observer.py", "src/learned_det_dataset.py", "src/learned_det_train.py",
           "src/learned_det_infer.py", "src/dpjait_record.py", "src/dpjait_sim_record.py",
           "src/dpjait_prereg_runner.py"]
REAL, SIM = v1.REAL, v1.SIM
CFG = dict(
    video_fps=25.0, gate_px=60.0, t_primary=400, grid_primary=v1.CFG["grid_primary"],
    ladder_a_px=[0, 4, 8, 16, 32, 64], ladder_a_reps=4, seed=20460926, lag_max=150,
    band=[2.0 / 3.0, 1.5], h1_slope=[-0.70, -0.30], h1_min_levels=4,
    h3_min_fraction=0.80, h6a_min_fraction=0.80, h6a_native_min_families=2,
    h6b_min_families=3,
)


# ------------------------------------------------------------ correction math
def autocorr_sums(e: np.ndarray, lag_max: int):
    """e: (n, 3) residuals with NaN rows -> (num[1..L], den) for rho = num/den."""
    ok = ~np.isnan(e).any(axis=1)
    if ok.sum() < 10:
        return np.zeros(lag_max), 0.0
    x = np.where(ok[:, None], e - np.nanmean(e[ok], axis=0), 0.0)
    den = float((x[ok] ** 2).sum())
    num = np.zeros(lag_max)
    for lag in range(1, lag_max + 1):
        if lag >= len(x):
            break
        both = ok[:-lag] & ok[lag:]
        num[lag - 1] = float((x[:-lag][both] * x[lag:][both]).sum())
    return num, den


def closed_form(T_s: float, kappa: float, v: float, sigma: float) -> float:
    return float(T_s * np.sqrt(kappa * v * v / (7.0 * sigma))) if sigma > 0 else float("nan")


def corrected(T_s, T_frames, kappa, v, sigma, rho, lag_max):
    """One-step correction: m0 -> lag -> C(lag) -> sigma_eff -> m1."""
    m0 = closed_form(T_s, kappa, v, sigma)
    if not np.isfinite(m0) or m0 <= 0:
        return m0, float("nan"), None, float("nan")
    lag = int(min(max(round(T_frames / m0), 1), lag_max))
    C = float(rho[lag - 1])
    s_eff = sigma * np.sqrt(max(0.0, 1.0 - C))
    return m0, closed_form(T_s, kappa, v, s_eff), lag, C


# ------------------------------------------------------------------- steps
def step_freeze(out: Path) -> None:
    if out.exists() and (out / "freeze.json").exists():
        raise SystemExit("refusing to re-freeze %s" % out)
    out.mkdir(parents=True, exist_ok=True)
    v1.save_new(out / "config.json", dict(CFG, real=REAL, sim=SIM, utc=v1.utc()))
    v1.save_new(out / "freeze.json", dict(hashes={s: v1.sha(ROOT / s) for s in SOURCES}, utc=v1.utc()))
    for s in SOURCES:
        dst = out / "source_snapshot" / s
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / s, dst)
    print("frozen %d sources under %s" % (len(SOURCES), out))


def verify(out: Path) -> None:
    for s, h in json.loads((out / "freeze.json").read_text())["hashes"].items():
        if v1.sha(ROOT / s) != h:
            raise SystemExit("frozen source changed: %s" % s)


def runs():
    out = []
    for li, px in enumerate(CFG["ladder_a_px"]):
        for r in range(1 if px == 0 else CFG["ladder_a_reps"]):
            out.append((li, px, r))
    return out


def step_predict(out: Path, name: str) -> None:
    """ESTIMATES only.  Never computes a reference path length."""
    verify(out)
    from own_blob_observer import load
    rec, sim = v1.open_record(name)
    meta, obs = load(out / "observations" / (name + ".npz"))
    n = min(min(meta["frames_read"].values()), rec.n_video_frames)
    mats = [v1.cam_matrices(rec)[c] for c in rec.cam_ids]
    drones = sorted(rec.reference)
    rec_idx = (REAL + SIM).index(name)
    bound = v1.bind(rec, obs, 1, n, CFG["gate_px"])
    Tp = CFG["t_primary"]
    wins = [w for w in v1.eligible_windows(rec, Tp) if (w + 1) * Tp < n]
    rows, dense = [], {}
    for li, px, rep in runs():
        uv = bound.copy()
        if px > 0:
            uv = uv + np.random.default_rng([CFG["seed"], rec_idx, li, rep]).normal(0.0, float(px), uv.shape)
        for d, dr in enumerate(drones):
            P = v1.triangulate_dense(uv[:, d], mats)
            dense["A|%d|%d|%s" % (li, rep, dr)] = P.astype(np.float32)
            for w in wins:
                f0 = w * Tp
                for m in CFG["grid_primary"]:
                    pts = P[v1.instants(f0, Tp, m)]
                    ok = ~np.isnan(pts).any(axis=1)
                    failed = (not ok[0]) or (not ok[-1]) or ok.sum() < 2
                    rows.append(dict(ladder="A", level=li, value=px, rep=rep, drone=dr, T=Tp,
                                     window=w, m=m, n_used=int(ok.sum()), failed=bool(failed),
                                     est=None if failed else v1.polyline(pts[ok])))
    pred = dict(record=name, sim=sim, n_frames=n, drones=drones, cameras=rec.cam_ids,
                windows_primary=wins, rows=rows, utc=v1.utc(),
                observation_rate={c: float(np.mean(~np.isnan(bound[:, :, j, 0])))
                                  for j, c in enumerate(rec.cam_ids)})
    path = out / "predictions" / (name + ".json")
    v1.save_new(path, pred)
    dpath = out / "predictions" / (name + "_dense.npz")
    if dpath.exists():
        raise SystemExit("refusing to overwrite %s" % dpath)
    np.savez_compressed(dpath, **dense)
    sp = out / "seal.json"
    seals = json.loads(sp.read_text()) if sp.exists() else {}
    seals[name] = dict(sha256=v1.sha(path), dense_sha256=v1.sha(dpath), utc=v1.utc(), n_rows=len(rows))
    sp.write_text(json.dumps(seals, indent=1), encoding="utf-8")
    print("%s: %d rows sealed, %d windows" % (name, len(rows), len(wins)))


def step_score(out: Path, name: str) -> None:
    rec, sim = v1.open_record(name)
    path = out / "predictions" / (name + ".json")
    dpath = out / "predictions" / (name + "_dense.npz")
    seal = json.loads((out / "seal.json").read_text())[name]
    if v1.sha(path) != seal["sha256"] or v1.sha(dpath) != seal["dense_sha256"]:
        raise SystemExit("prediction files changed after sealing: %s" % name)
    pred = json.loads(path.read_text())
    dense = np.load(dpath)
    per = v1.ref_per_frame(sim)
    dt = 1.0 / (CFG["video_fps"] * per)
    Tp = CFG["t_primary"]
    refs, kin = {}, {}
    for r in pred["rows"]:
        key = (r["drone"], r["window"])
        if key not in refs:
            f0, f1 = r["window"] * Tp, (r["window"] + 1) * Tp
            refs[key] = rec.reference_path_length(r["drone"], f0, f1)
            kin[key] = v1.curvature_speed(np.asarray(rec.reference[r["drone"]][f0 * per: f1 * per + 1], float), dt)
        r["reference_m"] = refs[key]
        r["err"] = (r["est"] - refs[key]) if (r["est"] is not None and np.isfinite(refs[key])) else None
    sig, acn, acd = {}, {}, {}
    L = CFG["lag_max"]
    for key in dense.files:
        _, li, rep, dr = key.split("|")
        P = dense[key].astype(float)
        for w in pred["windows_primary"]:
            f0, f1 = w * Tp, (w + 1) * Tp
            e = P[f0:f1] - np.asarray([rec.reference_at(dr, f) for f in range(f0, f1)])
            ok = ~np.isnan(e).any(axis=1)
            if ok.sum() > 10:
                c = e[ok] - e[ok].mean(axis=0)
                sig.setdefault(li, []).append(float(np.sqrt((c ** 2).mean())))
            num, den = autocorr_sums(e, L)
            acn[li] = acn.get(li, np.zeros(L)) + num
            acd[li] = acd.get(li, 0.0) + den
    pred["sigma_a_m"] = {k: float(np.mean(v)) for k, v in sig.items()}
    pred["autocorr_num"] = {k: v.tolist() for k, v in acn.items()}
    pred["autocorr_den"] = {k: float(v) for k, v in acd.items()}
    pred["kinematics"] = [dict(drone=k[0], window=k[1], kappa=v[0], speed=v[1]) for k, v in kin.items()]
    v1.save_new(out / "scored" / (name + ".json"), pred)
    print("%s scored: sigma_a mm %s" % (name, {k: round(v * 1000, 1) for k, v in sorted(pred["sigma_a_m"].items())}))


def step_decide(out: Path) -> None:
    cfg = json.loads((out / "config.json").read_text())
    scored = {p.stem: json.loads(p.read_text()) for p in sorted((out / "scored").glob("*.json"))}
    fams = {"real": [n for n in REAL if n in scored]}
    fams.update({s: [s] for s in SIM if s in scored})
    Tp, Ts = cfg["t_primary"], cfg["t_primary"] / cfg["video_fps"]
    grid, (lo, hi), L = cfg["grid_primary"], cfg["band"], cfg["lag_max"]
    res, cells = {"families": {}}, []
    for fam, names in fams.items():
        if not names:
            continue
        rows = [r for n in names for r in scored[n]["rows"]]
        kin = [k for n in names for k in scored[n]["kinematics"] if np.isfinite(k["kappa"]) and np.isfinite(k["speed"])]
        kap = float(np.median([k["kappa"] for k in kin]))
        spd = float(np.median([k["speed"] for k in kin]))
        levels = []
        for li, px in enumerate(cfg["ladder_a_px"]):
            sub = [r for r in rows if r["level"] == li]
            sv = [scored[n]["sigma_a_m"].get(str(li)) for n in names]
            sigma = float(np.mean([s for s in sv if s]))
            num = sum(np.asarray(scored[n]["autocorr_num"][str(li)]) for n in names)
            den = sum(scored[n]["autocorr_den"][str(li)] for n in names)
            rho = num / den if den > 0 else np.zeros(L)
            a_mean, s_mean = v1._curve(sub, "mean")
            mstar = min(a_mean, key=a_mean.get)
            m0, m1, lag, C = corrected(Ts, Tp, kap, spd, sigma, rho, L)
            lv = dict(value=px, sigma_a_mm=sigma * 1000, m_star=mstar, crossing=v1._crossing(s_mean),
                      pred_uncorrected=m0, pred_corrected=m1, lag=lag, C=C, rho1=float(rho[0]),
                      censored=bool(mstar in (grid[0], grid[-1])),
                      fail_rate=float(np.mean([r["failed"] for r in sub])))
            levels.append(lv)
            cells.append(dict(family=fam, **lv))
        ok_lv = [l for l in levels if not l["censored"]]
        slope = None
        if len(ok_lv) >= cfg["h1_min_levels"]:
            slope = float(np.polyfit(np.log([l["sigma_a_mm"] for l in ok_lv]),
                                     np.log([l["m_star"] for l in ok_lv]), 1)[0])
        res["families"][fam] = dict(records=names, kappa_median=kap, speed_median=spd,
                                    levels=levels, h1_slope=slope)
    inr = lambda s: s is not None and cfg["h1_slope"][0] <= s <= cfg["h1_slope"][1]  # noqa: E731
    sims = [f for f in res["families"] if f != "real"]
    res["H1"] = dict(real=res["families"].get("real", {}).get("h1_slope"),
                     sim={f: res["families"][f]["h1_slope"] for f in sims},
                     pass_=bool(inr(res["families"].get("real", {}).get("h1_slope"))
                                and sum(inr(res["families"][f]["h1_slope"]) for f in sims) >= 2))
    band = lambda x, m: np.isfinite(x) and lo <= x / m <= hi  # noqa: E731
    wc = [c for c in cells if c["crossing"]]
    ok3 = [band(c["crossing"], c["m_star"]) for c in wc]
    res["H3"] = dict(cells=len(ok3), within=int(sum(ok3)),
                     pass_=bool(ok3) and len(wc) >= len(cells) / 2 and float(np.mean(ok3)) >= cfg["h3_min_fraction"])
    ok6 = [band(c["pred_corrected"], c["m_star"]) for c in cells]
    native = [c for c in cells if c["value"] == 0]
    nat_ok = sum(band(c["pred_corrected"], c["m_star"]) for c in native)
    res["H6a"] = dict(cells=len(ok6), within=int(sum(ok6)), native_within=int(nat_ok),
                      pass_=bool(ok6) and float(np.mean(ok6)) >= cfg["h6a_min_fraction"]
                      and nat_ok >= cfg["h6a_native_min_families"])
    better = [abs(np.log(c["pred_corrected"] / c["m_star"])) < abs(np.log(c["pred_uncorrected"] / c["m_star"]))
              for c in native if np.isfinite(c["pred_corrected"])]
    res["H6b"] = dict(native_families=len(better), corrected_closer=int(sum(better)),
                      pass_=sum(better) >= cfg["h6b_min_families"])
    ok2 = [band(c["pred_uncorrected"], c["m_star"]) for c in cells]
    res["uncorrected_for_reference"] = dict(cells=len(ok2), within=int(sum(ok2)))
    v1.save_new(out / "decision.json", dict(utc=v1.utc(), result=res))
    for h in ("H1", "H3", "H6a", "H6b", "uncorrected_for_reference"):
        print(h, {k: v for k, v in res[h].items() if k != "detail"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["freeze", "predict", "score", "decide"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    {"freeze": lambda: step_freeze(out), "predict": lambda: step_predict(out, a.record),
     "score": lambda: step_score(out, a.record), "decide": lambda: step_decide(out)}[a.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
