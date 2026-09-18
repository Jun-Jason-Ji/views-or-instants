"""Pre-registered view/instant allocation on the LBMC Lyon rig (protocol/LYON_PREREG_v0.3.md).

Steps (each writes only into --out and never modifies earlier outputs):
  freeze      hash protocol + sources + calibration, snapshot sources
  predict     per trial: labelled 2D observations (src/lyon_observe.py output) ->
              all targets x windows x subsets x m x grids x arms -> path predictions, sealed
  score       read the C3D reference LENGTHS, attach errors
  select-estimator  (development slice only) choose the m-piecewise primary estimator

Observations come from src/lyon_observe.py (identity from the reference
projection window, position from the image).  Reference path lengths are
computed only by `score`, after predictions are sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lyon_observe import PARAMS as OBS_PARAMS, VERSION as OBS_VERSION  # noqa: E402
from mcalib_cache_observer import triangulate  # noqa: E402
from three_view_cpu import three_view_triangulate  # noqa: E402
from robust_triangulation_cpu import robust_triangulate  # noqa: E402
from continuous_motion_cpu import estimate as state_estimate  # noqa: E402
from limited_arc_cpu import limited_arc  # noqa: E402
from gated_state_cpu import gate  # noqa: E402
from analyze_allocation import scene_center, subset_score  # noqa: E402

CAL = ROOT / "experiments/lyon_calibration_2026-09-17/calibration.json"
PROTOCOL = ROOT / "protocol/LYON_PREREG_v0.3.md"
SOURCES = ["protocol/LYON_PREREG_v0.3.md", "src/lyon_prereg_runner.py", "src/lyon_observe.py", "src/three_view_cpu.py",
           "src/robust_triangulation_cpu.py", "src/mcalib_cache_observer.py", "src/continuous_motion_cpu.py",
           "src/limited_arc_cpu.py", "src/gated_state_cpu.py", "src/analyze_allocation.py",
           "experiments/lyon_calibration_2026-09-17/calibration.json"]

CFG = dict(
    fps=60.0, marker_rate=120.0, window_frames=240,
    # m grid and budget points chosen on the development slice: the Goldilocks
    # sampling density on this rig lies near m = 60 (head) to m = 120 (foot),
    # far above MCalib's, so the equal-budget comparisons must straddle it.
    moments=[8, 12, 16, 20, 24, 30, 40, 48, 60, 72, 80, 90, 120, 240],
    budgets={240: [(2, 120), (3, 80), (4, 60), (5, 48)], 360: [(3, 120), (4, 90), (5, 72), (9, 40)]},
    phase_offsets=[-0.4, -0.2, 0.2, 0.4], jitter_seeds=[10401, 10402, 10403, 10404, 10405, 10406], jitter_amp=0.4,
    random_subset_seeds=list(range(20401, 20411)), sigma_m=0.003, nis=16.27,
    arms=["dlt", "spatial", "spatial_temporal", "spatial_rig"],
    # v0.3 secondary arm: the same spatial gate with thresholds scaled to this rig's labelling+calibration residual
    rig_gate=dict(px=8.0, spread_m=0.04, angle_deg=5.0),
    targets=["RFM5", "SV", "RFCC", "LSAT"],  # primary RFM5 (foot, ~6 m per window), secondary SV (head, ~0.6 m)
    # per-target primary estimator, chosen on the development slice (select-estimator) and then frozen:
    # state_mean for m <= state_mean_max_m, limited_arc above; all three estimators are always reported
    estimator_rule={"RFM5": {"state_mean_max_m": 60}, "RFCC": {"state_mean_max_m": 60}, "SV": {"state_mean_max_m": 0}, "LSAT": {"state_mean_max_m": 0}},
)


# ------------------------------------------------------------------ helpers
def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_new(p: Path, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as fh:
        json.dump(v, fh, indent=1)


def load_cams():
    return json.loads(CAL.read_text())["cameras"]


def subset_name(sub):
    return "+".join(sub)


def protocol_subsets(cams):
    ids = sorted(cams)
    center = scene_center(cams)
    subs = [tuple(ids)]
    subs += list(combinations(ids, 2)) + list(combinations(ids, 3))
    rule = {}
    for k in (2, 3, 4, 5, 7):
        rule[k] = min(combinations(ids, k), key=lambda s: subset_score(cams, s, center))
    for k in (4, 5, 7):
        subs.append(rule[k])
        rng = np.random.default_rng(CFG["random_subset_seeds"])
        seen = {rule[k]}
        while len(seen) < 11:
            s = tuple(sorted(rng.choice(ids, k, replace=False)))
            seen.add(s)
        subs += [s for s in sorted(seen) if s != rule[k]]
    return [tuple(s) for s in dict.fromkeys(subs)], {k: subset_name(v) for k, v in rule.items()}, center.tolist()


def grids(m, n=None):
    """Uniform grid plus, for budget arms, phase-offset and jitter grids. Returns {name: indices}."""
    n = n or CFG["window_frames"]
    base = np.linspace(0, n - 1, m)
    out = {"uniform": np.rint(base).astype(int)}
    if m < n:
        step = (n - 1) / (m - 1)
        for off in CFG["phase_offsets"]:
            g = base.copy()
            g[1:-1] += off * step
            out[f"phase{off:+.1f}"] = np.clip(np.rint(g), 0, n - 1).astype(int)
        for seed in CFG["jitter_seeds"]:
            rng = np.random.default_rng(seed)
            g = base.copy()
            g[1:-1] += rng.uniform(-CFG["jitter_amp"], CFG["jitter_amp"], m - 2) * step
            out[f"jitter{seed}"] = np.clip(np.rint(g), 0, n - 1).astype(int)
    return out


def three_view_rig(observations, cams, px, spread_m, angle_deg):
    """three_view_cpu.three_view_triangulate with rig-scaled thresholds (frozen file left untouched)."""
    from robust_triangulation_cpu import reprojection_errors
    valid = {c: np.asarray(v, float) for c, v in observations.items() if v is not None}
    if len(valid) != 3:
        return robust_triangulate(observations, cams, px)[0]
    p = triangulate(valid, cams)
    if max(reprojection_errors(p, valid, cams).values()) > px:
        raise ValueError("three_view_inconsistent")
    pairpoints = []
    for a, b in combinations(sorted(valid), 2):
        pairpoints.append(triangulate({a: valid[a], b: valid[b]}, cams))
        centers = [-np.asarray(cams[c]["R"]).T @ np.asarray(cams[c]["t_m"]) for c in (a, b)]
        rays = [(p - c) / np.linalg.norm(p - c) for c in centers]
        if np.degrees(np.arccos(np.clip(abs(float(rays[0] @ rays[1])), -1, 1))) < angle_deg:
            raise ValueError("three_view_small_ray_angle")
    if max(np.linalg.norm(a - b) for a, b in combinations(pairpoints, 2)) > spread_m:
        raise ValueError("three_view_pair_disagreement")
    return p


def tri_arm(obs, cams, k, arm):
    valid = {c: v for c, v in obs.items() if v is not None}
    if len(valid) < 2:
        raise ValueError("insufficient_visible_views")
    if arm == "dlt":
        return triangulate(valid, cams)
    if arm == "spatial_rig":
        g = CFG["rig_gate"]
        if k >= 4:
            return robust_triangulate(obs, cams, g["px"])[0]
        if k == 3 and len(valid) == 3:
            return three_view_rig(obs, cams, g["px"], g["spread_m"], g["angle_deg"])
        return triangulate(valid, cams)
    if k >= 4:
        return robust_triangulate(obs, cams)[0]
    if k == 3 and len(valid) == 3:
        return three_view_triangulate(obs, cams)[0]
    return triangulate(valid, cams)


# ------------------------------------------------------------------ steps
def step_freeze(out: Path):
    out.mkdir(parents=True, exist_ok=False)
    cams = load_cams()
    subs, rule, center = protocol_subsets(cams)
    cfg = dict(CFG, observer=OBS_VERSION, observer_params=OBS_PARAMS, rule_subsets=rule, scene_center_m=center,
               n_subsets=len(subs), subsets=[subset_name(s) for s in subs], utc=utc())
    save_new(out / "config.json", cfg)
    save_new(out / "freeze.json", dict(hashes={s: sha(ROOT / s) for s in SOURCES}, utc=cfg["utc"]))
    for s in SOURCES:
        d = out / "source_snapshot" / s
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / s, d)
    print(f"frozen: {len(subs)} subsets, rule {rule}")


def verify(out: Path):
    for s, h in json.loads((out / "freeze.json").read_text())["hashes"].items():
        if sha(ROOT / s) != h:
            raise SystemExit(f"frozen source changed: {s}")


def load_observations(obs_dir: Path, cams):
    meta = json.loads((obs_dir / "observe_meta.json").read_text())
    uv = {n: np.load(obs_dir / f"uv_{n}.npy") for n in sorted(cams)}
    return meta, uv


def step_predict(out: Path, trial: str, obs_dir: Path, dev: bool, targets=None, only_rule=False):
    if not dev:
        verify(out)
    cams = load_cams()
    cfg = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else CFG
    rule_max = cfg.get("estimator_rule")
    targets = targets or cfg.get("targets", CFG["targets"])
    meta, UV = load_observations(obs_dir, cams)
    subs, rule, _ = protocol_subsets(cams)
    if only_rule:
        subs = [s for s in subs if subset_name(s) in rule.values() or len(s) == len(cams)]
    T = min(v.shape[0] for v in UV.values())
    nwin = T // CFG["window_frames"]
    rows = []
    t0 = perf_counter()
    for target in targets:
        j = meta["markers"].index(target)
        for sub in subs:
            k = len(sub)
            for w in range(nwin):
                start = w * CFG["window_frames"]
                for m in CFG["moments"]:
                    budget_arm = any((k, m) in v for v in CFG["budgets"].values())
                    for gname, g in grids(m).items():
                        if gname != "uniform" and not budget_arm:
                            continue
                        sel = np.unique(start + g)  # rounding can duplicate an index on jitter grids
                        t = sel / CFG["fps"]
                        obs = [{c: (None if not np.isfinite(UV[c][i, j, 0]) else UV[c][i, j]) for c in sub} for i in sel]
                        for arm in CFG["arms"]:
                            pts = np.full((len(sel), 3), np.nan)
                            reasons = {}
                            for a, o in enumerate(obs):
                                try:
                                    pts[a] = tri_arm(o, cams, k, arm)
                                except ValueError as e:
                                    reasons[str(e)] = reasons.get(str(e), 0) + 1
                            ok = np.isfinite(pts).all(axis=1)
                            base = dict(trial=trial, target=target, start=int(start), cameras=subset_name(sub), views=k, moments=m,
                                        moments_actual=int(len(sel)), frames=k * m, grid=gname, arm=arm, missing=int((~ok).sum()), reasons=reasons)
                            if not ok[0] or not ok[-1] or ok.sum() < 2:
                                rows.append(dict(base, failure="endpoint_or_too_few", method=None, value_m=None))
                                continue
                            q, qt = pts[ok], t[ok]
                            gated = 0
                            if arm == "spatial_temporal":
                                try:
                                    acc, log = gate(q, qt, sigma=CFG["sigma_m"], nis_threshold=CFG["nis"])
                                except ValueError as e:
                                    rows.append(dict(base, failure=f"gate:{e}", method=None, value_m=None))
                                    continue
                                gated = len(log)
                                if acc[-1] != len(q) - 1:
                                    rows.append(dict(base, failure="endpoint_gated", method=None, value_m=None, temporal_rejected=gated))
                                    continue
                                q, qt = q[acc], qt[acc]
                            base["temporal_rejected"] = gated
                            rows.append(dict(base, method="polygon", value_m=float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())))
                            try:
                                rows.append(dict(base, method="limited", value_m=float(limited_arc(q, qt))))
                            except Exception as e:  # noqa: BLE001
                                rows.append(dict(base, method="limited", value_m=None, failure=f"limited:{e}"))
                            if len(q) <= 60:
                                try:
                                    rows.append(dict(base, method="state_mean", value_m=state_estimate(q, qt, sigma=CFG["sigma_m"])["mean_path_m"]))
                                except Exception as e:  # noqa: BLE001
                                    rows.append(dict(base, method="state_mean", value_m=None, failure=f"state:{e}"))
            print(f"{trial} {target} {subset_name(sub)}: {len(rows)} rows, {perf_counter() - t0:.0f} s", flush=True)
    pd = out / "predictions" / f"{trial.replace('/', '_')}.json"
    pd.parent.mkdir(parents=True, exist_ok=True)
    pd.write_text(json.dumps(dict(trial=trial, observations=str(obs_dir), observer_meta_sha256=sha(obs_dir / "observe_meta.json"),
                                  targets=targets, estimator_rule_state_mean_max_m=rule_max, rows=rows), indent=0))
    seal = dict(file=str(pd.relative_to(ROOT)), sha256=sha(pd), rows=len(rows), utc=utc(), dev=dev)
    with (out / "seal.json").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(seal) + "\n")
    print("sealed", seal)


def reference_lengths(c3d_path: Path, T_video: int, targets):
    from lyon_geometry_selfcheck import load_markers
    M, labels, _ = load_markers(c3d_path)
    ratio = int(round(CFG["marker_rate"] / CFG["fps"]))
    out = {}
    for target in targets:
        tr = M[:, labels.index(target)] / 1000.0
        for w in range(T_video // CFG["window_frames"]):
            s = w * CFG["window_frames"]
            seg = tr[ratio * s: ratio * (s + CFG["window_frames"] - 1) + 1]
            out[(target, s)] = dict(ref_m=float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()),
                                    ref_decimated_m=float(np.linalg.norm(np.diff(seg[::2], axis=0), axis=1).sum()))
    return out


def step_score(out: Path, trial: str, c3d: Path, dev: bool):
    pd = out / "predictions" / f"{trial.replace('/', '_')}.json"
    seals = [json.loads(l) for l in (out / "seal.json").read_text().splitlines()]
    assert any(s["sha256"] == sha(pd) for s in seals), "prediction file not sealed or modified"
    P = json.loads(pd.read_text())
    T = max(r["start"] for r in P["rows"]) + CFG["window_frames"]
    ref = reference_lengths(c3d, T, P["targets"])
    for r in P["rows"]:
        r["ref_m"] = ref[(r["target"], r["start"])]["ref_m"]
        r["error_m"] = None if r.get("value_m") is None else r["value_m"] - r["ref_m"]
    sd = out / "scored"
    sd.mkdir(exist_ok=True)
    (sd / f"{trial.replace('/', '_')}.json").write_text(json.dumps(dict(trial=trial, c3d_sha256=sha(c3d),
        reference={f"{k[0]}@{k[1]}": v for k, v in ref.items()}, rows=P["rows"]), indent=0))
    print(f"scored {trial}: {len(P['rows'])} rows, {len(ref)} target-windows")


def step_select_estimator(out: Path, trial: str):
    """Development slice only: per m, which estimator has the lowest MAE on k=9 and the rule triple (uniform grid, spatial arm)."""
    S = json.loads((out / "scored" / f"{trial.replace('/', '_')}.json").read_text())["rows"]
    cams = load_cams()
    _, rule, _ = protocol_subsets(cams)
    targets = sorted({r["target"] for r in S})
    result = {}
    for target in targets:
        tab = {}
        for m in CFG["moments"]:
            for cam in (subset_name(tuple(sorted(cams))), rule[3]):
                for meth in ("polygon", "limited", "state_mean"):
                    e = [abs(r["error_m"]) for r in S if r["target"] == target and r["moments"] == m and r["cameras"] == cam
                         and r["grid"] == "uniform" and r["arm"] == "spatial" and r["method"] == meth and r.get("error_m") is not None]
                    tab.setdefault(m, {})[f"{'k9' if cam != rule[3] else 'k3rule'}:{meth}"] = (round(1000 * float(np.mean(e)), 2), len(e)) if e else None
        best_state = [m for m in CFG["moments"] if m <= 60 and tab[m].get("k9:state_mean") and tab[m].get("k9:limited")
                      and tab[m]["k9:state_mean"][0] <= tab[m]["k9:limited"][0]]
        rule_m = max(best_state) if best_state else 0
        result[target] = dict(table={str(m): v for m, v in tab.items()}, state_mean_max_m=rule_m)
        print(f"== {target}: state_mean_max_m -> {rule_m}")
        for m in CFG["moments"]:
            print("  ", m, tab[m])
    save_new(out / "estimator_selection.json", dict(per_target=result, utc=utc()))



# ------------------------------------------------------------------ hypotheses
def primary_method(cfg, target, m):
    rule = (cfg.get("estimator_rule") or CFG["estimator_rule"]).get(target, {"state_mean_max_m": 0})
    return "state_mean" if m <= rule["state_mean_max_m"] else "limited"


def _errors(rows, cfg, target, cameras, m, grid="uniform", arm="spatial"):
    """{(trial,start): error_m} for the primary estimator; failed windows absent."""
    meth = primary_method(cfg, target, m)
    return {(r["trial"], r["start"]): r["error_m"] for r in rows
            if r["target"] == target and r["cameras"] == cameras and r["moments"] == m and r["grid"] == grid
            and r["arm"] == arm and r["method"] == meth and r.get("error_m") is not None}


def _windows(rows, target, cameras, m, grid="uniform", arm="spatial"):
    return {(r["trial"], r["start"]) for r in rows if r["target"] == target and r["cameras"] == cameras
            and r["moments"] == m and r["grid"] == grid and r["arm"] == arm}


def paired(ea, eb, seed=0, B=2000):
    """MAE(a) - MAE(b) on common windows, cluster bootstrap by trial. Negative favours a."""
    common = sorted(set(ea) & set(eb))
    if not common:
        return dict(n=0)
    da = np.array([abs(ea[w]) for w in common]); db = np.array([abs(eb[w]) for w in common])
    trials = sorted({w[0] for w in common})
    idx = {t: [i for i, w in enumerate(common) if w[0] == t] for t in trials}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        pick = rng.choice(trials, len(trials), replace=True)
        ii = np.concatenate([idx[t] for t in pick])
        boots.append(da[ii].mean() - db[ii].mean())
    return dict(n=len(common), mae_a_mm=1000 * float(da.mean()), mae_b_mm=1000 * float(db.mean()),
                diff_mm=1000 * float(da.mean() - db.mean()), ci95_mm=[1000 * float(v) for v in np.percentile(boots, [2.5, 97.5])],
                a_better_windows=int((da < db).sum()))


def _mae_by_subset(rows, cfg, target, subsets, m, grid="uniform", arm="spatial", min_windows=3):
    out = {}
    for c in subsets:
        e = _errors(rows, cfg, target, c, m, grid, arm)
        if len(e) >= min_windows:
            out[c] = 1000 * float(np.mean([abs(v) for v in e.values()]))
    return out


def _completion(rows, cfg, target, subsets, m, arm):
    done = tot = 0
    for c in subsets:
        w = _windows(rows, target, c, m, "uniform", arm)
        tot += len(w)
        done += len(_errors(rows, cfg, target, c, m, "uniform", arm))
    return done / tot if tot else None


def step_decide(out: Path):
    """Evaluate the pre-registered hypotheses of protocol/LYON_PREREG_v0.3.md section 7."""
    cfg = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else dict(CFG)
    cams = load_cams()
    allc = subset_name(tuple(sorted(cams)))
    _, rule, _ = protocol_subsets(cams)
    rows = []
    for f in sorted((out / "scored").glob("*.json")):
        rows += json.loads(f.read_text())["rows"]
    trials = sorted({r["trial"] for r in rows})
    targets = sorted({r["target"] for r in rows})
    subsets_by_k = {k: sorted({r["cameras"] for r in rows if r["views"] == k}) for k in (2, 3, 4, 5, 7, 9)}
    grids_all = ["uniform"] + [f"phase{o:+.1f}" for o in CFG["phase_offsets"]] + [f"jitter{sd}" for sd in CFG["jitter_seeds"]]
    # direction predicted for H2, per target and budget: 'instants' (3-view better), 'instants_not_worse', 'views'
    H2_PRED = {"RFM5": {"240": "instants", "360": "instants"}, "SV": {"240": "tie", "360": "views"}}
    D = dict(trials=trials, targets=targets, n_subsets={str(k): len(v) for k, v in subsets_by_k.items()}, utc=utc(), hypotheses={})
    for target in targets:
        H = {}
        # ---- H1: non-saturation + visibility mechanism
        h1 = {}
        for m in (60, 80, 90, 120):
            e9 = _errors(rows, cfg, target, allc, m)
            mae9 = 1000 * float(np.mean([abs(v) for v in e9.values()])) if e9 else None
            tri = _mae_by_subset(rows, cfg, target, subsets_by_k[3], m)
            if not (mae9 and tri):
                continue
            diffs = np.array([v - mae9 for v in tri.values()])
            missing = {}
            for c in tri:
                meth = primary_method(cfg, target, m)
                ms = [r["missing"] for r in rows if r["target"] == target and r["cameras"] == c and r["moments"] == m
                      and r["grid"] == "uniform" and r["arm"] == "spatial" and r["method"] == meth and r.get("error_m") is not None]
                missing[c] = float(np.mean(ms)) if ms else None
            mm = np.array([missing[c] for c in tri if missing[c] is not None])
            mv = np.array([tri[c] for c in tri if missing[c] is not None])
            corr = float(np.corrcoef(mv, mm)[0, 1]) if len(mm) > 2 and mm.std() > 0 else None
            vis = [tri[c] for c in tri if missing[c] is not None and missing[c] <= 2]
            h1[str(m)] = dict(mae_k9_mm=mae9, n_triples=len(tri), median_diff_mm=float(np.median(diffs)),
                              not_saturated=bool(np.median(diffs) > 0.15 * mae9), corr_mae_missing=corr,
                              corr_pass=bool(corr is not None and corr > 0.5),
                              visible_triples=dict(n=len(vis), median_mae_mm=float(np.median(vis)) if vis else None,
                                                   not_worse_than_k9=bool(vis and np.median(vis) - mae9 <= 0.15 * mae9)),
                              rule_triple_mae_mm=tri.get(rule[3]))
        curve9 = {}
        for m in cfg["moments"]:
            e = _errors(rows, cfg, target, allc, m)
            if e:
                curve9[m] = 1000 * float(np.mean([abs(v) for v in e.values()]))
        mstar9 = min(curve9, key=curve9.get) if curve9 else None
        m_b = min(h1, key=lambda k: abs(int(k) - mstar9)) if (h1 and mstar9) else None
        H["H1"] = dict(per_m=h1, pass_a=all(v["not_saturated"] for v in h1.values()) if h1 else None,
                       m_star_k9=mstar9, corr_at_m_star=h1[m_b]["corr_mae_missing"] if m_b else None,
                       pass_b=h1[m_b]["corr_pass"] if m_b else None)
        # ---- H2 / H3: equal budget with median-over-subsets statistics
        h2 = {}
        for B, arms in cfg["budgets"].items():
            arms = [tuple(a) for a in arms]
            three = next(a for a in arms if a[0] == 3)
            kmax = max(arms)
            per_grid = {}
            for g in grids_all:
                tri = _mae_by_subset(rows, cfg, target, subsets_by_k[3], three[1], g)
                if kmax[0] == 9:
                    e9 = _errors(rows, cfg, target, allc, kmax[1], g)
                    other = 1000 * float(np.mean([abs(v) for v in e9.values()])) if e9 else None
                else:
                    oth = _mae_by_subset(rows, cfg, target, subsets_by_k[kmax[0]], kmax[1], g)
                    other = float(np.median(list(oth.values()))) if oth else None
                med3 = float(np.median(list(tri.values()))) if tri else None
                per_grid[g] = dict(median_triple_mae_mm=med3, n_triples=len(tri), other_mae_mm=other,
                                   rule_triple_mae_mm=tri.get(rule[3]))
            pred = H2_PRED.get(target, {}).get(str(B))
            def verdict(v):
                if v["median_triple_mae_mm"] is None or v["other_mae_mm"] is None:
                    return None
                d = v["median_triple_mae_mm"] - v["other_mae_mm"]
                if pred == "instants":
                    return d < 0
                if pred == "tie":
                    return abs(d) <= 0.25 * max(v["median_triple_mae_mm"], v["other_mae_mm"])
                if pred == "views":
                    return d > 0.15 * v["median_triple_mae_mm"]
                return None
            verdicts = {g: verdict(v) for g, v in per_grid.items()}
            fails = sum(1 for v in verdicts.values() if v is False)
            three_better = sum(1 for v in per_grid.values() if v["median_triple_mae_mm"] is not None and v["other_mae_mm"] is not None
                               and v["median_triple_mae_mm"] < v["other_mae_mm"])
            if pred == "tie":
                h3 = bool(verdicts["uniform"] and 3 <= three_better <= 8)
            else:
                h3 = bool(verdicts["uniform"] and fails == 0)
            h2[str(B)] = dict(comparison=f"3x{three[1]} vs {kmax[0]}x{kmax[1]}", predicted=pred, uniform=per_grid["uniform"],
                              H2_pass=verdicts["uniform"], grids_failing=fails, three_view_better_grids=three_better, H3_pass=h3,
                              per_grid=per_grid, verdicts=verdicts)
        H["H2_H3"] = h2
        H["H2_pass"] = all(v["H2_pass"] for v in h2.values()) if h2 else None
        H["H3_pass"] = all(v["H3_pass"] for v in h2.values()) if h2 else None
        # ---- H4: gate is rig-specific
        h4 = {}
        for m in (60, 120):
            c_pair_dlt = _completion(rows, cfg, target, subsets_by_k[2], m, "dlt")
            c_tri_dlt = _completion(rows, cfg, target, subsets_by_k[3], m, "dlt")
            c_tri_sp = _completion(rows, cfg, target, subsets_by_k[3], m, "spatial")
            mae_dlt = _mae_by_subset(rows, cfg, target, subsets_by_k[3], m, arm="dlt")
            mae_sp = _mae_by_subset(rows, cfg, target, subsets_by_k[3], m, arm="spatial")
            c_tri_rig = _completion(rows, cfg, target, subsets_by_k[3], m, "spatial_rig")
            mae_rig = _mae_by_subset(rows, cfg, target, subsets_by_k[3], m, arm="spatial_rig")
            h4[str(m)] = dict(pairs_dlt=c_pair_dlt, triples_dlt=c_tri_dlt, triples_spatial=c_tri_sp, triples_spatial_rig=c_tri_rig,
                              median_triple_mae_spatial_rig_mm=float(np.median(list(mae_rig.values()))) if mae_rig else None,
                              d_pass=bool(c_tri_rig is not None and c_tri_rig >= 0.85),
                              a_pass=bool(c_tri_dlt is not None and c_pair_dlt is not None and c_tri_dlt >= c_pair_dlt),
                              b_pass=bool(c_tri_dlt is not None and c_tri_sp is not None and c_tri_dlt - c_tri_sp >= 0.15),
                              median_triple_mae_dlt_mm=float(np.median(list(mae_dlt.values()))) if mae_dlt else None,
                              median_triple_mae_spatial_mm=float(np.median(list(mae_sp.values()))) if mae_sp else None)
        H["H4"] = h4
        # ---- H5: Goldilocks point
        curve = {}
        for m in cfg["moments"]:
            e = _errors(rows, cfg, target, allc, m)
            if e:
                curve[str(m)] = 1000 * float(np.mean([abs(v) for v in e.values()]))
        mstar = int(min(curve, key=curve.get)) if curve else None
        pred5 = {"RFM5": lambda x: x >= 90, "SV": lambda x: 48 <= x <= 80}.get(target)
        H["H5"] = dict(k9_curve_mm=curve, m_star=mstar, pass_=bool(pred5(mstar)) if (pred5 and mstar) else None)
        D["hypotheses"][target] = H
    (out / "decision.json").write_text(json.dumps(D, indent=1))
    for target in targets:
        H = D["hypotheses"][target]
        print(f"== {target}")
        for m, v in H["H1"]["per_m"].items():
            vt = v["visible_triples"]
            print(f"  H1 m={m}: k9 {v['mae_k9_mm']:.1f}; median triple diff {v['median_diff_mm']:+.1f} (not saturated {v['not_saturated']}); corr(MAE,missing) {v['corr_mae_missing']}; visible triples n={vt['n']} mae {vt['median_mae_mm']}")
        for B, v in H["H2_H3"].items():
            u = v["uniform"]
            print(f"  H2 B={B} {v['comparison']} pred={v['predicted']}: median triple {u['median_triple_mae_mm']:.1f} vs {u['other_mae_mm']:.1f} -> {v['H2_pass']}; grids failing {v['grids_failing']}/11, 3-view better on {v['three_view_better_grids']}/11 -> H3 {v['H3_pass']}")
        print(f"  H1 pass_a {H['H1']['pass_a']}, corr at m*={H['H1']['m_star_k9']}: {H['H1']['corr_at_m_star']} -> pass_b {H['H1']['pass_b']}")
        for m, v in H["H4"].items():
            print(f"  H4 m={m}: completion pairs_dlt {v['pairs_dlt']:.2f} triples_dlt {v['triples_dlt']:.2f} spatial {v['triples_spatial']:.2f} rig-gate {v['triples_spatial_rig'] if v['triples_spatial_rig'] is None else round(v['triples_spatial_rig'], 2)} (a {v['a_pass']}, b {v['b_pass']}, d {v['d_pass']}); triple MAE dlt {v['median_triple_mae_dlt_mm']} spatial {v['median_triple_mae_spatial_mm']} rig {v['median_triple_mae_spatial_rig_mm']}")
        print(f"  H5 m*={H['H5']['m_star']} pass={H['H5']['pass_']}")
        print(f"  => H2 {H['H2_pass']}  H3 {H['H3_pass']}")
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["freeze", "predict", "score", "select-estimator", "decide"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--obs", default=None, help="predict: folder written by src/lyon_observe.py")
    ap.add_argument("--trial", default=None)
    ap.add_argument("--c3d", default=None)
    ap.add_argument("--targets", default=None, help="predict: comma list overriding config targets (development only)")
    ap.add_argument("--only-rule", action="store_true", help="predict: rule subsets and all-cameras only")
    ap.add_argument("--dev", action="store_true", help="development slice: no freeze verification")
    a = ap.parse_args()
    out = ROOT / a.out
    if a.step == "freeze":
        step_freeze(out)
    elif a.step == "predict":
        step_predict(out, a.trial, ROOT / a.obs, a.dev, a.targets.split(",") if a.targets else None, a.only_rule)
    elif a.step == "score":
        step_score(out, a.trial, Path(a.c3d), a.dev)
    elif a.step == "select-estimator":
        step_select_estimator(out, a.trial)
    elif a.step == "decide":
        step_decide(out)


if __name__ == "__main__":
    main()
