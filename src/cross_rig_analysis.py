"""Cross-rig analyses that need no new data (2026-09-18).

1. Closest-to-Goldilocks rule tested on every (approximately) equal-budget
   pair of configurations on MCalib (confirmation records 15-18) and on the
   Lyon development slice.
2. Window-level bootstrap of the Goldilocks density m*.
3. Pilot-window estimate of m*: estimate m* from window 1 only and check how
   well it predicts the best allocation on the remaining windows.
4. Figures: Goldilocks curves of both rigs; allocation surfaces.

Outputs: experiments/cross_rig_analysis_2026-09-18/{results.json, *.png}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/cross_rig_analysis_2026-09-18"
OUT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- loading
def load_mcalib():
    S = json.loads((ROOT / "experiments/allocation_confirm_v1_2026-09-15/scored.json").read_text())
    rows = S["rows"] if isinstance(S, dict) else S
    # primary estimator as in the MCalib confirmation: state_mean for m<=33, limited above
    out = []
    for r in rows:
        meth = "state_mean" if r["moments"] <= 33 else "limited"
        if r["method"] != meth:
            continue
        out.append(dict(rig="MCalib", target="marker", k=r["views"], m=r["moments"], cameras=r["cameras"],
                        window=(r["record"], r["start"]), err=r["error_m"], missing=r.get("missing", 0)))
    return out


def load_lyon():
    S = json.loads((ROOT / "experiments/lyon_prereg_dev_full_2026-09-17/merged/scored/participant_02_gait.json").read_text())
    out = []
    for r in S["rows"]:
        if r["grid"] != "uniform" or r["arm"] != "spatial":
            continue
        meth = "state_mean" if (r["target"] in ("RFM5", "RFCC") and r["moments"] <= 60) else "limited"
        if r["method"] != meth:
            continue
        out.append(dict(rig="Lyon", target=r["target"], k=r["views"], m=r["moments"], cameras=r["cameras"],
                        window=(r["trial"], r["start"]), err=r["error_m"], missing=r.get("missing", 0)))
    return out


def mae_table(rows, min_windows=3):
    """{(k, m): median over subsets of MAE (mm)}, subsets with >= min_windows completed; also per-subset dict."""
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["err"] is not None:
            per[(r["k"], r["m"])][r["cameras"]].append(abs(r["err"]) * 1000)
    med, full = {}, {}
    for key, subs in per.items():
        vals = {c: float(np.mean(v)) for c, v in subs.items() if len(v) >= min_windows}
        if vals:
            med[key] = float(np.median(list(vals.values())))
            full[key] = vals
    return med, full


def kmax_curve(rows):
    kmax = max(r["k"] for r in rows)
    curve = {}
    for m in sorted({r["m"] for r in rows if r["k"] == kmax}):
        e = [abs(r["err"]) * 1000 for r in rows if r["k"] == kmax and r["m"] == m and r["err"] is not None]
        if e:
            curve[m] = float(np.mean(e))
    return kmax, curve


# ----------------------------------------------------------------- 1. rule over all equal-budget pairs
def rule_test(rows, label, tol=0.06):
    med, _ = mae_table(rows)
    kmax, curve = kmax_curve(rows)
    mstar = min(curve, key=curve.get)
    keys = sorted(med)
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] == b[0]:
                continue
            Ba, Bb = a[0] * a[1], b[0] * b[1]
            if abs(np.log(Ba / Bb)) > tol:
                continue
            lo, hi = (a, b) if a[0] < b[0] else (b, a)  # lo = fewer views, more instants
            d_lo, d_hi = abs(np.log(lo[1] / mstar)), abs(np.log(hi[1] / mstar))
            if abs(d_lo - d_hi) <= 0.1:
                pred = "tie"
            else:
                pred = "fewer_views" if d_lo < d_hi else "more_views"
            r = med[lo] / med[hi]
            obs = "tie" if abs(np.log(r)) <= np.log(1.25) else ("fewer_views" if r < 1 else "more_views")
            pairs.append(dict(fewer=lo, more=hi, budget=(Ba, Bb), mae_fewer=med[lo], mae_more=med[hi],
                              logdist=(d_lo, d_hi), predicted=pred, observed=obs, hit=(pred == obs)))
    hits = sum(p["hit"] for p in pairs)
    # baseline: "always fewer views" and "always more views"
    base_f = sum(p["observed"] == "fewer_views" for p in pairs)
    base_m = sum(p["observed"] == "more_views" for p in pairs)
    print(f"[{label}] m*(k={kmax})={mstar}; equal-budget pairs {len(pairs)}: rule hits {hits} ({hits / max(1, len(pairs)):.2f}); "
          f"baseline always-fewer-views {base_f}, always-more-views {base_m}")
    for p in pairs:
        print(f"   {p['fewer'][0]}x{p['fewer'][1]:<3} vs {p['more'][0]}x{p['more'][1]:<3} B={p['budget'][0]}/{p['budget'][1]}: "
              f"{p['mae_fewer']:7.1f} vs {p['mae_more']:7.1f}  pred {p['predicted']:11s} obs {p['observed']:11s} {'OK' if p['hit'] else '--'}")
    return dict(mstar=mstar, kmax=kmax, n_pairs=len(pairs), hits=hits, baseline_fewer=base_f, baseline_more=base_m,
                pairs=[{**p, "fewer": list(p["fewer"]), "more": list(p["more"])} for p in pairs])


# ----------------------------------------------------------------- 2. bootstrap of m*
def bootstrap_mstar(rows, label, B=2000, seed=0):
    kmax = max(r["k"] for r in rows)
    wins = sorted({r["window"] for r in rows if r["k"] == kmax})
    ms = sorted({r["m"] for r in rows if r["k"] == kmax})
    E = np.full((len(wins), len(ms)), np.nan)
    wi = {w: i for i, w in enumerate(wins)}
    for r in rows:
        if r["k"] == kmax and r["err"] is not None:
            E[wi[r["window"]], ms.index(r["m"])] = abs(r["err"]) * 1000
    rng = np.random.default_rng(seed)
    stars = []
    for _ in range(B):
        idx = rng.integers(0, len(wins), len(wins))
        curve = np.nanmean(E[idx], axis=0)
        stars.append(ms[int(np.nanargmin(curve))])
    stars = np.array(stars)
    dist = {int(m): float((stars == m).mean()) for m in ms if (stars == m).any()}
    lo, hi = np.percentile(stars, [5, 95])
    print(f"[{label}] m* bootstrap over {len(wins)} windows: point {ms[int(np.nanargmin(np.nanmean(E, axis=0)))]}, 90% interval [{lo:.0f}, {hi:.0f}], distribution {dist}")
    return dict(windows=len(wins), point=int(ms[int(np.nanargmin(np.nanmean(E, axis=0)))]), p5=float(lo), p95=float(hi), distribution=dist)


# ----------------------------------------------------------------- 3. pilot-window m*
def pilot_test(rows, label):
    kmax = max(r["k"] for r in rows)
    wins = sorted({r["window"] for r in rows if r["k"] == kmax})
    ms = sorted({r["m"] for r in rows if r["k"] == kmax})
    E = {}
    for r in rows:
        if r["k"] == kmax and r["err"] is not None:
            E[(r["window"], r["m"])] = abs(r["err"]) * 1000
    med, _ = mae_table(rows)
    res = []
    for pilot in wins:
        curve = {m: E.get((pilot, m)) for m in ms}
        curve = {m: v for m, v in curve.items() if v is not None}
        if not curve:
            continue
        m_pilot = min(curve, key=curve.get)
        rest = [w for w in wins if w != pilot]
        rc = {m: np.mean([E[(w, m)] for w in rest if (w, m) in E]) for m in ms if any((w, m) in E for w in rest)}
        m_rest = min(rc, key=rc.get)
        # does the pilot's m* pick the right side of each equal-budget pair on the rest?
        res.append(dict(pilot=list(pilot), m_pilot=int(m_pilot), m_rest=int(m_rest), log_ratio=float(abs(np.log(m_pilot / m_rest)))))
    ok = sum(1 for r in res if r["log_ratio"] <= np.log(1.5))
    print(f"[{label}] pilot-window m*: within x1.5 of the held-out m* in {ok}/{len(res)} windows; " + ", ".join(f"{r['m_pilot']}->{r['m_rest']}" for r in res))
    return dict(within_1p5x=ok, n=len(res), detail=res)


# ----------------------------------------------------------------- 4. figures
def figures(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (label, rows) in zip(axes, data.items()):
        med, _ = mae_table(rows)
        kmax, curve = kmax_curve(rows)
        ks = sorted({k for k, m in med})
        for k in ks:
            xs = sorted(m for kk, m in med if kk == k)
            ax.plot(xs, [med[(k, m)] for m in xs], marker="o", ms=3, lw=1.2,
                    label=f"k={k}" + (" (all)" if k == kmax else " (median subset)"))
        ax.axvline(min(curve, key=curve.get), color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("instants per 4 s window, m")
        ax.set_ylabel("MAE of path length (mm)")
        ax.set_title(label)
        ax.grid(True, which="both", lw=0.3, alpha=0.5)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "goldilocks_curves.png", dpi=170)
    # allocation surface: MAE over (k, m) with iso-budget lines
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (label, rows) in zip(axes, data.items()):
        med, _ = mae_table(rows)
        ks = sorted({k for k, m in med})
        ms = sorted({m for k, m in med})
        Z = np.full((len(ks), len(ms)), np.nan)
        for (k, m), v in med.items():
            Z[ks.index(k), ms.index(m)] = np.log10(v)
        im = ax.imshow(Z, aspect="auto", origin="lower", cmap="viridis_r")
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels(ms, fontsize=7)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels(ks)
        ax.set_xlabel("m")
        ax.set_ylabel("k")
        ax.set_title(f"{label}: log10 MAE (mm)")
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "allocation_surfaces.png", dpi=170)


def main():
    mc = load_mcalib()
    ly = load_lyon()
    data = {"MCalib (records 15-18)": mc,
            "Lyon RFM5 (dev)": [r for r in ly if r["target"] == "RFM5"],
            "Lyon SV (dev)": [r for r in ly if r["target"] == "SV"]}
    results = {}
    for label, rows in data.items():
        print(f"\n===== {label}")
        results[label] = dict(rule=rule_test(rows, label), mstar_bootstrap=bootstrap_mstar(rows, label), pilot=pilot_test(rows, label))
    figures(data)
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
