"""Build the v1.1.0 reproducibility archive for Zenodo (adds the DPJAIT rig).

Superset of `build_zenodo_archive.py` (v1.0.x, MCalib + Lyon pre-registration):
this version also carries the DPJAIT pre-registrations (v0.1, the two noise-ladder
protocols and the reference-free predictor) (protocol v0.1, the frozen
runner and its source snapshot, the sealed predictions, the scored results, the
decision and both errata), the cross-rig mechanism analysis, and the current TIM
manuscript with its figures.

What stays out: the third-party datasets themselves (MCalib, LBMC Lyon, DPJAIT).
None is ours to redistribute; the archive carries input hash lists instead.

Usage:
    python src/build_zenodo_archive_v110.py                 # build
    python src/build_zenodo_archive_v110.py --verify        # re-check
    python src/build_zenodo_archive_v110.py --no-sim-rows   # drop the three
        simulated records' prediction/scored files (~700 MB raw) and keep their
        seals, decision and error structure only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/zenodo"
VERSION = "1.1.0"
NAME = f"view_instant_allocation_v{VERSION}"

# --- code -------------------------------------------------------------------
CODE_KEYS = ("allocation", "adaptive", "ctsd", "bearing", "gated", "uncertainty", "paired_view",
             "view_geometry", "view_saturation", "tum_dense", "make_mst_figures", "make_ieee_figures",
             "check_mst", "verify_manuscript", "score_allocation", "fit_allocation", "plot_allocation",
             "plot_tum", "analyze_allocation", "dpjait", "lyon", "cross_rig", "make_tim_figures",
             "make_dpjait_figures", "make_mechanism_figure", "own_blob", "mstar_closed_form",
             "check_tim_compliance", "dpjait_noise", "refree", "bootstrap_noise", "diagnose_mstar",
             "learned_det", "crossref_lookup", "make_elsevier_refs", "check_measurement")
CODE_CORE = ("limited_arc_cpu.py", "continuous_motion_cpu.py", "three_view_cpu.py",
             "robust_triangulation_cpu.py", "mcalib_cache_observer.py", "audit_mcalib_sample.py",
             "prepare_mcalib_cache.py", "calibrated_measurement_core_v1.py", "rgbd_rigid_route_cpu.py",
             "rgbd_odometry_cpu.py", "pose_propagation_cpu.py", "feature_chain_core.py",
             "xfeat_cpu_adapter.py", "run_fixed_observer_replication_cpu.py", "run_tum_rgbd_cpu.py",
             "submission_sensitivity_analysis.py", "audit_submission.py", "audit_citation_targets.py",
             "build_submission_package.py", "build_zenodo_archive.py",
             "build_zenodo_archive_v110.py", "publish_zenodo_version.py",
             "make_revision_budget_figure.py", "verify_budget_revision.py",
             "make_grid_robustness_figure.py", "verify_grid_revision.py",
             "extract_deflate64_zip.py")
TESTS = ("test_dpjait_prereg_v1.py", "test_dpjait_chain_v1.py", "test_lyon_prereg_v1.py",
         "test_dpjait_noise_runner.py")


# --- experiment directories -------------------------------------------------
EXP_KEYS = ("allocation_", "adaptive_moments", "ctsd_baseline", "view_saturation",
            "uncertainty_budget", "tum_dense_cheap", "allocator_dev", "submission_sensitivity_",
            "dpjait_prereg_v0.1", "dpjait_chain_dev", "dpjait_geometry", "lyon_prereg_v0.2",
            "lyon_prereg_v0.3", "lyon_prereg_dev_full", "cross_rig_analysis",
            "dpjait_noise_prereg_v0.1", "dpjait_noise_dev", "dpjait_noise_prereg_v0.2",
            "refree_mstar_prereg_v0.1", "refree_mstar_dev", "refree_identity_sensitivity",
            "noise_bootstrap_ci", "mstar_underprediction_diag", "dpjait_learned_det")
SIM_RECORDS = ("S08_D8", "S09_D6", "S10_D6")

# --- documents --------------------------------------------------------------
DOCS = ("protocol/DPJAIT_PREREG_v0.1.md", "protocol/DPJAIT_NOISE_PREREG_v0.1.md",
        "protocol/DPJAIT_NOISE_PREREG_v0.2.md", "protocol/REFREE_MSTAR_PREREG_v0.1.md",
        "reports/DPJAIT_NOISE_PREREG_v0.2_RESULT_2026-09-26.md", "reports/NOISE_BOOTSTRAP_CI_2026-09-26.md",
        "reports/MSTAR_UNDERPREDICTION_DIAGNOSIS_2026-09-26.md", "reports/REFREE_MSTAR_DEV_2026-09-26.md",
        "reports/REFREE_MSTAR_PREREG_RESULT_2026-09-26.md", "reports/REFREE_IDENTITY_SENSITIVITY_2026-09-26.md",
        "protocol/RIG_D_SELFBUILT_DESIGN_v0.1.md",
        "reports/DPJAIT_NOISE_PREREG_RESULT_2026-09-25.md",
        "paper/tim/fig_noise_knob.pdf", "paper/tim/fig_noise_knob_a.pdf", "paper/tim/edics.txt", "protocol/LYON_PREREG_v0.2.md",
        "protocol/LYON_PREREG_v0.3.md",
        "paper/tim/tim_manuscript.tex", "paper/tim/tim_manuscript.pdf",
        "paper/tim/fig_mechanism.pdf", "paper/tim/fig_dpjait_goldilocks.pdf",
        "paper/tim/fig_dpjait_views.pdf", "paper/tim/fig_dpjait_identity.pdf",
        "paper/tim/fig_dpjait_bias.pdf", "paper/tim/fig_goldilocks.pdf", "paper/tim/fig_budget.pdf",
        "paper/tim/figure_values.json", "paper/tim/figure_values_dpjait.json",
        "paper/tim/figure_values_mechanism.json",
        "paper/mst/mst_manuscript.tex", "paper/mst/novelty_statement.txt",
        "reports/OBSERVATION_ALLOCATION_DENSE_FEWVIEW_2026-09-15.md",
        "reports/ALLOCATION_FOLLOWUPS_RULE_LAW_TUM_2026-09-15.md",
        "reports/ALLOCATION_CLOSEOUT_AND_PUBLISHABILITY_2026-09-15.md",
        "reports/LYON_GEOMETRY_SELFCHECK_2026-09-17.md",
        "reports/LYON_DETECTOR_FREEZE_2026-09-17.md",
        "reports/LYON_DEV_SWEEP_2026-09-18.md",
        "reports/CROSS_RIG_SYNTHESIS_2026-09-18.md",
        "reports/CROSS_RIG_RULE_TEST_2026-09-18.md",
        "reports/DPJAIT_AUDIT_2026-09-19.md",
        "reports/DPJAIT_CHAIN_DEV_2026-09-19.md",
        "reports/DPJAIT_PREREG_RESULT_2026-09-20.md")
REPO_FILES = (".zenodo.json", "CITATION.cff", "LICENSE", "NOTICE.md",
              "experiments/mcalib_3d_cache_2026-09-13_verified/public/calibration.json")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def collect_code():
    src = ROOT / "src"
    code = {p for p in src.glob("*.py") if any(k in p.name for k in CODE_KEYS)}
    code |= {src / c for c in CODE_CORE if (src / c).exists()}
    code |= {ROOT / "tests" / t for t in TESTS if (ROOT / "tests" / t).exists()}
    return sorted(code)


# regenerable or bulky intermediates kept out of the archive: the detector training
# set (2.9 GB of DPJAIT frames, rebuilt by learned_det_dataset.py), the last-epoch
# and failed-run weights, and the split-camera cache of the M4 development
EXCLUDE_PARTS = ("dataset", "run_failed_2026-09-25_no_polars", "m4_cache")
EXCLUDE_NAMES = ("last.pt",)


def wanted(p: Path, keep_sim_rows: bool) -> bool:
    parts = p.parts
    if "dpjait_learned_det" in parts and (any(x in parts for x in EXCLUDE_PARTS) or p.name in EXCLUDE_NAMES):
        return False
    if "m4_cache" in parts:
        return False
    if keep_sim_rows:
        return True
    if "dpjait_prereg_v0.1" in parts and parts[-2] in ("predictions", "scored"):
        return not any(p.name.startswith(r) for r in SIM_RECORDS)
    return True


def build(keep_sim_rows: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    zip_path = OUT / f"{NAME}.zip"
    if zip_path.exists():
        raise SystemExit(f"{zip_path} already exists; delete it first to rebuild")
    exps = sorted(d for d in (ROOT / "experiments").iterdir()
                  if d.is_dir() and any(k in d.name for k in EXP_KEYS))
    manifest, skipped = {}, 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in collect_code():
            arc = f"{NAME}/{p.relative_to(ROOT)}".replace("\\", "/")
            z.write(p, arc)
            manifest[arc] = sha(p)
        for d in exps:
            for p in sorted(d.rglob("*")):
                if not p.is_file():
                    continue
                if not wanted(p, keep_sim_rows):
                    skipped += 1
                    continue
                arc = f"{NAME}/experiments/{p.relative_to(ROOT / 'experiments')}".replace("\\", "/")
                z.write(p, arc)
                manifest[arc] = sha(p)
        for rel in DOCS:
            p = ROOT / rel
            if p.exists():
                arc = f"{NAME}/{rel}"
                z.write(p, arc)
                manifest[arc] = sha(p)
        repo = ROOT / "paper/repo"
        for rel in REPO_FILES:
            p = repo / rel
            if p.exists():
                arc = f"{NAME}/{rel}"
                z.write(p, arc)
                manifest[arc] = sha(p)
        readme = repo / "README.md"
        if readme.exists():
            z.write(readme, f"{NAME}/README.md")
            manifest[f"{NAME}/README.md"] = sha(readme)
        mpath = OUT / f"manifest_v{VERSION}.json"
        mpath.write_text(json.dumps({"archive": zip_path.name, "version": VERSION,
                                     "sim_rows_included": keep_sim_rows,
                                     "files": manifest,
                                     "built_utc": datetime.now(timezone.utc).isoformat()},
                                    indent=1), encoding="utf-8")
        z.write(mpath, f"{NAME}/manifest.json")
    digest = sha(zip_path)
    (OUT / f"{NAME}.sha256").write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    print(f"archive: {zip_path}")
    print(f"  size    {zip_path.stat().st_size / 1e6:.1f} MB")
    print(f"  files   {len(manifest)} ({skipped} simulated-record files skipped)")
    print(f"  sha256  {digest}")


def verify() -> None:
    zip_path = OUT / f"{NAME}.zip"
    man = json.loads((OUT / f"manifest_v{VERSION}.json").read_text(encoding="utf-8"))["files"]
    bad = []
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for arc, want in man.items():
            if arc not in names:
                bad.append(f"missing from archive: {arc}")
                continue
            h = hashlib.sha256()
            with z.open(arc) as f:
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
            if h.hexdigest() != want:
                bad.append(f"hash mismatch: {arc}")
    print(f"verified {len(man)} entries, {len(bad)} problems")
    for b in bad:
        print("  ", b)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-sim-rows", action="store_true",
                    help="omit the three simulated records' prediction/scored files")
    a = ap.parse_args()
    verify() if a.verify else build(keep_sim_rows=not a.no_sim_rows)
