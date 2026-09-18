"""One-shot orchestration of the pre-registered Lyon confirmation (protocol v0.3).

    python src/run_lyon_confirmation.py --archive <Video_Data.zip> --markers <Markers.zip or folder> \
        --workers 4 [--dry-run]

Steps, in the order the protocol fixes them:
  0. verify the freeze (experiments/lyon_prereg_v0.3/freeze.json) still matches the sources;
  1. extract Video_Data.zip (Deflate64-aware) for the confirmation trials only, into the
     encrypted data folder; skip the excluded trials (participant_02/gait, exotic, dance);
  2. check Calib.toml is byte-identical to the public one (else abort: a new protocol version is needed);
  3. run the frozen src/lyon_observe.py per trial (reads the C3D only to place the +/-8 px windows);
  4. predict per trial, per target, sharded over subsets, in parallel (sharding lives in this
     script: the frozen runner is imported unchanged and its subset list is sliced); the shard
     files are merged into one prediction file per trial and sealed;
  5. score (reference lengths computed only now), then decide (H1-H5);
  6. write reports/LYON_PREREG_RESULT_<date>.md skeleton with the decision tables.
Everything is written under experiments/lyon_prereg_v0.3/confirmation/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/lyon_prereg_v0.3"
DATA = ROOT / "data_external/lbmc_lyon_2026-09-15"
PUBLIC_CALIB = DATA / "Example_Anonymized_Video_Data/calibration/Calib.toml"
TRIALS = {  # confirmation trials of protocol v0.3 section 6
    "participant_01/gait": "gait.c3d", "participant_01/sit-stand": "sit-stand.c3d", "participant_01/mmh": "mmh.c3d",
    "participant_02/sit-stand": "sit-stand.c3d", "participant_02/mmh": "mmh.c3d",
}
TARGETS = ["RFM5", "SV", "RFCC", "LSAT"]
PY = sys.executable


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(cmd, log: Path | None = None, cwd=ROOT):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    if log:
        with log.open("a", encoding="utf-8") as fh:
            return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    return subprocess.run(cmd, cwd=cwd)


def extract(archive: Path, dest: Path, wanted_prefixes: list[str]):
    """Deflate64-aware selective extraction (same logic as src/extract_deflate64_zip.py)."""
    import struct
    import zlib
    try:
        import inflate64
    except ImportError:
        raise SystemExit("pip install inflate64")
    z = zipfile.ZipFile(archive)
    n = 0
    with archive.open("rb") as fp:
        for info in z.infolist():
            if info.is_dir() or not any(w in info.filename for w in wanted_prefixes):
                continue
            fp.seek(info.header_offset)
            sig, _, _, _, _, _, _, _, _, nlen, elen = struct.unpack("<IHHHHHIIIHH", fp.read(30))
            assert sig == 0x04034B50
            fp.seek(info.header_offset + 30 + nlen + elen)
            out = dest / info.filename
            out.parent.mkdir(parents=True, exist_ok=True)
            remaining, crc = info.compress_size, 0
            inf = inflate64.Inflater() if info.compress_type == 9 else None
            with out.open("wb") as fo:
                while remaining > 0:
                    chunk = fp.read(min(1 << 22, remaining))
                    remaining -= len(chunk)
                    data = inf.inflate(chunk) if inf else chunk
                    crc = zlib.crc32(data, crc)
                    fo.write(data)
            if (crc & 0xFFFFFFFF) != info.CRC:
                raise SystemExit(f"CRC failure: {info.filename}")
            n += 1
    print(f"extracted {n} files to {dest}")


def shard_worker(out: Path, obs: Path, trial: str, target: str, i: int, n: int):
    """Run the FROZEN runner's step_predict on a slice of the subset list. Does not modify the runner."""
    out, obs = Path(out).resolve(), Path(obs).resolve()  # the frozen runner seals paths relative to ROOT
    sys.path.insert(0, str(ROOT / "src"))
    import lyon_prereg_runner as R
    orig = R.protocol_subsets

    def sliced(cams):
        subs, rule, center = orig(cams)
        return subs[i::n], rule, center
    R.protocol_subsets = sliced
    shard_out = out / "shards" / f"{trial.replace('/', '_')}__{target}__{i}of{n}"
    shard_out.mkdir(parents=True, exist_ok=True)
    for f in ("freeze.json", "config.json"):  # the frozen step_predict verifies sources against freeze.json in its out dir
        shutil.copyfile(FREEZE / f, shard_out / f)
    R.step_predict(shard_out, trial, obs, dev=False, targets=[target], only_rule=False)


def merge_shards(out: Path, trial: str):
    """Concatenate shard predictions into the single per-trial file the frozen `score` expects, and seal it."""
    out = Path(out).resolve()
    tag = trial.replace("/", "_")
    rows, targets, meta = [], set(), None
    for d in sorted((out / "shards").glob(f"{tag}__*")):
        P = json.loads((d / "predictions" / f"{tag}.json").read_text())
        rows += P["rows"]
        targets |= set(P["targets"])
        meta = meta or P
    pd = out / "predictions" / f"{tag}.json"
    pd.parent.mkdir(parents=True, exist_ok=True)
    pd.write_text(json.dumps(dict(trial=trial, observations=meta["observations"], observer_meta_sha256=meta["observer_meta_sha256"],
                                  targets=sorted(targets), estimator_rule_state_mean_max_m=meta.get("estimator_rule_state_mean_max_m"),
                                  merged_from_shards=True, rows=rows), indent=0))
    seal = dict(file=str(pd.relative_to(ROOT)), sha256=sha(pd), rows=len(rows), utc=datetime.now(timezone.utc).isoformat(), dev=False)
    with (out / "seal.json").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(seal) + "\n")
    print("merged + sealed", trial, len(rows), "rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-worker", nargs=6, metavar=("OUT", "OBS", "TRIAL", "TARGET", "I", "N"), help="internal")
    ap.add_argument("--archive", required=True, help="Video_Data.zip from the restricted download")
    ap.add_argument("--markers", required=True, help="Markers.zip or the extracted Markers folder")
    ap.add_argument("--dest", default=str(DATA / "restricted"), help="where to extract (use an encrypted volume)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.shard_worker:
        o, ob, t, tg, i, n = a.shard_worker
        shard_worker(Path(o), Path(ob), t, tg, int(i), int(n))
        return
    dest = Path(a.dest)
    out = FREEZE / "confirmation"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "orchestration.log"

    # 0. freeze intact
    fz = json.loads((FREEZE / "freeze.json").read_text())
    for src, h in fz["hashes"].items():
        if sha(ROOT / src) != h:
            raise SystemExit(f"frozen source changed since freeze: {src}")
    print("freeze verified:", fz["utc"])
    if a.dry_run:
        print("dry run: would extract", list(TRIALS), "from", a.archive)
        return

    # 1. extract videos + calibration for the confirmation trials only
    wanted = ["calibration/Calib.toml"] + [f"{t}/videos/" for t in TRIALS]
    extract(Path(a.archive), dest, wanted)
    root = next(p for p in dest.rglob("Calib.toml")).parents[1]
    # 2. calibration identical?
    if sha(root / "calibration/Calib.toml") != sha(PUBLIC_CALIB):
        raise SystemExit("Calib.toml differs from the public slice: STOP and issue protocol v0.4")
    print("Calib.toml identical to the public slice")
    # markers
    mk = Path(a.markers)
    if mk.suffix == ".zip":
        mdir = dest / "Markers_extracted"
        zipfile.ZipFile(mk).extractall(mdir)
        mk = mdir / "Markers"
    inputs = {t: dict(c3d_sha256=sha(mk / t.split("/")[0] / c)) for t, c in TRIALS.items()}
    (out / "inputs.json").write_text(json.dumps(dict(archive_sha256=sha(Path(a.archive)), c3d=inputs,
                                                     utc=datetime.now(timezone.utc).isoformat()), indent=1))

    # 3. frozen observer per trial
    for t, c in TRIALS.items():
        obs = out / "observations" / t.replace("/", "_")
        if (obs / "observe_meta.json").exists():
            continue
        run([PY, "-u", str(ROOT / "src/lyon_observe.py"), "--root", str(root), "--trial", t,
             "--c3d", str(mk / t.split("/")[0] / c), "--out", str(obs), "--markers", ",".join(TARGETS)], log)

    # 4. predictions: per trial, per target, sharded over subsets, at most `workers` processes at once
    jobs = [(t, target, i) for t in TRIALS for target in TARGETS for i in range(a.workers)]
    running = []
    import time
    for t, target, i in jobs:
        obs = out / "observations" / t.replace("/", "_")
        cmd = [PY, "-u", str(Path(__file__)), "--archive", a.archive, "--markers", a.markers,
               "--shard-worker", str(out), str(obs), t, target, str(i), str(a.workers)]
        running.append(subprocess.Popen(cmd, cwd=ROOT, stdout=log.open("a"), stderr=subprocess.STDOUT))
        while sum(p.poll() is None for p in running) >= a.workers:
            time.sleep(5)
    for p in running:
        p.wait()
    for t in TRIALS:
        merge_shards(out, t)

    # 5. score and decide
    for t, c in TRIALS.items():
        run([PY, "-u", str(ROOT / "src/lyon_prereg_runner.py"), "score", "--out", str(out.relative_to(ROOT)),
             "--trial", t, "--c3d", str(mk / t.split("/")[0] / c)], log)
    run([PY, "-u", str(ROOT / "src/lyon_prereg_runner.py"), "decide", "--out", str(out.relative_to(ROOT))], log)

    # 6. report skeleton
    D = json.loads((out / "decision.json").read_text())
    date = datetime.now().strftime("%Y-%m-%d")
    rep = ROOT / f"reports/LYON_PREREG_RESULT_{date}.md"
    lines = [f"# Lyon 预注册确认结果（协议 v0.3）\n", f"生成时间 {D['utc']}；试验 {D['trials']}；目标 {D['targets']}。\n",
             "自动生成的判定摘要，逐条与 protocol/LYON_PREREG_v0.3.md §7 对照；叙述与表格待人工补全。\n"]
    for target, H in D["hypotheses"].items():
        lines.append(f"\n## {target}\n")
        lines.append(f"- H1: not saturated at all m: {H['H1']['pass_a']}; corr at m*={H['H1']['m_star_k9']}: {H['H1']['corr_at_m_star']} -> {H['H1']['pass_b']}")
        for B, v in H["H2_H3"].items():
            u = v["uniform"]
            lines.append(f"- H2 B={B} {v['comparison']} (pred {v['predicted']}): {u['median_triple_mae_mm']} vs {u['other_mae_mm']} -> {v['H2_pass']}; H3 grids failing {v['grids_failing']}/11, 3-view better {v['three_view_better_grids']}/11 -> {v['H3_pass']}")
        for m, v in H["H4"].items():
            lines.append(f"- H4 m={m}: completion pairs_dlt {v['pairs_dlt']}, triples_dlt {v['triples_dlt']}, spatial {v['triples_spatial']}, rig {v['triples_spatial_rig']} (a {v['a_pass']} b {v['b_pass']} d {v['d_pass']})")
        lines.append(f"- H5: m*={H['H5']['m_star']} pass={H['H5']['pass_']}")
        lines.append(f"- **H2 {H['H2_pass']}, H3 {H['H3_pass']}**")
    rep.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", rep)


if __name__ == "__main__":
    main()
