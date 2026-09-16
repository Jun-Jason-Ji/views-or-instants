"""Prospectively frozen, new-sequence fixed-observer CPU replication."""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import run_fixed_bidirectional_cpu as old
from run_tum_rgbd_cpu import timestamps, save
from pose_propagation_cpu import interpolate_pose, pose_errors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/fixed_observer_replication_2026-09-13'
SEQUENCES = ['desk2', 'plant']
METHODS = ['forward', 'reverse', 'mean_fallback']


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def data_dir(seq):
    return ROOT / f'data_external/tum_rgbd_{seq}_2026-09-13'


def freeze():
    old.verify()
    OUT.mkdir(exist_ok=False)
    config = dict(
        utc=stamp(), sequences=SEQUENCES, methods=METHODS, budgets=[16, 12],
        scope='Prospective new-sequence replication; same Freiburg1 acquisition setting, camera egomotion, not cross-domain or moving-object task.',
        prior_contact='xyz/rpy raw and reference seen; desk reference seen and excluded. No desk2/plant data or result contact found in project before freeze. Official descriptive metadata consulted only.',
        acquisition='Official complete archives; each <=768 MiB. No substitute sequence after viewing results; acquisition failures retained.',
        windows='Public RGB time base +1s, then consecutive non-overlapping half-open [start,end) 4s windows, all complete windows ending <= last RGB timestamp, at most 12 per sequence. Trailing incomplete interval excluded by public duration only. >=33 paired frames, span>=3.8s, max pair gap<=.1s; invalid public windows retained as failures, no replacement.',
        association='Unchanged each RGB nearest depth within .02s, nearest RGB per depth retained, sorted by time.',
        sampling='Unchanged rounded linspace of full paired frame indices, endpoints included; identical timestamps across methods at each budget.',
        observer='Call frozen run_fixed_bidirectional_cpu.query without changes: forward, inverse reverse, equal SO3 midpoint + translation mean with forward fallback; same ORB/depth/PnP/K/seed/sigma and q fitting. No joint fused reprojection gate.',
        primary='B16 mean_fallback versus forward. Pooled paired path absolute-error gain >0 and mean position RMSE gain >=0; same signs in EACH sequence; no added failed query on any forward success. At least 3 reference-valid common successes per sequence and >=6 total, otherwise insufficient evidence. Minimum count is a descriptive evidence floor, not a power calculation.',
        secondary='B12 stress; reverse versus forward; fusion versus reverse. All pairwise common-success and triple-common tables. No method choice or threshold tuning with new GT. Fusion versus reverse gains do not establish equal-compute superiority.',
        evaluation='Save all sequences predictions before parsing any GT; same dense native-meter reference polygon, gap>.1s rejected, no scale fit or full trajectory alignment; pose errors at selected timestamps anchored only at first GT pose.',
        accounting='Three methods run separately; per-query cache only, failed work charged. Cyclic order rotation by window/budget, single OpenCV thread; no strict runtime benchmark. Each successful directional chain B-1 attempts, fusion at most 2(B-1).',
        decision='CPU replication only, no automatic training/GPU. Mixed signs/new failure fail replication screen; inadequate common cases insufficient, never silently replace windows.',
        official_sources=['https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download', 'https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats'])
    save(OUT / 'config.json', config)
    paths = list((ROOT / 'src').glob('*.py')) + list((ROOT / 'tests').glob('test*.py')) + [OUT / 'config.json']
    save(OUT / 'freeze.json', dict(utc=stamp(), hashes={str(p.relative_to(ROOT)): digest(p) for p in paths}))
    for p in paths:
        if p.suffix == '.py':
            target = OUT / 'source_snapshot' / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
    print('Protocol and implementation frozen before data acquisition.', flush=True)


def verify():
    old.verify()
    for p, h in json.loads((OUT / 'freeze.json').read_text())['hashes'].items():
        assert digest(ROOT / p) == h, p


def associate(rgb, depth):
    dt = np.array([t for t, _ in depth])
    candidates = []
    for i, (t, _) in enumerate(rgb):
        j = int(np.argmin(abs(dt-t)))
        if abs(dt[j]-t) <= .02:
            candidates.append((float(abs(dt[j]-t)), i, j))
    used, pairs = set(), []
    for delta, i, j in sorted(candidates):
        if j not in used:
            used.add(j)
            pairs.append(dict(t=rgb[i][0], rgb=rgb[i][1], depth=depth[j][1], sync_delta_s=delta))
    return sorted(pairs, key=lambda p: p['t'])


def windows_for(rgb, pairs):
    windows = []
    for k in range(12):
        offset = 1. + 4*k
        begin, end = rgb[0][0]+offset, rgb[0][0]+offset+4.
        if end > rgb[-1][0]:
            break
        rows = [p for p in pairs if begin <= p['t'] < end]
        invalid = None
        if len(rows) < 33:
            invalid = 'insufficient_paired_frames'
        elif rows[-1]['t']-rows[0]['t'] < 3.8 or max(np.diff([p['t'] for p in rows])) > .1:
            invalid = 'insufficient_public_temporal_coverage'
        windows.append(dict(offset=offset, requested_times=[begin, end], frames=rows, public_failure=invalid))
    return windows


def prepare():
    verify()
    bindings, audit = {}, {}
    for seq in SEQUENCES:
        data = data_dir(seq)
        manifest = json.loads((data / 'manifest.json').read_text())
        assert manifest['gzip_crc_verified']
        for f in manifest['saved']:
            assert digest(data / f['path']) == f['sha256'], f['path']
            bindings[str((data / f['path']).relative_to(ROOT))] = f['sha256']
        # Hashing reference bytes for integrity is not parsing them for design/scoring.
        bindings[str((data / 'manifest.json').relative_to(ROOT))] = digest(data / 'manifest.json')
        rgb, depth = timestamps(data / 'public/rgb.txt'), timestamps(data / 'public/depth.txt')
        pairs = associate(rgb, depth)
        windows = windows_for(rgb, pairs)
        dest = OUT / seq
        dest.mkdir(exist_ok=False)
        save(dest / 'public_windows.json', windows)
        bindings[str((dest / 'public_windows.json').relative_to(ROOT))] = digest(dest / 'public_windows.json')
        audit[seq] = dict(rgb_count=len(rgb), depth_count=len(depth), paired_count=len(pairs),
                          public_duration_s=rgb[-1][0]-rgb[0][0], windows=len(windows),
                          invalid_public_windows=sum(w['public_failure'] is not None for w in windows),
                          trailing_interval_s=rgb[-1][0]-(windows[-1]['requested_times'][1] if windows else rgb[0][0]))
    save(OUT / 'input_bindings.json', dict(utc=stamp(), hashes=bindings, audit=audit))
    print(json.dumps(audit), flush=True)


def verify_inputs():
    verify()
    for p, h in json.loads((OUT / 'input_bindings.json').read_text())['hashes'].items():
        assert digest(ROOT / p) == h, p


def run():
    verify_inputs()
    cv2.setNumThreads(1)
    started = stamp()
    rows = []
    for seq in SEQUENCES:
        windows = json.loads((OUT / seq / 'public_windows.json').read_text())
        for wi, w in enumerate(windows):
            for bi, budget in enumerate([16, 12]):
                start = (wi + bi) % 3
                order = METHODS[start:] + METHODS[:start]
                for method in order:
                    if w['public_failure']:
                        r = dict(sequence=seq, offset=w['offset'], budget=budget, method=method,
                                 failure=w['public_failure'], selected=[], read_indices=[], times=[],
                                 values_m={}, poses=[], accepted_edges=[], events=[], fallbacks=[],
                                 rgbd_pairs_decoded=0, match_attempts=0, wall_s=0, q=None)
                    else:
                        r = old.query(w, seq, budget, method)
                    r['execution_order'] = order
                    rows.append(r)
            print(seq, w['offset'], 'all methods predicted', flush=True)
    save(OUT / 'predictions.json', rows)
    save(OUT / 'prediction_seal.json', dict(started_utc=started, completed_utc=stamp(), sha256=digest(OUT / 'predictions.json'), queries=len(rows)))
    verify_inputs()


def reference_metrics(gt, times):
    if not times:
        return None, 'query_no_times'
    begin, end = times[0], times[-1]
    if begin < gt[0, 0] or end > gt[-1, 0]:
        return None, 'outside_reference'
    lo, hi = max(0, np.searchsorted(gt[:, 0], begin)-1), min(len(gt), np.searchsorted(gt[:, 0], end)+1)
    if np.max(np.diff(gt[lo:hi, 0])) > .1:
        return None, 'reference_gap'
    endpoints = np.array([[np.interp(t, gt[:, 0], gt[:, k]) for k in (1, 2, 3)] for t in (begin, end)])
    ref = np.vstack([endpoints[0], gt[(gt[:, 0]>begin)&(gt[:, 0]<end), 1:4], endpoints[1]])
    length = float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum())
    ii = np.unique(np.r_[np.arange(0, len(ref), 2), len(ref)-1])
    return dict(path_m=length, displacement_m=float(np.linalg.norm(endpoints[1]-endpoints[0])),
                duration_s=end-begin, stride2_path_delta_m=float(np.linalg.norm(np.diff(ref[ii], axis=0), axis=1).sum()-length)), None


def means(rows):
    if not rows:
        return dict(n=0)
    return dict(n=len(rows), mae_m={k:float(np.mean([abs(r['errors_m'][k]) for r in rows])) for k in ['state_mean','limited','polygon','displacement']},
                mean_position_rmse_m=float(np.mean([r['pose_errors']['position_rmse_m'] for r in rows])),
                mean_orientation_deg=float(np.mean([r['pose_errors']['orientation_mean_deg'] for r in rows])))


def validate_query(r, window):
    if r['selected']:
        expected = np.rint(np.linspace(0,len(window['frames'])-1,r['budget'])).astype(int).tolist()
        assert r['selected'] == expected
        assert r['rgbd_pairs_decoded'] == len(r['read_indices']) <= r['budget']
        assert r['read_indices'] == r['selected'][:len(r['read_indices'])]
        assert len(set(r['selected'])) == r['budget']
    assert r['match_attempts'] <= (2 if r['method']=='mean_fallback' else 1)*(r['budget']-1)
    if r['failure'] is None:
        assert r['rgbd_pairs_decoded'] == r['budget']
        assert len(r['poses']) == r['budget'] and len(r['accepted_edges']) == r['budget']-1
    else:
        assert not r['values_m']


def evaluate():
    verify_inputs()
    seal = json.loads((OUT / 'prediction_seal.json').read_text())
    assert digest(OUT / 'predictions.json') == seal['sha256']
    scoring_started = stamp()
    predictions = json.loads((OUT / 'predictions.json').read_text())
    windows = {(seq,w['offset']):w for seq in SEQUENCES for w in json.loads((OUT / seq / 'public_windows.json').read_text())}
    assert len(predictions) == 6*len(windows)
    assert len({(r['sequence'],r['offset'],r['budget'],r['method']) for r in predictions}) == len(predictions)
    gt = {seq:np.loadtxt(data_dir(seq) / 'reference/groundtruth.txt') for seq in SEQUENCES}
    evaluated = []
    for r in predictions:
        validate_query(r, windows[r['sequence'],r['offset']])
        truth, invalid = reference_metrics(gt[r['sequence']], r['times'])
        errors, pose = None, None
        if r['failure'] is None:
            poses = np.array(r['poses'])
            assert np.allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-10)
            if invalid is None:
                errors = {k:v-truth['displacement_m' if k=='displacement' else 'path_m'] for k,v in r['values_m'].items()}
                gp = np.array([interpolate_pose(gt[r['sequence']], t) for t in r['times']])
                pose = pose_errors(poses, np.linalg.inv(gp[0]) @ gp)
        else:
            assert not r['values_m']
        evaluated.append(dict(sequence=r['sequence'],offset=r['offset'],budget=r['budget'],method=r['method'],
                              failure=r['failure'],reference_failure=invalid,truth=truth,errors_m=errors,pose_errors=pose))
    summary, paired = {}, []
    index = {(r['sequence'],r['offset'],r['budget'],r['method']):r for r in evaluated}
    for budget in [16, 12]:
        summary[str(budget)] = {}
        for seq in SEQUENCES + ['pooled']:
            rows = [r for r in evaluated if r['budget']==budget and (seq=='pooled' or r['sequence']==seq)]
            keys = sorted({(r['sequence'], r['offset']) for r in rows})
            common = [(s,o) for s,o in keys if all(index[s,o,budget,m]['errors_m'] is not None for m in METHODS)]
            own = {}
            for method in METHODS:
                rr = [r for r in rows if r['method']==method]
                pr = [r for r in predictions if r['budget']==budget and r['method']==method and (seq=='pooled' or r['sequence']==seq)]
                own[method] = dict(total=len(rr),success=sum(r['failure'] is None for r in rr),
                                   reference_invalid=sum(r['reference_failure'] is not None for r in rr),
                                   own_valid=means([r for r in rr if r['errors_m'] is not None]),
                                   triple_common=means([index[s,o,budget,method] for s,o in common]),
                                   mean_attempts=float(np.mean([r['match_attempts'] for r in pr])),
                                   median_wall_s=float(np.median([r['wall_s'] for r in pr])),
                                   fallback_edges=sum(len(r['fallbacks']) for r in pr))
            comparisons = {}
            for baseline, method in [('forward','mean_fallback'),('forward','reverse'),('reverse','mean_fallback')]:
                cases = []
                for s,o in keys:
                    a,z = index[s,o,budget,baseline],index[s,o,budget,method]
                    if a['errors_m'] is not None and z['errors_m'] is not None:
                        cases.append(dict(sequence=s,offset=o,budget=budget,baseline=baseline,method=method,
                                          path_gain_m=abs(a['errors_m']['state_mean'])-abs(z['errors_m']['state_mean']),
                                          position_gain_m=a['pose_errors']['position_rmse_m']-z['pose_errors']['position_rmse_m']))
                newfail = sum(index[s,o,budget,baseline]['failure'] is None and index[s,o,budget,method]['failure'] is not None for s,o in keys)
                comparisons[f'{method}_vs_{baseline}'] = dict(n=len(cases), added_failures=newfail,
                    mean_path_gain_m=float(np.mean([c['path_gain_m'] for c in cases])) if cases else None,
                    mean_position_gain_m=float(np.mean([c['position_gain_m'] for c in cases])) if cases else None,
                    improved=sum(c['path_gain_m']>.001 for c in cases),worsened=sum(c['path_gain_m']<-.001 for c in cases))
                if seq != 'pooled':
                    paired.extend(cases)
            summary[str(budget)][seq] = dict(methods=own,common_windows=common,comparisons=comparisons)
    main = [summary['16'][s]['comparisons']['mean_fallback_vs_forward'] for s in SEQUENCES]
    enough = all(r['n']>=3 for r in main) and sum(r['n'] for r in main)>=6
    signs = all(r['mean_path_gain_m'] is not None and r['mean_path_gain_m']>0 and r['mean_position_gain_m']>=0 and r['added_failures']==0 for r in main)
    decision = 'insufficient_evidence' if not enough else ('passed' if signs else 'failed')
    save(OUT / 'evaluation.json', evaluated)
    save(OUT / 'paired_comparisons.json', paired)
    save(OUT / 'summary.json', summary)
    save(OUT / 'validation.json', dict(scoring_started_utc=scoring_started,predictions_sealed_before_scoring=True,
        queries=len(predictions),primary_outcome=decision,primary_evidence_floor_met=enough,
        primary_direction_and_failure_criteria_met=signs,counts_and_pose_checks=True,
        total_rgbd_pairs_decoded=sum(r['rgbd_pairs_decoded'] for r in predictions),
        total_match_attempts=sum(r['match_attempts'] for r in predictions),query_wall_sum_s=sum(r['wall_s'] for r in predictions),
        predictions_sha256=seal['sha256']))
    print(json.dumps(dict(decision=decision,main=main)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze','verify','prepare','run','evaluate'])
    globals()[parser.parse_args().action]()
