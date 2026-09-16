"""Fixed, low-cost uncertainty-structure candidate versus all cached controls."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
import numpy as np
from propagated_uncertainty_cpu_v1 import registration_proxy, propagate_position_proxy, estimate_path, MODES
from continuous_motion_cpu import smooth
from pose_propagation_cpu import interpolate_pose

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'experiments/fr3_weighted_confirmation_v1_2026-09-13'
OUT = ROOT/'experiments/propagated_uncertainty_v1_2026-09-14'


def stamp(): return datetime.now(timezone.utc).isoformat()
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def save(p, value):
    with p.open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, allow_nan=False)
def key(r): return r['sequence'], r['offset'], r['budget']
def refpath(s): return ROOT/f'data_external/tum_rgbd_fr3_{s}_2026-09-13/reference/groundtruth.txt'
def seal(name, paths):
    save(OUT/name, dict(utc=stamp(), hashes={p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(paths))}))
def verify(path):
    for rel, sha in load(path)['hashes'].items():
        p = (ROOT/rel).resolve(); assert p.is_relative_to(ROOT) and digest(p) == sha, rel


def freeze():
    OUT.mkdir(exist_ok=False); old = load(SOURCE/'config.json')
    cfg = dict(utc=stamp(), stage='Seen fr3 development candidate test; no unseen confirmation, no model training.',
        sequences=old['sequences'], offsets=old['offsets_seconds'], budgets=[16, 32],
        geometries=['lg_baseline', 'unit_refit', 'depth_weighted', 'huber_refit', 'depth_huber_refit'], modes=list(MODES),
        primary=dict(geometry='depth_weighted', mode='propagated_full', budget=16),
        estimator='Same inherited Wiener-velocity prior, initialposition/velocity variance100,13 original q grid,8 quadrature nodes. Dense Gaussian conditioning with full observation covariance. q picked solely by own observation likelihood. iid replays old RTS, q exact/path<=1e-7m.',
        proxy='Invert cached target_from_source; y=A target, J=[I,-skew(y)], H=sum wJtJ; local C=(sum w||residual||²/(3N-6))*H^-1; weights normalizedmean1 on same frozen rawconsensus, frozenpermethod; collinear/condition>1e12/invalid inputs failclosed.',
        propagation='Independent edge leftSE3 perturbations. Jji=[Ri,-skew(pj-pi)Ri] forj>i; sum JCiJt acrossedges creates temporal positioncorrelation. Does not model adjacent-edge sharedframe or correspondence correlation; conditional firstorder proxy, not calibratedcovariance.',
        scale='Normalize propagated nonanchor covariance so mean coordinate marginal variance=0.003²;95% normalizedpropagation+5%iidfloor. Anchor noise remains0.003²I3. Totalmarginalvariance equalsiid for all modes, no GTfit or scale sweep.',
        controls='iid=old3mm; isotropic_marginal=eachfull3x3block replaced by trace/3I; block_marginal=full3x3blocks without cross-time blocks; propagated_full=entire matrix. Same geometry/fits/rawpositions across4modes.',
        gate='Primary full-depth B16 compared against EACH old LG/depth/Huber/depthHuber iid control: sequence-equal mean absolute path improvement>=5mm AND5%, everysequencepath>0, rawpositionRMSE notworse eachsequence (tolerance1e-12m), nofailed/fallbackcandidate,all15paired. All4comparisons required toadvance. Diagnosticonly evenifpass; no posthoc candidate promotion.',
        failures='Any new mode/proxy numericalfailure returns original whole-query path/rawposes with explicit fallback+failure, and blocks gate. iid parityfailure stops experiment as implementationfailure. Allrows preserved.',
        position='Original published positionmetric remains rawrigid, unchanged within eachgeometry. Smoothedpositions secondary, compare only with same prior iid smoothed output. No replacement of published rawmetric.',
        reference='All600 predictions sealed before rereading alreadyseen GT. No GT in covariance/qselection. Evaluate originaldensepath, originalrawposeerror and separately common-time smoothedRMSE.',
        scientific_limits='Normalization discards absolute covariance calibration; heuristicWLS weights/consensus selection/conditioning/nonlinearerrors/bias and adjacent-edgecovariance remain unmodeled. Not novel uncertainty theory. Frozenfitdirection and3mm matched-energyscale are engineering choices.',
        sources=['https://censi.science/pub/research/2007-icra-icpcov.pdf', 'https://www.research-collection.ethz.ch/entities/publication/ba3a20a6-c07e-4535-a435-10b2dbccb582', 'https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf'],
        new_gpu_calls=0, new_image_decodes=0, new_huh7_02_access=False)
    save(OUT/'config.json', cfg)
    paths = {OUT/'config.json', ROOT/'tests/test_propagated_uncertainty_cpu_v1.py'}
    paths |= {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, '__file__', None) and Path(m.__file__).resolve().is_relative_to(ROOT/'src')}
    paths |= {SOURCE/n for n in ['config.json', 'baseline_predictions.json', 'predictions.json', 'evaluation.json', 'prediction_seal.json', 'reference_seal.json']}
    paths |= {refpath(s) for s in cfg['sequences']}
    for row in load(SOURCE/'baseline_predictions.json'):
        paths |= {SOURCE/'baseline_arrays'/e['array_file'] for e in row['chains'][0]['events']}
    for row in load(SOURCE/'predictions.json'):
        if row['method'] != 'lg_baseline': paths |= {SOURCE/'arrays'/e['array_file'] for e in row['refit_events']}
    verified = 0
    prior_seals = [SOURCE/'prediction_seal.json', SOURCE/'reference_seal.json', ROOT/'experiments/q_error_structure_v1_2026-09-14/freeze.json']
    for prior in prior_seals:
        hashes = load(prior)['hashes']
        for p in paths:
            rel = p.relative_to(ROOT).as_posix()
            if rel in hashes: assert digest(p) == hashes[rel], rel; verified += 1
    for p in paths:
        if p.suffix == '.py':
            dest = OUT/'source_snapshot'/p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as f: f.write(p.read_bytes())
    seal('freeze.json', paths); print(json.dumps(dict(files=len(paths), prior_hashes_checked=verified)), flush=True)


def run():
    verify(OUT/'freeze.json'); cfg = load(OUT/'config.json'); tick = perf_counter(); started = stamp()
    base = {key(r): r for r in load(SOURCE/'baseline_predictions.json')}
    originals = {(*key(r), r['method']): r for r in load(SOURCE/'predictions.json')}
    rows = []; maxparity = 0.; proxy_edges = 0; failures = []
    for k, b in base.items():
        cache = []
        for event in b['chains'][0]['events']:
            with np.load(SOURCE/'baseline_arrays'/event['array_file'], allow_pickle=False) as z:
                cache.append({name: z[name].copy() for name in z.files})
        for geometry in cfg['geometries']:
            original = originals[*k, geometry]; assert original['failure'] is None
            points = np.asarray(original['poses'])[:, :3, 3]; times = np.asarray(original['times'])
            identity = estimate_path(points, times, .003**2*np.eye(3*len(points)))
            parity = abs(identity['path_m']-original['values_m']['state_mean']); maxparity = max(maxparity, parity)
            assert parity <= 1e-7 and identity['q'] == original['q'], (*k, geometry, 'iid_replay_failure', parity)
            local = []; metadata = []; failure = None
            try:
                for i, arrays in enumerate(cache):
                    if geometry == 'lg_baseline':
                        transform = arrays['transform']; weights = np.ones(len(arrays['raw_inlier_indices']))
                    else:
                        event = original['refit_events'][i]
                        assert event['source_array'] == b['chains'][0]['events'][i]['array_file']
                        with np.load(SOURCE/'arrays'/event['array_file'], allow_pickle=False) as z:
                            assert np.array_equal(z['raw_inlier_indices'], arrays['raw_inlier_indices'])
                            transform, weights = z['transform'].copy(), z['weights'].copy()
                    index = arrays['raw_inlier_indices']
                    c, info = registration_proxy(arrays['source_xyz'][index], arrays['target_xyz'][index], transform, weights)
                    local.append(c); metadata.append(info); proxy_edges += 1
                noises, propagation = propagate_position_proxy(original['poses'], local)
            except (ValueError, np.linalg.LinAlgError) as exc:
                failure = f'{type(exc).__name__}: {exc}'; noises = {}; propagation = None
            for mode in cfg['modes']:
                why = None
                if mode == 'iid':
                    result = identity
                else:
                    try:
                        if failure: raise ValueError(failure)
                        result = estimate_path(points, times, noises[mode])
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        why = f'{type(exc).__name__}: {exc}'
                        result = {**identity, 'path_m': original['values_m']['state_mean'], 'q': original['q']}
                        failures.append(dict(sequence=k[0], offset=k[1], budget=k[2], geometry=geometry, mode=mode, reason=why))
                rows.append(dict(sequence=k[0], offset=k[1], budget=k[2], geometry=geometry, mode=mode,
                    **result, fallback=why is not None, failure=why, raw_position_source_geometry=geometry))
            name = f'{k[0]}_{k[1]}_{k[2]}_{geometry}'
            with (OUT/'arrays'/f'{name}.npz').open('xb') as f:
                np.savez_compressed(f, local_covariances=np.array(local), **noises)
            save(OUT/'proxy_metadata'/f'{name}.json', dict(sequence=k[0], offset=k[1], budget=k[2], geometry=geometry,
                edges=metadata, propagation=propagation, proxy_failure=failure))
        print(k, 'complete', len(rows), 'rows', flush=True)
    assert len(rows) == 600
    save(OUT/'predictions.json', rows)
    save(OUT/'runtime.json', dict(started_utc=started, completed_utc=stamp(), wall_s=perf_counter()-tick, rows=len(rows),
        original_iid_path_replay_max_delta_m=maxparity, original_iid_q_all_equal=True, registration_proxy_edge_evaluations=proxy_edges,
        new_gpu_calls=0, new_image_decodes=0, failures=failures, numpy=np.__version__))
    verify(OUT/'freeze.json')
    seal('prediction_seal.json', [OUT/'predictions.json', OUT/'runtime.json', *list((OUT/'arrays').glob('*.npz')), *list((OUT/'proxy_metadata').glob('*.json'))])
    print(json.dumps(load(OUT/'runtime.json')), flush=True)


def evaluate():
    verify(OUT/'freeze.json'); verify(OUT/'prediction_seal.json'); started = stamp(); tick = perf_counter()
    cfg = load(OUT/'config.json'); predictions = load(OUT/'predictions.json')
    old = {(*key(r), r['method']): r for r in load(SOURCE/'evaluation.json')}
    oldpred = {(*key(r), r['method']): r for r in load(SOURCE/'predictions.json')}
    refs = {s: np.loadtxt(refpath(s)) for s in cfg['sequences']}; truth = {}
    rows = []
    for p in predictions:
        k = key(p); original = old[*k, p['geometry']]
        if k not in truth:
            times = oldpred[*k, p['geometry']]['times']
            g = np.array([interpolate_pose(refs[k[0]], t) for t in times]); g = np.linalg.inv(g[0])@g
            truth[k] = g[:, :3, 3]
        smoothed = np.asarray(p['positions'])
        rows.append(dict(sequence=k[0], offset=k[1], budget=k[2], geometry=p['geometry'], mode=p['mode'],
            failure=p['failure'], fallback=p['fallback'], path_m=p['path_m'], q=p['q'],
            signed_path_error_m=p['path_m']-original['truth']['path_m'],
            raw_position_rmse_m=original['pose_errors']['position_rmse_m'],
            smoothed_position_rmse_m=float(np.sqrt(np.mean(np.sum((smoothed-truth[k])**2, axis=1))))))
    lookup = {(*key(r), r['geometry'], r['mode']): r for r in rows}; groups = []
    for budget in cfg['budgets']:
        for geometry in cfg['geometries']:
            for mode in cfg['modes']:
                for sequence in cfg['sequences']:
                    part = [r for r in rows if (r['budget'], r['geometry'], r['mode'], r['sequence']) == (budget, geometry, mode, sequence)]
                    gains = []
                    for r in part:
                        iid = lookup[*key(r), geometry, 'iid']
                        gains.append(dict(path=abs(iid['signed_path_error_m'])-abs(r['signed_path_error_m']),
                            smoothed_position=iid['smoothed_position_rmse_m']-r['smoothed_position_rmse_m']))
                    groups.append(dict(budget=budget, geometry=geometry, mode=mode, sequence=sequence, n=len(part),
                        path_mae_m=mean(abs(r['signed_path_error_m']) for r in part), signed_path_error_mean_m=mean(r['signed_path_error_m'] for r in part),
                        raw_position_rmse_mean_m=mean(r['raw_position_rmse_m'] for r in part),
                        smoothed_position_rmse_mean_m=mean(r['smoothed_position_rmse_m'] for r in part),
                        path_gain_vs_same_geometry_iid_m=mean(r['path'] for r in gains),
                        smoothed_position_gain_vs_same_geometry_iid_m=mean(r['smoothed_position'] for r in gains),
                        path_win_count_vs_same_geometry_iid=sum(r['path'] > 1e-10 for r in gains), failure_count=sum(r['fallback'] for r in part)))
    primary = [r for r in rows if (r['budget'], r['geometry'], r['mode']) == (16, 'depth_weighted', 'propagated_full')]
    comparisons = []
    for baseline in ['lg_baseline', 'depth_weighted', 'huber_refit', 'depth_huber_refit']:
        per_sequence = []; paired = []
        for r in primary:
            base = old[*key(r), baseline]
            paired.append(dict(sequence=r['sequence'], offset=r['offset'], budget=16,
                baseline_absolute_path_error_m=abs(base['errors_m']['state_mean']), candidate_absolute_path_error_m=abs(r['signed_path_error_m']),
                path_gain_m=abs(base['errors_m']['state_mean'])-abs(r['signed_path_error_m']),
                raw_position_gain_m=base['pose_errors']['position_rmse_m']-r['raw_position_rmse_m'], fallback=r['fallback']))
        for sequence in cfg['sequences']:
            pp = [r for r in paired if r['sequence'] == sequence]
            per_sequence.append(dict(sequence=sequence, n=len(pp), path_gain_m=mean(r['path_gain_m'] for r in pp),
                raw_position_gain_m=mean(r['raw_position_gain_m'] for r in pp)))
        gain = mean(r['path_gain_m'] for r in per_sequence); base_error = mean(r['baseline_absolute_path_error_m'] for r in paired)
        checks = dict(all15paired=len(paired) == 15, no_candidate_failure=not any(r['fallback'] for r in paired),
            absolute_gain_at_least_5mm=gain >= .005, relative_gain_at_least_5percent=gain >= .05*base_error,
            each_sequence_path_positive=all(r['path_gain_m'] > 0 for r in per_sequence),
            each_sequence_raw_position_not_worse=all(r['raw_position_gain_m'] >= -1e-12 for r in per_sequence))
        comparisons.append(dict(baseline=baseline, sequence_equal_gain_m=gain, relative_gain=gain/base_error,
            checks=checks, passed=all(checks.values()), per_sequence=per_sequence, pairs=paired))
    decision = dict(status='development_gate_pass_requires_new_confirmation' if all(c['passed'] for c in comparisons) else 'development_gate_failed_do_not_promote',
        primary_geometry='depth_weighted', primary_mode='propagated_full', primary_budget=16, comparisons=comparisons,
        secondary_modes_and_geometries='Diagnostic only; no posthoc switch to a better mode/geometry.',
        scale_calibrated=False, training_authorized_by_this_result=False)
    save(OUT/'evaluation.json', rows); save(OUT/'summary.json', groups); save(OUT/'decision.json', decision)
    save(OUT/'validation.json', dict(technical_pass=len(rows) == 600 and len(groups) == 120,
        predictions=600, groups=120, evaluation_started_utc=started, prediction_seal_utc=load(OUT/'prediction_seal.json')['utc'],
        evaluation_wall_s=perf_counter()-tick, original_raw_positions_unchanged=True, new_gpu_calls=0, new_image_decodes=0, new_huh7_02_reads=0))
    seal('evaluation_seal.json', [OUT/n for n in ['evaluation.json', 'summary.json', 'decision.json', 'validation.json']])
    print(json.dumps(dict(status=decision['status'], comparisons=[{k:c[k] for k in ['baseline', 'sequence_equal_gain_m', 'relative_gain', 'checks']} for c in comparisons])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['freeze', 'run', 'evaluate']); action = p.parse_args().action
    if action == 'run':
        (OUT/'arrays').mkdir(exist_ok=False); (OUT/'proxy_metadata').mkdir(exist_ok=False)
    globals()[action]()
