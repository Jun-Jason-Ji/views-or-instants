"""Fixed-frame feature-chain core. Prediction functions never load reference data."""
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from rgbd_odometry_cpu import K, solve_motion, advance
from continuous_motion_cpu import estimate
from limited_arc_cpu import limited_arc
from pose_propagation_cpu import polygon, interpolate_pose, pose_errors


def classic_extractor(method):
    if method not in ('orb', 'sift', 'rootsift'):
        raise ValueError('unknown_classic_method')
    detector = cv2.ORB_create(nfeatures=2000) if method == 'orb' else cv2.SIFT_create(nfeatures=2000)
    def extract(gray):
        keys, desc = detector.detectAndCompute(gray, None)
        if desc is None:
            raise ValueError('no_features')
        if method == 'rootsift':
            desc = np.sqrt(desc / np.maximum(desc.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)
        return np.array([k.pt for k in keys]), desc
    return extract


def classic_matcher(method):
    norm = cv2.NORM_HAMMING if method == 'orb' else cv2.NORM_L2
    def match(first, second):
        rows = cv2.BFMatcher(norm).knnMatch(first[1], second[1], k=2)
        indices = [(a.queryIdx, a.trainIdx) for pair in rows if len(pair) == 2
                   for a, b in [pair] if a.distance < .75 * b.distance]
        return np.asarray(indices, dtype=np.int64).reshape(-1, 2)
    return match


def depth_correspondences(source_keys, target_keys, depth, pairs):
    xyz, uv, keep = [], [], []
    for q, t in pairs:
        x, y = source_keys[q]; u, v = int(round(x)), int(round(y))
        if not (1 <= u < 639 and 1 <= v < 479):
            continue
        patch = depth[v-1:v+2, u-1:u+2].astype(float) / 5000
        valid = patch[patch > 0]
        if len(valid) < 5 or valid.max()-valid.min() > .05:
            continue
        z = float(np.median(valid))
        if not .3 <= z <= 4:
            continue
        xyz.append([(x-K[0,2])*z/K[0,0], (y-K[1,2])*z/K[1,1], z])
        uv.append(target_keys[t]); keep.append((int(q), int(t)))
    return np.asarray(xyz).reshape(-1, 3), np.asarray(uv).reshape(-1, 2), keep


def query(window, sequence, budget, method, extract, match, data_root):
    start = perf_counter(); frame_s = feature_s = match_s = solver_s = 0.
    ids = np.rint(np.linspace(0, len(window['frames'])-1, budget)).astype(int).tolist()
    times = [window['frames'][i]['t'] for i in ids]
    poses, events, reads, cache = [np.eye(4)], [], [], {}
    values = {}; failure = None; q = None
    try:
        for i in ids:
            f = window['frames'][i]; reads.append(i); tick = perf_counter()
            gray = cv2.imread(str(Path(data_root)/f['rgb']), cv2.IMREAD_GRAYSCALE)
            depth = cv2.imread(str(Path(data_root)/f['depth']), cv2.IMREAD_UNCHANGED)
            frame_s += perf_counter()-tick
            if gray is None or depth is None or gray.shape != (480,640) or depth.shape != (480,640) or depth.dtype != np.uint16:
                raise ValueError('invalid_rgbd')
            tick = perf_counter()
            try:
                key, desc = extract(gray)
            finally:
                feature_s += perf_counter()-tick
            cache[i] = (key, desc, depth)
        for a, b in zip(ids[:-1], ids[1:]):
            event = dict(a=a, b=b, failure=None, pairs_before_depth=0, pairs_after_depth=0,
                         unique_source=0, unique_target=0, solver_attempts=0)
            events.append(event)
            try:
                tick = perf_counter()
                try:
                    pairs = match(cache[a], cache[b])
                    event['pairs_before_depth'] = len(pairs)
                    xyz, uv, kept = depth_correspondences(cache[a][0], cache[b][0], cache[a][2], pairs)
                finally:
                    match_s += perf_counter()-tick
                event.update(pairs_after_depth=len(kept), unique_source=len({p[0] for p in kept}),
                             unique_target=len({p[1] for p in kept}))
                tick = perf_counter(); event['solver_attempts'] = int(len(xyz) >= 20)
                try:
                    transform, stats = solve_motion(xyz, uv)
                finally:
                    solver_s += perf_counter()-tick
                event['stats'] = stats
                poses.append(advance(poses[-1], transform))
            except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
                event['failure'] = str(exc)
                raise
        pp = np.asarray(poses); tt = np.asarray(times)
        state = estimate(pp[:,:3,3], tt, .003); q = state['q']
        values = dict(state_mean=state['mean_path_m'], limited=limited_arc(pp[:,:3,3],tt),
                      polygon=polygon(pp), displacement=float(np.linalg.norm(pp[-1,:3,3])))
    except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
        failure = str(exc)
    return dict(sequence=sequence, offset=window['offset'], budget=budget, method=method,
                selected=ids, times=times, failure=failure, poses=[p.tolist() for p in poses],
                values_m=values, q=q, events=events, read_indices=reads, rgbd_pairs_decoded=len(cache),
                feature_counts={str(i):len(f[0]) for i,f in cache.items()},
                wall_seconds=dict(read=frame_s,features=feature_s,matching_and_depth=match_s,
                                  solver=solver_s,total=perf_counter()-start))


def reference_metrics(gt, times):
    begin, end = times[0], times[-1]
    if begin < gt[0,0] or end > gt[-1,0]:
        return None, 'outside_reference'
    lo, hi = max(0,np.searchsorted(gt[:,0],begin)-1), min(len(gt),np.searchsorted(gt[:,0],end)+1)
    if np.max(np.diff(gt[lo:hi,0])) > .1:
        return None, 'reference_gap'
    ends = np.array([[np.interp(t,gt[:,0],gt[:,k]) for k in (1,2,3)] for t in (begin,end)])
    ref = np.vstack([ends[0],gt[(gt[:,0]>begin)&(gt[:,0]<end),1:4],ends[1]])
    return dict(path_m=float(np.linalg.norm(np.diff(ref,axis=0),axis=1).sum()),
                displacement_m=float(np.linalg.norm(ends[1]-ends[0]))), None


def score(rows, references):
    output = []
    for row in rows:
        truth, invalid = reference_metrics(references[row['sequence']],row['times'])
        errors = pose = None
        if row['failure'] is None and invalid is None:
            errors = {k:v-truth['displacement_m' if k=='displacement' else 'path_m'] for k,v in row['values_m'].items()}
            gtposes = np.array([interpolate_pose(references[row['sequence']],t) for t in row['times']])
            pose = pose_errors(np.array(row['poses']),np.linalg.inv(gtposes[0])@gtposes)
        output.append({**{k:row[k] for k in ['sequence','offset','budget','method','failure']},
                       'reference_failure':invalid,'truth':truth,'errors_m':errors,'pose_errors':pose})
    return output


def summarize(rows):
    result = {}
    methods = sorted({r['method'] for r in rows})
    for budget in sorted({r['budget'] for r in rows}):
        result[str(budget)] = {}
        for seq in sorted({r['sequence'] for r in rows})+['pooled']:
            part = [r for r in rows if r['budget']==budget and (seq=='pooled' or r['sequence']==seq)]
            idx = {(r['sequence'],r['offset'],r['method']):r for r in part}
            stats = {}
            for method in methods:
                own = [r for r in part if r['method']==method]; valid = [r for r in own if r['errors_m'] is not None]
                stats[method] = dict(total=len(own),success=sum(r['failure'] is None for r in own),valid=len(valid),
                    own_path_mae_m=float(np.mean([abs(r['errors_m']['state_mean']) for r in valid])) if valid else None,
                    own_position_mean_rmse_m=float(np.mean([r['pose_errors']['position_rmse_m'] for r in valid])) if valid else None)
            comparisons = {}
            for a in methods:
                for b in methods:
                    if a==b: continue
                    pairs = []; new = []; lost = []
                    for s,o in sorted({(r['sequence'],r['offset']) for r in part}):
                        x,y = idx[s,o,a],idx[s,o,b]
                        if x['failure'] is not None and y['failure'] is None: new.append([s,o])
                        if x['failure'] is None and y['failure'] is not None: lost.append([s,o])
                        if x['errors_m'] is not None and y['errors_m'] is not None:
                            pairs.append(dict(sequence=s,offset=o,baseline_path_error_m=abs(x['errors_m']['state_mean']),
                                method_path_error_m=abs(y['errors_m']['state_mean']),baseline_position_rmse_m=x['pose_errors']['position_rmse_m'],
                                method_position_rmse_m=y['pose_errors']['position_rmse_m']))
                    common = {k:float(np.mean([r[k] for r in pairs])) for k in ['baseline_path_error_m','method_path_error_m','baseline_position_rmse_m','method_position_rmse_m']} if pairs else {}
                    comparisons[a+'__'+b] = dict(paired_n=len(pairs),pairs=pairs,new=new,lost=lost,common_means=common)
            result[str(budget)][seq] = dict(methods=stats,comparisons=comparisons)
    return result
