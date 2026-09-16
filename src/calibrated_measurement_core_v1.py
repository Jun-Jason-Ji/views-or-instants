"""Explicit-calibration TUM pinhole RGB-D version of the frozen LG rigid query.

Only projection calibration is parameterized. Original selected-frame rules,
depth filtering, matching, RANSAC/SVD, pose composition and path estimators stay
unchanged. No reference loading, weight fitting or global calibration mutation.
"""
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from rgbd_rigid_route_cpu import solve_rigid
from rgbd_odometry_cpu import advance
from continuous_motion_cpu import estimate
from limited_arc_cpu import limited_arc
from pose_propagation_cpu import polygon


def validate_calibration(K, *, depth_scale=5000, D=(0, 0, 0, 0, 0), image_size=(640, 480)):
    """Return a private K copy; accept only the declared TUM PNG pinhole contract.

Zero distortion can be given as scalar 0 or five zero coefficients. The caller
must establish that the RGB and depth pixels correspond to this pinhole image;
this function does not undistort, register depth, or infer camera calibration.
"""
    matrix = np.array(K, dtype=float, copy=True)
    distortion = np.asarray(D, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or matrix[0, 1] != 0 or matrix[1, 0] != 0
            or not np.array_equal(matrix[2], [0., 0., 1.])
            or not 0 <= matrix[0, 2] < 640 or not 0 <= matrix[1, 2] < 480):
        raise ValueError('invalid_pinhole_intrinsics')
    if distortion.shape not in ((), (5,)) or not np.isfinite(distortion).all() or np.any(distortion != 0):
        raise ValueError('only_zero_distortion_supported')
    if tuple(image_size) != (640, 480):
        raise ValueError('only_640x480_supported')
    if not np.isscalar(depth_scale) or not np.isfinite(depth_scale) or depth_scale != 5000:
        raise ValueError('only_TUM_uint16_depth_scale_5000_supported')
    matrix.flags.writeable = False
    return matrix


def depth_correspondences(source_keys, target_keys, depth, pairs, *, K, depth_scale=5000):
    """Original 3x3 rounded median filter and original subpixel projection rays."""
    matrix = validate_calibration(K, depth_scale=depth_scale)
    source_keys, target_keys, depth, pairs = map(np.asarray, (source_keys, target_keys, depth, pairs))
    if (source_keys.ndim != 2 or source_keys.shape[1:] != (2,)
            or target_keys.ndim != 2 or target_keys.shape[1:] != (2,)
            or not np.isfinite(source_keys).all() or not np.isfinite(target_keys).all()):
        raise ValueError('invalid_feature_locations')
    if depth.shape != (480, 640) or depth.dtype != np.uint16:
        raise ValueError('invalid_TUM_depth_image')
    if pairs.ndim != 2 or pairs.shape[1:] != (2,) or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError('invalid_match_indices')
    if len(pairs) and (np.any(pairs < 0) or np.any(pairs[:, 0] >= len(source_keys))
                      or np.any(pairs[:, 1] >= len(target_keys))):
        raise ValueError('match_index_outside_features')
    xyz, uv, keep = [], [], []
    for q, t in pairs:
        x, y = source_keys[q]
        u, v = int(round(x)), int(round(y))
        if not (1 <= u < 639 and 1 <= v < 479):
            continue
        patch = depth[v-1:v+2, u-1:u+2].astype(float) / depth_scale
        valid = patch[patch > 0]
        if len(valid) < 5 or valid.max()-valid.min() > .05:
            continue
        z = float(np.median(valid))
        if not .3 <= z <= 4:
            continue
        xyz.append([(x-matrix[0, 2])*z/matrix[0, 0],
                    (y-matrix[1, 2])*z/matrix[1, 1], z])
        uv.append(target_keys[t])
        keep.append((int(q), int(t)))
    return np.asarray(xyz).reshape(-1, 3), np.asarray(uv).reshape(-1, 2), keep


def paired_depth(source_keys, target_keys, source_depth, target_depth, pairs, *, K, depth_scale=5000):
    """Two original depth filters with exactly the original correspondence ordering."""
    xyz, _, kept = depth_correspondences(source_keys, target_keys, source_depth, pairs,
                                         K=K, depth_scale=depth_scale)
    first = np.asarray(kept, np.int64).reshape(-1, 2)
    target_xyz, _, target_kept = depth_correspondences(
        target_keys, source_keys, target_depth, first[:, ::-1], K=K, depth_scale=depth_scale)
    source_map = {(int(q), int(t)): i for i, (q, t) in enumerate(first)}
    original_pairs = np.asarray([(q, t) for t, q in target_kept], np.int64).reshape(-1, 2)
    ii = np.asarray([source_map[int(q), int(t)] for q, t in original_pairs], np.int64)
    return xyz[ii], target_xyz, original_pairs, len(first)


def query(window, sequence, budget, method, model, data_root, arrays_root=None, *, K,
          depth_scale=5000, D=(0, 0, 0, 0, 0), image_size=(640, 480)):
    """Frozen lighterglue_rigid query with an explicitly supplied pinhole K.

Images and features are cached once per query. All B images are read before any
feature or geometry failure; inference stops at the first failed edge. The
matching model must implement extract(gray) and match_features(full_features).
"""
    matrix = validate_calibration(K, depth_scale=depth_scale, D=D, image_size=image_size)
    if method != 'lighterglue_rigid':
        raise ValueError('only_lighterglue_rigid_supported')
    if not isinstance(budget, (int, np.integer)) or not 2 <= budget <= len(window['frames']):
        raise ValueError('invalid_budget')
    if model is None:
        raise ValueError('missing_model')
    started = perf_counter()
    selected = np.rint(np.linspace(0, len(window['frames'])-1, budget)).astype(int).tolist()
    times = [window['frames'][i]['t'] for i in selected]
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('invalid_selected_timestamps')
    calibration = dict(K=matrix.tolist(), depth_scale=5000, D=[0.]*5, image_size=[640, 480],
                       contract='TUM registered uint16 PNG depth; caller-established zero-distortion pinhole RGB pixels')
    common = dict(sequence=sequence, offset=window['offset'], budget=int(budget), method=method,
                  selected=selected, times=times, calibration=calibration)
    images, reads = {}, []
    read_s = 0.
    try:
        for index in selected:
            frame = window['frames'][index]
            reads.append(index)
            tick = perf_counter()
            gray = cv2.imread(str(Path(data_root)/frame['rgb']), 0)
            depth = cv2.imread(str(Path(data_root)/frame['depth']), -1)
            read_s += perf_counter()-tick
            if (gray is None or depth is None or gray.shape != (480, 640)
                    or depth.shape != (480, 640) or depth.dtype != np.uint16):
                raise ValueError('invalid_rgbd')
            images[index] = (gray, depth)
    except (ValueError, cv2.error) as exc:
        return dict(**common, failure=str(exc), poses=[np.eye(4).tolist()], values_m={}, q=None,
                    chains=[], selected_chain=None, fallback_triggered=False, read_indices=reads,
                    rgbd_pairs_decoded=len(images), wall_seconds=dict(read=read_s, total=perf_counter()-started))

    start = perf_counter()
    features, events, poses = {}, [], [np.eye(4)]
    failure, values, q = None, {}, None
    feature_s = match_s = solver_s = 0.
    extract_attempts = match_attempts = solver_attempts = 0
    try:
        for i in selected:
            tick = perf_counter()
            extract_attempts += 1
            try:
                features[i] = model.extract(images[i][0])
            finally:
                feature_s += perf_counter()-tick
        for a, b in zip(selected[:-1], selected[1:]):
            event = dict(a=a, b=b, failure=None)
            events.append(event)
            tick = perf_counter()
            try:
                match_attempts += 1
                pairs = model.match_features(features[a], features[b])
                event['pairs_before_depth'] = len(pairs)
                source, target, kept, source_count = paired_depth(
                    features[a]['keypoints'], features[b]['keypoints'], images[a][1], images[b][1], pairs,
                    K=matrix, depth_scale=depth_scale)
                event.update(source_depth_survivors=source_count, paired_depth_survivors=len(kept))
            except (ValueError, cv2.error, np.linalg.LinAlgError, RuntimeError) as exc:
                event['failure'] = str(exc)
                raise
            finally:
                match_s += perf_counter()-tick
            tick = perf_counter()
            try:
                solver_attempts += 1
                result = solve_rigid(source, target, kept)
                event['metadata'] = result['metadata']
                if arrays_root is not None:
                    name = f'{sequence}_{window["offset"]}_{budget}_{method}_{method}_{a}_{b}.npz'
                    with (Path(arrays_root)/name).open('xb') as stream:
                        np.savez_compressed(stream, **result['arrays'])
                    event['array_file'] = name
                if not result['metadata']['success']:
                    raise ValueError(result['metadata']['failure'])
                transform = result['arrays']['transform']
            except (ValueError, cv2.error, np.linalg.LinAlgError, RuntimeError) as exc:
                event['failure'] = str(exc)
                raise
            finally:
                solver_s += perf_counter()-tick
            poses.append(advance(poses[-1], transform))
        pp, tt = np.array(poses), np.array(times)
        state = estimate(pp[:, :3, 3], tt, .003)
        q = state['q']
        values = dict(state_mean=state['mean_path_m'], limited=limited_arc(pp[:, :3, 3], tt),
                      polygon=polygon(pp), displacement=float(np.linalg.norm(pp[-1, :3, 3])))
    except (ValueError, cv2.error, np.linalg.LinAlgError, RuntimeError) as exc:
        failure = str(exc)
    chain = dict(kind=method, failure=failure, poses=[p.tolist() for p in poses], values_m=values, q=q,
                 events=events, feature_extract_attempts=extract_attempts, match_attempts=match_attempts,
                 solver_attempts=solver_attempts,
                 feature_counts={str(i): len(f['keypoints']) for i, f in features.items()},
                 model_device=str(getattr(model, 'dev', 'unspecified')),
                 wall_seconds=dict(features=feature_s, matching_and_depth=match_s,
                                   solver=solver_s, total=perf_counter()-start))
    return dict(**common, **{k: chain[k] for k in ('failure', 'poses', 'values_m', 'q')},
                chains=[chain], selected_chain=method, fallback_triggered=False, read_indices=reads,
                rgbd_pairs_decoded=len(images), wall_seconds=dict(read=read_s, total=perf_counter()-started))
