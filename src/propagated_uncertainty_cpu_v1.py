"""Conditional registration/propagation proxy and fixed-prior path inference.

This is an uncalibrated, traditional first-order inverse-normal-matrix proxy.
Condition on fixed consensus/weights, assume independent edges, and normalize
the mean position marginal variance to the inherited 3 mm scale. No GT input.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from continuous_motion_cpu import Q_GRID

SIGMA_M = .003
FLOOR_FRACTION = .05
MODES = ('iid', 'isotropic_marginal', 'block_marginal', 'propagated_full')


def skew(x):
    x = np.asarray(x, float)
    if x.shape != (3,) or not np.isfinite(x).all(): raise ValueError('invalid_skew_vector')
    a, b, c = x
    return np.array([[0., -c, b], [c, 0., -a], [-b, a, 0.]])


def registration_proxy(source, target, target_from_source, weights):
    """6x6 left perturbation of the INVERSE transform in the source frame.

    Inverse predicts source y=A target; J=[I,-skew(y)]. Conditional WLS proxy
    C=(sum w||e||²/(3N-6)) inv(sum w J'J). Weights have mean one. This does not
    model matching/consensus selection or correlation among depth samples.
    """
    a, b, transform, w = map(lambda x: np.asarray(x, float), (source, target, target_from_source, weights))
    if a.ndim != 2 or a.shape != b.shape or a.shape[1:] != (3,) or len(a) < 4 or w.shape != (len(a),): raise ValueError('invalid_registration_inputs')
    if not all(np.isfinite(x).all() for x in (a, b, transform, w)) or np.any(w <= 0): raise ValueError('nonfinite_or_nonpositive_registration_input')
    if transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-10, rtol=0): raise ValueError('invalid_transform')
    rr = transform[:3, :3]
    if not np.allclose(rr.T@rr, np.eye(3), atol=1e-8, rtol=0) or abs(np.linalg.det(rr)-1) > 1e-8: raise ValueError('non_se3_transform')
    w = w/w.mean(); inverse = np.linalg.inv(transform)
    fitted = b@inverse[:3, :3].T+inverse[:3, 3]
    residual = fitted-a
    jac = np.stack([np.column_stack((np.eye(3), -skew(y))) for y in fitted])
    normal = np.einsum('n,nji,njk->ik', w, jac, jac)
    eig = np.linalg.eigvalsh(normal)
    if eig[0] <= 1e-12*eig[-1]: raise ValueError('ill_conditioned_registration_proxy')
    variance = float(np.sum(w[:, None]*residual**2)/(3*len(a)-6))
    covariance = variance*np.linalg.solve(normal, np.eye(6)); covariance = (covariance+covariance.T)/2
    return covariance, dict(correspondences=len(a), weighted_residual_coordinate_variance_proxy_m2=variance,
        normal_condition_number=float(eig[-1]/eig[0]), effective_weight_n=float(w.sum()**2/(w@w)),
        covariance_trace_mixed_units=float(np.trace(covariance)))


def propagate_position_proxy(poses, local_covariances, sigma=SIGMA_M):
    """Independent local left-SE3 errors induce temporal position correlation.

    J[j,i]=[R_i, -skew(p_j-p_i)R_i] for j>i. The same edge contributes to all
    subsequent poses. Adjacent-edge shared-image noise is NOT modeled here.
    """
    p, cs = np.asarray(poses, float), np.asarray(local_covariances, float)
    if p.ndim != 3 or p.shape[1:] != (4, 4) or len(p) < 2 or cs.shape != (len(p)-1, 6, 6): raise ValueError('invalid_propagation_shapes')
    if not np.isfinite(p).all() or not np.isfinite(cs).all() or not np.isfinite(sigma) or sigma <= 0: raise ValueError('invalid_propagation_values')
    if not np.allclose(cs, cs.transpose(0, 2, 1), atol=1e-12, rtol=1e-9) or np.linalg.eigvalsh(cs).min() < -1e-12: raise ValueError('invalid_local_covariance')
    rotations = p[:, :3, :3]
    if not np.allclose(rotations.transpose(0, 2, 1)@rotations, np.eye(3), atol=1e-8, rtol=0) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-8, rtol=0): raise ValueError('invalid_pose_rotation')
    if not np.allclose(p[:, 3], [0, 0, 0, 1], atol=1e-10, rtol=0): raise ValueError('invalid_pose_row')
    n = len(p); covariance = np.zeros((3*n, 3*n))
    for i, c in enumerate(cs):
        jac = np.zeros((3*n, 6)); rotation = p[i, :3, :3]
        for j in range(i+1, n):
            jac[3*j:3*j+3] = np.column_stack((rotation, -skew(p[j, :3, 3]-p[i, :3, 3])@rotation))
        covariance += jac@c@jac.T
    covariance = (covariance+covariance.T)/2
    average = float(np.trace(covariance[3:, 3:])/(3*(n-1)))
    if average <= 0 or not np.isfinite(average): raise ValueError('zero_or_invalid_uncertainty_proxy')
    # Equal total marginal variance to iid, not physically calibrated covariance.
    scaled = covariance*(1-FLOOR_FRACTION)*sigma**2/average
    scaled[3:, 3:] += FLOOR_FRACTION*sigma**2*np.eye(3*(n-1))
    scaled[:3, :3] = sigma**2*np.eye(3)
    blocks = np.zeros_like(scaled); isotropic = np.zeros_like(scaled)
    for j in range(n):
        sl = slice(3*j, 3*j+3); blocks[sl, sl] = scaled[sl, sl]
        isotropic[sl, sl] = np.trace(scaled[sl, sl])/3*np.eye(3)
    noises = dict(iid=sigma**2*np.eye(3*n), isotropic_marginal=isotropic, block_marginal=blocks, propagated_full=scaled)
    for noise in noises.values():
        cho_factor(noise, lower=True, check_finite=True)
        if abs(np.trace(noise)/(3*n)-sigma**2) > 1e-14: raise ValueError('noise_energy_normalization_failure')
    marginal_std = [float(np.sqrt(np.trace(scaled[3*j:3*j+3, 3*j:3*j+3])/3)) for j in range(n)]
    return noises, dict(unnormalized_mean_nonanchor_position_variance_m2=average,
        covariance_multiplier=(1-FLOOR_FRACTION)*sigma**2/average, floor_fraction=FLOOR_FRACTION,
        marginal_coordinate_std_m=marginal_std, mean_coordinate_variance_m2=float(np.trace(scaled)/(3*n)),
        min_full_eigenvalue_m2=float(np.linalg.eigvalsh(scaled)[0]),
        interpretation='Relative covariance shape only; inherited average3mm; no calibrated intervals, no bias correction, no adjacent-edge cross covariance')


def integrated_wiener_kernel(t, s):
    t, s = np.asarray(t, float)[:, None], np.asarray(s, float)[None, :]
    lo, hi = np.minimum(t, s), np.maximum(t, s)
    return lo*lo*(3*hi-lo)/6


def velocity_position_kernel(t, s):
    t, s = np.asarray(t, float)[:, None], np.asarray(s, float)[None, :]
    return np.where(t <= s, t*s-t*t/2, s*s/2)


def fit_fixed_q(points, times, noise, q, query_times=None):
    """Exact Gaussian conditioning; same broad proper prior as existing RTS.

    Separate six initial position/velocity coefficients (variance100) using
    Woodbury to avoid subtracting large diffuse-prior contributions numerically.
    Return posterior mean only; uncertainty proxy supplies no coverage claim.
    """
    points, times, noise = np.asarray(points, float), np.asarray(times, float), np.asarray(noise, float)
    n = len(times)
    if points.shape != (n, 3) or n < 2 or noise.shape != (3*n, 3*n) or not all(np.isfinite(x).all() for x in (points, times, noise)): raise ValueError('invalid_gp_shapes_or_values')
    if not np.all(np.diff(times) > 0) or not np.isfinite(q) or q <= 0: raise ValueError('invalid_gp_time_or_q')
    if not np.allclose(noise, noise.T, atol=1e-12, rtol=1e-10): raise ValueError('nonsymmetric_noise')
    cho_factor(noise, lower=True)
    t = times-times[0]
    query = times if query_times is None else np.asarray(query_times, float)
    if query.ndim != 1 or not np.isfinite(query).all() or np.any(query < times[0]) or np.any(query > times[-1]): raise ValueError('query_extrapolation')
    u = query-times[0]; y = (points-points[0]).reshape(-1)
    design = np.kron(np.column_stack((np.ones(n), t)), np.eye(3))
    process = q*np.kron(integrated_wiener_kernel(t, t), np.eye(3))
    chol = cho_factor(process+noise, lower=True)
    inv_design = cho_solve(chol, design); inv_y = cho_solve(chol, y)
    precision = .01*np.eye(6)+design.T@inv_design
    chol_beta = cho_factor((precision+precision.T)/2, lower=True)
    beta = cho_solve(chol_beta, design.T@inv_y)
    residual = y-design@beta; alpha = cho_solve(chol, residual)
    logdet = 2*np.log(np.diag(chol[0])).sum()+6*np.log(100.)+2*np.log(np.diag(chol_beta[0])).sum()
    quadratic = residual@alpha+beta@beta/100.
    nll = float(.5*(3*n*np.log(2*np.pi)+logdet+quadratic))
    mean_position = (y-noise@alpha).reshape(n, 3)+points[0]
    velocity = np.tile(beta[3:6], (len(u), 1))+(q*np.kron(velocity_position_kernel(u, t), np.eye(3))@alpha).reshape(len(u), 3)
    if not np.isfinite(velocity).all() or not np.isfinite(nll): raise ValueError('nonfinite_gp_prediction')
    return dict(nll=nll, positions=mean_position, velocities=velocity)


def estimate_path(points, times, noise, q_grid=Q_GRID, order=8):
    qs = np.asarray(q_grid, float)
    if qs.ndim != 1 or len(qs) == 0 or not np.isfinite(qs).all() or np.any(qs <= 0) or np.any(np.diff(qs) <= 0): raise ValueError('invalid_q_grid')
    t = np.asarray(times, float); nodes, weights = np.polynomial.legendre.leggauss(order)
    dt = np.diff(t); query = (t[:-1, None]+dt[:, None]*(nodes+1)/2).reshape(-1)
    quadrature = (dt[:, None]*weights/2).reshape(-1)
    scores = [fit_fixed_q(points, t, noise, float(q))['nll'] for q in qs]
    best = int(np.argmin(scores)); result = fit_fixed_q(points, t, noise, float(qs[best]), query)
    return dict(path_m=float(np.linalg.norm(result['velocities'], axis=1)@quadrature),
        q=float(qs[best]), q_index=best, q_at_boundary=best in (0, len(qs)-1), nll=result['nll'],
        q_nll=scores, positions=result['positions'].tolist())
