"""Reference-free prediction of the path-length error curve and the optimal observation allocation.

Idea: everything needed to predict the bias-variance U-curve in the number of
sampled moments m is estimable from the observations themselves.

1. fit_state_params: joint (q, sigma) maximum-likelihood on the Kalman
   innovations of a cheap probe. q is the acceleration white-noise spectral
   density of the motion, sigma the per-observation 3D position noise.
2. predict_error_curve: parametric bootstrap. Simulate Wiener-velocity paths
   with the fitted q on a dense grid (the simulated dense polyline plays the
   role of the reference), sample them at m moments, add sigma noise, run the
   SAME deployed estimator, and average the absolute path error.
3. choose_allocation: given a frame budget and per-view noise estimates
   sigma(k), pick the (k, m) with the lowest predicted error.

No reference trajectory, no ground-truth path, no tuning against the answer.
"""
import numpy as np
from continuous_motion_cpu import filter_states,transition,estimate as state_estimate
from limited_arc_cpu import limited_arc

Q_GRID=np.logspace(-6,3,19)
S_GRID=np.logspace(np.log10(2e-4),np.log10(2e-1),19)


def fit_two_part_probe(burst_points,burst_times,spread_points,spread_times,q_grid=Q_GRID,s_grid=S_GRID):
    """Noise from a dense consecutive burst, motion from a spread-out set.

    A sparse probe cannot separate measurement noise from motion: over a long
    gap the Wiener process can explain any displacement. A burst at the native
    frame interval inverts the ratio, so sigma becomes identifiable there; q is
    then fitted on the spread with sigma held fixed.
    """
    b=fit_state_params(burst_points,burst_times,q_grid,s_grid)
    sigma=b['sigma'];best=(np.inf,None)
    for qi,q in enumerate(q_grid):
        nll=filter_states(np.asarray(spread_points,float),np.asarray(spread_times,float),float(q),sigma)[-1]
        if nll<best[0]:best=(nll,qi)
    return dict(q=float(q_grid[best[1]]),sigma=float(sigma),burst_q=b['q'],nll=float(best[0]),
                q_at_boundary=best[1] in (0,len(q_grid)-1),sigma_at_boundary=b['sigma_at_boundary'])


def fit_state_params(points,times,q_grid=Q_GRID,s_grid=S_GRID):
    """Joint (q, sigma) by innovation NLL. Returns dict with fitted values and boundary flags."""
    p=np.asarray(points,float);t=np.asarray(times,float)
    best=(np.inf,None,None)
    for qi,q in enumerate(q_grid):
        for si,s in enumerate(s_grid):
            nll=filter_states(p,t,float(q),float(s))[-1]
            if nll<best[0]:best=(nll,qi,si)
    nll,qi,si=best
    return dict(q=float(q_grid[qi]),sigma=float(s_grid[si]),nll=float(nll),
                q_at_boundary=qi in (0,len(q_grid)-1),sigma_at_boundary=si in (0,len(s_grid)-1))


def simulate_path(q,duration,dense,rng):
    """Wiener-velocity sample path on `dense` uniform steps; returns times, positions."""
    t=np.linspace(0,duration,dense);state=np.zeros((2,3));out=np.empty((dense,3));out[0]=0
    for i in range(1,dense):
        F,Q=transition(t[i]-t[i-1],q)
        state=F@state+np.linalg.cholesky(Q+1e-18*np.eye(2))@rng.normal(size=(2,3))
        out[i]=state[0]
    return t,out


def burst_noise_sample(points):
    """Empirical 3D noise draws from a dense consecutive burst, via scaled second differences.

    For samples at the native frame interval the second difference is dominated by
    measurement noise; dividing by sqrt(6) gives draws with the same covariance and
    the same tails as the per-sample noise, without assuming a Gaussian.
    """
    p=np.asarray(points,float)
    if len(p)<3:return None
    return (p[:-2]-2*p[1:-1]+p[2:])/np.sqrt(6)


def predict_error_curve(q,sigma,duration,m_grid,realizations=32,dense=401,seed=0,estimator='state_mean',sigma_assumed=.003,
                        noise_pool=None,miss_rate=0.):
    """Bootstrap prediction of |estimated path - true path| for each m.

    noise_pool: empirical noise draws (Nx3) resampled with replacement; falls back
    to isotropic Gaussian(sigma) when None. miss_rate: probability a moment yields
    no usable position, dropped from the polyline as in the real chain.
    """
    rng=np.random.default_rng(seed);errs={int(m):[] for m in m_grid};pool=None if noise_pool is None or len(noise_pool)<3 else np.asarray(noise_pool,float)
    for _ in range(realizations):
        t,x=simulate_path(q,duration,dense,rng)
        truth=float(np.linalg.norm(np.diff(x,axis=0),axis=1).sum())
        noise=pool[rng.integers(0,len(pool),len(x))] if pool is not None else rng.normal(0,sigma,x.shape)
        for m in m_grid:
            idx=np.rint(np.linspace(0,dense-1,int(m))).astype(int)
            obs=x[idx]+noise[idx];tt=t[idx]
            if miss_rate>0:
                keep=rng.random(len(idx))>=miss_rate;keep[0]=keep[-1]=True
                if keep.sum()>=3:obs,tt=obs[keep],tt[keep]
            v=state_estimate(obs,tt,sigma_assumed)['mean_path_m'] if estimator=='state_mean' else limited_arc(obs,tt)
            errs[int(m)].append(abs(v-truth))
    return {m:float(np.mean(v)) for m,v in errs.items()},{m:float(np.std(v)/np.sqrt(len(v))) for m,v in errs.items()}


def choose_allocation(curves,budget):
    """curves: {k: {m: predicted_error}}. Return the (k,m) with lowest predicted error at cost k*m<=budget."""
    best=None
    for k,c in curves.items():
        for m,e in c.items():
            if k*m<=budget and (best is None or e<best[2]):best=(k,m,e)
    return best
