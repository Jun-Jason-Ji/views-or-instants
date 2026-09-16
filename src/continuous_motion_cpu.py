"""Wiener-velocity Kalman/RTS baseline, selected positions only, isotropic R.

q is acceleration white-noise spectral density (m^2/s^3), not a bound.
Posterior mean path is model conditional, with plug-in q and assumed sigma.
"""
from math import erf
import numpy as np

Q_GRID=np.logspace(-6,3,13)


def transition(dt,q):
    return np.array([[1.,dt],[0.,1.]]),q*np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]])


def filter_states(points,times,q,sigma,query_times=None):
    p=np.asarray(points,float);t=np.asarray(times,float)
    if p.ndim!=2 or p.shape!=(len(t),3) or len(t)<2 or not np.all(np.diff(t)>0) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(t)) or not np.isfinite(q) or q<=0 or not np.isfinite(sigma) or sigma<=0:
        raise ValueError('invalid_observations_or_noise')
    grid=t if query_times is None else np.unique(np.r_[t,query_times])
    if not np.all(np.isfinite(grid)) or grid[0]<t[0] or grid[-1]>t[-1]:raise ValueError('query_extrapolation')
    obs={float(s):v for s,v in zip(t,p-p[0])}
    # Translation-invariant centering; broad proper prior, NOT fitted noise.
    m=np.zeros((2,3));P=np.diag([100.,100.]);nll=0.;mf=[];pf=[];mp=[];pp=[];fs=[]
    for k,s in enumerate(grid):
        F,Q=transition(0 if k==0 else s-grid[k-1],q)
        m=F@m;P=F@P@F.T+Q
        mp.append(m.copy());pp.append(P.copy());fs.append(F)
        if float(s) in obs:
            residual=obs[float(s)]-m[0];S=P[0,0]+sigma**2;K=P[:,0]/S
            nll+=.5*(3*np.log(2*np.pi*S)+residual@residual/S)
            m=m+K[:,None]*residual
            A=np.eye(2);A[:,0]-=K
            P=A@P@A.T+sigma**2*np.outer(K,K)
        mf.append(m.copy());pf.append((P+P.T)/2)
    return grid,np.array(mf),np.array(pf),np.array(mp),np.array(pp),np.array(fs),float(nll)


def smooth(points,times,q,sigma,query_times):
    grid,m,P,mp,pp,F,nll=filter_states(points,times,q,sigma,query_times)
    for k in range(len(grid)-2,-1,-1):
        G=np.linalg.solve(pp[k+1],F[k+1]@P[k]).T
        m[k]+=G@(m[k+1]-mp[k+1])
        P[k]+=G@(P[k+1]-pp[k+1])@G.T
        P[k]=(P[k]+P[k].T)/2
    if np.linalg.eigvalsh(P).min() < -1e-8:raise ValueError('non_psd_smoother')
    m[:,0]+=np.asarray(points)[0]
    return grid,m,P,nll


def expected_norm_isotropic(mean,variance):
    """Exact E||N(mean, variance*I_3)||, stable at zero noncentrality."""
    mean=np.asarray(mean,float);var=np.maximum(np.asarray(variance,float),0)
    r=np.linalg.norm(mean,axis=-1);s=np.sqrt(var);a=r/np.where(s>0,s,1)
    out=r.copy();small=(s>0)&(a<1e-3);large=(s>0)&~small
    out[small]=2*s[small]*np.sqrt(2/np.pi)*(1+a[small]**2/6-a[small]**4/120)
    al=a[large];sl=s[large];rl=r[large]
    out[large]=sl*np.sqrt(2/np.pi)*np.exp(-al**2/2)+(rl+sl**2/rl)*np.array([erf(v/np.sqrt(2)) for v in al])
    return out


def estimate(points,times,sigma=.003,q_grid=Q_GRID,order=8):
    scores=np.array([filter_states(points,times,float(q),sigma)[-1] for q in q_grid])
    best=int(np.argmin(scores));q=float(q_grid[best]);t=np.asarray(times,float)
    nodes,w=np.polynomial.legendre.leggauss(order)
    dt=np.diff(t);qt=(t[:-1,None]+dt[:,None]*(nodes+1)/2).ravel();qw=(dt[:,None]*w/2).ravel()
    grid,m,P,_=smooth(points,t,q,sigma,qt);idx=np.searchsorted(grid,qt)
    velocity=m[idx,1];variance=P[idx,1,1]
    mean_path=float(np.linalg.norm(velocity,axis=1)@qw)
    posterior_path=float(expected_norm_isotropic(velocity,variance)@qw)
    return dict(mean_path_m=mean_path,posterior_path_m=posterior_path,q=q,q_at_boundary=best in (0,len(q_grid)-1),
                nll=float(scores[best]),sigma_m=sigma,posterior_minus_mean_m=posterior_path-mean_path,
                interpretation='Conditional posterior mean of path; plug-in q, assumed isotropic independent noise; NO coverage guarantee')
