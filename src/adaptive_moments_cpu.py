"""Closed-form non-uniform temporal allocation for path-length measurement.

The dominant error at a realistic budget is motion missed between samples. For a
smooth curve the deficit of one segment is

    arc - chord  ~=  (dt^3 / 24) * kappa^2 |v|^3 ,

so with sum(dt_i) = T fixed, total deficit sum(dt_i^3 g_i) is minimised (Lagrange,
g_i > 0) at dt_i proportional to g_i^(-1/3). Equivalently: place the moments at
equal increments of the cumulative density

    rho(t) = ( kappa(t)^2 |v(t)|^3 )^(1/3) = |v x a|^(2/3) / |v| .

Unlike a GPS collar, a video observer can choose WHEN to look, so this allocation
is actually available. rho is computed from a smoothed trajectory fitted to a
uniform probe, so nothing here reads the reference.
"""
import numpy as np
from continuous_motion_cpu import smooth,filter_states,Q_GRID


def fit_q(points,times,sigma):
    best=(np.inf,None)
    for qi,q in enumerate(Q_GRID):
        nll=filter_states(np.asarray(points,float),np.asarray(times,float),float(q),sigma)[-1]
        if nll<best[0]:best=(nll,qi)
    return float(Q_GRID[best[1]])


def error_density(points,times,sigma,grid,q=None,floor_frac=.05):
    """rho(t) = |v x a|^(2/3) / |v| on `grid`, from the smoothed probe trajectory."""
    t=np.asarray(times,float);p=np.asarray(points,float)
    if q is None:q=fit_q(p,t,sigma)
    g=np.asarray(grid,float);g=np.clip(g,t[0],t[-1])
    gr,m,_,_=smooth(p,t,q,sigma,g);idx=np.searchsorted(gr,g)
    x=m[idx,0];v=m[idx,1]
    # acceleration from the smoothed velocity (the model has no explicit accel state)
    a=np.gradient(v,g,axis=0,edge_order=2)
    cross=np.linalg.norm(np.cross(v,a),axis=1);speed=np.linalg.norm(v,axis=1)
    rho=np.where(speed>1e-9,cross**(2/3)/np.maximum(speed,1e-9),0.)
    if not np.all(np.isfinite(rho)):rho=np.nan_to_num(rho)
    rho=rho+floor_frac*(rho.max() if rho.max()>0 else 1.)   # floor keeps coverage where the fit says straight
    return rho,q,x


def allocate(rho,grid,count,must_keep=()):
    """Indices into `grid` at equal increments of the cumulative density, endpoints forced."""
    g=np.asarray(grid,float);c=np.concatenate([[0],np.cumsum(.5*(rho[1:]+rho[:-1])*np.diff(g))])
    if c[-1]<=0:return np.rint(np.linspace(0,len(g)-1,count)).astype(int)
    targets=np.linspace(0,c[-1],count)
    idx=np.unique(np.clip(np.searchsorted(c,targets),0,len(g)-1))
    idx=np.unique(np.concatenate([idx,[0,len(g)-1],np.asarray(must_keep,int)])).astype(int)
    while len(idx)>count:                      # drop the moment whose removal costs least density
        gaps=[(c[idx[i+1]]-c[idx[i-1]],i) for i in range(1,len(idx)-1)]
        if not gaps:break
        idx=np.delete(idx,min(gaps)[1])
    while len(idx)<count:                      # split the largest remaining density gap
        gaps=[(c[idx[i+1]]-c[idx[i]],i) for i in range(len(idx)-1)]
        _,i=max(gaps);lo,hi=idx[i],idx[i+1]
        mid=int(np.searchsorted(c,(c[lo]+c[hi])/2))
        mid=min(max(mid,lo+1),hi-1) if hi-lo>1 else None
        if mid is None:break
        idx=np.unique(np.concatenate([idx,[mid]]))
    return idx
