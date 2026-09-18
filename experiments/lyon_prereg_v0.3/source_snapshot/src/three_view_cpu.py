"""Conservative three-view fallback, retaining the frozen >=4-view rule."""
from itertools import combinations
import numpy as np
from robust_triangulation_cpu import robust_triangulate,reprojection_errors
from mcalib_cache_observer import triangulate


def three_view_triangulate(observations,cameras,threshold_px=4.):
    valid={c:np.asarray(p,float) for c,p in observations.items() if p is not None and np.asarray(p).shape==(2,) and np.all(np.isfinite(p))}
    if len(valid)!=3:return robust_triangulate(observations,cameras,threshold_px)
    p=triangulate(valid,cameras)
    if max(reprojection_errors(p,valid,cameras).values())>threshold_px:raise ValueError('three_view_inconsistent')
    pairpoints=[]
    for a,b in combinations(sorted(valid),2):
        pairpoints.append(triangulate({a:valid[a],b:valid[b]},cameras))
        centers=[-np.asarray(cameras[c]['R']).T@np.asarray(cameras[c]['t_m']) for c in [a,b]]
        rays=[(p-c)/np.linalg.norm(p-c) for c in centers]
        angle=np.degrees(np.arccos(np.clip(abs(float(rays[0]@rays[1])),-1,1)))
        if angle<5:raise ValueError('three_view_small_ray_angle')
    spread=max(np.linalg.norm(a-b) for a,b in combinations(pairpoints,2))
    if spread>.02:raise ValueError('three_view_pair_disagreement')
    return p,dict(inliers=sorted(valid),rejected=[],fast_path=False,three_view_fallback=True,pair_spread_m=float(spread))
