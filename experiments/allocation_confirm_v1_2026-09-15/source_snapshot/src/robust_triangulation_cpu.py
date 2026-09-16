"""Fixed pixel-consensus triangulation; no reference or unselected-time input."""
from itertools import combinations
import numpy as np
import cv2
from mcalib_cache_observer import triangulate


def reprojection_errors(point,observations,cameras):
    errors={}
    for cid,uv in observations.items():
        c=cameras[cid];R=np.asarray(c['R']);t=np.asarray(c['t_m'])
        if (R@point+t)[2]<=0:errors[cid]=float('inf');continue
        q=cv2.projectPoints(np.asarray(point).reshape(1,3),cv2.Rodrigues(R)[0],t,np.asarray(c['K']),np.asarray(c['D']))[0].reshape(2)
        errors[cid]=float(np.linalg.norm(q-uv))
    return errors


def robust_triangulate(observations,cameras,threshold_px=4.):
    if not np.isfinite(threshold_px) or threshold_px<=0:raise ValueError('invalid_threshold')
    valid={c:np.asarray(v,float) for c,v in observations.items() if v is not None and np.asarray(v).shape==(2,) and np.all(np.isfinite(v))}
    required=max(4,len(valid)//2+1)
    if len(valid)<required:raise ValueError('insufficient_visible_views')
    try:
        p=triangulate(valid,cameras);err=reprojection_errors(p,valid,cameras)
        if max(err.values())<=threshold_px:return p,dict(inliers=sorted(valid),rejected=[],fast_path=True)
    except ValueError:pass
    best=None
    for pair in combinations(sorted(valid),2):
        try:
            p=triangulate({c:valid[c] for c in pair},cameras);err=reprojection_errors(p,valid,cameras)
        except ValueError:continue
        support=sorted(c for c,e in err.items() if e<=threshold_px)
        if len(support)<required:continue
        score=(len(support),-float(np.median([err[c] for c in support])))
        if best is None or score>best[0]:best=(score,support)
    if best is None:raise ValueError('no_multiview_consensus')
    support=best[1]
    for _ in range(3):
        p=triangulate({c:valid[c] for c in support},cameras);err=reprojection_errors(p,valid,cameras)
        updated=sorted(c for c,e in err.items() if e<=threshold_px)
        if len(updated)<required:raise ValueError('refinement_lost_consensus')
        if updated==support:return p,dict(inliers=support,rejected=sorted(set(valid)-set(support)),fast_path=False)
        support=updated
    raise ValueError('unstable_consensus')
