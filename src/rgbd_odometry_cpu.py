"""Selected RGB-D ORB/PnP camera odometry; never uses reference poses."""
import cv2
import numpy as np

K=np.array([[525.,0,319.5],[0,525.,239.5],[0,0,1.]])


def solve_motion(xyz,uv):
    xyz=np.asarray(xyz,float);uv=np.asarray(uv,float)
    if len(xyz)<20 or xyz.shape!=(len(uv),3) or not np.isfinite(xyz).all() or not np.isfinite(uv).all():raise ValueError('insufficient_or_invalid_correspondences')
    cv2.setRNGSeed(1309)
    ok,r,t,inside=cv2.solvePnPRansac(xyz,uv,K,None,iterationsCount=200,reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inside is None or len(inside)<20 or len(inside)/len(xyz)<.25:raise ValueError('pnp_consensus_failure')
    ii=inside[:,0];ok,r,t=cv2.solvePnP(xyz[ii],uv[ii],K,None,r,t,True,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:raise ValueError('pnp_refine_failure')
    R=cv2.Rodrigues(r)[0];points=xyz[ii]@R.T+t.ravel()
    residual=np.linalg.norm(cv2.projectPoints(xyz[ii],r,t,K,None)[0].reshape(-1,2)-uv[ii],axis=1)
    if np.any(points[:,2]<=0) or np.median(residual)>1.5:raise ValueError('invalid_refined_geometry')
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=t.ravel()
    return T,dict(matches=len(xyz),inliers=len(ii),median_reprojection_px=float(np.median(residual)))


def features(gray,depth):
    if gray is None or depth is None or gray.shape!=(480,640) or depth.shape!=(480,640) or depth.dtype!=np.uint16:raise ValueError('invalid_rgbd')
    detector=cv2.ORB_create(nfeatures=2000)
    key,desc=detector.detectAndCompute(gray,None)
    if desc is None:raise ValueError('no_features')
    return np.array([k.pt for k in key]),desc,depth


def relative(previous,current):
    key,desc,depth=previous;newkey,newdesc,_=current
    matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc,newdesc,k=2);xyz=[];uv=[]
    for pair in matches:
        if len(pair)!=2:continue
        a,b=pair
        if a.distance>=.75*b.distance:continue
        x,y=key[a.queryIdx];u,v=int(round(x)),int(round(y))
        if not (1<=u<639 and 1<=v<479):continue
        patch=depth[v-1:v+2,u-1:u+2].astype(float)/5000;valid=patch[patch>0]
        if len(valid)<5 or valid.max()-valid.min()>.05:continue
        z=float(np.median(valid))
        if not .3<=z<=4:continue
        xyz.append([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z]);uv.append(newkey[a.trainIdx])
    return solve_motion(xyz,uv)


def advance(world_from_previous,current_from_previous):
    R=current_from_previous[:3,:3];t=current_from_previous[:3,3]
    inverse=np.eye(4);inverse[:3,:3]=R.T;inverse[:3,3]=-R.T@t
    return world_from_previous@inverse


def flow_indices(grays,budget):
    """Fully charged low-resolution preview, cumulative image-motion spacing."""
    small=[cv2.resize(g,(160,120),interpolation=cv2.INTER_AREA) for g in grays]
    increment=[]
    for a,b in zip(small[:-1],small[1:]):
        flow=cv2.calcOpticalFlowFarneback(a,b,None,.5,3,15,3,5,1.2,0)
        increment.append(max(float(np.quantile(np.linalg.norm(flow,axis=2),.75)),1e-6))
    cum=np.r_[0,np.cumsum(increment)];targets=np.linspace(0,cum[-1],budget);chosen={0,len(grays)-1}
    for target in targets[1:-1]:
        available=[i for i in range(1,len(grays)-1) if i not in chosen]
        chosen.add(min(available,key=lambda i:(abs(cum[i]-target),i)))
    return np.array(sorted(chosen)),increment
