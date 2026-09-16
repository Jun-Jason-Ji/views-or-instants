"""Reference-only pose diagnostics and exact chain decomposition helpers."""
import numpy as np


def rotation(q):
    q=np.asarray(q,float);q=q/np.linalg.norm(q);x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def interpolate_pose(gt,t):
    if t<gt[0,0] or t>gt[-1,0]:raise ValueError('reference_extrapolation')
    j=min(len(gt)-1,max(1,int(np.searchsorted(gt[:,0],t))));a,b=gt[j-1],gt[j]
    if b[0]-a[0]>.1:raise ValueError('reference_gap')
    u=(t-a[0])/(b[0]-a[0]);qa=a[4:8]/np.linalg.norm(a[4:8]);qb=b[4:8]/np.linalg.norm(b[4:8])
    dot=float(qa@qb)
    if dot<0:qb=-qb;dot=-dot
    if dot>.9995:q=(1-u)*qa+u*qb
    else:
        theta=np.arccos(np.clip(dot,-1,1));q=(np.sin((1-u)*theta)*qa+np.sin(u*theta)*qb)/np.sin(theta)
    T=np.eye(4);T[:3,:3]=rotation(q);T[:3,3]=(1-u)*a[1:4]+u*b[1:4]
    return T


def rotation_error(A,B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T@B)-1)/2,-1,1))))


def pose_errors(poses,truth):
    return dict(position_rmse_m=float(np.sqrt(np.mean(np.sum((poses[:,:3,3]-truth[:,:3,3])**2,axis=1)))),
                orientation_mean_deg=float(np.mean([rotation_error(p[:3,:3],q[:3,:3]) for p,q in zip(poses,truth)])))


def relative_errors(A,B):
    return dict(translation_m=float(np.linalg.norm(A[:3,3]-B[:3,3])),rotation_deg=rotation_error(A[:3,:3],B[:3,:3]))


def polygon(poses):return float(np.linalg.norm(np.diff(np.asarray(poses)[:,:3,3],axis=0),axis=1).sum())
