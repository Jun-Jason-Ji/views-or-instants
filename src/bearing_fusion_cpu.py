"""Dense single-camera bearings + sparse multi-view metric anchors -> path length.

Two fusion estimators, both CPU, both using only selected/paid observations:

1. ray_depth: interpolate anchor depth along time, re-project every dense
   bearing to 3D at the interpolated depth, then apply a path functional.
2. bearing_smoother: Wiener-velocity Kalman/RTS smoother in 3D with two
   measurement types (full 3D anchors, and 2-row ray constraints from
   bearings). Plug-in q by innovation NLL; assumed noises are engineering
   inputs, not calibrated uncertainties.

No mocap or unselected-time information is used anywhere here.
"""
import numpy as np
import cv2
from scipy.interpolate import PchipInterpolator,CubicSpline
from limited_arc_cpu import limited_arc

Q_GRID=np.logspace(-6,3,13)


def camera_geometry(cam):
    R=np.asarray(cam['R'],float);t=np.asarray(cam['t_m'],float).reshape(3)
    return R,t,-R.T@t


def ray_world(uv,cam):
    """Unit direction in world coordinates of the back-projected pixel."""
    K=np.asarray(cam['K'],float);D=np.asarray(cam['D'],float)
    x,y=cv2.undistortPoints(np.asarray(uv,float).reshape(1,1,2),K,D).reshape(2)
    R,t,o=camera_geometry(cam);d=R.T@np.array([x,y,1.])
    return o,d  # d has unit z-component in camera frame, so depth z scales it directly


def depth_of(point,cam):
    R,t,_=camera_geometry(cam)
    return float((R@np.asarray(point,float)+t)[2])


def polygon(points):
    p=np.asarray(points,float);return float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())


def ray_depth_reconstruct(bearing_times,bearing_uv,anchor_times,anchor_points,cam,method='pchip',space='z'):
    """Return dense 3D points at bearing_times from interpolated anchor depth."""
    bt=np.asarray(bearing_times,float);at=np.asarray(anchor_times,float)
    if len(at)<2 or not np.all(np.diff(at)>0) or not np.all(np.diff(bt)>0):raise ValueError('invalid_times')
    if bt[0]<at[0]-1e-9 or bt[-1]>at[-1]+1e-9:raise ValueError('bearing_outside_anchor_span')
    z=np.array([depth_of(p,cam) for p in anchor_points])
    if np.any(z<=0):raise ValueError('anchor_behind_camera')
    y=np.log(z) if space=='log' else (1/z if space=='inverse' else z)
    if method=='linear':f=lambda s:np.interp(s,at,y)
    elif method=='pchip':f=PchipInterpolator(at,y)
    elif method=='cubic':f=CubicSpline(at,y,bc_type='natural')
    else:raise ValueError('unknown_method')
    yi=np.asarray(f(bt),float);zi=np.exp(yi) if space=='log' else (1/yi if space=='inverse' else yi)
    if np.any(~np.isfinite(zi)) or np.any(zi<=0):raise ValueError('invalid_interpolated_depth')
    pts=[]
    for uv,zz in zip(bearing_uv,zi):
        o,d=ray_world(uv,cam);pts.append(o+zz*d)
    return np.array(pts),zi


def _basis_perp(r):
    r=r/np.linalg.norm(r);a=np.array([1.,0,0]) if abs(r[0])<.9 else np.array([0,1.,0])
    e1=np.cross(r,a);e1/=np.linalg.norm(e1);e2=np.cross(r,e1);return e1,e2


def bearing_smoother(anchor_times,anchor_points,bearing_times,bearing_uv,cam,sigma_p=.003,sigma_px=1.,q=None,q_grid=Q_GRID,order=8,depth_guess=None):
    """3D Wiener-velocity RTS smoother with anchor and ray measurements.

    Returns dict with mean-trajectory path (Gauss-Legendre over dense grid),
    dense mean positions at bearing times, chosen q and NLL.
    """
    at=np.asarray(anchor_times,float);ap=np.asarray(anchor_points,float);bt=np.asarray(bearing_times,float)
    if ap.shape!=(len(at),3) or len(at)<2 or not np.all(np.diff(at)>0):raise ValueError('invalid_anchors')
    K=np.asarray(cam['K'],float);f=float((K[0,0]+K[1,1])/2)
    # Lateral bearing noise scales with depth; depth guess from linear anchor depth.
    z_anchor=np.array([depth_of(p,cam) for p in ap])
    zg=np.interp(bt,at,z_anchor) if depth_guess is None else np.asarray(depth_guess,float)
    bear=[]
    for s,uv,zz in zip(bt,bearing_uv,zg):
        o,d=ray_world(uv,cam);r=d/np.linalg.norm(d);e1,e2=_basis_perp(r)
        H=np.zeros((2,6));H[0,:3]=e1;H[1,:3]=e2;y=np.array([e1@o,e2@o]);sig=max(float(zz),.3)*sigma_px/f
        bear.append((float(s),H,y,sig**2))
    anchors={float(s):p for s,p in zip(at,ap)}
    bmap={}
    for item in bear:bmap.setdefault(item[0],[]).append(item)
    t0,t1=at[0],at[-1];nodes,w=np.polynomial.legendre.leggauss(order)
    dense=np.unique(np.r_[at,bt]);dense=dense[(dense>=t0)&(dense<=t1)]
    dt=np.diff(dense);qt=(dense[:-1,None]+dt[:,None]*(nodes+1)/2).ravel();qw=(dt[:,None]*w/2).ravel()
    grid=np.unique(np.r_[dense,qt]);p0=ap[0]

    def run(qv,want_smooth):
        m=np.zeros(6);P=np.diag([100.]*3+[100.]*3);nll=0.;mf=[];pf=[];mp=[];pp=[];fs=[]
        I3=np.eye(3)
        for k,s in enumerate(grid):
            d=0. if k==0 else s-grid[k-1]
            F=np.eye(6);F[:3,3:]=d*I3;Q=qv*np.block([[d**3/3*I3,d**2/2*I3],[d**2/2*I3,d*I3]])
            m=F@m;P=F@P@F.T+Q;mp.append(m.copy());pp.append(P.copy());fs.append(F)
            key=float(s)
            if key in anchors:
                H=np.zeros((3,6));H[:,:3]=I3;y=anchors[key]-p0;S=H@P@H.T+sigma_p**2*I3;res=y-H@m
                nll+=.5*(np.log(np.linalg.det(2*np.pi*S))+res@np.linalg.solve(S,res))
                Kg=P@H.T@np.linalg.inv(S);m=m+Kg@res;A=np.eye(6)-Kg@H;P=A@P@A.T+Kg@(sigma_p**2*I3)@Kg.T
            for _,H,y,var in bmap.get(key,[]):
                yy=y-H[:,:3]@p0;S=H@P@H.T+var*np.eye(2);res=yy-H@m
                nll+=.5*(np.log(np.linalg.det(2*np.pi*S))+res@np.linalg.solve(S,res))
                Kg=P@H.T@np.linalg.inv(S);m=m+Kg@res;A=np.eye(6)-Kg@H;P=A@P@A.T+Kg@(var*np.eye(2))@Kg.T
            P=(P+P.T)/2;mf.append(m.copy());pf.append(P.copy())
        if not want_smooth:return float(nll)
        mf=np.array(mf);pf=np.array(pf);mp=np.array(mp);pp=np.array(pp)
        for k in range(len(grid)-2,-1,-1):
            G=np.linalg.solve(pp[k+1],fs[k+1]@pf[k]).T
            mf[k]+=G@(mf[k+1]-mp[k+1]);pf[k]+=G@(pf[k+1]-pp[k+1])@G.T;pf[k]=(pf[k]+pf[k].T)/2
        mf[:,:3]+=p0;return mf,pf,float(nll)

    if q is None:
        scores=np.array([run(float(v),False) for v in q_grid]);best=int(np.argmin(scores));q=float(q_grid[best]);boundary=best in (0,len(q_grid)-1)
    else:boundary=None
    mf,pf,nll=run(q,True);idx=np.searchsorted(grid,qt);vel=mf[idx,3:]
    mean_path=float(np.linalg.norm(vel,axis=1)@qw)
    di=np.searchsorted(grid,dense);dense_pts=mf[di,:3]
    return dict(mean_path_m=mean_path,q=q,q_at_boundary=boundary,nll=nll,dense_times=dense,dense_points=dense_pts,
                polygon_m=polygon(dense_pts),limited_m=limited_arc(dense_pts,dense))
