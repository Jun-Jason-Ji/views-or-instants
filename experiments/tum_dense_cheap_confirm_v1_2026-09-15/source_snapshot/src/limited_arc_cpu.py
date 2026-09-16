"""Rotation-equivariant segmentwise Hermite limiter; selected points only."""
import numpy as np


def limited_arc(points,times,order=32):
    p=np.asarray(points,float);t=np.asarray(times,float)
    if p.ndim!=2 or len(p)<2 or len(t)!=len(p) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(t)) or not np.all(np.diff(t)>0):raise ValueError('invalid_samples')
    if len(p)==2:return float(np.linalg.norm(p[1]-p[0]))
    v=np.gradient(p,t,axis=0,edge_order=2);nodes,w=np.polynomial.legendre.leggauss(order);u=(nodes[:,None]+1)/2;w=w/2
    total=0.
    for i,h in enumerate(np.diff(t)):
        d=p[i+1]-p[i];length=np.linalg.norm(d)
        if length==0:continue
        direction=d/length;tangents=[]
        for vi in (v[i],v[i+1]):
            tangent=h*vi;long=float(tangent@direction)
            # Remove backward longitudinal motion, keep transverse curvature.
            tangent=tangent-min(long,0)*direction
            norm=np.linalg.norm(tangent)
            if norm>3*length:tangent*=3*length/norm
            tangents.append(tangent)
        a,b=tangents;longsum=float((a+b)@direction)
        if longsum>3*length:
            factor=3*length/longsum;a*=factor;b*=factor
        # Bezier derivative control points have nonnegative chord projection.
        derivative=(3*u*u-4*u+1)*a+(-6*u*u+6*u)*d+(3*u*u-2*u)*b
        total+=float(np.linalg.norm(derivative,axis=1)@w)
    return total
