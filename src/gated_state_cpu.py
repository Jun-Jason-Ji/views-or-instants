"""Temporal-consensus gate for moments that lack view consensus.

Forward Wiener-velocity Kalman pass with the q chosen by the ungated NLL rule
(continuous_motion_cpu.estimate). An observation whose normalised innovation
squared (3 dof, isotropic) exceeds `nis_threshold` is rejected. The first two
observations are always accepted. Returns accepted indices; estimators are then
re-run on the accepted points only. No reference is used.
"""
import numpy as np
from continuous_motion_cpu import transition,estimate,filter_states


def gate(points,times,sigma=.003,nis_threshold=16.27,q=None):
    p=np.asarray(points,float);t=np.asarray(times,float)
    if len(p)<3:return np.arange(len(p)),[]
    if q is None:q=estimate(p,t,sigma)['q']
    m=np.zeros((2,3));P=np.diag([100.,100.]);accepted=[];nis_log=[];last_t=t[0];n_acc=0
    for k in range(len(t)):
        F,Q=transition(0 if k==0 else t[k]-last_t,q);m=F@m;P=F@P@F.T+Q
        res=(p[k]-p[0])-m[0];S=P[0,0]+sigma**2;nis=float(res@res/S)
        if n_acc>=2 and nis>nis_threshold:nis_log.append((k,nis));last_t=t[k];continue
        K=P[:,0]/S;m=m+K[:,None]*res;A=np.eye(2);A[:,0]-=K;P=A@P@A.T+sigma**2*np.outer(K,K)
        accepted.append(k);n_acc+=1;last_t=t[k]
    return np.array(accepted),nis_log
