import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
from bearing_fusion_cpu import ray_depth_reconstruct,bearing_smoother,depth_of,ray_world,polygon
from run_allocation_sweep import subset_points
from analyze_allocation import scene_center,subset_score


def make_cam(rvec,tvec,f=800.):
    R=cv2.Rodrigues(np.asarray(rvec,float))[0]
    return dict(K=[[f,0,320.],[0,f,240.],[0,0,1.]],D=[0.]*5,R=R.tolist(),t_m=list(map(float,tvec)))


def project(p,cam):
    R=np.asarray(cam['R']);t=np.asarray(cam['t_m']);q=R@p+t;K=np.asarray(cam['K'])
    return np.array([K[0,0]*q[0]/q[2]+K[0,2],K[1,1]*q[1]/q[2]+K[1,2]])


class BearingFusionTests(unittest.TestCase):
    def setUp(self):
        self.cam=make_cam([0.1,-0.2,0.05],[0.3,-0.1,4.0]);self.t=np.linspace(0,4,33)
        self.circle=np.column_stack([np.cos(self.t*1.5),np.sin(self.t*1.5),0.2*self.t])
    def test_ray_and_depth_consistency(self):
        p=np.array([.4,-.2,.9]);uv=project(p,self.cam);o,d=ray_world(uv,self.cam);z=depth_of(p,self.cam)
        np.testing.assert_allclose(o+z*d,p,atol=1e-9)
    def test_reconstruction_exact_when_depth_known_at_all_times(self):
        uv=[project(p,self.cam) for p in self.circle]
        pts,_=ray_depth_reconstruct(self.t,uv,self.t,self.circle,self.cam,'linear','z')
        np.testing.assert_allclose(pts,self.circle,atol=1e-8)
    def test_sparse_anchor_reconstruction_beats_anchor_polygon(self):
        uv=[project(p,self.cam) for p in self.circle];aid=np.rint(np.linspace(0,32,8)).astype(int)
        pts,_=ray_depth_reconstruct(self.t,uv,self.t[aid],self.circle[aid],self.cam,'pchip','z')
        truth=polygon(self.circle);self.assertLess(abs(polygon(pts)-truth),abs(polygon(self.circle[aid])-truth))
    def test_smoother_recovers_straight_line(self):
        line=np.outer(self.t,[.5,.2,-.1])+[1,1,1];uv=[project(p,self.cam) for p in line];aid=[0,32]
        r=bearing_smoother(self.t[aid],line[aid],self.t,uv,self.cam)
        self.assertAlmostEqual(r['mean_path_m'],polygon(line),places=3);np.testing.assert_allclose(r['dense_points'],line,atol=5e-3)
    def test_invalid_inputs(self):
        uv=[project(p,self.cam) for p in self.circle]
        with self.assertRaises(ValueError):ray_depth_reconstruct(self.t,uv,self.t[[5,20]],self.circle[[5,20]],self.cam)
        with self.assertRaises(ValueError):ray_depth_reconstruct(self.t,uv,self.t[[0,32]],self.circle[[0,32]],self.cam,'unknown')
    def test_subset_points_and_calibration_rule(self):
        cams={'a':make_cam([0,0,0],[0,0,4.]),'b':make_cam([0,-0.6,0],[2.5,0,3.]),'c':make_cam([0,0.02,0],[0.05,0,4.])}
        pts=np.column_stack([np.linspace(-.2,.2,5),np.linspace(.1,-.1,5),np.linspace(0,.3,5)])
        uv={c:[project(p,cams[c]) for p in pts] for c in cams};uv['b'][2]=None
        out,maxerr=subset_points(uv,cams,('a','b'),5)
        np.testing.assert_allclose(out[[0,1,3,4]],pts[[0,1,3,4]],atol=1e-6);self.assertTrue(np.all(np.isnan(out[2])))
        center=scene_center(cams);self.assertLess(subset_score(cams,('a','b'),center),subset_score(cams,('a','c'),center))


if __name__=='__main__':unittest.main()
