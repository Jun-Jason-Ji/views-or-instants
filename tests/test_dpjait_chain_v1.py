"""Tests for the DPJAIT loaders, identity assignment and comparison runner.

The geometry tests build a synthetic four-camera rig with known ground truth so
they run without the 16 GB dataset; the tests that need the real files skip
themselves when it is absent.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

import dpjait_chain_compare as cc  # noqa: E402
import dpjait_identity as di  # noqa: E402
from dpjait_record import Record, load_cameras, project  # noqa: E402

DATA = ROOT / "data_external" / "dpjait_2026-09-19" / "Real_Data"


def synthetic_rig():
    """Four cameras on a 6 m ring looking at the origin, no distortion."""
    cams = {}
    for i, ang in enumerate(np.deg2rad([0, 90, 180, 270])):
        C = np.array([6 * np.cos(ang), 6 * np.sin(ang), 3.0])
        fwd = -C / np.linalg.norm(C)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        R = np.vstack([right, -up, fwd])          # world -> camera
        K = np.array([[1100.0, 0, 960.0], [0, 1100.0, 540.0], [0, 0, 1.0]])
        cams["cam%d" % i] = {"K": K, "D": np.zeros(4), "R": R, "t_m": -R @ C}
    return cams


class FakeRecord:
    def __init__(self, cams, dets):
        self.cameras = cams
        self.cam_ids = sorted(cams)
        self.detections = dets


def image_of(cams, X):
    return {c: project(cams[c], np.asarray(X).reshape(1, 3))[0][0] for c in cams}


class TestCalibrationParsing(unittest.TestCase):
    def test_units_and_convention(self):
        """mm and m calibration files both land in metres; rotation is Rodrigues."""
        mm = ROOT / "tests" / "_tmp_calib_mm.csv"
        mm.write_text(
            "cam_name;cam_x[mm];cam_y[mm];cam_z[mm];cam_or_x[rad];cam_or_y[rad];"
            "cam_or_z[rad];focal_length;cx;cy;coeff_1;coeff_2;coeff_3;coeff_4\n"
            "A;1000;2000;3000;0.1;0.2;0.3;1158.7;962;541;-0.15;0.08;-0.0006;0.00008\n")
        try:
            cams = load_cameras(mm)
        finally:
            mm.unlink()
        cam = cams["A"]
        np.testing.assert_allclose(cam["t_m"], [1.0, 2.0, 3.0])
        expect, _ = cv2.Rodrigues(np.array([0.1, 0.2, 0.3]).reshape(3, 1))
        np.testing.assert_allclose(cam["R"], expect, atol=1e-12)
        self.assertAlmostEqual(cam["K"][0, 0], 1158.7)
        self.assertEqual(len(cam["D"]), 4)

    def test_metre_file_not_rescaled(self):
        m = ROOT / "tests" / "_tmp_calib_m.csv"
        m.write_text(
            "cam_name;cam_x[m];cam_y[m];cam_z[m];cam_or_x[rad];cam_or_y[rad];"
            "cam_or_z[rad];focal_length;cx;cy\n"
            "A;1.5;2.5;3.5;0;0;0;1156.3;960;540\n")
        try:
            cams = load_cameras(m)
        finally:
            m.unlink()
        np.testing.assert_allclose(cams["A"]["t_m"], [1.5, 2.5, 3.5])
        np.testing.assert_allclose(cams["A"]["D"], np.zeros(4))


class TestEpipolar(unittest.TestCase):
    def setUp(self):
        self.cams = synthetic_rig()

    def test_true_correspondence_is_on_the_epipolar_line(self):
        X = np.array([0.4, -0.3, 1.2])
        uv = image_of(self.cams, X)
        rec = FakeRecord(self.cams, {})
        d = di.epipolar_px(rec, "cam0", uv["cam0"], "cam1", uv["cam1"])
        self.assertLess(d, 1e-6)

    def test_wrong_correspondence_is_rejected(self):
        uv_a = image_of(self.cams, np.array([0.4, -0.3, 1.2]))["cam0"]
        uv_b = image_of(self.cams, np.array([-1.5, 1.0, 2.4]))["cam1"]
        rec = FakeRecord(self.cams, {})
        self.assertGreater(di.epipolar_px(rec, "cam0", uv_a, "cam1", uv_b), 20.0)


class TestFreeAssignment(unittest.TestCase):
    def setUp(self):
        self.cams = synthetic_rig()
        self.A = np.array([0.5, 0.2, 1.0])
        self.B = np.array([-1.2, -0.8, 1.8])
        dets = {}
        for c in self.cams:
            ua = project(self.cams[c], self.A.reshape(1, 3))[0][0]
            ub = project(self.cams[c], self.B.reshape(1, 3))[0][0]
            dets[c] = {0: np.vstack([ua, ub])}
        self.rec = FakeRecord(self.cams, dets)

    def test_two_targets_are_separated_without_reference(self):
        groups = di.assign_free(self.rec, 0, self.rec.cam_ids, 20.0)
        self.assertEqual(len(groups), 2)
        from mcalib_cache_observer import triangulate
        got = sorted((triangulate(g, self.cams) for g in groups), key=lambda p: p[0])
        np.testing.assert_allclose(got[0], self.B, atol=1e-3)
        np.testing.assert_allclose(got[1], self.A, atol=1e-3)

    def test_groups_are_disjoint_in_camera(self):
        for g in di.assign_free(self.rec, 0, self.rec.cam_ids, 20.0):
            self.assertEqual(len(g), len(set(g)))

    def test_single_camera_yields_nothing(self):
        self.assertEqual(di.assign_free(self.rec, 0, ["cam0"], 20.0), [])


class TestLinkTracks(unittest.TestCase):
    def test_two_separated_targets_stay_separate(self):
        times = [0.0, 0.5, 1.0]
        a = [np.array([0.0, 0, 0]), np.array([0.2, 0, 0]), np.array([0.4, 0, 0])]
        b = [np.array([3.0, 0, 0]), np.array([3.2, 0, 0]), np.array([3.4, 0, 0])]
        tracks = di.link_tracks([[a[i], b[i]] for i in range(3)], times)
        self.assertEqual(len(tracks), 2)
        for tr in tracks:
            self.assertEqual(len(tr), 3)
            xs = [p[0] for _, p in tr]
            self.assertLess(max(xs) - min(xs), 1.0)

    def test_gap_beyond_speed_limit_starts_a_new_track(self):
        times = [0.0, 0.04]
        far = di.MAX_SPEED_MPS * 0.04 + di.LINK_SLACK_M + 5.0
        tracks = di.link_tracks([[np.zeros(3)], [np.array([far, 0, 0])]], times)
        self.assertEqual(len(tracks), 2)


class TestRunnerHelpers(unittest.TestCase):
    def test_instants_span_the_window_and_are_unique(self):
        for m in cc.M_GRID:
            idx = cc.instants(100, m)
            self.assertEqual(idx[0], 100)
            self.assertEqual(idx[-1], 100 + cc.WINDOW_FRAMES)
            self.assertEqual(len(idx), len(set(idx)))
            self.assertLessEqual(len(idx), m)

    def test_polyline_length(self):
        pts = [np.zeros(3), np.array([3.0, 4.0, 0]), np.array([3.0, 4.0, 12.0])]
        self.assertAlmostEqual(cc.polyline(pts), 17.0)

    def test_polyline_is_translation_invariant(self):
        """The lever arm is a constant offset, so it must not change a length."""
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(12, 3))
        shifted = pts + np.array([0.021, 0.006, -0.116])
        self.assertAlmostEqual(cc.polyline(pts), cc.polyline(shifted), places=12)


@unittest.skipUnless((DATA / "R02_D1").is_dir(), "DPJAIT dataset not extracted")
class TestRealRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = Record(DATA / "R02_D1")

    def test_shape(self):
        self.assertEqual(len(self.rec.cam_ids), 4)
        self.assertEqual(self.rec.n_drones, 1)
        self.assertGreater(self.rec.n_video_frames, 1500)

    def test_detections_are_complete_and_unique(self):
        """The single-drone records are what make a reference-free chain possible."""
        for c in self.rec.cam_ids:
            counts = [len(v) for v in self.rec.detections[c].values()]
            self.assertEqual(set(counts), {1})

    def test_reference_path_length_is_positive_and_additive(self):
        whole = self.rec.reference_path_length("drone", 0, 200)
        half1 = self.rec.reference_path_length("drone", 0, 100)
        half2 = self.rec.reference_path_length("drone", 100, 200)
        self.assertGreater(whole, 0.1)
        self.assertAlmostEqual(whole, half1 + half2, places=6)

    def test_published_convention_beats_the_alternatives(self):
        """cam_x/y/z is the translation vector t, not the camera centre C."""
        from mcalib_cache_observer import triangulate
        errs = []
        for f in range(0, 400, 20):
            obs = di.assign_annotated(self.rec, "drone", f, self.rec.cam_ids, 60.0)
            if len(obs) < 3:
                continue
            p = triangulate(obs, self.rec.cameras)
            X = self.rec.reference_at("drone", f)
            errs.append(np.linalg.norm(p - X))
        self.assertGreater(len(errs), 10)
        # the 116 mm lever arm dominates; a wrong convention gives metres
        self.assertLess(float(np.median(errs)), 0.30)


if __name__ == "__main__":
    unittest.main()
