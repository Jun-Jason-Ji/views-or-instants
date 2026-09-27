"""Tests for the DPJAIT noise-knob pre-registration (protocol/DPJAIT_NOISE_PREREG_v0.1.md)."""
import inspect
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dpjait_noise_runner as nr  # noqa: E402
import own_blob_observer as ob  # noqa: E402


# --- the separation that makes the pre-registration mean anything -----------
@pytest.mark.parametrize("fn", [nr.step_predict, nr.bind, nr.triangulate_dense,
                                nr.eligible_windows, nr.runs, nr.instants])
def test_predict_path_never_reads_a_reference_length(fn):
    assert "reference_path_length" not in inspect.getsource(fn)


def test_observer_never_reads_the_reference():
    src = inspect.getsource(ob)
    assert "reference_at" not in src and "reference_path_length" not in src


def test_score_refuses_changed_predictions():
    src = inspect.getsource(nr.step_score)
    assert "changed after sealing" in src


# --- sampling -----------------------------------------------------------------
@pytest.mark.parametrize("m", [4, 5, 17, 134, 400])
def test_instants_include_both_ends(m):
    idx = nr.instants(800, 400, m)
    assert idx[0] == 800 and idx[-1] == 1200
    assert len(idx) == len(set(idx)) and len(idx) <= m


def test_run_order_is_fixed_and_complete():
    r = nr.runs()
    a = [x for x in r if x[0] == "A"]
    b = [x for x in r if x[0] == "B"]
    assert len(a) == 1 + (len(nr.CFG["ladder_a_px"]) - 1) * nr.CFG["ladder_a_reps"]
    assert [x[2] for x in b] == [2, 4, 8]
    assert r == nr.runs()


def test_noise_is_reproducible():
    g1 = np.random.default_rng([nr.CFG["seed"], 3, 2, 1]).normal(0, 8, (5, 2))
    g2 = np.random.default_rng([nr.CFG["seed"], 3, 2, 1]).normal(0, 8, (5, 2))
    g3 = np.random.default_rng([nr.CFG["seed"], 3, 2, 2]).normal(0, 8, (5, 2))
    assert np.array_equal(g1, g2) and not np.array_equal(g1, g3)


# --- geometry -----------------------------------------------------------------
def _synthetic_rig():
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    D = np.zeros(4)
    mats = []
    for yaw in (-0.5, 0.0, 0.6):
        R, _ = cv2.Rodrigues(np.array([0.0, yaw, 0.0]))
        C = np.array([4 * np.sin(yaw), 0.0, -4 * np.cos(yaw)])
        t = -R @ C
        mats.append((K, D, np.column_stack([R, t])))
    return mats


def test_triangulate_dense_recovers_points_and_marks_gaps():
    mats = _synthetic_rig()
    X = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0], [-0.3, 0.2, 0.1]])
    uv = np.zeros((3, 3, 2))
    for j, (K, D, P) in enumerate(mats):
        h = (K @ (P @ np.c_[X, np.ones(3)].T)).T
        uv[:, j] = h[:, :2] / h[:, 2:]
    uv[2, 1:] = np.nan                       # only one view at frame 2
    Y = nr.triangulate_dense(uv, mats)
    assert np.allclose(Y[:2], X[:2], atol=1e-6)
    assert np.isnan(Y[2]).all()


# --- the blob centroid mapping back to full resolution --------------------------
@pytest.mark.parametrize("s", [1, 2, 4, 8])
def test_blob_centroid_maps_back_to_full_resolution(s):
    bg = np.full((1080, 1920), 40, np.uint8)
    img = bg.copy()
    cx, cy = 1000.3, 500.7
    cv2.circle(img, (int(round(cx * 16)), int(round(cy * 16))), 12 * 16, 220, -1,
               lineType=cv2.LINE_AA, shift=4)
    b = ob.blobs(img, bg, s)
    assert b, "no blob found"
    x, y, _ = b[0]
    tol = 0.15 if s == 1 else 0.3 * s
    assert abs(x - cx) < tol and abs(y - cy) < tol


# --- decision helpers ----------------------------------------------------------
def test_crossing_interpolates_in_log_m():
    sgn = {4: -10.0, 8: -2.0, 16: 6.0, 32: 20.0}
    x = nr._crossing(sgn)
    assert 8 < x < 16
    assert nr._crossing({4: 1.0, 8: 2.0}) is None
