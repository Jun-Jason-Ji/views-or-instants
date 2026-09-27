"""Tests for protocol/DPJAIT_NOISE_PREREG_v0.2.md (learned chain + correlated-noise correction)."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dpjait_noise_runner_v02 as v2  # noqa: E402
import learned_det_dataset as ds  # noqa: E402
import learned_det_infer as inf  # noqa: E402


@pytest.mark.parametrize("fn", [v2.step_predict, v2.runs])
def test_predict_never_reads_a_reference_length(fn):
    assert "reference_path_length" not in inspect.getsource(fn)


def test_inference_never_reads_the_reference():
    src = inspect.getsource(inf)
    assert "reference_at" not in src and "reference_path_length" not in src


def test_detector_trains_only_on_development_records():
    used = set(sum(ds.TRAIN.values(), []) + sum(ds.VAL.values(), []))
    confirm = set(v2.REAL + v2.SIM)
    assert not (used & confirm)
    assert used == {"R01_D2", "R02_D1", "R12_D4", "R13_D3", "S07_D8", "R03_D1", "S05_D8"}


def test_autocorr_of_white_noise_is_near_zero_and_of_a_random_walk_is_high():
    rng = np.random.default_rng(0)
    white = rng.normal(size=(4000, 3))
    num, den = v2.autocorr_sums(white, 5)
    assert abs(num[0] / den) < 0.05
    walk = np.cumsum(rng.normal(size=(4000, 3)), axis=0)
    num, den = v2.autocorr_sums(walk, 5)
    assert num[0] / den > 0.9


def test_autocorr_ignores_missing_rows():
    rng = np.random.default_rng(1)
    e = rng.normal(size=(500, 3))
    e[::7] = np.nan
    num, den = v2.autocorr_sums(e, 3)
    assert np.isfinite(num).all() and den > 0


def test_correction_raises_the_prediction_when_noise_is_correlated():
    rho = np.full(150, 0.5)
    m0, m1, lag, C = v2.corrected(16.0, 400, 2.0, 0.4, 0.015, rho, 150)
    assert C == 0.5 and m1 > m0
    assert np.isclose(m1 / m0, 0.5 ** -0.25)            # m ~ sigma^-1/2, sigma_eff = sigma*sqrt(0.5)
    assert lag == round(400 / m0)


def test_correction_is_neutral_for_uncorrelated_or_negative_rho():
    m0, m1, _, _ = v2.corrected(16.0, 400, 2.0, 0.4, 0.015, np.zeros(150), 150)
    assert np.isclose(m0, m1)
    m0, m1n, _, _ = v2.corrected(16.0, 400, 2.0, 0.4, 0.015, np.full(150, -0.3), 150)
    assert m1n < m0                                      # 1-C > 1 lowers the prediction; not clipped


def test_lag_is_clipped():
    _, _, lag, _ = v2.corrected(16.0, 400, 50.0, 5.0, 1e-6, np.zeros(150), 150)
    assert lag == 1
    _, _, lag, _ = v2.corrected(16.0, 400, 1e-4, 0.01, 1.0, np.zeros(150), 150)
    assert lag == 150


def test_run_order_fixed():
    r = v2.runs()
    assert r == v2.runs() and len(r) == 1 + 5 * 4
