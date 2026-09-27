"""Tests for the DPJAIT preregistration runner's structural guarantees.

The scientific value of the runner rests on three properties that are easy to
break silently, so they are tested rather than trusted:

1. `predict` output carries no reference-derived field (the seal would be
   meaningless if a reference length could leak into a prediction);
2. a prediction file altered after sealing is refused by `score`;
3. a frozen source altered after `freeze` is refused by `predict`.

The eligibility proxy is tested against the property it was calibrated for:
it must be computable without any reference and must rank a moving window
above a still one.
"""
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import dpjait_prereg_runner as pr  # noqa: E402

DATA = ROOT / "data_external" / "dpjait_2026-09-19" / "Real_Data"


class FakeRec:
    """Minimal record exposing only what the proxy needs: detections."""
    def __init__(self, tracks):
        self.cam_ids = sorted(tracks)
        self.detections = {c: {f: np.asarray([uv]) for f, uv in enumerate(seq)}
                           for c, seq in tracks.items()}


class TestEligibilityProxy(unittest.TestCase):
    def test_moving_window_ranks_above_still_window(self):
        still = FakeRec({"a": [(100.0, 100.0)] * 21, "b": [(50.0, 50.0)] * 21})
        moving = FakeRec({"a": [(100.0 + 8 * i, 100.0) for i in range(21)],
                          "b": [(50.0 + 8 * i, 50.0) for i in range(21)]})
        self.assertAlmostEqual(pr.window_proxy_px(still, 0, 20), 0.0)
        self.assertGreater(pr.window_proxy_px(moving, 0, 20),
                           pr.window_proxy_px(still, 0, 20))

    def test_proxy_is_total_travel_not_displacement(self):
        """A target that returns to its start still counts as having moved."""
        out = [(0.0, 0.0), (60.0, 0.0), (0.0, 0.0)]
        rec = FakeRec({"a": out, "b": out})
        self.assertAlmostEqual(pr.window_proxy_px(rec, 0, 2), 120.0)

    def test_threshold_is_the_frozen_value(self):
        self.assertEqual(pr.CFG["eligibility_proxy_px"], 75.0)

    def test_empty_detections_give_nan(self):
        rec = FakeRec({"a": [], "b": []})
        self.assertTrue(np.isnan(pr.window_proxy_px(rec, 0, 20)))


class TestConfig(unittest.TestCase):
    def test_confirmation_and_development_sets_are_disjoint(self):
        dev = {"R02_D1", "R03_D1", "R01_D2", "R12_D4", "R13_D3", "S05_D8", "S07_D8"}
        confirm = set(pr.CFG["confirm_real_single"]) | set(pr.CFG["confirm_real_multi"]) \
            | set(pr.CFG["confirm_sim"])
        self.assertEqual(dev & confirm, set())

    def test_three_view_record_excluded(self):
        """R17_D1_A ships only three videos and cannot carry the k=4 arm."""
        allrecs = set(pr.CFG["confirm_real_single"]) | set(pr.CFG["confirm_real_multi"])
        self.assertNotIn("R17_D1_A", allrecs)

    def test_sources_include_the_protocol_and_the_runner(self):
        self.assertIn(pr.PROTOCOL, pr.SOURCES)
        self.assertIn("src/dpjait_prereg_runner.py", pr.SOURCES)
        for s in pr.SOURCES:
            self.assertTrue((ROOT / s).is_file(), s)


class TestFreezeAndSeal(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "run"
        pr.step_freeze(self.out)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_freeze_records_every_source_hash(self):
        h = json.loads((self.out / "freeze.json").read_text())["hashes"]
        self.assertEqual(set(h), set(pr.SOURCES))
        for s, v in h.items():
            self.assertEqual(hashlib.sha256((ROOT / s).read_bytes()).hexdigest(), v)

    def test_freeze_refuses_to_reuse_a_directory(self):
        with self.assertRaises(SystemExit):
            pr.step_freeze(self.out)

    def test_verify_passes_when_sources_are_untouched(self):
        pr.verify(self.out)

    def test_verify_fails_when_a_frozen_source_changes(self):
        f = json.loads((self.out / "freeze.json").read_text())
        f["hashes"]["src/dpjait_identity.py"] = "0" * 64
        (self.out / "freeze.json").write_text(json.dumps(f))
        with self.assertRaises(SystemExit):
            pr.verify(self.out)

    def test_save_new_refuses_to_overwrite(self):
        with self.assertRaises(SystemExit):
            pr.save_new(self.out / "config.json", {"x": 1})


@unittest.skipUnless((DATA / "R02_D1").is_dir(), "DPJAIT dataset not extracted")
class TestPredictScoreSeparation(unittest.TestCase):
    """The core guarantee: no reference length can reach a prediction."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.out = cls.tmp / "run"
        pr.step_freeze(cls.out)
        pr.step_predict(cls.out, DATA / "R02_D1", sim=False, dev=False, max_windows=1)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_predictions_contain_no_reference_field(self):
        rows = json.loads((self.out / "predictions" / "R02_D1.json").read_text())["rows"]
        self.assertTrue(rows)
        for r in rows:
            for key in r:
                self.assertNotIn("reference", key)
                self.assertFalse(key.startswith("err_"))

    def test_predictions_do_contain_estimates(self):
        rows = json.loads((self.out / "predictions" / "R02_D1.json").read_text())["rows"]
        self.assertTrue(any("est_polygon" in r for r in rows))

    def test_both_arms_are_present(self):
        rows = json.loads((self.out / "predictions" / "R02_D1.json").read_text())["rows"]
        self.assertEqual({r["chain"] for r in rows}, {"annotated", "free"})

    def test_score_refuses_a_tampered_prediction_file(self):
        p = self.out / "predictions" / "R02_D1.json"
        original = p.read_text()
        d = json.loads(original)
        d["rows"][0]["est_polygon"] = 999.0
        p.write_text(json.dumps(d, default=float))
        try:
            with self.assertRaises(SystemExit):
                pr.step_score(self.out, DATA / "R02_D1", sim=False)
        finally:
            p.write_text(original)

    def test_score_adds_errors_and_keeps_estimates(self):
        pr.step_score(self.out, DATA / "R02_D1", sim=False)
        rows = json.loads((self.out / "scored" / "R02_D1.json").read_text())["rows"]
        done = [r for r in rows if not r.get("failed")]
        self.assertTrue(done)
        for r in done:
            self.assertIn("reference_m", r)
            self.assertIn("err_polygon", r)
            self.assertAlmostEqual(r["err_polygon"],
                                   abs(r["est_polygon"] - r["reference_m"]), places=9)


if __name__ == "__main__":
    unittest.main()
