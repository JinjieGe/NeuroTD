import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotd.simulation_validation import KnownDelayAudit, METHODS, delay_metrics


def record(audit, window, E, error, centers=(32, 64, 96), truth=None):
    truth = np.zeros(len(centers)) if truth is None else np.asarray(truth)
    audit.record(0.02, window, window // 2, E, {
        method: (np.array(centers), truth + error, truth) for method in METHODS
    })


class TestKnownDelayAudit(unittest.TestCase):
    def test_actual_estimator_recovers_known_positive_delay(self):
        from neurotd.time_varying_shift import compute_windowed_mse_with_hop
        x = np.random.default_rng(7).normal(size=512)
        y = np.concatenate([np.zeros(8), x[:-8]])
        centers, shifts = compute_windowed_mse_with_hop(x, y, 65, 32)
        self.assertGreater(len(centers), 0)
        np.testing.assert_array_equal(shifts, np.full(len(centers), 8))
        self.assertEqual(delay_metrics(shifts, np.full(len(centers), 8), 5120)["delay_rmse_ms"], 0)

    def test_delay_metrics_units_and_bias(self):
        m = delay_metrics([10, 14], [10, 10], 2000)
        self.assertAlmostEqual(m["delay_rmse_ms"], np.sqrt(2))
        self.assertEqual(m["delay_mae_ms"], 1)
        self.assertEqual(m["delay_bias_ms"], 1)

    def test_invalid_arrays_fail_instead_of_truncating(self):
        for estimated, truth in (([], []), ([1, 2], [1]), ([np.nan], [0]), ([[1]], [[1]])):
            with self.assertRaises(ValueError):
                delay_metrics(estimated, truth, 1000)
        with self.assertRaises(ValueError):
            delay_metrics([1], [1], 0)

    def test_E_selection_does_not_use_truth(self):
        a = KnownDelayAudit("test", 0, 1000)
        record(a, 33, 0.1, 5)
        record(a, 65, 0.2, 1)
        candidates, traces, selections = a.tables()
        self.assertEqual(len(traces), 24)
        self.assertEqual(len(candidates), 8)
        self.assertTrue((selections.E_selected_window == 33).all())
        self.assertTrue((selections.oracle_window == 65).all())
        self.assertTrue((selections.excess_rmse_ms == 4).all())
        self.assertFalse(selections.E_choice_is_rmse_optimal.any())
        b = KnownDelayAudit("test", 0, 1000)
        record(b, 33, 0.1, 1)
        record(b, 65, 0.2, 5)
        self.assertTrue((b.tables()[2].E_selected_window == 33).all())

    def test_E_tie_does_not_use_RMSE_tiebreaker(self):
        a = KnownDelayAudit("test", 0, 1000)
        record(a, 65, 0.1, 0)
        record(a, 33, 0.1, 2)
        s = a.tables()[2]
        self.assertTrue((s.E_selected_window == 33).all())
        self.assertTrue(s.rmse_ratio.isna().all())
        self.assertTrue((s.excess_rmse_ms == 2).all())

    def test_exact_center_intersection_no_interpolation(self):
        a = KnownDelayAudit("test", 0, 1000)
        record(a, 33, 0.1, np.array([9, 0, 9]), centers=(16, 32, 48))
        record(a, 65, 0.2, 1, centers=(32, 64))
        candidates, _, s = a.tables()
        self.assertTrue((candidates.n_shared_centers == 1).all())
        shared = s[s.evaluation == "shared_centers"].iloc[0]
        native = s[s.evaluation == "native_centers"].iloc[0]
        self.assertEqual(shared.oracle_window, 33)
        self.assertEqual(native.oracle_window, 65)

    def test_no_shared_centers_reports_missing(self):
        a = KnownDelayAudit("test", 0, 1000)
        record(a, 33, 0.1, 1, centers=(16, 48))
        record(a, 65, 0.2, 2, centers=(32, 64))
        candidates, _, s = a.tables()
        self.assertTrue(candidates.delay_rmse_shared_ms.isna().all())
        self.assertEqual(list(s.evaluation), ["native_centers"])

    def test_true_delay_outside_search_is_flagged(self):
        a = KnownDelayAudit("test", 0, 1000)
        record(a, 33, 0.1, 0, truth=[0, 16, 17])
        c = a.tables()[0]
        self.assertAlmostEqual(c[c.method == "NeuroTD"].iloc[0].true_outside_neurotd_search_fraction, 1 / 3)
        self.assertTrue(c[c.method != "NeuroTD"].true_outside_neurotd_search_fraction.isna().all())

    def test_duplicate_or_nonfinite_score_rejected(self):
        a = KnownDelayAudit("test", 0, 1000)
        record(a, 33, 0.1, 0)
        with self.assertRaises(ValueError):
            record(a, 33, 0.2, 1)
        with self.assertRaises(ValueError):
            record(a, 65, np.nan, 1)

    def test_write_outputs_and_refuse_overwrite(self):
        a = KnownDelayAudit("test", 0, 1000)
        record(a, 33, 0.1, 0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit"
            a.write(out)
            for name in ("candidate_metrics.csv", "delay_traces.csv.gz", "window_selection.csv",
                         "summary.md", "selection_vs_truth.png", "four_methods_known_delay_rmse.png"):
                self.assertGreater((out / name).stat().st_size, 0)
            with self.assertRaises(FileExistsError):
                a.write(out)


if __name__ == "__main__":
    unittest.main()
