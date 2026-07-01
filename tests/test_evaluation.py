import unittest
import numpy as np

from src.evaluation import (
    normalize_oracle_scores,
    normalized_score_summary,
    offline_scores_for_indices,
    sample_proposal_diagnostic_indices,
)


class OracleNormalizationTests(unittest.TestCase):
    def test_normalize_tfbind10_with_design_bench_oracle_bounds(self):
        scores = np.array([-1.8585268, 2.1287067], dtype=np.float32)
        normalized = normalize_oracle_scores(scores, "TFBind10-Exact-v0")
        np.testing.assert_allclose(normalized, np.array([0.0, 1.0], dtype=np.float32), atol=1e-6)

    def test_normalized_summary_reports_mean_and_percentiles(self):
        scores = np.array([-880.4585, 199.36252, 340.90985], dtype=np.float32)
        summary = normalized_score_summary(scores, "DKittyMorphology-Exact-v0")
        self.assertAlmostEqual(summary["normalized_mean"], float(np.mean([0.0, 0.8841, 1.0])), places=3)
        self.assertAlmostEqual(summary["normalized_p100"], 1.0, places=6)

    def test_offline_scores_for_indices_reuses_dataset_labels(self):
        y = np.array([[10.0], [20.0], [30.0], [40.0]], dtype=np.float32)
        scores = offline_scores_for_indices(y, np.array([3, 1]))
        np.testing.assert_allclose(scores, np.array([40.0, 20.0], dtype=np.float32))

    def test_sample_proposal_diagnostic_indices_keeps_selected_and_limits_raw(self):
        selected = np.array([10, 20, 30], dtype=np.int64)
        indices = sample_proposal_diagnostic_indices(100, selected, max_raw=5, seed=7)

        self.assertEqual(len(indices), 8)
        self.assertTrue(set(selected).issubset(set(indices.tolist())))
        self.assertEqual(len(set(indices.tolist()) - set(selected.tolist())), 5)
        self.assertEqual(len(indices), len(set(indices.tolist())))

    def test_sample_proposal_diagnostic_indices_zero_keeps_all(self):
        selected = np.array([1, 3], dtype=np.int64)
        indices = sample_proposal_diagnostic_indices(5, selected, max_raw=0, seed=7)
        np.testing.assert_array_equal(indices, np.arange(5, dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
