import unittest
import numpy as np

from src.quality import (
    compute_quality_weights,
    conservative_rerank_scores,
    select_top_indices,
)


class QualityScoringTests(unittest.TestCase):
    def test_quality_weights_reward_good_trajectories(self):
        weights = compute_quality_weights(
            improvement=np.array([1.0, 0.1, -0.2], dtype=np.float32),
            monotonicity=np.array([1.0, 0.5, 1.0], dtype=np.float32),
            uncertainty=np.array([0.0, 0.2, 0.1], dtype=np.float32),
            manifold_distance=np.array([0.0, 0.5, 0.1], dtype=np.float32),
        )

        self.assertEqual(weights.shape, (3,))
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])
        self.assertTrue(np.all(weights >= 0.05))
        self.assertTrue(np.all(weights <= 1.0))

    def test_quality_weight_modes_separate_score_and_geometry_signals(self):
        improvement = np.array([1.0, 0.2], dtype=np.float32)
        monotonicity = np.array([1.0, 0.5], dtype=np.float32)
        uncertainty = np.array([0.0, 10.0], dtype=np.float32)
        manifold_distance = np.array([0.0, 10.0], dtype=np.float32)

        score_only = compute_quality_weights(
            improvement,
            monotonicity,
            uncertainty,
            manifold_distance,
            mode="score",
        )
        score_only_with_changed_geometry = compute_quality_weights(
            improvement,
            monotonicity,
            uncertainty[::-1],
            manifold_distance[::-1],
            mode="score",
        )
        np.testing.assert_allclose(score_only, score_only_with_changed_geometry)

        geometry_only = compute_quality_weights(
            improvement,
            monotonicity,
            uncertainty,
            manifold_distance,
            mode="geometry",
        )
        geometry_only_with_changed_score = compute_quality_weights(
            improvement[::-1],
            monotonicity[::-1],
            uncertainty,
            manifold_distance,
            mode="geometry",
        )
        np.testing.assert_allclose(geometry_only, geometry_only_with_changed_score)

        none = compute_quality_weights(
            improvement,
            monotonicity,
            uncertainty,
            manifold_distance,
            mode="none",
        )
        np.testing.assert_allclose(none, np.ones_like(improvement, dtype=np.float32))

        with self.assertRaises(ValueError):
            compute_quality_weights(improvement, monotonicity, uncertainty, manifold_distance, mode="bad")

    def test_conservative_rerank_penalizes_uncertain_ood_candidates(self):
        scores = conservative_rerank_scores(
            predicted_score=np.array([0.9, 0.95, 0.8], dtype=np.float32),
            uncertainty=np.array([0.05, 0.5, 0.0], dtype=np.float32),
            manifold_distance=np.array([0.1, 0.8, 0.0], dtype=np.float32),
            uncertainty_weight=0.5,
            distance_weight=0.2,
        )

        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[2], scores[1])

    def test_select_top_indices_returns_sorted_descending_candidates(self):
        idx = select_top_indices(np.array([0.1, 0.7, 0.4, 0.9]), k=2)
        np.testing.assert_array_equal(idx, np.array([3, 1]))


if __name__ == "__main__":
    unittest.main()

class GPSampleQualityTests(unittest.TestCase):
    def test_quality_from_gp_samples_uses_endpoint_improvement_and_distance(self):
        import torch
        from src.quality import quality_from_gp_samples

        samples = {
            "f0": [
                [(torch.tensor([2.0, 0.0]), torch.tensor(1.2)), (torch.tensor([0.0, 0.0]), torch.tensor(0.2))],
                [(torch.tensor([0.2, 0.0]), torch.tensor(0.25)), (torch.tensor([0.0, 0.0]), torch.tensor(0.2))],
            ]
        }
        quality = quality_from_gp_samples(samples)

        self.assertEqual(quality["weights"].shape, (2,))
        self.assertGreater(quality["improvement"][0], quality["improvement"][1])
        self.assertGreater(quality["manifold_distance"][0], quality["manifold_distance"][1])
        self.assertGreater(quality["weights"][0], quality["weights"][1])

class OfflineProxyTests(unittest.TestCase):
    def test_knn_proxy_scores_prefers_near_high_scoring_training_points(self):
        from src.quality import knn_proxy_scores

        train_x = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]], dtype=np.float32)
        train_y = np.array([0.1, 0.9, 0.2], dtype=np.float32)
        candidates = np.array([[1.1, 1.0], [2.9, 3.0]], dtype=np.float32)

        proxy = knn_proxy_scores(candidates, train_x, train_y, k=1)

        self.assertGreater(proxy["predicted_score"][0], proxy["predicted_score"][1])
        self.assertLess(proxy["manifold_distance"][0], 0.2)
        self.assertEqual(proxy["predicted_score"].shape, (2,))

class SeedDiverseSelectionTests(unittest.TestCase):
    def test_select_best_per_seed_indices_keeps_best_from_each_seed(self):
        from src.quality import select_best_per_seed_indices

        scores = np.array([0.2, 0.9, 0.8, 0.1, 0.7], dtype=np.float32)
        seed_ids = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        idx = select_best_per_seed_indices(scores, seed_ids, k=3)

        np.testing.assert_array_equal(idx, np.array([1, 2, 4]))

class RecordedPathQualityTests(unittest.TestCase):
    def test_quality_from_recorded_path_uses_path_monotonicity(self):
        import torch
        from src.quality import quality_from_gp_samples

        samples = {"f0": [
            {
                "trajectory": torch.tensor([[0.0], [0.5], [1.0]]),
                "trajectory_scores": torch.tensor([0.0, 0.5, 1.0]),
                "x_low": torch.tensor([0.0]),
                "x_high": torch.tensor([1.0]),
                "y_low": torch.tensor(0.0),
                "y_high": torch.tensor(1.0),
            },
            {
                "trajectory": torch.tensor([[0.0], [0.5], [1.0]]),
                "trajectory_scores": torch.tensor([0.0, -0.2, 1.0]),
                "x_low": torch.tensor([0.0]),
                "x_high": torch.tensor([1.0]),
                "y_low": torch.tensor(0.0),
                "y_high": torch.tensor(1.0),
            },
        ]}
        quality = quality_from_gp_samples(samples)

        self.assertEqual(quality["monotonicity"][0], 1.0)
        self.assertLess(quality["monotonicity"][1], 1.0)
        self.assertGreater(quality["weights"][0], quality["weights"][1])

class BootstrapKNNUncertaintyTests(unittest.TestCase):
    def test_bootstrap_knn_proxy_scores_returns_ensemble_uncertainty(self):
        from src.quality import bootstrap_knn_proxy_scores

        train_x = np.array([[0.0], [0.1], [0.2], [2.0]], dtype=np.float32)
        train_y = np.array([0.0, 1.0, 2.0, 10.0], dtype=np.float32)
        candidates = np.array([[0.1], [2.0]], dtype=np.float32)

        proxy = bootstrap_knn_proxy_scores(
            candidates,
            train_x,
            train_y,
            k=3,
            num_bootstrap=8,
            seed=0,
        )

        self.assertEqual(proxy["predicted_score"].shape, (2,))
        self.assertEqual(proxy["uncertainty"].shape, (2,))
        self.assertTrue(np.all(proxy["uncertainty"] >= 0.0))
        self.assertGreater(proxy["uncertainty"][0], 0.0)

class QualityFilteringTests(unittest.TestCase):
    def test_filter_trajectories_by_quality_keeps_top_weighted_samples(self):
        from src.quality import filter_trajectories_by_quality

        trajectories = np.arange(12, dtype=np.float32).reshape(4, 3, 1)
        quality = {
            "weights": np.array([0.2, 0.9, 0.1, 0.7], dtype=np.float32),
            "improvement": np.array([2.0, 9.0, 1.0, 7.0], dtype=np.float32),
        }

        filtered_trajs, filtered_quality = filter_trajectories_by_quality(
            trajectories,
            quality,
            keep_ratio=0.5,
        )

        self.assertEqual(filtered_trajs.shape[0], 2)
        np.testing.assert_array_equal(filtered_quality["weights"], np.array([0.9, 0.7], dtype=np.float32))
        np.testing.assert_array_equal(filtered_quality["improvement"], np.array([9.0, 7.0], dtype=np.float32))
        np.testing.assert_array_equal(filtered_trajs[:, 0, 0], np.array([3.0, 9.0], dtype=np.float32))

class TrustRegionRerankTests(unittest.TestCase):
    def test_trust_region_scores_penalize_large_seed_displacement(self):
        from src.quality import trust_region_rerank_scores

        scores = trust_region_rerank_scores(
            predicted_score=np.array([1.0, 1.2], dtype=np.float32),
            uncertainty=np.array([0.0, 0.0], dtype=np.float32),
            manifold_distance=np.array([0.0, 0.0], dtype=np.float32),
            seed_displacement=np.array([0.1, 5.0], dtype=np.float32),
            uncertainty_weight=0.0,
            distance_weight=0.0,
            displacement_weight=0.1,
        )

        self.assertGreater(scores[0], scores[1])


    def test_clip_proposals_to_trust_region_limits_seed_displacement(self):
        from src.quality import clip_proposals_to_trust_region

        proposals = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
        seeds = np.array([[0.0, 0.0]], dtype=np.float32)
        seed_ids = np.array([0, 0], dtype=np.int64)

        clipped = clip_proposals_to_trust_region(
            proposals,
            seed_ids,
            seeds,
            max_displacement=1.0,
        )

        distances = np.linalg.norm(clipped - seeds[seed_ids], axis=1)
        self.assertTrue(np.all(distances <= 1.0 + 1e-6))
        np.testing.assert_allclose(clipped[0], np.array([0.6, 0.8], dtype=np.float32), atol=1e-6)

    def test_select_best_per_seed_can_fallback_to_seed(self):
        from src.quality import select_best_per_seed_with_fallback

        proposals = np.array([[10.0], [20.0], [1.5], [1.8]], dtype=np.float32)
        seeds = np.array([[0.0], [1.0]], dtype=np.float32)
        scores = np.array([-5.0, -6.0, 0.2, 0.1], dtype=np.float32)
        seed_ids = np.array([0, 0, 1, 1], dtype=np.int64)

        selected, fallback_mask = select_best_per_seed_with_fallback(
            proposals,
            scores,
            seed_ids,
            seeds,
            min_score=0.0,
        )

        np.testing.assert_allclose(selected, np.array([[0.0], [1.5]], dtype=np.float32))
        np.testing.assert_array_equal(fallback_mask, np.array([True, False]))
