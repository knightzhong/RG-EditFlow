import unittest
import numpy as np

from src.trajectory_sources import generate_knn_mixup_trajectories


class KNNMixupTrajectoryTests(unittest.TestCase):
    def test_knn_mixup_trajectories_connect_lower_to_higher_offline_points(self):
        x_train = np.array([
            [0.0, 0.0],
            [0.2, 0.0],
            [1.0, 0.0],
            [1.2, 0.0],
            [2.0, 0.0],
        ], dtype=np.float32)
        y_train = np.array([0.0, 0.1, 0.6, 0.8, 1.0], dtype=np.float32)

        trajectories, quality = generate_knn_mixup_trajectories(
            x_train,
            y_train,
            num_pairs=4,
            num_steps=3,
            seed=0,
            high_quantile=0.6,
            low_quantile=0.6,
            neighbor_pool=2,
        )

        self.assertEqual(trajectories.shape, (4, 4, 2))
        self.assertEqual(quality["weights"].shape, (4,))
        self.assertTrue(np.all(quality["improvement"] > 0.0))
        self.assertTrue(np.all(quality["monotonicity"] == 1.0))
        np.testing.assert_allclose(trajectories[:, 0, :], quality["x_low"], atol=1e-6)
        np.testing.assert_allclose(trajectories[:, -1, :], quality["x_high"], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
