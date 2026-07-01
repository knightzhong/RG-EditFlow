import unittest
import numpy as np
import torch

from src.generator import generate_trajectories_from_GP_samples


class GPRecordedPathTests(unittest.TestCase):
    def test_generate_trajectories_can_use_recorded_paths_as_endpoint_bridges(self):
        recorded = torch.tensor([
            [0.0, 0.0],
            [0.1, 0.4],
            [1.0, 1.0],
        ], dtype=torch.float32)
        samples = {"f0": [{
            "trajectory": recorded,
            "x_low": recorded[0],
            "x_high": recorded[-1],
            "y_low": torch.tensor(0.0),
            "y_high": torch.tensor(1.0),
        }]}

        trajs = generate_trajectories_from_GP_samples(
            samples,
            torch.device("cpu"),
            num_steps=2,
            path_mode="endpoint",
        )

        self.assertEqual(trajs.shape, (1, 3, 2))
        np.testing.assert_allclose(trajs[0, 0], recorded[0].numpy(), atol=1e-6)
        np.testing.assert_allclose(trajs[0, 1], [0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(trajs[0, 2], recorded[-1].numpy(), atol=1e-6)

    def test_generate_trajectories_uses_recorded_path_samples(self):
        recorded = torch.tensor([
            [0.0, 0.0],
            [0.1, 0.4],
            [1.0, 1.0],
        ], dtype=torch.float32)
        samples = {"f0": [{
            "trajectory": recorded,
            "x_low": recorded[0],
            "x_high": recorded[-1],
            "y_low": torch.tensor(0.0),
            "y_high": torch.tensor(1.0),
        }]}

        trajs = generate_trajectories_from_GP_samples(samples, torch.device("cpu"), num_steps=2)

        self.assertEqual(trajs.shape, (1, 3, 2))
        np.testing.assert_allclose(trajs[0], recorded.numpy(), atol=1e-6)
        self.assertNotAlmostEqual(float(trajs[0, 1, 1]), 0.5)


if __name__ == "__main__":
    unittest.main()

class FakeKernel:
    def __init__(self):
        self.lengthscale = torch.tensor(1.0)


class FakeGP:
    def __init__(self):
        self.kernel = FakeKernel()
        self.variance = torch.tensor(1.0)

    def set_hyper(self, lengthscale, variance):
        self.kernel.lengthscale = lengthscale
        self.variance = variance

    def mean_posterior(self, x):
        return torch.sum(x * x, dim=-1)


class GPSamplingRecordedPathTests(unittest.TestCase):
    def test_sampling_data_can_record_full_gradient_paths(self):
        from src.generator import sampling_data_from_GP

        samples = sampling_data_from_GP(
            x_train=torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32),
            device=torch.device("cpu"),
            GP_Model=FakeGP(),
            num_gradient_steps=3,
            num_functions=1,
            num_points=2,
            learning_rate=0.1,
            threshold_diff=-1.0,
            record_paths=True,
        )

        first = samples["f0"][0]
        self.assertIsInstance(first, dict)
        self.assertEqual(first["trajectory"].shape, (7, 2))
        self.assertEqual(first["trajectory_scores"].shape, (7,))
        np.testing.assert_allclose(first["trajectory"][0], first["x_low"], atol=1e-6)
        np.testing.assert_allclose(first["trajectory"][-1], first["x_high"], atol=1e-6)
