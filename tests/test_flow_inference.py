import unittest
import numpy as np
import torch

from src.flow import inference_ode_multi
from src.config import Config


class ConstantVectorField(torch.nn.Module):
    def forward(self, x, t, x_0):
        return torch.ones_like(x) * 0.5


class MultiTrajectoryInferenceTests(unittest.TestCase):
    def test_inference_ode_multi_returns_flattened_proposals_and_seed_ids(self):
        old_steps = Config.INFERENCE_STEPS
        Config.INFERENCE_STEPS = 2
        try:
            proposals, seed_ids = inference_ode_multi(
                ConstantVectorField(),
                np.zeros((3, 2), dtype=np.float32),
                torch.device("cpu"),
                num_samples=4,
                noise_scale=0.0,
            )

            self.assertEqual(proposals.shape, (12, 2))
            self.assertEqual(seed_ids.shape, (12,))
            np.testing.assert_array_equal(seed_ids[:4], np.array([0, 0, 0, 0]))
            np.testing.assert_allclose(proposals[:4], np.full((4, 2), 0.5), atol=1e-6)
        finally:
            Config.INFERENCE_STEPS = old_steps


if __name__ == "__main__":
    unittest.main()
