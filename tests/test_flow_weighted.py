import unittest
import numpy as np
import torch

from src.flow import train_cfm_step, build_velocity_targets
from src.config import Config


class TinyVectorField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 2)

    def forward(self, x, t, x_0):
        return self.linear(torch.cat([x, t, x_0], dim=-1))


class WeightedFlowTrainingTests(unittest.TestCase):
    def test_build_velocity_targets_can_mix_path_and_endpoint_velocity(self):
        trajectories = torch.tensor([
            [[0.0], [2.0], [3.0]],
        ], dtype=torch.float32)
        k = torch.tensor([0])
        alpha = torch.tensor([[0.25]], dtype=torch.float32)
        idx_range = torch.tensor([0])

        path_velocity = build_velocity_targets(
            trajectories,
            k,
            alpha,
            idx_range,
            endpoint_mix=0.0,
        )
        mixed_velocity = build_velocity_targets(
            trajectories,
            k,
            alpha,
            idx_range,
            endpoint_mix=0.5,
        )

        np.testing.assert_allclose(path_velocity.numpy(), [[4.0]], atol=1e-6)
        np.testing.assert_allclose(mixed_velocity.numpy(), [[3.5]], atol=1e-6)

    def test_train_cfm_step_accepts_trajectory_weights(self):
        old_batch_size = Config.FM_BATCH_SIZE
        Config.FM_BATCH_SIZE = 2
        try:
            model = TinyVectorField()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            trajectories = np.array([
                [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]],
            ], dtype=np.float32)
            weights = np.array([1.0, 0.1], dtype=np.float32)

            loss = train_cfm_step(
                model,
                trajectories,
                optimizer,
                torch.device("cpu"),
                trajectory_weights=weights,
            )

            self.assertGreaterEqual(loss, 0.0)
            self.assertTrue(np.isfinite(loss))
        finally:
            Config.FM_BATCH_SIZE = old_batch_size


if __name__ == "__main__":
    unittest.main()
