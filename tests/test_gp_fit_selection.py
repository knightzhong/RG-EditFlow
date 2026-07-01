import unittest
import torch

from src.generator import select_gp_fit_tensors


class GPFitSelectionTests(unittest.TestCase):
    def test_select_gp_fit_tensors_caps_large_datasets(self):
        x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        y = torch.arange(10, dtype=torch.float32)

        selected_x, selected_y = select_gp_fit_tensors(x, y, max_samples=4, seed=0)

        self.assertEqual(selected_x.shape, (4, 2))
        self.assertEqual(selected_y.shape, (4,))
        self.assertTrue(torch.all(selected_y < 10))

    def test_select_gp_fit_tensors_keeps_small_datasets(self):
        x = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        y = torch.arange(3, dtype=torch.float32)

        selected_x, selected_y = select_gp_fit_tensors(x, y, max_samples=10, seed=0)

        torch.testing.assert_close(selected_x, x)
        torch.testing.assert_close(selected_y, y)


if __name__ == "__main__":
    unittest.main()
