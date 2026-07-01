import pickle
import unittest

import numpy as np

from src.compat import patch_numpy_legacy_pickle_aliases


class NumpyCompatTests(unittest.TestCase):
    def test_patch_adds_numpy_loads_for_legacy_design_bench_oracles(self):
        original = getattr(np, "loads", None)
        if hasattr(np, "loads"):
            delattr(np, "loads")
        try:
            patch_numpy_legacy_pickle_aliases()
            payload = {"ok": True}
            self.assertEqual(np.loads(pickle.dumps(payload)), payload)
        finally:
            if original is None:
                if hasattr(np, "loads"):
                    delattr(np, "loads")
            else:
                np.loads = original


if __name__ == "__main__":
    unittest.main()
