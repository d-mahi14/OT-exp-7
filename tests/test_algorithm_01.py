import unittest
import numpy as np

from algorithms import __init__  # keeps package recognized


class TestWeightedSum(unittest.TestCase):

    def test_basic_weighted_sum(self):
        from importlib import import_module

        module = import_module("algorithms.01_weighted_sum")

        objectives = np.array([2.0, 5.0])
        weights = np.array([0.4, 0.6])

        result = module.weighted_sum(objectives, weights)

        self.assertAlmostEqual(result, 3.8)

    def test_equal_weights(self):
        from importlib import import_module

        module = import_module("algorithms.01_weighted_sum")

        objectives = np.array([2.0, 6.0])
        weights = np.array([0.5, 0.5])

        result = module.weighted_sum(objectives, weights)

        self.assertAlmostEqual(result, 4.0)

    def test_weights_must_sum_to_one(self):
        from importlib import import_module

        module = import_module("algorithms.01_weighted_sum")

        with self.assertRaises(ValueError):
            module.weighted_sum(
                np.array([2.0, 5.0]),
                np.array([0.2, 0.2])
            )

    def test_negative_weight_rejected(self):
        from importlib import import_module

        module = import_module("algorithms.01_weighted_sum")

        with self.assertRaises(ValueError):
            module.weighted_sum(
                np.array([2.0, 5.0]),
                np.array([1.2, -0.2])
            )

    def test_dimension_mismatch(self):
        from importlib import import_module

        module = import_module("algorithms.01_weighted_sum")

        with self.assertRaises(ValueError):
            module.weighted_sum(
                np.array([2.0, 5.0]),
                np.array([1.0])
            )


if __name__ == "__main__":
    unittest.main()