import numpy as np
import unittest

from src.optimizer import AdamOptimizer


class TestAdamOptimizer(unittest.TestCase):
    def test_params_update(self):
        params = {"w": np.array([1.0, 2.0, 3.0])}
        opt = AdamOptimizer(params, lr=0.1)
        grads = {"w": np.array([1.0, 1.0, 1.0])}
        old_w = params["w"].copy()
        opt.step(grads)
        self.assertFalse(np.allclose(params["w"], old_w))

    def test_convergence_simple(self):
        params = {"w": np.array([5.0])}
        opt = AdamOptimizer(params, lr=0.1)
        for _ in range(200):
            grad = 2.0 * params["w"]
            opt.step({"w": grad})
        self.assertAlmostEqual(params["w"][0], 0.0, places=1)

    def test_bias_correction(self):
        params = {"w": np.array([1.0])}
        opt = AdamOptimizer(params, lr=0.01)
        grads = {"w": np.array([0.5])}
        opt.step(grads)
        self.assertEqual(opt.t, 1)
        self.assertFalse(np.allclose(opt.m["w"], 0))


if __name__ == "__main__":
    unittest.main()
