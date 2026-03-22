import numpy as np
import unittest

from src.transformer import init_transformer, forward_with_cache
from src.loss import cross_entropy_loss
from src.backward import full_backward
from src.optimizer import collect_params, AdamOptimizer


class TestTrainingConvergence(unittest.TestCase):
    def test_loss_decreases(self):
        np.random.seed(42)
        model = init_transformer(vocab_size=50, d_model=32, d_ff=64, n_layers=1)
        params = collect_params(model)
        optimizer = AdamOptimizer(params, lr=1e-3)

        enc = np.array([[1, 2, 3, 4, 5]])
        dec = np.array([[10, 1, 2, 3, 4]])
        tgt = np.array([[1, 2, 3, 4, 5]])

        losses = []
        for _ in range(50):
            logits, cache = forward_with_cache(enc, dec, model)
            loss, _ = cross_entropy_loss(logits, tgt)
            grads = full_backward(logits, tgt, cache, model)
            max_g = max(np.abs(g).max() for g in grads.values())
            if max_g > 5.0:
                grads = {k: v * 5.0 / max_g for k, v in grads.items()}
            optimizer.step(grads)
            losses.append(loss)

        self.assertGreater(losses[0], losses[-1])
        self.assertGreater(losses[0] - losses[-1], 0.5)

    def test_overfit_single_example(self):
        np.random.seed(42)
        model = init_transformer(vocab_size=30, d_model=16, d_ff=32, n_layers=1)
        params = collect_params(model)
        optimizer = AdamOptimizer(params, lr=1e-3)

        enc = np.array([[1, 2, 3]])
        dec = np.array([[10, 4, 5]])
        tgt = np.array([[4, 5, 6]])

        for _ in range(100):
            logits, cache = forward_with_cache(enc, dec, model)
            loss, _ = cross_entropy_loss(logits, tgt)
            grads = full_backward(logits, tgt, cache, model)
            max_g = max(np.abs(g).max() for g in grads.values())
            if max_g > 5.0:
                grads = {k: v * 5.0 / max_g for k, v in grads.items()}
            optimizer.step(grads)

        self.assertLess(loss, 1.0)


if __name__ == "__main__":
    unittest.main()
