import numpy as np
import unittest

from src.loss import cross_entropy_loss


class TestCrossEntropyLoss(unittest.TestCase):
    def test_perfect_prediction(self):
        logits = np.zeros((1, 2, 5))
        logits[0, 0, 1] = 100.0
        logits[0, 1, 2] = 100.0
        targets = np.array([[1, 2]])
        loss, probs = cross_entropy_loss(logits, targets)
        self.assertAlmostEqual(loss, 0.0, places=4)

    def test_uniform_prediction(self):
        vocab = 100
        logits = np.zeros((1, 1, vocab))
        targets = np.array([[5]])
        loss, _ = cross_entropy_loss(logits, targets)
        expected = np.log(vocab)
        self.assertAlmostEqual(loss, expected, places=3)

    def test_padding_ignored(self):
        logits = np.random.randn(1, 3, 10)
        targets = np.array([[5, 0, 0]])
        loss1, _ = cross_entropy_loss(logits, targets, pad_id=0)

        targets2 = np.array([[5, 3, 7]])
        loss2, _ = cross_entropy_loss(logits, targets2, pad_id=0)

        self.assertNotAlmostEqual(loss1, loss2, places=2)


if __name__ == "__main__":
    unittest.main()
