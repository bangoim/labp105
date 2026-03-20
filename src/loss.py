import numpy as np

from src.utils import softmax


def cross_entropy_loss(logits, targets, pad_id=0):
    batch_size, seq_len, vocab_size = logits.shape
    probs = softmax(logits)
    probs = np.clip(probs, 1e-12, 1.0)

    mask = (targets != pad_id).astype(np.float64)
    n_valid = mask.sum()
    if n_valid == 0:
        return 0.0, probs

    targets_flat = targets.reshape(-1)
    probs_flat = probs.reshape(-1, vocab_size)
    log_probs = -np.log(probs_flat[np.arange(len(targets_flat)), targets_flat])
    log_probs = log_probs.reshape(batch_size, seq_len)
    loss = (log_probs * mask).sum() / n_valid

    return loss, probs
