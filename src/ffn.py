import numpy as np


def init_ffn_weights(d_model, d_ff):
    scale = np.sqrt(2.0 / d_model)
    W1 = np.random.randn(d_model, d_ff) * scale
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * scale
    b2 = np.zeros(d_model)
    return W1, b1, W2, b2


def feed_forward(X, W1, b1, W2, b2):
    hidden = np.maximum(0, X @ W1 + b1)
    return hidden @ W2 + b2
