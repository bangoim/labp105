import numpy as np

from src.utils import softmax


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores)
    return weights @ V


def init_attention_weights(d_model):
    scale = np.sqrt(2.0 / d_model)
    Wq = np.random.randn(d_model, d_model) * scale
    Wk = np.random.randn(d_model, d_model) * scale
    Wv = np.random.randn(d_model, d_model) * scale
    return Wq, Wk, Wv


def project_qkv(X, Wq, Wk, Wv):
    return X @ Wq, X @ Wk, X @ Wv
