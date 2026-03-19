import numpy as np


EPSILON = 1e-6


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, eps=EPSILON):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm


def residual_add_norm(x, sublayer_out, eps=EPSILON):
    return layer_norm(x + sublayer_out, eps)
