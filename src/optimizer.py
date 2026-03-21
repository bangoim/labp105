import numpy as np


def collect_params(model):
    params = {}
    params["embedding_table"] = model["embedding_table"]
    params["W_out"] = model["W_out"]

    for i, (Wq, Wk, Wv, W1, b1, W2, b2) in enumerate(model["encoder_layers"]):
        params[f"enc_{i}_Wq"] = Wq
        params[f"enc_{i}_Wk"] = Wk
        params[f"enc_{i}_Wv"] = Wv
        params[f"enc_{i}_W1"] = W1
        params[f"enc_{i}_b1"] = b1
        params[f"enc_{i}_W2"] = W2
        params[f"enc_{i}_b2"] = b2

    for i, (self_attn_w, cross_attn_w, ffn_w) in enumerate(model["decoder_layers"]):
        Wq_s, Wk_s, Wv_s = self_attn_w
        Wq_c, Wk_c, Wv_c = cross_attn_w
        W1, b1, W2, b2 = ffn_w
        params[f"dec_{i}_Wq_s"] = Wq_s
        params[f"dec_{i}_Wk_s"] = Wk_s
        params[f"dec_{i}_Wv_s"] = Wv_s
        params[f"dec_{i}_Wq_c"] = Wq_c
        params[f"dec_{i}_Wk_c"] = Wk_c
        params[f"dec_{i}_Wv_c"] = Wv_c
        params[f"dec_{i}_W1"] = W1
        params[f"dec_{i}_b1"] = b1
        params[f"dec_{i}_W2"] = W2
        params[f"dec_{i}_b2"] = b2

    return params


class AdamOptimizer:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.t += 1
        for key in self.params:
            if key not in grads:
                continue
            g = grads[key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
