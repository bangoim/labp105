import numpy as np

from src.utils import softmax, EPSILON


def backward_layer_norm(dout, x_input, eps=EPSILON):
    mean = np.mean(x_input, axis=-1, keepdims=True)
    var = np.var(x_input, axis=-1, keepdims=True)
    std_inv = 1.0 / np.sqrt(var + eps)
    x_centered = x_input - mean
    x_norm = x_centered * std_inv
    D = x_input.shape[-1]

    dx_norm = dout
    dvar = np.sum(dx_norm * x_centered * (-0.5) * (std_inv ** 3), axis=-1, keepdims=True)
    dmean = np.sum(dx_norm * (-std_inv), axis=-1, keepdims=True) + dvar * np.mean(-2.0 * x_centered, axis=-1, keepdims=True)
    dx = dx_norm * std_inv + dvar * 2.0 * x_centered / D + dmean / D
    return dx


def backward_cross_entropy_softmax(logits, targets, pad_id=0):
    batch_size, seq_len, vocab_size = logits.shape
    probs = softmax(logits)
    mask = (targets != pad_id).astype(np.float64)
    n_valid = mask.sum()
    if n_valid == 0:
        return np.zeros_like(logits)

    dlogits = probs.copy()
    for b in range(batch_size):
        for t in range(seq_len):
            if mask[b, t] > 0:
                dlogits[b, t, targets[b, t]] -= 1.0
            else:
                dlogits[b, t, :] = 0.0
    dlogits /= n_valid
    return dlogits


def backward_attention(dout, Q, K, V, attn_weights, mask=None):
    d_k = Q.shape[-1]
    scale = np.sqrt(d_k)

    dV = attn_weights.transpose(0, 2, 1) @ dout
    d_attn_w = dout @ V.transpose(0, 2, 1)

    d_scores = attn_weights * (d_attn_w - np.sum(d_attn_w * attn_weights, axis=-1, keepdims=True))
    d_scores /= scale

    dQ = d_scores @ K
    dK = d_scores.transpose(0, 2, 1) @ Q

    return dQ, dK, dV


def backward_ffn(dout, X_in, W1, b1, W2, b2, hidden_pre, hidden):
    dW2 = np.einsum("bsi,bsj->ij", hidden, dout)
    db2 = dout.sum(axis=(0, 1))
    dhidden = dout @ W2.T

    dhidden_pre = dhidden * (hidden_pre > 0).astype(np.float64)
    dW1 = np.einsum("bsi,bsj->ij", X_in, dhidden_pre)
    db1 = dhidden_pre.sum(axis=(0, 1))
    dX = dhidden_pre @ W1.T

    return dX, dW1, db1, dW2, db2


def backward_decoder(dY, cache, model):
    grads = {}
    dec_layers = model["decoder_layers"]
    Z = cache["Z"]

    dZ_total = np.zeros_like(Z)

    for layer_idx in reversed(range(len(dec_layers))):
        lc = cache["dec_layers"][layer_idx]
        self_attn_w, cross_attn_w, ffn_w = dec_layers[layer_idx]
        Wq_s, Wk_s, Wv_s = self_attn_w
        Wq_c, Wk_c, Wv_c = cross_attn_w
        W1, b1, W2, b2 = ffn_w

        d_add3 = backward_layer_norm(dY, lc["add3"])
        d_norm2_ffn = d_add3.copy()
        d_ffn_out = d_add3.copy()

        dX_ffn, dW1, db1, dW2, db2 = backward_ffn(
            d_ffn_out, lc["norm2"], W1, b1, W2, b2, lc["hidden_pre"], lc["hidden"]
        )
        d_norm2 = d_norm2_ffn + dX_ffn

        grads[f"dec_{layer_idx}_W1"] = dW1
        grads[f"dec_{layer_idx}_b1"] = db1
        grads[f"dec_{layer_idx}_W2"] = dW2
        grads[f"dec_{layer_idx}_b2"] = db2

        d_add2 = backward_layer_norm(d_norm2, lc["add2"])
        d_norm1_cross = d_add2.copy()
        d_cross_out = d_add2.copy()

        dQc, dKc, dVc = backward_attention(
            d_cross_out, lc["Qc"], lc["Kc"], lc["Vc"], lc["attn_w_c"]
        )

        grads[f"dec_{layer_idx}_Wq_c"] = np.einsum("bsi,bsj->ij", lc["norm1"], dQc)
        grads[f"dec_{layer_idx}_Wk_c"] = np.einsum("bsi,bsj->ij", Z, dKc)
        grads[f"dec_{layer_idx}_Wv_c"] = np.einsum("bsi,bsj->ij", Z, dVc)

        d_norm1_from_cross = dQc @ Wq_c.T
        dZ_total += dKc @ Wk_c.T + dVc @ Wv_c.T

        d_norm1 = d_norm1_cross + d_norm1_from_cross

        d_add1 = backward_layer_norm(d_norm1, lc["add1"])
        d_input_self = d_add1.copy()
        d_self_attn_out = d_add1.copy()

        dQs, dKs, dVs = backward_attention(
            d_self_attn_out, lc["Qs"], lc["Ks"], lc["Vs"], lc["attn_w_s"]
        )

        Y_in = lc["input"]
        grads[f"dec_{layer_idx}_Wq_s"] = np.einsum("bsi,bsj->ij", Y_in, dQs)
        grads[f"dec_{layer_idx}_Wk_s"] = np.einsum("bsi,bsj->ij", Y_in, dKs)
        grads[f"dec_{layer_idx}_Wv_s"] = np.einsum("bsi,bsj->ij", Y_in, dVs)

        dY = d_input_self + dQs @ Wq_s.T + dKs @ Wk_s.T + dVs @ Wv_s.T

    return dY, dZ_total, grads


def backward_encoder(dZ, cache, model):
    grads = {}
    enc_layers = model["encoder_layers"]

    dX = dZ
    for layer_idx in reversed(range(len(enc_layers))):
        lc = cache["enc_layers"][layer_idx]
        Wq, Wk, Wv, W1, b1, W2, b2 = enc_layers[layer_idx]

        d_add2 = backward_layer_norm(dX, lc["add2"])
        d_norm1_ffn = d_add2.copy()
        d_ffn_out = d_add2.copy()

        dX_ffn, dW1, db1, dW2, db2 = backward_ffn(
            d_ffn_out, lc["norm1"], W1, b1, W2, b2, lc["hidden_pre"], lc["hidden"]
        )
        d_norm1 = d_norm1_ffn + dX_ffn

        grads[f"enc_{layer_idx}_W1"] = dW1
        grads[f"enc_{layer_idx}_b1"] = db1
        grads[f"enc_{layer_idx}_W2"] = dW2
        grads[f"enc_{layer_idx}_b2"] = db2

        d_add1 = backward_layer_norm(d_norm1, lc["add1"])
        d_input_attn = d_add1.copy()
        d_attn_out = d_add1.copy()

        dQ, dK, dV = backward_attention(
            d_attn_out, lc["Q"], lc["K"], lc["V"], lc["attn_weights"]
        )

        X_in = lc["input"]
        grads[f"enc_{layer_idx}_Wq"] = np.einsum("bsi,bsj->ij", X_in, dQ)
        grads[f"enc_{layer_idx}_Wk"] = np.einsum("bsi,bsj->ij", X_in, dK)
        grads[f"enc_{layer_idx}_Wv"] = np.einsum("bsi,bsj->ij", X_in, dV)

        dX = d_input_attn + dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T

    return dX, grads


def backward_embeddings(d_enc_input, d_dec_input, cache, embedding_table):
    d_emb = np.zeros_like(embedding_table)
    enc_ids = cache["encoder_input_ids"]
    dec_ids = cache["decoder_input_ids"]

    for b in range(enc_ids.shape[0]):
        for s in range(enc_ids.shape[1]):
            d_emb[enc_ids[b, s]] += d_enc_input[b, s]

    for b in range(dec_ids.shape[0]):
        for s in range(dec_ids.shape[1]):
            d_emb[dec_ids[b, s]] += d_dec_input[b, s]

    return d_emb


def full_backward(logits, targets, cache, model, pad_id=0):
    grads = {}

    dlogits = backward_cross_entropy_softmax(logits, targets, pad_id)

    grads["W_out"] = np.einsum("bsi,bsj->ij", cache["dec_out"], dlogits)
    d_dec_out = dlogits @ model["W_out"].T

    d_dec_emb, dZ_from_dec, dec_grads = backward_decoder(d_dec_out, cache, model)
    grads.update(dec_grads)

    d_enc_emb, enc_grads = backward_encoder(dZ_from_dec, cache, model)
    grads.update(enc_grads)

    grads["embedding_table"] = backward_embeddings(
        d_enc_emb, d_dec_emb, cache, model["embedding_table"]
    )

    return grads
