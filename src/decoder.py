import numpy as np

from src.attention import scaled_dot_product_attention, init_attention_weights, project_qkv
from src.ffn import feed_forward, init_ffn_weights
from src.masks import create_causal_mask
from src.utils import softmax, residual_add_norm


def decoder_masked_self_attention(Y, Wq, Wk, Wv, mask, eps=1e-6):
    Q, K, V = project_qkv(Y, Wq, Wk, Wv)
    attn_out = scaled_dot_product_attention(Q, K, V, mask)
    return residual_add_norm(Y, attn_out, eps)


def init_decoder_block(d_model, d_ff):
    Wq_s, Wk_s, Wv_s = init_attention_weights(d_model)
    Wq_c, Wk_c, Wv_c = init_attention_weights(d_model)
    W1, b1, W2, b2 = init_ffn_weights(d_model, d_ff)
    return (Wq_s, Wk_s, Wv_s), (Wq_c, Wk_c, Wv_c), (W1, b1, W2, b2)


def decoder_block(Y, Z, self_attn_weights, cross_attn_weights, ffn_weights, mask, eps=1e-6):
    Wq_s, Wk_s, Wv_s = self_attn_weights
    Wq_c, Wk_c, Wv_c = cross_attn_weights
    W1, b1, W2, b2 = ffn_weights

    out = decoder_masked_self_attention(Y, Wq_s, Wk_s, Wv_s, mask, eps)

    Q = out @ Wq_c
    K = Z @ Wk_c
    V = Z @ Wv_c
    cross_out = scaled_dot_product_attention(Q, K, V)
    out = residual_add_norm(out, cross_out, eps)

    ffn_out = feed_forward(out, W1, b1, W2, b2)
    out = residual_add_norm(out, ffn_out, eps)

    return out


def init_decoder_stack(n_layers, d_model, d_ff):
    layers = []
    for _ in range(n_layers):
        block_weights = init_decoder_block(d_model, d_ff)
        layers.append(block_weights)
    return layers


def decoder(Y, Z, layers, mask):
    for self_attn_w, cross_attn_w, ffn_w in layers:
        Y = decoder_block(Y, Z, self_attn_w, cross_attn_w, ffn_w, mask)
    return Y


def output_projection(decoder_out, W_out):
    logits = decoder_out @ W_out
    return logits
