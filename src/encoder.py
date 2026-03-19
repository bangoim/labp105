from src.attention import scaled_dot_product_attention, init_attention_weights, project_qkv
from src.ffn import feed_forward, init_ffn_weights
from src.utils import residual_add_norm


def init_encoder_block(d_model, d_ff):
    Wq, Wk, Wv = init_attention_weights(d_model)
    W1, b1, W2, b2 = init_ffn_weights(d_model, d_ff)
    return Wq, Wk, Wv, W1, b1, W2, b2


def encoder_block(X, Wq, Wk, Wv, W1, b1, W2, b2, eps=1e-6):
    Q, K, V = project_qkv(X, Wq, Wk, Wv)
    attn_out = scaled_dot_product_attention(Q, K, V)
    X = residual_add_norm(X, attn_out, eps)

    ffn_out = feed_forward(X, W1, b1, W2, b2)
    X = residual_add_norm(X, ffn_out, eps)

    return X


def init_encoder_stack(n_layers, d_model, d_ff):
    layers = []
    for _ in range(n_layers):
        block_weights = init_encoder_block(d_model, d_ff)
        layers.append(block_weights)
    return layers


def encoder(X, layers):
    for Wq, Wk, Wv, W1, b1, W2, b2 in layers:
        X = encoder_block(X, Wq, Wk, Wv, W1, b1, W2, b2)
    return X
