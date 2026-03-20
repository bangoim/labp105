import numpy as np

from src.embeddings import get_embeddings, positional_encoding, create_embedding_table
from src.encoder import init_encoder_stack
from src.decoder import init_decoder_stack
from src.masks import create_causal_mask
from src.utils import softmax, layer_norm, EPSILON
from src.attention import scaled_dot_product_attention


def init_transformer(vocab_size, d_model, d_ff, n_layers):
    embedding_table = create_embedding_table(vocab_size, d_model)
    encoder_layers = init_encoder_stack(n_layers, d_model, d_ff)
    decoder_layers = init_decoder_stack(n_layers, d_model, d_ff)
    scale = np.sqrt(2.0 / d_model)
    W_out = np.random.randn(d_model, vocab_size) * scale
    return {
        "embedding_table": embedding_table,
        "encoder_layers": encoder_layers,
        "decoder_layers": decoder_layers,
        "W_out": W_out,
        "d_model": d_model,
        "vocab_size": vocab_size,
    }


def forward_with_cache(encoder_input_ids, decoder_input_ids, model):
    """Forward pass completo retornando cache para backpropagation.

    Args:
        encoder_input_ids: (batch, src_len) IDs de entrada do encoder
        decoder_input_ids: (batch, tgt_len) IDs de entrada do decoder
        model: dicionário com pesos do modelo

    Returns:
        logits: (batch, tgt_len, vocab_size) logits de saída
        cache: dicionário com valores intermediários para backprop
    """
    cache = {}
    emb_table = model["embedding_table"]
    d_model = model["d_model"]

    cache["encoder_input_ids"] = encoder_input_ids
    cache["decoder_input_ids"] = decoder_input_ids

    enc_emb = get_embeddings(encoder_input_ids, emb_table)
    enc_pe = positional_encoding(enc_emb.shape[1], d_model)
    enc_input = enc_emb + enc_pe

    cache["enc_layers"] = []
    X = enc_input
    for layer_idx, (Wq, Wk, Wv, W1, b1, W2, b2) in enumerate(model["encoder_layers"]):
        lc = {}
        lc["input"] = X

        Q, K, V = X @ Wq, X @ Wk, X @ Wv
        lc["Q"], lc["K"], lc["V"] = Q, K, V

        d_k = Q.shape[-1]
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_k)
        attn_weights = softmax(scores)
        attn_out = attn_weights @ V
        lc["attn_weights"] = attn_weights
        lc["attn_out"] = attn_out

        add1 = X + attn_out
        lc["add1"] = add1
        norm1 = layer_norm(add1)
        lc["norm1"] = norm1

        hidden_pre = norm1 @ W1 + b1
        hidden = np.maximum(0, hidden_pre)
        ffn_out = hidden @ W2 + b2
        lc["hidden_pre"] = hidden_pre
        lc["hidden"] = hidden
        lc["ffn_out"] = ffn_out

        add2 = norm1 + ffn_out
        lc["add2"] = add2
        X = layer_norm(add2)
        lc["output"] = X

        cache["enc_layers"].append(lc)

    Z = X
    cache["Z"] = Z

    dec_emb = get_embeddings(decoder_input_ids, emb_table)
    dec_pe = positional_encoding(dec_emb.shape[1], d_model)
    dec_input = dec_emb + dec_pe
    causal_mask = create_causal_mask(dec_emb.shape[1])

    cache["dec_layers"] = []
    Y = dec_input
    for layer_idx, (self_attn_w, cross_attn_w, ffn_w) in enumerate(model["decoder_layers"]):
        lc = {}
        Wq_s, Wk_s, Wv_s = self_attn_w
        Wq_c, Wk_c, Wv_c = cross_attn_w
        W1, b1, W2, b2 = ffn_w

        lc["input"] = Y

        Qs, Ks, Vs = Y @ Wq_s, Y @ Wk_s, Y @ Wv_s
        lc["Qs"], lc["Ks"], lc["Vs"] = Qs, Ks, Vs

        d_k = Qs.shape[-1]
        scores_s = (Qs @ Ks.transpose(0, 2, 1)) / np.sqrt(d_k) + causal_mask
        attn_w_s = softmax(scores_s)
        attn_out_s = attn_w_s @ Vs
        lc["attn_w_s"] = attn_w_s
        lc["attn_out_s"] = attn_out_s

        add1 = Y + attn_out_s
        lc["add1"] = add1
        norm1 = layer_norm(add1)
        lc["norm1"] = norm1

        Qc = norm1 @ Wq_c
        Kc = Z @ Wk_c
        Vc = Z @ Wv_c
        lc["Qc"], lc["Kc"], lc["Vc"] = Qc, Kc, Vc

        scores_c = (Qc @ Kc.transpose(0, 2, 1)) / np.sqrt(d_k)
        attn_w_c = softmax(scores_c)
        cross_out = attn_w_c @ Vc
        lc["attn_w_c"] = attn_w_c
        lc["cross_out"] = cross_out

        add2 = norm1 + cross_out
        lc["add2"] = add2
        norm2 = layer_norm(add2)
        lc["norm2"] = norm2

        hidden_pre = norm2 @ W1 + b1
        hidden = np.maximum(0, hidden_pre)
        ffn_out = hidden @ W2 + b2
        lc["hidden_pre"] = hidden_pre
        lc["hidden"] = hidden
        lc["ffn_out"] = ffn_out

        add3 = norm2 + ffn_out
        lc["add3"] = add3
        Y = layer_norm(add3)
        lc["output"] = Y

        cache["dec_layers"].append(lc)

    cache["dec_out"] = Y
    logits = Y @ model["W_out"]
    cache["logits"] = logits

    return logits, cache


def transformer_forward(encoder_input_ids, decoder_input_ids, model):
    logits, _ = forward_with_cache(encoder_input_ids, decoder_input_ids, model)
    return logits
