import numpy as np

from src.embeddings import get_embeddings, positional_encoding, create_embedding_table
from src.encoder import init_encoder_stack, encoder
from src.decoder import init_decoder_stack, decoder, output_projection
from src.masks import create_causal_mask


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


def transformer_forward(encoder_input_ids, decoder_input_ids, model):
    emb_table = model["embedding_table"]
    d_model = model["d_model"]

    enc_emb = get_embeddings(encoder_input_ids, emb_table)
    enc_pe = positional_encoding(enc_emb.shape[1], d_model)
    enc_input = enc_emb + enc_pe

    Z = encoder(enc_input, model["encoder_layers"])

    dec_emb = get_embeddings(decoder_input_ids, emb_table)
    dec_pe = positional_encoding(dec_emb.shape[1], d_model)
    dec_input = dec_emb + dec_pe

    mask = create_causal_mask(dec_emb.shape[1])
    dec_out = decoder(dec_input, Z, model["decoder_layers"], mask)

    logits = output_projection(dec_out, model["W_out"])
    return logits
