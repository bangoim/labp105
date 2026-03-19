import numpy as np


def create_embedding_table(vocab_size, d_model):
    scale = np.sqrt(1.0 / d_model)
    return np.random.randn(vocab_size, d_model) * scale


def get_embeddings(token_ids, embedding_table):
    return embedding_table[token_ids]


def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.power(10000.0, np.arange(0, d_model, 2) / d_model)

    pe[:, 0::2] = np.sin(position / div_term)
    pe[:, 1::2] = np.cos(position / div_term)

    return pe
