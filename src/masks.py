import numpy as np


def create_causal_mask(seq_len):
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    return mask


def create_padding_mask(token_ids, pad_id=0):
    return (token_ids == pad_id).astype(np.float64)
