import numpy as np
from transformers import AutoTokenizer


def load_tokenizer(model_name="bert-base-multilingual-cased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    start_id = tokenizer.convert_tokens_to_ids("[unused1]")
    eos_id = tokenizer.convert_tokens_to_ids("[unused2]")
    pad_id = tokenizer.pad_token_id
    return tokenizer, start_id, eos_id, pad_id


def tokenize_pairs(pairs, tokenizer, start_id, eos_id, max_len=64):
    src_sequences = []
    tgt_sequences = []

    for src_text, tgt_text in pairs:
        src_ids = tokenizer.encode(src_text, add_special_tokens=False)[:max_len]
        tgt_ids = tokenizer.encode(tgt_text, add_special_tokens=False)[:max_len - 2]
        tgt_ids = [start_id] + tgt_ids + [eos_id]
        src_sequences.append(src_ids)
        tgt_sequences.append(tgt_ids)

    return src_sequences, tgt_sequences


def pad_sequences(sequences, pad_id=0):
    max_len = max(len(s) for s in sequences)
    padded = np.full((len(sequences), max_len), pad_id, dtype=np.int64)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    return padded
