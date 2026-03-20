import numpy as np

from src.dataset import load_opus_books
from src.tokenizer import load_tokenizer, tokenize_pairs, pad_sequences


def prepare_data(n_samples=1000, max_len=64, lang_pair="en-fr"):
    pairs = load_opus_books(lang_pair=lang_pair, n_samples=n_samples)
    tokenizer, start_id, eos_id, pad_id = load_tokenizer()
    src_seqs, tgt_seqs = tokenize_pairs(pairs, tokenizer, start_id, eos_id, max_len)

    src_padded = pad_sequences(src_seqs, pad_id)
    tgt_padded = pad_sequences(tgt_seqs, pad_id)

    decoder_input = tgt_padded[:, :-1]
    targets = tgt_padded[:, 1:]

    vocab_size = tokenizer.vocab_size
    return src_padded, decoder_input, targets, vocab_size, pad_id, tokenizer, start_id, eos_id


def create_batches(src, dec_input, targets, batch_size=32):
    n = len(src)
    indices = np.arange(n)
    np.random.shuffle(indices)
    batches = []
    for i in range(0, n, batch_size):
        idx = indices[i:i + batch_size]
        batches.append((src[idx], dec_input[idx], targets[idx]))
    return batches
