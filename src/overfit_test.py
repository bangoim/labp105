import numpy as np

from src.transformer import init_transformer, forward_with_cache
from src.loss import cross_entropy_loss
from src.backward import full_backward
from src.optimizer import collect_params, AdamOptimizer
from src.tokenizer import load_tokenizer, tokenize_pairs, pad_sequences
from src.inference import autoregressive_generate


def overfit_test():
    tokenizer, start_id, eos_id, pad_id = load_tokenizer()
    vocab_size = tokenizer.vocab_size

    pairs = [
        ("The cat is on the table", "Le chat est sur la table"),
        ("I love music", "J'aime la musique"),
        ("Good morning", "Bonjour"),
        ("The sky is blue", "Le ciel est bleu"),
        ("She reads a book", "Elle lit un livre"),
    ]

    src_seqs, tgt_seqs = tokenize_pairs(pairs, tokenizer, start_id, eos_id, max_len=32)
    src_padded = pad_sequences(src_seqs, pad_id)
    tgt_padded = pad_sequences(tgt_seqs, pad_id)
    dec_input = tgt_padded[:, :-1]
    targets = tgt_padded[:, 1:]

    d_model = 64
    d_ff = 128
    n_layers = 2
    lr = 5e-4
    epochs = 200

    np.random.seed(42)
    model = init_transformer(vocab_size, d_model, d_ff, n_layers)
    params = collect_params(model)
    optimizer = AdamOptimizer(params, lr=lr)

    print(f"Overfitting em {len(pairs)} frases por {epochs} epochs...")
    print(f"d_model={d_model}, n_layers={n_layers}\n")

    for epoch in range(epochs):
        logits, cache = forward_with_cache(src_padded, dec_input, model)
        loss, _ = cross_entropy_loss(logits, targets, pad_id)
        grads = full_backward(logits, targets, cache, model, pad_id)

        max_grad = max(np.abs(g).max() for g in grads.values())
        if max_grad > 5.0:
            grads = {k: v * 5.0 / max_grad for k, v in grads.items()}

        optimizer.step(grads)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

    print("\n--- Teste de memorização ---\n")

    for i, (src_text, expected) in enumerate(pairs):
        enc_ids = src_padded[i:i+1]
        generated_ids = autoregressive_generate(enc_ids, model, start_id, eos_id, max_len=32)

        if generated_ids[0] == start_id:
            generated_ids = generated_ids[1:]
        if generated_ids and generated_ids[-1] == eos_id:
            generated_ids = generated_ids[:-1]

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Input:    {src_text}")
        print(f"Esperado: {expected}")
        print(f"Gerado:   {generated_text}")
        print()


if __name__ == "__main__":
    overfit_test()
