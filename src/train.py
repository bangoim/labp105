import numpy as np

from src.transformer import init_transformer, forward_with_cache
from src.loss import cross_entropy_loss
from src.backward import full_backward
from src.optimizer import collect_params, AdamOptimizer
from src.data_pipeline import prepare_data, create_batches


def train(n_samples=1000, d_model=128, d_ff=256, n_layers=2,
          epochs=15, batch_size=32, lr=1e-3, max_len=40):

    print("Preparando dados...")
    src, dec_input, targets, vocab_size, pad_id, tokenizer, start_id, eos_id = prepare_data(
        n_samples=n_samples, max_len=max_len
    )
    print(f"Dados: {src.shape[0]} pares, vocab_size={vocab_size}")
    print(f"Encoder input shape: {src.shape}, Decoder input shape: {dec_input.shape}")

    print("Inicializando modelo...")
    model = init_transformer(vocab_size, d_model, d_ff, n_layers)
    params = collect_params(model)
    optimizer = AdamOptimizer(params, lr=lr)

    print(f"Treinando por {epochs} epochs...")
    for epoch in range(epochs):
        batches = create_batches(src, dec_input, targets, batch_size)
        epoch_loss = 0.0
        n_batches = 0

        for batch_src, batch_dec, batch_tgt in batches:
            logits, cache = forward_with_cache(batch_src, batch_dec, model)
            loss, _ = cross_entropy_loss(logits, batch_tgt, pad_id)
            grads = full_backward(logits, batch_tgt, cache, model, pad_id)

            max_grad = max(np.abs(g).max() for g in grads.values())
            if max_grad > 5.0:
                clip_factor = 5.0 / max_grad
                grads = {k: v * clip_factor for k, v in grads.items()}

            optimizer.step(grads)
            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    return model, tokenizer, start_id, eos_id, pad_id


if __name__ == "__main__":
    np.random.seed(42)
    train(n_samples=1000, d_model=128, d_ff=256, n_layers=2, epochs=15, batch_size=32)
