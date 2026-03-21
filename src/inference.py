import numpy as np

from src.transformer import transformer_forward


def autoregressive_generate(encoder_input_ids, model, start_id, eos_id, max_len=64):
    decoder_ids = np.array([[start_id]])

    for _ in range(max_len):
        logits = transformer_forward(encoder_input_ids, decoder_ids, model)
        next_id = int(np.argmax(logits[0, -1, :]))
        decoder_ids = np.concatenate([decoder_ids, np.array([[next_id]])], axis=1)
        if next_id == eos_id:
            break

    return decoder_ids[0].tolist()
