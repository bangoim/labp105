from datasets import load_dataset


def load_opus_books(lang_pair="en-fr", n_samples=1000):
    ds = load_dataset("Helsinki-NLP/opus_books", lang_pair, split="train")
    ds = ds.select(range(min(n_samples, len(ds))))
    pairs = []
    src, tgt = lang_pair.split("-")
    for item in ds:
        pairs.append((item["translation"][src], item["translation"][tgt]))
    return pairs
