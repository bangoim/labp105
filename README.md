# Lab P105 - Training Loop do Transformer

O dataset usado é o [opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books) (EN→FR), com um subset de 1000 frases pra não demorar uma eternidade.

## Como rodar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Treinamento completo:
```bash
python3 -m src.train
```

Teste de overfitting (5 frases, pra provar que o backprop funciona):
```bash
python3 -m src.overfit_test
```

Testes:
```bash
python3 -m unittest discover -s tests -v
```

## O que tem aqui

- `src/train.py` — training loop (forward → loss → backward → adam step)
- `src/backward.py` — backpropagation manual por todo o transformer
- `src/optimizer.py` — adam do zero
- `src/overfit_test.py` — overfitting em 5 frases pra validar os gradientes
- `src/data_pipeline.py` — carrega opus_books, tokeniza e monta batches
- o resto em `src/` é o modelo em si (encoder, decoder, attention, ffn, etc.)

## Aviso

O modelo não vai traduzir textos novos direito. A ideia deste lab é só mostrar que a loss cai e que ele consegue decorar um conjunto pequeno de frases. Pra tradução de verdade ia precisar de muito mais dado e GPU.
