# Mini Transformer

Japanese literature text generation project for a learning portfolio.

The goal is to build a small GPT-style Transformer decoder from scratch, train it
on Japanese literary texts such as works by Natsume Soseki, Akutagawa Ryunosuke,
and Dazai Osamu, and compare how generation changes by author and sampling
settings.

## Learning Goals

- Prepare Japanese literary text data in a reproducible way.
- Start with a self-made character tokenizer.
- Later compare the character tokenizer with SentencePiece.
- Implement a small Transformer decoder in plain PyTorch.
- Train with a visible loss curve.
- Compare generation with temperature, top-k, and top-p sampling.
- Record author-style examples, failure cases, and why the Japanese output can become strange.

## Current Status

Ticket 7 is complete. The project now has a small reproducible Aozora Bunko data
preparation pipeline, a self-made character tokenizer, and next-token
language-modeling batches, plus a small GPT-style decoder for forward-pass
smoke checks, a plain PyTorch training loop, and checkpoint-based text
generation.

Style comparison experiments will be added in a later ticket.

## Workflow

See `AGENTS.md` for repository rules and `tickets.md` for the ticket roadmap.

Each ticket should be implemented, verified, committed locally, and then stopped
before moving to the next ticket.

## Data Preparation

Ticket 2 adds a reproducible Aozora Bunko-style data preparation pipeline. The
tracked manifest is [manifests/aozora_works.json](manifests/aozora_works.json),
and the generated text files are intentionally ignored by Git.

Run:

```bash
uv run python scripts/prepare_aozora.py
```

The script downloads the text zip linked from each Aozora card, extracts the
Shift-JIS text, removes common Aozora header/footer text, ruby markup, and
annotation markup, then writes cleaned UTF-8 text under `data/processed/`.

The smoke output prints `author_id,title,characters,processed_text` so the next
ticket can confirm there is usable text for each author. The manifest uses three
works per author:

- 夏目 漱石: `坊っちゃん`, `こころ`, `吾輩は猫である`
- 芥川 竜之介: `羅生門`, `鼻`, `蜘蛛の糸`
- 太宰 治: `走れメロス`, `斜陽`, `人間失格`

Limitations:

- The script is a small learning pipeline, not a complete Aozora parser.
- Cleaning rules are intentionally simple and should be inspected before using a
  larger corpus.
- Full raw and processed texts are local generated artifacts and are not
  committed.

## Character Tokenizer

Ticket 3 adds a self-made character tokenizer. It learns one token per character
from the cleaned text files and reserves four special tokens at the beginning of
the vocabulary:

- `<pad>`: padding token for future batches
- `<unk>`: fallback token for unseen characters
- `<bos>`: beginning-of-sequence marker
- `<eos>`: end-of-sequence marker

Run the inspection command after data preparation:

```bash
uv run python scripts/inspect_char_tokenizer.py
```

The command builds a vocabulary from `data/processed/**/*.txt`, saves it to
`data/tokenizers/char_vocab.json`, reloads it, and prints a small
text -> token ids -> text round trip. The vocabulary file is generated under
`data/`, so it is intentionally ignored by Git.

## Dataset And Batching

Ticket 4 adds next-token language-modeling batches. Each example is a fixed
length window where `input_ids` is the current token sequence and `target_ids` is
the same sequence shifted one token to the right.

Run:

```bash
uv run python scripts/inspect_lm_batches.py
```

The command loads the cleaned text records, creates the character vocabulary if
needed, creates train/validation `DataLoader` objects, and prints batch shapes
plus author/source metadata. The metadata is kept in each batch so later tickets
can compare generation by author.

## Mini Transformer Decoder

Ticket 5 adds a small GPT-style decoder in plain PyTorch. The model includes
token embeddings, positional embeddings, causal self-attention, feed-forward
layers, residual connections, layer normalization, and a language-modeling head.

Run the forward-pass smoke check:

```bash
uv run python scripts/inspect_model_forward.py
```

The command creates random token ids, runs a forward pass, and checks that the
logits have shape `[batch, block, vocab]`. The default smoke model is deliberately
small so later local training experiments can run on a laptop.

## Training Loop And Loss Curve

Ticket 6 adds a plain PyTorch training loop. It loads the processed text records,
builds or loads the character vocabulary, creates train/validation batches,
trains the mini Transformer with next-token cross entropy loss, and saves basic
learning artifacts under `outputs/`.

Run a tiny smoke training job:

```bash
uv run python scripts/train.py \
  --output-dir outputs/ticket6_smoke \
  --block-size 16 \
  --stride 256 \
  --batch-size 2 \
  --embedding-dim 32 \
  --num-layers 1 \
  --num-heads 4 \
  --feed-forward-dim 64 \
  --dropout 0.0 \
  --epochs 1 \
  --max-train-batches 2 \
  --max-validation-batches 1
```

The smoke run is only a pipeline check. It should produce:

- `outputs/ticket6_smoke/checkpoint.pt`
- `outputs/ticket6_smoke/metrics.json`
- `outputs/ticket6_smoke/loss_curve.png`

For a longer local run, remove the batch limits and use the default model size:

```bash
uv run python scripts/train.py --output-dir outputs/local_run --epochs 5
```

The training script prefers Apple Silicon `mps` when available, then falls back
to CUDA or CPU. Generated checkpoints, metrics, plots, vocabulary files, and text
data are ignored by Git.

## Generation Sampling

Ticket 7 adds checkpoint-based text generation. The script loads
`checkpoint.pt`, restores the model config and weights, loads the character
vocabulary, encodes a prompt, and repeatedly samples the next token from the last
position's logits.

Greedy decoding uses the highest-logit token each step:

```bash
uv run python scripts/generate.py \
  --checkpoint outputs/ticket6_smoke/checkpoint.pt \
  --prompt 吾輩は \
  --max-new-tokens 80 \
  --temperature 0
```

Temperature sampling makes generation more or less random. Higher values sample
more freely; `0` means greedy decoding:

```bash
uv run python scripts/generate.py \
  --checkpoint outputs/ticket6_smoke/checkpoint.pt \
  --prompt 吾輩は \
  --max-new-tokens 80 \
  --temperature 0.8
```

Top-k sampling keeps only the `k` highest-logit candidates before sampling:

```bash
uv run python scripts/generate.py \
  --checkpoint outputs/ticket6_smoke/checkpoint.pt \
  --prompt 吾輩は \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 20
```

Top-p sampling keeps the smallest set of candidates whose cumulative probability
reaches `p`:

```bash
uv run python scripts/generate.py \
  --checkpoint outputs/ticket6_smoke/checkpoint.pt \
  --prompt 吾輩は \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-p 0.9
```

You can combine `--top-k` and `--top-p`, and you can save the generation log
under ignored output paths:

```bash
uv run python scripts/generate.py \
  --checkpoint outputs/ticket6_smoke/checkpoint.pt \
  --prompt 吾輩は \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 20 \
  --top-p 0.9 \
  --output-path outputs/generation/sample.txt
```

Representative smoke examples from a short local checkpoint are below. These
only prove that the generation path works; they are not quality claims.

```text
greedy:
吾輩はまた。
　私は、その時、その時、その時、

top-k + top-p:
吾輩はあの顔を見ている。
「それであなた。
```

## Planned Outputs

- A reproducible data preparation pipeline for Aozora Bunko-style texts.
- A character tokenizer and a later SentencePiece tokenizer comparison.
- A small GPT-style Transformer decoder.
- Training metrics and loss curve plots.
- Generated text examples by author and sampling settings.
- A written failure analysis suitable for a portfolio README or experiment note.
