# Mini Transformer

Japanese literature text generation project for a learning portfolio.

The goal is to build a small GPT-style Transformer decoder from scratch and
train it on Japanese literary texts. The current reproducible dataset uses ten
works by Natsume Soseki so the model can learn from one consistent author corpus
before any cross-author comparison.

## Learning Goals

- Prepare Japanese literary text data in a reproducible way.
- Start with a self-made character tokenizer.
- Compare the character tokenizer with SentencePiece.
- Implement a small Transformer decoder in plain PyTorch.
- Train with a visible loss curve.
- Compare generation with temperature, top-k, and top-p sampling.
- Record author-specific examples, failure cases, and why the Japanese output can become strange.

## Current Status

Ticket 17 is complete. The project now has a reproducible Aozora Bunko data
pipeline, character and SentencePiece tokenizers, language-modeling batches, a
small GPT-style decoder, a plain PyTorch training loop, checkpoint-based
generation, sampling comparisons, optional input/output embedding tying, and a
separate Hugging Face pretrained-model fine-tuning baseline.

The current best baseline is a 100 epoch tied SentencePiece run with stable
small-weight initialization. It reached validation loss `4.5810` and produced
phrase-like Japanese fragments, but it still repeats phrases, breaks grammar,
and loses long-range coherence. Detailed training logs and comparisons are in
[experiments/results.md](experiments/results.md).
For a concise portfolio or interview-oriented evaluation, see
[experiments/generation_quality_summary.md](experiments/generation_quality_summary.md).

## Best Result Summary

Best run:

- Tokenizer: `data/tokenizers/sentencepiece_unigram_vocab3000_cov998.model`
- Model: mini Transformer decoder with tied input/output embeddings
- Initialization: `Embedding` and `Linear` weights initialized with `std=0.02`
- Training: 100 epochs with cosine decay to `0.00002`
- Best validation loss: `4.5810`

Improvement path:

```text
character tokenizer
-> SentencePiece
-> embedding tying
-> stable initialization
-> 100 epoch training
-> temperature comparison
```

The most useful qualitative change is that generation moved from rare-character
jumps and repeated token collapse toward sentence-like fragments. The model is
still a learning-scale generator, not a coherent prose model.

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
ticket can confirm there is usable text. The current manifest uses ten works by
夏目 漱石: `坊っちゃん`, `こころ`, `吾輩は猫である`, `草枕`, `三四郎`,
`それから`, `門`, `彼岸過迄`, `行人`, and `道草`.

This author-specific corpus is intentional. It prevents the model's training
examples from mixing the prose of multiple authors while assessing whether the
small model can learn a consistent style. Earlier multi-author experiments in
`experiments/results.md` remain historical results, not the current dataset.

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
- `outputs/ticket6_smoke/best_checkpoint.pt`
- `outputs/ticket6_smoke/metrics.json`
- `outputs/ticket6_smoke/loss_curve.png`

For a longer local run, remove the batch limits and use the default model size.
This example uses warmup plus cosine learning-rate decay and stops early when
validation loss no longer improves:

```bash
uv run python scripts/train.py \
  --output-dir outputs/local_run \
  --epochs 30 \
  --scheduler cosine \
  --warmup-ratio 0.05 \
  --min-learning-rate 0.00005 \
  --early-stopping-patience 10 \
  --early-stopping-min-delta 0.001
```

The training script prefers Apple Silicon `mps` when available, then falls back
to CUDA or CPU. Generated checkpoints, metrics, plots, vocabulary files, and text
data are ignored by Git.

When scheduler support is enabled with `--scheduler cosine`, the learning rate
warms up for `--warmup-steps` or `--warmup-ratio` of total optimizer steps, then
decays toward `--min-learning-rate`. The metrics file records the learning rate
for each epoch, `best_validation_loss`, `best_epoch`, and whether training
stopped early. `checkpoint.pt` stores the final model state, while
`best_checkpoint.pt` stores the model state from the best validation epoch.

## Colab And W&B

For longer runs on Google Colab, select a GPU runtime, clone the repository, and
set up the project with `uv`:

```bash
git clone <your-repository-url>
cd mini-transformer
pip install uv
uv sync --locked
```

Log in to Weights & Biases once per Colab runtime:

```bash
uv run wandb login
```

Then prepare the local Colab dataset and train with W&B logging enabled:

```bash
uv run python scripts/prepare_aozora.py

uv run python scripts/train.py \
  --output-dir outputs/local_run_colab \
  --block-size 64 \
  --stride 64 \
  --batch-size 32 \
  --embedding-dim 128 \
  --num-layers 4 \
  --num-heads 4 \
  --feed-forward-dim 512 \
  --dropout 0.2 \
  --epochs 50 \
  --learning-rate 0.001 \
  --scheduler cosine \
  --warmup-ratio 0.05 \
  --min-learning-rate 0.00005 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.001 \
  --use-wandb \
  --wandb-project mini-transformer \
  --wandb-run-name block64-char-v1
```

W&B logging is opt-in. When `--use-wandb` is set, the script logs the run config,
epoch losses, learning rate, `loss_curve.png`, `metrics.json`, and
`best_checkpoint.pt` as a model artifact. The final `checkpoint.pt` stays local
to avoid storing every checkpoint artifact.

Generate from the best Colab checkpoint:

```bash
uv run python scripts/generate.py \
  --checkpoint outputs/local_run_colab/best_checkpoint.pt \
  --prompt 吾輩は \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 20 \
  --top-p 0.9
```

Colab runtimes are temporary, so copy any local `outputs/` files you need to
Google Drive or rely on W&B for the logged metrics and best checkpoint artifact.

## Generation Sampling

Ticket 7 adds checkpoint-based text generation. The script loads
`checkpoint.pt`, restores the model config and weights, loads the character
vocabulary, encodes a prompt, and repeatedly samples the next token from the last
position's logits.

Recommended command for the current best checkpoint:

```bash
uv run python scripts/generate.py \
  --checkpoint outputs/sentencepiece_unigram_tied_dropout0.1_lr3e-4_v1/best_checkpoint.pt \
  --prompt 私は \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 20 \
  --top-p 0.9 \
  --seed 42
```

The general default is `temperature=0.8`, `top-k=20`, and `top-p=0.9`. For the
prompt `吾輩は`, `temperature=0.7` was more stable in the latest comparison
because it reduced phrase repetition and topic drift.

Representative examples from the 100 epoch tied SentencePiece checkpoint:

```text
prompt=吾輩は, temperature=0.7
吾輩はこの長が、ただ、やが、ただ、一言とうとうときにあろう。
二尺の中で、三度に上へ出し、これも、三度に、下りとうつまって、
三十五円の中を、その後に向って、下女に、この時の日本の上って...

prompt=私は, temperature=0.8
私はこの 一 先生はその日とうた。そうしてその日日、奥さんの中の前に
その下宿へ行ってくれと、この晩から帰って来て、この家へ行った。
私はいつまでも、先生をお前から私の室に着いているといった...
```

These examples are better than the early smoke outputs, but they still show the
main limitations: repeated phrases, broken grammar, unnatural word connections,
and weak long-range meaning.

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

## Style Comparison Experiments

Ticket 8 adds [experiments/results.md](experiments/results.md). It records a
small comparable run for each author using the same model settings, generation
examples for greedy, temperature, top-k, and top-p sampling, and an honest
failure analysis.

The local artifacts are intentionally ignored by Git:

- `outputs/ticket8_style/<author_id>/metrics.json`
- `outputs/ticket8_style/<author_id>/loss_curve.png`
- `outputs/ticket8_style/<author_id>/best_checkpoint.pt`
- `outputs/ticket8_style/<author_id>/*.txt`

The runs show the training and generation pipeline working, but they do not yet
show reliable author style. The documented failures include repeated characters,
rare kanji jumps, punctuation loops, small per-author datasets, character
tokenizer limits, short context length, small model size, short training time,
and sampling randomness.

## SentencePiece Comparison

Ticket 9 adds an optional SentencePiece tokenizer path. The character tokenizer
remains the default, so older commands still work. To train a SentencePiece
Unigram model from the processed local texts:

```bash
uv run python scripts/train_sentencepiece_tokenizer.py \
  --model-path data/tokenizers/ticket9_sentencepiece_unigram.model \
  --vocab-size 4000 \
  --model-type unigram
```

The generated `.model` and `.vocab` files live under `data/`, which is ignored
by Git. The default `vocab-size` is 4000 because this Japanese corpus has enough
distinct characters that smaller vocabularies can be smaller than the required
covered character set.

Train with SentencePiece by passing the tokenizer type and model path:

```bash
uv run python scripts/train.py \
  --tokenizer-type sentencepiece \
  --vocab-path data/tokenizers/ticket9_sentencepiece_unigram.model \
  --output-dir outputs/ticket9_smoke/sentencepiece \
  --block-size 64 \
  --stride 64 \
  --batch-size 4 \
  --embedding-dim 64 \
  --num-layers 2 \
  --num-heads 4 \
  --feed-forward-dim 128 \
  --dropout 0.1 \
  --epochs 2 \
  --learning-rate 0.001 \
  --scheduler cosine \
  --warmup-ratio 0.1 \
  --min-learning-rate 0.0001 \
  --max-train-batches 5 \
  --max-validation-batches 2
```

Checkpoints save `tokenizer_type`, so generation can usually restore the right
tokenizer automatically:

```bash
uv run python scripts/generate.py \
  --checkpoint outputs/ticket9_smoke/sentencepiece/best_checkpoint.pt \
  --prompt 吾輩は \
  --max-new-tokens 30 \
  --temperature 0.8 \
  --top-k 20
```

See [experiments/results.md](experiments/results.md) for the smoke comparison.
The short run proves the SentencePiece path works, but it is not a final quality
claim. Token-level losses from character and SentencePiece tokenizers are not
directly equivalent because the tokens cover different amounts of text.

## Embedding Tying

Ticket 10 adds optional input/output embedding tying. This shares the token
embedding table with the output language-modeling head. Explicit small-weight
initialization is important for tied embeddings; without it, the first larger
tied run started with an unstable loss scale.

```bash
uv run python scripts/inspect_model_forward.py --tie-embeddings
```

The current best run used the SentencePiece tokenizer, tied embeddings, stable
initialization, and 100 epochs:

```bash
uv run python scripts/train.py \
  --tokenizer-type sentencepiece \
  --vocab-path data/tokenizers/sentencepiece_unigram_vocab3000_cov998.model \
  --output-dir outputs/sentencepiece_unigram_tied_dropout0.1_lr3e-4_v1 \
  --block-size 384 \
  --stride 384 \
  --batch-size 64 \
  --embedding-dim 128 \
  --num-layers 4 \
  --num-heads 4 \
  --feed-forward-dim 512 \
  --dropout 0.1 \
  --tie-embeddings \
  --epochs 100 \
  --learning-rate 0.0003 \
  --scheduler cosine \
  --warmup-ratio 0.05 \
  --min-learning-rate 0.00002
```

This run reached validation loss `4.5810`, improving over the earlier untied
SentencePiece baseline of `4.9311`. See
[experiments/results.md](experiments/results.md) for the full epoch log,
temperature comparison, generation examples, and failure analysis.

## Hugging Face Pretrained-Model Baseline

Ticket 17 keeps the self-made Transformer above intact and adds a comparison
baseline based on [rinna/japanese-gpt2-small](https://huggingface.co/rinna/japanese-gpt2-small).
It is an approximately 0.1B-parameter Japanese GPT-2 model with an MIT license.
The model card specifies `use_fast=False` and `do_lower_case=True`; both scripts
apply those settings. The pretrained model and its tokenizer are from rinna, not
created by this project.

For this baseline, validation is split by work rather than by random windows:
`道草` (`michikusa.txt`) is validation-only and the remaining nine Natsume works
are used for training. This prevents text windows from the held-out work leaking
into the training set.

Run the local smoke check after data preparation. It downloads the pretrained
model on first use and intentionally processes only a few batches, so it proves
the pipeline rather than generation quality:

```bash
uv sync --locked
uv run python scripts/prepare_aozora.py

uv run python scripts/train_huggingface.py \
  --output-dir outputs/huggingface_rinna_smoke \
  --block-size 64 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --epochs 1 \
  --max-train-batches 1 \
  --max-validation-batches 1

uv run python scripts/generate_huggingface.py \
  --model-dir outputs/huggingface_rinna_smoke/best_model \
  --prompt 吾輩は \
  --max-new-tokens 20 \
  --temperature 0.7 \
  --top-k 20 \
  --top-p 0.9 \
  --seed 42 \
  --output-path outputs/huggingface_rinna_smoke/generation.txt
```

The fine-tuning defaults are block size 256, batch size 2, gradient accumulation
8, 3 epochs, AdamW with learning rate `2e-5`, 10% linear warmup followed by
cosine decay, and seed 42. The scripts prefer MPS, then CUDA, then CPU. Best
artifacts are stored under the selected output directory:

- `best_model/`: model and tokenizer in Hugging Face `save_pretrained` format
- `metrics.json`: split, configuration, and per-epoch losses
- `loss_curve.png`: train and validation loss plot

For a full GPU run in Google Colab, use separate cells after cloning the
repository and selecting a GPU runtime:

```bash
pip install uv
uv sync --locked
```

```bash
uv run python scripts/prepare_aozora.py
```

```bash
uv run python scripts/train_huggingface.py \
  --output-dir outputs/huggingface_rinna_natsume_v1
```

```bash
uv run python scripts/generate_huggingface.py \
  --model-dir outputs/huggingface_rinna_natsume_v1/best_model \
  --prompt 吾輩は \
  --max-new-tokens 120 \
  --temperature 0.7 \
  --top-k 20 \
  --top-p 0.9 \
  --seed 42 \
  --output-path outputs/huggingface_rinna_natsume_v1/generation_wagahai.txt

uv run python scripts/generate_huggingface.py \
  --model-dir outputs/huggingface_rinna_natsume_v1/best_model \
  --prompt 私は \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 20 \
  --top-p 0.9 \
  --seed 42 \
  --output-path outputs/huggingface_rinna_natsume_v1/generation_watashi.txt
```

The Hugging Face generator uses `repetition_penalty=1.0` by default so decoding
does not receive extra repetition control unavailable to the existing baseline.
You can opt in to it with, for example, `--repetition-penalty 1.1`.

Do not compare the Hugging Face and self-made models by their loss values: they
use different tokenizers and initial training conditions. Compare generated text
using the same prompts, seed, temperature, top-k, and top-p settings instead.

## Project Outputs

- A reproducible data preparation pipeline for Aozora Bunko-style texts.
- A character tokenizer and a SentencePiece tokenizer comparison.
- A small GPT-style Transformer decoder.
- Optional input/output embedding tying.
- Training metrics and loss curve plots.
- Generated text examples by author and sampling settings.
- A written failure analysis suitable for a portfolio README or experiment note.
