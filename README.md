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

Ticket 2 is complete. The project now has a small reproducible Aozora Bunko data
preparation pipeline.

Implementation code for tokenization, modeling, training, and generation will be
added in later tickets.

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
ticket can confirm there is usable text for each author. The initial manifest
uses one work per author:

- 夏目 漱石: `坊っちゃん`
- 芥川 竜之介: `羅生門`
- 太宰 治: `走れメロス`

Limitations:

- The script is a small learning pipeline, not a complete Aozora parser.
- Cleaning rules are intentionally simple and should be inspected before using a
  larger corpus.
- Full raw and processed texts are local generated artifacts and are not
  committed.

## Planned Outputs

- A reproducible data preparation pipeline for Aozora Bunko-style texts.
- A character tokenizer and a later SentencePiece tokenizer comparison.
- A small GPT-style Transformer decoder.
- Training metrics and loss curve plots.
- Generated text examples by author and sampling settings.
- A written failure analysis suitable for a portfolio README or experiment note.
