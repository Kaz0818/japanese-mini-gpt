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

This repository is at the planning and workflow stage. The first task is to set
up `AGENTS.md` and `tickets.md` so the project can be built one ticket at a time.

Implementation code for data preparation, tokenization, modeling, training, and
generation will be added in later tickets.

## Workflow

See `AGENTS.md` for repository rules and `tickets.md` for the ticket roadmap.

Each ticket should be implemented, verified, committed locally, and then stopped
before moving to the next ticket.

## Planned Outputs

- A reproducible data preparation pipeline for Aozora Bunko-style texts.
- A character tokenizer and a later SentencePiece tokenizer comparison.
- A small GPT-style Transformer decoder.
- Training metrics and loss curve plots.
- Generated text examples by author and sampling settings.
- A written failure analysis suitable for a portfolio README or experiment note.
