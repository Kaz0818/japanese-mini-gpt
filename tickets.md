# Tickets

This project is implemented one ticket at a time. Each ticket must end with
verification and a local Git commit before the next ticket starts.

## Status Legend

- `TODO`: Not started.
- `IN_PROGRESS`: Active ticket.
- `DONE`: Verified and committed.

## Ticket 1: Project Rules And Ticket Board

Status: `DONE`

Goal:
Create the project workflow and ticket list without implementing model code.

Tasks:
- Add repo-specific agent instructions in `AGENTS.md`.
- Add this ticket board in `tickets.md`.
- Update `README.md` with the project purpose, learning goals, and roadmap.

Done criteria:
- `AGENTS.md`, `tickets.md`, and `README.md` describe the workflow clearly.
- No tokenizer, model, training, or data download implementation is added yet.
- `git diff --check` passes.
- Changes are committed locally.

Target commit:
`docs: initialize project workflow and tickets`

## Ticket 2: Data Download And Cleaning

Status: `DONE`

Goal:
Prepare reproducible Japanese literary text data without committing full raw
texts.

Tasks:
- Add an author/work manifest with three works each for Natsume Soseki, Akutagawa Ryunosuke, and Dazai Osamu.
- Add a download and cleaning script for Aozora Bunko-style text.
- Store generated files under ignored paths such as `data/raw/` and `data/processed/`.
- Remove common metadata, ruby annotations, and boilerplate where practical.
- Add a smoke command that reports character counts per author.

Done criteria:
- The script can prepare a small local dataset with three works per author from the manifest.
- Generated text data remains ignored by Git.
- The README documents the data command and limitations.
- Changes are verified and committed locally.

Target commit:
`feat: add aozora data preparation pipeline`

## Ticket 3: Character Tokenizer

Status: `DONE`

Goal:
Implement the first tokenizer as a self-made character tokenizer for learning.

Tasks:
- Add special tokens: `<pad>`, `<unk>`, `<bos>`, and `<eos>`.
- Implement encode, decode, save, and load behavior.
- Build a vocabulary from the prepared training text.
- Add a small inspection command or example showing text to ids to text.

Done criteria:
- Encode/decode round trips are validated on a small Japanese sample.
- Vocabulary files can be saved and loaded.
- Tokenizer behavior is documented.
- Changes are verified and committed locally.

Target commit:
`feat: add character tokenizer`

## Ticket 4: Dataset And Batching

Status: `TODO`

Goal:
Turn tokenized text into language-modeling batches for next-token prediction.

Tasks:
- Add a dataset that creates fixed-length input and target windows.
- Add `DataLoader` creation for train/validation splits.
- Keep the data author-aware so style comparison remains possible.
- Add a debug command that prints `input_ids`, `target_ids`, and tensor shapes.

Done criteria:
- A smoke command prints batch shapes successfully.
- The relationship between input and target tokens is documented.
- Changes are verified and committed locally.

Target commit:
`feat: add language modeling dataset`

## Ticket 5: Mini Transformer Decoder

Status: `TODO`

Goal:
Implement a small GPT-style decoder in plain PyTorch.

Tasks:
- Add token embeddings and positional embeddings.
- Add causal self-attention, feed-forward layers, residual connections, and layer normalization.
- Add a language-modeling head that returns logits for each sequence position.
- Include comments for tensor shapes and the causal mask.

Done criteria:
- A forward-pass smoke check returns logits shaped like `[batch, block, vocab]`.
- The model size is small enough for local smoke training.
- Changes are verified and committed locally.

Target commit:
`feat: implement mini transformer decoder`

## Ticket 6: Training Loop And Loss Curve

Status: `TODO`

Goal:
Train the mini Transformer and save basic learning artifacts.

Tasks:
- Add a plain PyTorch training loop with visible config.
- Support device selection with `mps` preference and CPU fallback.
- Save checkpoints, metrics JSON, and `loss_curve.png` under ignored output paths.
- Document a tiny smoke run and a longer local run.

Done criteria:
- A tiny smoke run completes.
- Metrics and a loss curve are produced.
- Output artifacts remain ignored unless explicitly documented otherwise.
- Changes are verified and committed locally.

Target commit:
`feat: add training loop and loss curve`

## Ticket 7: Generation Sampling

Status: `TODO`

Goal:
Generate Japanese text from a trained checkpoint with multiple sampling modes.

Tasks:
- Add generation from a prompt.
- Add temperature sampling.
- Add top-k sampling.
- Add top-p sampling.
- Save or document representative generated examples.

Done criteria:
- Generation works from a checkpoint in a smoke scenario.
- Temperature, top-k, and top-p options are documented.
- Changes are verified and committed locally.

Target commit:
`feat: add sampling based text generation`

## Ticket 8: Style Comparison Experiments

Status: `TODO`

Goal:
Compare generation behavior across authors and document the results honestly.

Tasks:
- Run small comparable experiments per author.
- Add `experiments/results.md`.
- Include loss curves, generation examples, temperature comparisons, and sampling comparisons.
- Include failure examples and analysis of strange Japanese output.

Done criteria:
- Results are tied to actual commands and artifacts.
- Failure analysis mentions dataset size, tokenizer limits, context length, model size, training time, and sampling randomness.
- Changes are verified and committed locally.

Target commit:
`docs: add style comparison results`

## Ticket 9: SentencePiece Comparison

Status: `TODO`

Goal:
Compare the self-made character tokenizer with SentencePiece.

Tasks:
- Add SentencePiece as a dependency.
- Add a SentencePiece training script.
- Add a tokenizer adapter with the same basic interface as the character tokenizer.
- Run a small comparison against the character tokenizer.
- Document differences in loss, readability, and failure modes.

Done criteria:
- SentencePiece tokenizer training works in a smoke scenario.
- The comparison is documented in `experiments/results.md` or a linked results file.
- Changes are verified and committed locally.

Target commit:
`feat: add sentencepiece tokenizer comparison`
