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

Status: `DONE`

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

Status: `DONE`

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

Status: `DONE`

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

Status: `DONE`

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

Status: `DONE`

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

Status: `DONE`

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

## Ticket 10: Improve Generation Quality

Status: `DONE`

Goal:
Improve generation quality from the SentencePiece baseline and document a clear
before/after comparison against the character tokenizer.

Tasks:
- Compare the best character-tokenizer checkpoint and the best SentencePiece
  checkpoint with the same prompts and sampling settings.
- Add optional input/output embedding tying to `MiniTransformerDecoder`.
- Train a SentencePiece model with embedding tying enabled.
- Generate examples with the same prompts, temperature, top-k, and top-p settings.
- Document improvements, remaining failure modes, and representative examples in
  `experiments/results.md`.

Done criteria:
- Embedding tying has a smoke check or forward-pass verification.
- A SentencePiece training run with embedding tying produces `best_checkpoint.pt`.
- Character and SentencePiece generations are compared using the same prompts.
- The result notes explain both improvements and remaining incoherence honestly.
- Changes are verified and committed locally.

Target commit:
`feat: improve generation quality`

## Ticket 11: Portfolio README Finalization

Status: `DONE`

Goal:
Summarize the project through Ticket 10 in `README.md` so a portfolio reader can
understand the learning value, experiment results, reproduction commands, and
remaining limitations without reading every detailed experiment log.

Tasks:
- Update the README `Current Status` section for the latest 100 epoch tied
  SentencePiece baseline.
- Add the final recommended generation command based on the current best
  checkpoint.
- Add short representative generation examples for at least `吾輩は` and `私は`
  from `experiments/results.md`.
- Summarize the improvement path: character tokenizer, SentencePiece, embedding
  tying, stable initialization, 100 epoch training, and temperature comparison.
- Clearly document remaining limitations: phrase repetition, grammar errors, and
  weak long-range coherence.
- Keep `experiments/results.md` as the detailed experiment log and the README as
  the concise portfolio summary.

Done criteria:
- The README alone explains what was built, how to run it, how far generation
  quality improved, and what still fails.
- The representative command targets the current best checkpoint.
- The README includes concise `吾輩は` and `私は` examples without duplicating the
  full experiment log.
- The README links to `experiments/results.md` for details.
- `git diff --check` passes.
- Changes are verified and committed locally.

Target commit:
`docs: finalize portfolio readme`

## Ticket 12: Generation Quality Summary

Status: `DONE`

Goal:
Create a concise generation-quality evaluation note that explains what improved,
what still fails, and how to discuss the final model in a portfolio review or
interview.

Tasks:
- Summarize the final baseline: 100 epoch tied SentencePiece run with validation
  loss `4.5810`.
- Compare the main qualitative stages: character tokenizer failures,
  SentencePiece phrase fragments, tied-embedding initialization failure, stable
  tied run, and temperature comparison.
- Identify the best representative examples and the sampling settings used for
  each prompt.
- Explain remaining failure modes in plain language: phrase repetition, broken
  grammar, unnatural word connections, and weak long-range coherence.
- Add interview-ready takeaways: what was learned, what was tried, what worked,
  and what would be improved next.

Done criteria:
- The summary is concise enough to review quickly but specific enough to support
  an interview explanation.
- The note references real results and commands already recorded in
  `experiments/results.md`.
- No new model training or generation implementation is added.
- `git diff --check` passes.
- Changes are verified and committed locally.

Target commit:
`docs: add generation quality summary`

## Ticket 13: Experiment Config Cleanup

Status: `TODO`

Goal:
Reduce command-copy mistakes by introducing a small, readable way to store and
reuse common training and generation settings.

Tasks:
- Add a lightweight config file for the current best SentencePiece training and
  generation settings.
- Document which settings belong to training, tokenizer paths, checkpoint paths,
  and generation sampling.
- Keep the format beginner-readable and avoid adding a heavy configuration
  framework.
- Add a command or documented workflow for using the config to reproduce the
  current best run or generation examples.
- Preserve existing CLI behavior so older commands still work.

Done criteria:
- The best-run settings are visible in one tracked config or documented config
  example.
- README or a linked document explains how to use the config.
- Existing direct CLI commands remain valid.
- `git diff --check` passes.
- Changes are verified and committed locally.

Target commit:
`feat: add experiment config workflow`

## Ticket 14: Repetition Reduction Sampling

Status: `TODO`

Goal:
Improve generation readability by adding a small sampling option that reduces
obvious repeated tokens or phrases without changing the trained model.

Tasks:
- Add an opt-in repetition-control option to `scripts/generate.py`.
- Keep the default generation behavior unchanged.
- Compare generation with and without the new option using the current best
  checkpoint and the same prompts.
- Document when the option helps and when it makes text worse.
- Record representative examples and limitations in `experiments/results.md`.

Done criteria:
- Existing generation commands still work unchanged.
- The new option has a smoke check or direct generation comparison.
- Results show whether repetition is reduced for prompts such as `吾輩は` and
  `私は`.
- The documentation explains that this is decoding control, not model retraining.
- `git diff --check` passes.
- Changes are verified and committed locally.

Target commit:
`feat: add repetition reduction sampling`

## Ticket 15: Natsume Soseki Five-Work Dataset

Status: `DONE`

Goal:
Replace the default mixed-author source manifest with a reproducible,
author-specific five-work Natsume Soseki corpus for the next training run.

Tasks:
- Keep `坊っちゃん`, `こころ`, and `吾輩は猫である`.
- Add `草枕` and `三四郎` from their Aozora Bunko cards.
- Remove other authors from the current default manifest so the preparation
  command downloads exactly the five intended works.
- Run the preparation command and confirm that five cleaned Natsume records are
  created with nonzero character counts.
- Update the README to distinguish the current author-specific corpus from the
  historical mixed-author experiments.

Done criteria:
- The tracked manifest contains exactly five Natsume Soseki works.
- `uv run python scripts/prepare_aozora.py` completes and reports five records.
- Raw and processed text remain ignored by Git.
- `git diff --check` passes.
- Changes are verified and committed locally.

Target commit:
`data: use five-work natsume soseki corpus`
