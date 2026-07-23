# Generation Quality Summary

This note is a concise evaluation of the final shared-corpus baseline. It is a
portfolio and interview companion to the detailed logs in
[results.md](results.md), not a claim that the model writes coherent Japanese
prose.

## Final Baseline

The current baseline is a mini Transformer decoder trained for 100 epochs on
the processed Aozora Bunko corpus with these choices:

- SentencePiece Unigram tokenizer:
  `data/tokenizers/sentencepiece_unigram_vocab3000_cov998.model`
- Tied input and output embeddings
- Explicit `std=0.02` initialization for `Embedding` and `Linear` weights
- `block_size=384`, 4 Transformer layers, 128 embedding dimensions, and 4
  attention heads
- Cosine learning-rate decay from `0.0003` to `0.00002`

Its best validation loss was `4.5810` at epoch 100. This is the best result in
this project, but token-level losses cannot be compared directly with the
character-tokenizer runs because their prediction units differ.

To reproduce the reported generation settings:

```bash
uv run python scripts/generate.py \
  --checkpoint outputs/sentencepiece_unigram_tied_dropout0.1_lr3e-4_v1/best_checkpoint.pt \
  --prompt <prompt> \
  --max-new-tokens 120 \
  --temperature <temperature> \
  --top-k 20 \
  --top-p 0.9 \
  --seed 42
```

## What Changed Across the Main Stages

| Stage | What happened | Main lesson |
| --- | --- | --- |
| Character tokenizer | Output often collapsed into repeated characters, punctuation, or rare-kanji jumps. | A character-level vocabulary makes phrase formation difficult at this model and data scale. |
| SentencePiece | Output began to contain phrase-like pieces such as `このひとが` and `そうして`, but repeated them. | Subword pieces improved local readability without solving sentence structure. |
| First tied-embedding run | Loss began near `80`, and output collapsed into a repeated token or phrase. | Reusing the embedding as an output projection needs suitable initialization. |
| Stable tied run | Small-weight initialization removed the loss explosion and improved the 60-epoch validation loss to `4.7479`. | Embedding tying was viable after the initialization fix. |
| 100-epoch stable tied run | Validation loss improved further to `4.5810`; gains were small near the end. | Longer training helped, but the remaining bottleneck is coherence rather than an obviously unfinished loss curve. |
| Temperature comparison | `0.7` worked better for `吾輩は`; `0.8` gave more useful vocabulary for `私は`. | Sampling should be chosen per prompt, not treated as a universal quality fix. |

## Representative Examples

Prompt `吾輩は` used `temperature=0.7`, `top-k=20`, and `top-p=0.9`:

```text
吾輩はこの長が、ただ、やが、ただ、一言とうとうときにあろう。
二尺の中で、三度に上へ出し、これも、三度に、下りとうつまって、
三十五円の中を、その後に向って、下女に、この時の日本の上って...
```

This sample has a prose-like rhythm and a few connected fragments, but its
grammar and meaning do not remain stable.

Prompt `私は` used `temperature=0.8`, `top-k=20`, and `top-p=0.9`:

```text
私はこの 一 先生はその日とうた。そうしてその日日、奥さんの中の前に
その下宿へ行ってくれと、この晩から帰って来て、この家へ行った。
私はいつまでも、先生をお前から私の室に着いているといった...
```

This sample includes coherent literary vocabulary such as `先生`, `奥さん`, and
`下宿`, but combines it in unnatural ways.

## What Still Fails

- Phrase repetition: common pieces such as `私は`, `父`, or `その後` recur too
  often.
- Broken grammar: particles, conjugations, and quotation structure do not form
  reliable sentences.
- Unnatural connections: locally plausible phrases are joined without a clear
  semantic relationship.
- Weak long-range coherence: the topic can change within a few clauses, even
  when the opening prompt is stable.

Temperature can trade repetition for variety, but it cannot repair the learned
probability distribution. Lower temperature can be safer; higher temperature
can introduce unrelated vocabulary and topic drift.

## Interview Takeaways

What was learned:

- A GPT-style decoder requires more than a correct forward pass; tokenizer,
  initialization, data scale, training duration, and decoding all affect what a
  reader sees.
- Loss is useful for comparing runs that use the same tokenizer, but generated
  text is essential for judging readability.

What was tried:

- A self-made character tokenizer, SentencePiece Unigram, embedding tying,
  stable initialization, longer training, and prompt-specific temperatures.

What worked:

- SentencePiece produced more phrase-like output than the character tokenizer.
- Explicit small-weight initialization made tied embeddings train stably.
- Extending the stable tied run to 100 epochs improved validation loss from
  `4.7479` to `4.5810`.

What to improve next:

- Train and evaluate author-specific corpora so style signals are not mixed.
- Split train and validation data by work rather than by adjacent text windows.
- Increase the amount of clean text and compare repetition-control decoding
  separately from model-quality changes.

For the full commands, loss history, and unabridged examples, see
[results.md](results.md).
