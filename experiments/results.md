# Experiment Results

This file records small, reproducible experiments for the mini Transformer
portfolio project. The numbers below are smoke-scale results, not quality claims.

## Ticket 8: Author Style Comparison

Date: 2026-06-30

Goal:
Compare generation behavior across Natsume Soseki, Akutagawa Ryunosuke, and
Dazai Osamu with the same small model settings.

### Setup

Data:

- Source: local Aozora Bunko processed text under `data/processed/`.
- Authors: three works each from the tracked manifest.
- Tokenizer: self-made character tokenizer at `data/tokenizers/char_vocab.json`.
- Device used: `mps`.
- Outputs: `outputs/ticket8_style/`, which is intentionally ignored by Git.

Training command pattern:

```bash
uv run python scripts/train.py \
  --data-dir data/processed/<author_id> \
  --vocab-path data/tokenizers/char_vocab.json \
  --output-dir outputs/ticket8_style/<author_id> \
  --block-size 48 \
  --stride 128 \
  --batch-size 8 \
  --embedding-dim 64 \
  --num-layers 2 \
  --num-heads 4 \
  --feed-forward-dim 128 \
  --dropout 0.1 \
  --epochs 3 \
  --learning-rate 0.001 \
  --scheduler cosine \
  --warmup-ratio 0.1 \
  --min-learning-rate 0.0001 \
  --max-train-batches 20 \
  --max-validation-batches 5
```

This intentionally limits training to at most 20 train batches and 5 validation
batches per epoch so the comparison can be rerun quickly on a laptop.

### Loss Summary

| Author | Dataset examples | Final train loss | Final validation loss | Best epoch | Loss curve |
| --- | ---: | ---: | ---: | ---: | --- |
| Natsume Soseki | 4,473 | 6.2934 | 6.1802 | 3 | `outputs/ticket8_style/natsume_soseki/loss_curve.png` |
| Akutagawa Ryunosuke | 114 | 7.0266 | 6.9697 | 3 | `outputs/ticket8_style/akutagawa_ryunosuke/loss_curve.png` |
| Dazai Osamu | 1,368 | 6.1333 | 5.9878 | 3 | `outputs/ticket8_style/dazai_osamu/loss_curve.png` |

All three runs improved during the three smoke epochs, but the losses remain
high. The comparison is therefore useful for checking the workflow and failure
modes, not for claiming that the model has learned an author's style.

### Generation Commands

Each author used the best checkpoint from the matching author-only run:

```bash
uv run python scripts/generate.py \
  --checkpoint outputs/ticket8_style/<author_id>/best_checkpoint.pt \
  --prompt <prompt> \
  --max-new-tokens 80 \
  --temperature <temperature> \
  --output-path outputs/ticket8_style/<author_id>/<name>.txt
```

Sampling comparison commands added either `--top-k 20` or `--top-p 0.9` with
`--temperature 0.8`.

Prompts:

- Natsume Soseki: `吾輩は`
- Akutagawa Ryunosuke: `ある日の事でございます`
- Dazai Osamu: `私は`

### Generation Examples

Greedy decoding (`--temperature 0`) collapses into repeated high-probability
characters.

```text
Natsume:
吾輩はだたたたたたたたたいたいたたいたいたたたいたたいたたたたたいたたいたいたたたたたたたたいたたたたたたたたたたたたたたたたたたたたたたたたたたたたたたたたたたた

Akutagawa:
ある日の事でございますた光いたたの光、のた、た光、たのたののたのいたたのたいたたした、のたたいたそいたたそいたたたそいたたたいたたたたいたいたたたそいたたそいたそいたそいたそいたそい

Dazai:
私は、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、
```

Temperature sampling without top-k or top-p keeps more variety, but the output
contains many rare kanji, punctuation artifacts, and broken sequences.

```text
Natsume temperature 0.8:
吾輩は夕初いもえ巣る宵腹はたに人女躇ちウる音傷だ江どて路限a婦随っ役あ再決―通ん囲評て親だみ、郷よ所道う時任物状たなま市丈能だ全け」忘だと烈低便帳畏髣う宵性て魔私役覘

Akutagawa temperature 0.8:
ある日の事でございます夕鹸いもえ巣徳宵宰俥O夏茫女躇眺楠る蟋傷蘭江頬谷路限a婦路県役ヴ再鈍―通慎畏評佇僅倒溢、恰よ所道劉改黴物売ぺな饑刀樗詈僧〕佳浣忘怖忽烈低便溢畏髣う宵逐廠魔騰役覘

Dazai temperature 0.8:
私は夕怖いもえトるた腹俥たに人女躇ち、、音傷だ江どて路限a婦シっ緒ヴ再鈍俸通嘴畏ごて僅にみ、郷よ所道うえ？例、たなま市務造っ全け」わ据や烈低れ分畏髣う宵籐て魔廻役覘
```

Top-k sampling reduces some of the rare-character jumps, but it still does not
produce coherent Japanese.

```text
Natsume top-k 20:
吾輩は」ういもたたもさははたに人。人てい、ててだてどてたたのだいっ十た。に十人ん。けてどだてそし」のもうえのよてたないたうくだなてのてだといもにのういうたのてないだも

Akutagawa top-k 20:
ある日の事でございますた十いたえ男したくつた時の。n、、、ててだてどて、たのだ―っけ、ご売だ人鋺事ふてどだて、、よ所そうえじい、たに饑光う門だ陞てどどだ動い陞光うくい事鋺のど陞窓そは

Dazai top-k 20:
私はてういもたたもさくしたに人分どち、、ててだてどて、、のだそっけ、。にだた、たちてどにて、し」とけう、とよ、たに、たうくっ、てどてだといも、う、いうたててさ、そも
```

Top-p at `0.9` behaved similarly to unconstrained temperature sampling in this
smoke run. That is a sign that the probability mass is still poorly shaped by
the undertrained model.

### Interpretation

The author-specific losses are not directly comparable as style scores because
the per-author dataset sizes differ greatly. Akutagawa has only 114 examples
with this block and stride setting, while Natsume has 4,473 and Dazai has 1,368.
The Akutagawa model therefore sees far fewer batches and has a weaker local
training signal.

The clearest common pattern is not author style yet. It is failure behavior:

- Greedy decoding repeats frequent characters or punctuation.
- Higher temperature produces diverse but incoherent character sequences.
- Top-k makes output less wild but still repetitive.
- Top-p does not fix rare-character artifacts at this training scale.

### Failure Analysis

The strange Japanese output is expected for this smoke experiment.

- Dataset size: each author run uses only three works, and the smoke command
  intentionally processes a small number of batches.
- Tokenizer limits: the character tokenizer can emit any seen character, but it
  does not know word boundaries, readings, phrases, or subword structure.
- Context length: `block_size=48` gives the model only a short context window.
- Model size: the model has 2 layers, 64 embedding dimensions, and a small
  feed-forward dimension, so its capacity is intentionally limited.
- Training time: three epochs with batch limits checks the pipeline, but it is
  not enough to learn stable Japanese prose.
- Sampling randomness: temperature, top-k, top-p, and the seed can change the
  visible output a lot when logits are still poorly calibrated.

Next useful experiment:
Run a longer shared configuration for all authors, then compare examples from
the same prompt and multiple seeds. Ticket 9 should also test whether
SentencePiece reduces the rare-character and word-boundary failure modes.

## Character Tokenizer Baseline Before SentencePiece

Date: 2026-07-02

Goal:
Establish a stronger character-tokenizer baseline before changing the tokenizer
in Ticket 9. Earlier runs suggested that `block_size=64` and `block_size=128`
plateaued near validation loss `3.1`, while `block_size=256` improved into the
`2.9` range. A later `block_size=256` run with `dropout=0.2` reached roughly
`2.74`, so this run checks whether the same setting still benefits from longer
training.

### Setup

Reported configuration:

- Tokenizer: self-made character tokenizer.
- `block_size=256`
- `dropout=0.2`
- `epochs=80`
- Learning-rate schedule: warmup followed by cosine decay to `0.00005`.

The full epoch log was pasted manually on 2026-07-02. The table below keeps the
main checkpoints instead of copying all 80 lines.

### Loss Summary

| Epoch | Train loss | Validation loss | Learning rate |
| ---: | ---: | ---: | ---: |
| 1 | 6.7059 | 5.2570 | 0.000251497 |
| 10 | 3.4817 | 3.3619 | 0.000985436 |
| 20 | 3.1014 | 2.9593 | 0.000899770 |
| 30 | 2.9245 | 2.7838 | 0.000750972 |
| 40 | 2.8176 | 2.6850 | 0.000564108 |
| 50 | 2.7504 | 2.6175 | 0.000370657 |
| 60 | 2.7097 | 2.5794 | 0.000203205 |
| 70 | 2.6858 | 2.5610 | 0.000089961 |
| 80 | 2.6752 | 2.5512 | 0.000050000 |

Best validation loss:

- Best epoch: 80
- Best validation loss: `2.5512`
- Final train loss: `2.6752`

### Interpretation

This run is a meaningful improvement over the earlier character-tokenizer
experiments. The validation loss continued to improve through epoch 80, so the
previous `dropout=0.2` run had probably not finished converging yet.

The improvement became much smaller near the end:

- Epoch 50 to 80 improved validation loss from `2.6175` to `2.5512`.
- Epoch 70 to 80 improved validation loss from `2.5610` to `2.5512`.

That pattern suggests that more epochs may still help a little, but the easy
gain from longer training is mostly used up. The current evidence supports this
working explanation: for the character tokenizer, the recent bottleneck was
partly context length and training time, not only tokenizer quality.

This does not prove that the character tokenizer is good enough. It only gives a
stronger baseline for the next comparison. Ticket 9 should compare
SentencePiece against this setting and check both validation loss and generation
readability.

### Dropout Comparison

After the `dropout=0.2` baseline, two lower-dropout runs were reported with the
same 80 epoch schedule. Both improved validation loss clearly.

| Dropout | Best epoch | Final train loss | Final validation loss |
| ---: | ---: | ---: | ---: |
| 0.2 | 80 | 2.6859 | 2.5160 |
| 0.1 | 80 | 2.3502 | 2.2053 |
| 0.05 | 80 | 2.1001 | 2.0069 |

The `dropout=0.05` run is the strongest character-tokenizer result so far. The
validation loss was still improving at the end:

- Epoch 50: `2.1109`
- Epoch 60: `2.0529`
- Epoch 70: `2.0213`
- Epoch 80: `2.0069`

This makes the earlier diagnosis more specific. The model was not only limited
by context length. With this data size and model size, `dropout=0.2` and
`dropout=0.1` were still regularizing the model more than necessary. Lowering
dropout allowed the model to fit useful character and phrase patterns without
showing validation loss degradation within 80 epochs.

### Next Comparison

Use `dropout=0.05` as the current character-tokenizer baseline, then check
generation examples from the best checkpoint before changing the tokenizer.
Ticket 9 can then compare SentencePiece against this stronger baseline.

Avoid tuning too many more character-tokenizer settings before Ticket 9. A
`dropout=0.0` run could be useful as an overfitting check, but it should be
treated as optional. The more important next question is whether SentencePiece
improves readability and failure modes beyond this lower-dropout character
baseline.

## Ticket 9: SentencePiece Comparison

Date: 2026-07-03

Goal:
Add a SentencePiece tokenizer path and compare it against the existing character
tokenizer in a small smoke run.

### Setup

SentencePiece tokenizer command:

```bash
uv run python scripts/train_sentencepiece_tokenizer.py \
  --model-path data/tokenizers/ticket9_sentencepiece_unigram.model \
  --vocab-size 4000 \
  --model-type unigram \
  --sample 吾輩は猫である。
```

The script trained a SentencePiece Unigram model from the 9 processed local text
records. The sample round trip succeeded:

```text
sample=吾輩は猫である。
token_ids=[2, 21, 350, 213, 45, 6, 3]
decoded=吾輩は猫である。
```

The first attempted smoke used `vocab_size=1000`, but that was too small for
this corpus at `character_coverage=0.9995`: SentencePiece needed about 3,000
required character pieces before it could add learned subword pieces. The smoke
therefore uses `vocab_size=4000`.

### Training Commands

Character tokenizer smoke:

```bash
uv run python scripts/train.py \
  --tokenizer-type char \
  --vocab-path data/tokenizers/char_vocab.json \
  --output-dir outputs/ticket9_smoke/char \
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

SentencePiece smoke used the same model and training settings, changing only:

```bash
--tokenizer-type sentencepiece
--vocab-path data/tokenizers/ticket9_sentencepiece_unigram.model
--output-dir outputs/ticket9_smoke/sentencepiece
```

### Smoke Metrics

These numbers only verify the comparison path. They are not a quality claim
because the run used only 2 epochs and 5 train batches per epoch.

| Tokenizer | Vocab size | Dataset examples | Best epoch | Final train loss | Final validation loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Character | 3,458 | 11,902 | 2 | 8.2018 | 8.1395 |
| SentencePiece Unigram | 4,000 | 7,836 | 2 | 8.3962 | 8.4458 |

The validation losses are also not perfectly comparable as absolute language
quality scores. Character tokens and SentencePiece tokens represent different
units, and one `block_size=64` window covers more original text when each token
can represent a subword. The useful signal from this smoke run is that both
tokenizer paths train and generate through the same model interface.

### Generation Examples

Both examples used:

```bash
uv run python scripts/generate.py \
  --checkpoint outputs/ticket9_smoke/<tokenizer>/best_checkpoint.pt \
  --prompt 吾輩は \
  --max-new-tokens 30 \
  --temperature 0.8 \
  --top-k 20
```

Character tokenizer:

```text
吾輩は肥多埒潮座楼陥増詣犇詳麼忌弁獅江長尾饉駝種版南癪細籐朱聡割碌
```

SentencePiece Unigram:

```text
吾輩は呈翰判T瞭乃窪芋んだろう綾脂のである吐年猛通り上葵暖虻れた後ありましたをもってとにかく範聡笑いながら箸毎日
```

### Interpretation

The short SentencePiece smoke output still contains broken text and rare
characters, but it also emits longer phrase-like pieces such as `んだろう`,
`のである`, `通り上`, `れた後`, and `ありました`. That is the expected difference
from the character tokenizer: SentencePiece can represent multi-character
patterns, while the character tokenizer must learn every phrase composition from
single-character steps.

This ticket should not replace the stronger character-tokenizer baseline above.
It adds the code path needed for a fair longer run. The next meaningful
experiment is to train SentencePiece with the stronger baseline settings used by
the best character run, then compare validation trend and generation readability
with the same prompts and sampling settings.

## SentencePiece Unigram Main Run

Date: 2026-07-03

Goal:
Train a stronger SentencePiece model after the smoke test and check whether it
improves generation behavior compared with the character tokenizer failures.

### Tokenizer

Tokenizer command used in Colab:

```bash
uv run python scripts/train_sentencepiece_tokenizer.py \
  --model-path data/tokenizers/sentencepiece_unigram_vocab3000_cov998.model \
  --vocab-size 3000 \
  --model-type unigram \
  --character-coverage 0.998
```

This tokenizer is smaller than the first 4000-vocabulary smoke tokenizer. The
change mattered: a previous run accidentally reused the old tokenizer path, so
the validation curve did not reflect the new vocabulary. The corrected training
run below used:

```text
vocab_path=data/tokenizers/sentencepiece_unigram_vocab3000_cov998.model
vocab_size=3000
```

### Training Command

```bash
uv run python scripts/train.py \
  --tokenizer-type sentencepiece \
  --vocab-path data/tokenizers/sentencepiece_unigram_vocab3000_cov998.model \
  --output-dir outputs/sentencepiece_unigram_dropout0.1_v2 \
  --block-size 384 \
  --stride 384 \
  --batch-size 64 \
  --embedding-dim 128 \
  --num-layers 4 \
  --num-heads 4 \
  --feed-forward-dim 512 \
  --dropout 0.1 \
  --epochs 50 \
  --learning-rate 0.0005 \
  --scheduler cosine \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.001 \
  --warmup-ratio 0.05 \
  --min-learning-rate 0.00005
```

### Loss Summary

| Epoch | Train loss | Validation loss | Learning rate |
| ---: | ---: | ---: | ---: |
| 1 | 8.0613 | 7.7877 | 0.000210000 |
| 10 | 6.0134 | 5.9516 | 0.000472526 |
| 20 | 5.3304 | 5.3040 | 0.000364700 |
| 30 | 5.0099 | 5.0500 | 0.000219045 |
| 40 | 4.8807 | 4.9606 | 0.000096987 |
| 50 | 4.8356 | 4.9311 | 0.000050000 |

Best validation loss:

- Best epoch: 50
- Best validation loss: `4.9311`
- Final train loss: `4.8356`

The validation loss was still improving at epoch 50, but the improvement had
become gradual. The last 10 epochs improved from `4.9606` to `4.9311`, so this
run is useful as a SentencePiece baseline even if longer training might still
help slightly.

Token-level loss is not directly comparable with the character-tokenizer loss,
because the two tokenizers define different prediction units. A SentencePiece
token can represent several characters, so the more important comparison here is
the generated text and failure pattern.

### Generation

Generation command:

```bash
uv run python scripts/generate.py \
  --checkpoint outputs/sentencepiece_unigram_dropout0.1_v2/best_checkpoint.pt \
  --prompt 吾輩は \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 20 \
  --top-p 0.9
```

Generated example:

```text
吾輩はこのひとが、いまは、と、私のき、自分は、そうして、ひとのひとのに、自分の、いま、お酒を、いまから、自分には、そのひとりに、とは、自分は、ひとり、つまって、それは、おか、いまに、そのひとが、お酒が、自分に、ひとりで、お母さまの、いまの、お母さまが、自分は、 「さまが、 「、おやらり、自分は、おっし、お母さまは、自分に、 「、そうして、おおく、自分に、そののお
```

### Interpretation

This output is still not coherent Japanese prose, but it is a clear improvement
over the earlier character-tokenizer failure mode. The character tokenizer often
collapsed into repeated characters, punctuation loops, or rare kanji sequences.
The SentencePiece output instead contains phrase-like units such as `このひとが`,
`自分は`, `そうして`, `お酒を`, `ひとりで`, and `お母さま`.

The remaining failure mode is now different:

- The model repeats common phrase fragments such as `自分`, `ひと`, and `いま`.
- Sentence-level structure is weak, so phrases are stitched together without a
  stable meaning.
- Some broken fragments remain, such as `私のき`, `おやらり`, and `そののお`.
- Quote punctuation appears, but the dialogue structure is not controlled.

This suggests that SentencePiece helped the tokenizer-level bottleneck: the
model can now emit multi-character Japanese fragments more easily. The next
bottleneck is likely model/data scale and sequence-level coherence, not only
tokenization.

### Next Step

This is good enough as the Ticket 9 SentencePiece baseline. More tuning could
improve it, but the most useful next work is not another small tokenizer tweak.
Prioritize one of these larger, easier-to-explain improvements:

1. Add tied input/output embeddings to reduce parameters and improve token
   sharing.
2. Add a cleaner config system so experiment settings and tokenizer paths cannot
   be mixed up.
3. Compare generation from the best character checkpoint and this SentencePiece
   checkpoint with the same prompts and sampling settings.
4. If doing one more SentencePiece experiment, try `vocab_size=2500` with
   `character_coverage=0.995`, but treat it as optional rather than blocking
   progress.

## Ticket 10: Embedding Tying And Generation Comparison

Date: 2026-07-03

Goal:
Add optional input/output embedding tying and compare character-tokenizer
generation with a SentencePiece checkpoint trained with embedding tying enabled.

### Code Change

`MiniTransformerDecoder` now accepts `tie_embeddings=True` in
`MiniTransformerConfig`. When enabled, the output language-modeling head reuses
the same weight tensor as the token embedding table. This reduces one large
parameter matrix and gives the input and output token representations a shared
space.

Smoke verification:

```bash
uv run python scripts/inspect_model_forward.py \
  --tie-embeddings \
  --vocab-size 64 \
  --block-size 16 \
  --batch-size 2 \
  --embedding-dim 32 \
  --num-layers 1 \
  --num-heads 4 \
  --feed-forward-dim 64
```

Result:

```text
logits.shape=(2, 16, 64)
expected_logits_shape=(2, 16, 64)
tie_embeddings=True
embedding_weights_shared=True
```

### SentencePiece Tied Smoke Run

Tokenizer command:

```bash
uv run python scripts/train_sentencepiece_tokenizer.py \
  --model-path data/tokenizers/ticket10_sentencepiece_unigram.model \
  --vocab-size 4000 \
  --sample 吾輩は猫である。
```

An attempted `vocab_size=1000` run failed because the required character set was
larger than the requested vocabulary. The successful run used `vocab_size=4000`
with the default `character_coverage=0.9995`.

Training command:

```bash
uv run python scripts/train.py \
  --tokenizer-type sentencepiece \
  --vocab-path data/tokenizers/ticket10_sentencepiece_unigram.model \
  --output-dir outputs/ticket10_quality/sentencepiece_tied \
  --block-size 64 \
  --stride 128 \
  --batch-size 8 \
  --embedding-dim 64 \
  --num-layers 2 \
  --num-heads 4 \
  --feed-forward-dim 128 \
  --dropout 0.05 \
  --tie-embeddings \
  --epochs 3 \
  --learning-rate 0.001 \
  --scheduler cosine \
  --warmup-ratio 0.1 \
  --min-learning-rate 0.0001 \
  --max-train-batches 20 \
  --max-validation-batches 5
```

Result:

| Run | Best epoch | Final train loss | Final validation loss | Best validation loss |
| --- | ---: | ---: | ---: | ---: |
| Character Ticket 9 smoke | 2 | 8.2018 | 8.1395 | 8.1395 |
| SentencePiece Ticket 9 smoke | 2 | 8.3962 | 8.4458 | 8.4458 |
| SentencePiece tied Ticket 10 smoke | 3 | 19.4714 | 18.7900 | 18.7900 |

The tied run produced `outputs/ticket10_quality/sentencepiece_tied/best_checkpoint.pt`.
However, the loss is much worse than the earlier smoke baselines. This means the
short tied run is useful as a wiring check, not as evidence of better model
quality.

### Generation Comparison

All examples used the same sampling settings:

```bash
uv run python scripts/generate.py \
  --checkpoint <checkpoint> \
  --prompt <prompt> \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 20 \
  --top-p 0.9 \
  --seed 42
```

Prompt `吾輩は`, character checkpoint:

```text
吾輩は肥多埒潮座楼陥増詣犇詳麼忌弁獅江長尾饉駝種版南癪細籐朱聡血醺鼻ヒ人導打私獰鏡3八蟀憬晰極羅栗罰碌台笹摧妹北巌e寒竣柴桐戒弛貌最鶯任類鮨璧餓雅袂葭貰ビ梁蟻作薨楽b
```

Prompt `吾輩は`, SentencePiece tied checkpoint:

```text
吾輩は体体体体体体迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は迷亭は
```

Prompt `私は`, character checkpoint:

```text
私は・鈴妻ぶ<unk>貪妬遅歌召匿興抽完マ朔揺畏坐速欺喫蟾絃憧副遮薩検聞十溌候邸貰強暖芒決忽鎬考聡羅箒蹟腥蜂插g府歌懲狆噂俺虫旅桐デ筍廷叔弛任類鮨璧餓雅否薪貰緬硝放俥壁習
```

Prompt `私は`, SentencePiece tied checkpoint:

```text
私は拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙拙
```

### Interpretation

Embedding tying itself is now implemented and verified, but this short
SentencePiece tied experiment did not improve generation quality. It changed
the failure mode:

- The character checkpoint still jumps through rare characters, Latin letters,
  and `<unk>`-like artifacts.
- The SentencePiece tied checkpoint emits recognizable pieces such as `迷亭は`,
  but collapses into repeated phrase pieces.
- The high validation loss suggests that this tied configuration needs a longer
  and better-tuned run before it can be judged fairly.

The honest conclusion is that Ticket 10 improved the code path and comparison
discipline, not the generated prose in this smoke run. For a real quality
attempt, reuse the stronger Ticket 9 main-run scale, lower the learning rate for
the tied model, and compare against the best character and best untied
SentencePiece checkpoints with multiple seeds.

### Stable Initialization Follow-up

Date: 2026-07-04

The first larger tied-embedding run exposed an initialization problem. Before the
fix, the tied model started with extremely large losses:

```text
epoch=1 train_loss=80.0055 validation_loss=78.1898
epoch=60 train_loss=7.2853 validation_loss=6.5515
```

The likely cause was that the default `nn.Embedding` initialization was too large
when reused as the output language-modeling head. The model now initializes
`Embedding` and `Linear` weights with a small normal distribution
(`std=0.02`) before tying the input and output weights.

After that fix, the same tied SentencePiece setup started at a normal loss scale
and improved beyond the previous untied SentencePiece baseline:

```text
epoch=1 train_loss=7.8879 validation_loss=7.7144
epoch=10 train_loss=5.8091 validation_loss=5.7352
epoch=20 train_loss=5.1862 validation_loss=5.1536
epoch=30 train_loss=4.9124 validation_loss=4.9140
epoch=40 train_loss=4.7836 validation_loss=4.8128
epoch=50 train_loss=4.7231 validation_loss=4.7662
epoch=60 train_loss=4.6959 validation_loss=4.7479
```

Updated comparison:

| Run | Best validation loss | Notes |
| --- | ---: | --- |
| SentencePiece untied main run | 4.9311 | Earlier Ticket 9 baseline |
| SentencePiece tied before init fix | 6.5515 | Started with loss explosion |
| SentencePiece tied after init fix | 4.7479 | Best SentencePiece result so far |

This is the first tied-embedding run that should be treated as a real candidate.
The validation loss kept improving through epoch 60, and the train/validation
gap stayed small, so there is no strong overfitting signal in this run.

Generation used:

```bash
uv run python scripts/generate.py \
  --checkpoint outputs/sentencepiece_unigram_tied_dropout0.1_lr3e-4_v1/best_checkpoint.pt \
  --prompt <prompt> \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 20 \
  --top-p 0.9 \
  --seed 42
```

Prompt `吾輩は`:

```text
吾輩はこの時のは ⁇ は、人間に対する時の方に、この人と、不思う。人間の ⁇ とかかと、大事の中ので、その時分は、人間があろう。人間の中であって、人間の中に、その男が、この所が、人間にかけたるるの人間の所が、その後にあろうとうら、いろしい事をしが、大にあるいならぬと、自慢を ⁇ りに極めずかりのの方
```

Prompt `私は`:

```text
私はこの 私は彼の人は、私はその私の心も私のKに、奥さんの生に思議なのです。私は私がお嬢さんに、Kに私はお嬢さんを知って、父の家に対して、父は奥さんに違いっきました。KにKの私の父の私の父が父の私は私の父は私はKの父は私はその後に、彼の父と、私は私はその奥さんと答えたた時も私の心を取り切っていました。私は私の父を書いたのです。私の母に私の父の母の外
```

Interpretation:

- Stable initialization fixed the tied-embedding loss explosion.
- The tied model now beats the earlier untied SentencePiece validation loss.
- Generation moved from repeated token collapse to phrase-like Japanese
  fragments.
- The `私は` sample includes corpus-specific terms such as `K`, `奥さん`,
  `お嬢さん`, `父`, `母`, and `心`, which is a stronger qualitative signal than
  the earlier smoke output.
- Remaining failures are still clear: unknown-token output appears as `⁇`,
  phrase repetition remains common, grammar is unstable, and long-range meaning
  is not coherent.

Current conclusion:
Embedding tying is worth keeping, but only with the explicit small-weight
initialization. The next bottleneck is no longer the tied-output implementation;
it is sequence-level coherence and decoding behavior.

### Longer 100 Epoch Tied Run

Date: 2026-07-04

The next run kept the same stable tied SentencePiece setup and extended training
to 100 epochs. This tests whether the epoch 60 model was still undertrained.

Training summary:

```text
epoch=1 train_loss=7.9275 validation_loss=7.7812 lr=6.3e-05
epoch=10 train_loss=5.8977 validation_loss=5.8185 lr=0.000298052
epoch=20 train_loss=5.2023 validation_loss=5.1661 lr=0.000283016
epoch=30 train_loss=4.8780 validation_loss=4.8808 lr=0.000254649
epoch=40 train_loss=4.7038 validation_loss=4.7500 lr=0.000216025
epoch=50 train_loss=4.5943 validation_loss=4.6699 lr=0.00017133
epoch=60 train_loss=4.5198 validation_loss=4.6301 lr=0.000125408
epoch=70 train_loss=4.4683 validation_loss=4.6045 lr=8.32336e-05
epoch=80 train_loss=4.4354 validation_loss=4.5932 lr=4.93783e-05
epoch=90 train_loss=4.4197 validation_loss=4.5848 lr=2.75106e-05
epoch=100 train_loss=4.4078 validation_loss=4.5810 lr=2e-05
```

Updated comparison:

| Run | Epochs | Best validation loss | Notes |
| --- | ---: | ---: | --- |
| SentencePiece untied main run | 50 | 4.9311 | Earlier Ticket 9 baseline |
| SentencePiece tied after init fix | 60 | 4.7479 | First stable tied candidate |
| SentencePiece tied after init fix | 100 | 4.5810 | Best SentencePiece result so far |

Interpretation:

- Extending training from 60 to 100 epochs improved validation loss from
  `4.7479` to `4.5810`.
- The improvement is real but diminishing: epoch 60 to 70 gained `0.0256`,
  while epoch 90 to 100 gained only `0.0038`.
- The train/validation gap at epoch 100 is small (`4.4078` vs `4.5810`), so
  this run still does not show severe overfitting.
- The learning rate has reached the floor, so another identical extension is
  unlikely to give a large gain.

Current conclusion:
Use this 100 epoch tied SentencePiece run as the new baseline. The next useful
work should evaluate generation quality from this checkpoint before changing
architecture or tokenizer settings again.

### Temperature Comparison From 100 Epoch Checkpoint

Date: 2026-07-04

The 100 epoch tied SentencePiece checkpoint was sampled with the same prompt,
`top-k`, `top-p`, and seed while changing only `temperature`.

Shared generation settings:

```bash
uv run python scripts/generate.py \
  --checkpoint outputs/sentencepiece_unigram_tied_dropout0.1_lr3e-4_v1/best_checkpoint.pt \
  --prompt <prompt> \
  --max-new-tokens 120 \
  --temperature <0.7|0.8|0.9> \
  --top-k 20 \
  --top-p 0.9 \
  --seed 42
```

For prompt `吾輩は`, `temperature=0.7` was the best of the three. It was still
not coherent prose, but it avoided the stronger `内供` repetition seen at
`temperature=0.8` and the topic drift toward `私` / `父` / `母` seen at
`temperature=0.9`.

Best `吾輩は` sample:

```text
吾輩はこの長が、ただ、やが、ただ、一言とうとうときにあろう。 二尺の中で、三度に上へ出し、これも、三度に、下りとうつまって、三十五円の中を、その後に向って、下女に、この時の日本の上って、一体が、三人たちまっている。 次第一度と、二十円の中から、その後を一杯をかに上へは、その上の中
```

For prompt `私は`, `temperature=0.8` was better than `temperature=0.7`. It kept
more scene-setting vocabulary such as `先生`, `奥さん`, `下宿`, `私の室`, `父`,
`母`, and `手紙`. The `temperature=0.7` sample was safer but more repetitive
around `その日`, `また`, and `自分の家`.

Best `私は` sample:

```text
私はこの 一 先生はその日とうた。そうしてその日日、奥さんの中の前にその下宿へ行ってくれと、この晩から帰って来て、この家へ行った。 私はいつまでも、先生をお前から私の室に着いているといった。私は私の父の間に、母のお父が父は、父の手紙を呼ぶっていらいなかった。 父を取り合いわざわざわせていた。 「私の父さまと、私の眼も、この母にお父の間に、母のお
```

Temperature interpretation:

- `0.7` is safer and worked better for `吾輩は`.
- `0.8` gives more expressive phrase variety and worked better for `私は`.
- `0.9` adds variety, but in this run it drifted away from the `吾輩は` prompt
  into unrelated `私` / `父` / `母` fragments.

Current default:
Use `temperature=0.8`, `top_k=20`, and `top_p=0.9` as the general generation
default, but use `temperature=0.7` when a prompt shows phrase repetition or topic
drift. For portfolio examples, it is acceptable to report prompt-specific
sampling settings instead of forcing one setting for every prompt.

## Hugging Face Fine-Tuning Baseline

Date: 2026-07-23

Ticket 17 adds a pretrained-model comparison baseline using
`rinna/japanese-gpt2-small`. The model and tokenizer are licensed under MIT by
rinna; this project only fine-tunes a local copy. It uses a whole-work split:
`michikusa.txt` (`道草`) is held out for validation and the remaining nine
Natsume Soseki works are training-only.

## Local Smoke Result

The local smoke run uses one epoch, one training batch, and one validation
batch. Its purpose is to verify downloading, tokenization, work-level splitting,
backpropagation, `save_pretrained` output, and generation. It is not evidence of
full-corpus quality or a meaningful validation score.

The tracked repository does not include generated `outputs/` artifacts. The
verified local run used MPS, block size 64, batch size 1, one accumulation step,
one epoch, and one train plus one validation batch. It recorded train loss
`6.0144` and validation loss `5.8487`. These are smoke-only numbers, not a
full-corpus baseline.

With `吾輩は`, temperature 0.7, top-k 20, top-p 0.9, and seed 42, the saved smoke
model produced the following continuation (the prompt is retained at the start):

```text
吾輩は猫である。 猫は人間に飼われ、人間は猫の姿で、人間は猫
```

The smoke run created:

- `outputs/huggingface_rinna_smoke/best_model/`
- `outputs/huggingface_rinna_smoke/metrics.json`
- `outputs/huggingface_rinna_smoke/loss_curve.png`

## Colab Full-Run Comparison Record

After the 3-epoch Colab run, append measured values only. Record the following
for both the Hugging Face model and the current self-made SentencePiece model:

- training command, hardware, runtime, best epoch, and held-out-data definition;
- the model's own validation loss, clearly labeled as not directly comparable
  across tokenizers or pretraining histories;
- outputs for `吾輩は` at temperature 0.7 and `私は` at temperature 0.8, both with
  top-k 20, top-p 0.9, and seed 42;
- qualitative observations: prompt continuation, repetition, grammar, and
  longer-range coherence;
- failures or surprising output, without converting a short example into a
  general quality claim.
