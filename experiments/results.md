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
