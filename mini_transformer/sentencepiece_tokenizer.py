from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import sentencepiece as spm


@dataclass(frozen=True)
class SentencePieceTokenizer:
    processor: spm.SentencePieceProcessor
    model_path: Path

    @classmethod
    def load(cls, model_path: Path) -> "SentencePieceTokenizer":
        processor = spm.SentencePieceProcessor()
        if not processor.Load(str(model_path)):
            raise RuntimeError(f"Failed to load SentencePiece model: {model_path}")
        return cls(processor=processor, model_path=model_path)

    @property
    def vocab_size(self) -> int:
        return int(self.processor.GetPieceSize())

    @property
    def pad_id(self) -> int:
        return int(self.processor.pad_id())

    @property
    def unk_id(self) -> int:
        return int(self.processor.unk_id())

    @property
    def bos_id(self) -> int:
        return int(self.processor.bos_id())

    @property
    def eos_id(self) -> int:
        return int(self.processor.eos_id())

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        token_ids = list(self.processor.EncodeAsIds(text))
        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            skipped_ids = {self.pad_id, self.bos_id, self.eos_id}
            token_ids = [token_id for token_id in token_ids if token_id not in skipped_ids]
        return str(self.processor.DecodeIds(token_ids))


def train_sentencepiece_tokenizer(
    *,
    texts: list[str],
    model_path: Path,
    vocab_size: int,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
    max_sentence_length: int = 40000,
) -> SentencePieceTokenizer:
    if not texts:
        raise ValueError("texts must contain at least one item")
    if vocab_size < 8:
        raise ValueError("vocab_size must be at least 8")
    if model_type not in {"unigram", "bpe"}:
        raise ValueError("model_type must be 'unigram' or 'bpe'")
    if not 0.0 < character_coverage <= 1.0:
        raise ValueError("character_coverage must be greater than 0 and at most 1")
    if max_sentence_length < 1:
        raise ValueError("max_sentence_length must be at least 1")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_prefix = model_path.with_suffix("")

    with TemporaryDirectory() as temporary_directory:
        corpus_path = Path(temporary_directory) / "sentencepiece_corpus.txt"
        corpus_path.write_text("\n".join(texts), encoding="utf-8")
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            max_sentence_length=max_sentence_length,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            hard_vocab_limit=False,
        )

    generated_model_path = model_prefix.with_suffix(".model")
    if generated_model_path != model_path:
        generated_model_path.replace(model_path)
    return SentencePieceTokenizer.load(model_path)
