from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_transformer.language_dataset import load_text_records
from mini_transformer.sentencepiece_tokenizer import train_sentencepiece_tokenizer


DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_MODEL_PATH = Path("data/tokenizers/sentencepiece_unigram.model")
DEFAULT_SAMPLE = "吾輩は猫である。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and inspect a SentencePiece tokenizer."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vocab-size", type=int, default=4000)
    parser.add_argument(
        "--model-type",
        choices=("unigram", "bpe"),
        default="unigram",
        help="SentencePiece algorithm. Unigram is the default comparison target.",
    )
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument("--max-sentence-length", type=int, default=40000)
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_text_records(args.data_dir)
    tokenizer = train_sentencepiece_tokenizer(
        texts=[record.text for record in records],
        model_path=args.model_path,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        max_sentence_length=args.max_sentence_length,
    )

    token_ids = tokenizer.encode(args.sample, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(token_ids)

    print(f"text_records={len(records)}")
    print(f"model_path={args.model_path}")
    print(f"model_type={args.model_type}")
    print(f"vocab_size={tokenizer.vocab_size}")
    print(
        "special_ids="
        f"pad:{tokenizer.pad_id},unk:{tokenizer.unk_id},"
        f"bos:{tokenizer.bos_id},eos:{tokenizer.eos_id}"
    )
    print(f"sample={args.sample}")
    print(f"token_ids={token_ids}")
    print(f"decoded={decoded}")

    if decoded != args.sample:
        raise RuntimeError("Decoded sample did not match the original sample.")


if __name__ == "__main__":
    main()
