from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_transformer.char_tokenizer import CharTokenizer
from mini_transformer.language_dataset import (
    LanguageModelingDataset,
    create_train_validation_loaders,
    load_text_records,
)


DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_VOCAB_PATH = Path("data/tokenizers/char_vocab.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect next-token language-modeling batches."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_text_records(args.data_dir)
    if args.vocab_path.exists():
        tokenizer = CharTokenizer.load(args.vocab_path)
    else:
        tokenizer = CharTokenizer.from_texts([record.text for record in records])
        tokenizer.save(args.vocab_path)

    dataset = LanguageModelingDataset(
        records,
        tokenizer,
        block_size=args.block_size,
        stride=args.stride,
    )
    train_loader, validation_loader = create_train_validation_loaders(
        dataset,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]
    shifted_targets_match = bool((input_ids[:, 1:] == target_ids[:, :-1]).all())

    print(f"text_records={len(records)}")
    print(f"vocab_path={args.vocab_path}")
    print(f"vocab_size={tokenizer.vocab_size}")
    print(f"dataset_examples={len(dataset)}")
    print(f"train_batches={len(train_loader)}")
    print(f"validation_batches={len(validation_loader)}")
    print(f"input_ids.shape={tuple(input_ids.shape)}")
    print(f"target_ids.shape={tuple(target_ids.shape)}")
    print(f"shifted_targets_match={shifted_targets_match}")
    print(f"author_id={batch['author_id']}")
    print(f"source_path={batch['source_path']}")
    print(f"start_index={batch['start_index'].tolist()}")
    print(f"input_ids[0,:12]={input_ids[0, :12].tolist()}")
    print(f"target_ids[0,:12]={target_ids[0, :12].tolist()}")
    print(f"input_text[0,:40]={tokenizer.decode(input_ids[0].tolist())[:40]}")
    print(f"target_text[0,:40]={tokenizer.decode(target_ids[0].tolist())[:40]}")

    if not shifted_targets_match:
        raise RuntimeError("target_ids should be input_ids shifted one token ahead.")


if __name__ == "__main__":
    main()
