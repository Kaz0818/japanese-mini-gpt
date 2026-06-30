from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_transformer.char_tokenizer import CharTokenizer


DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_VOCAB_PATH = Path("data/tokenizers/char_vocab.json")
DEFAULT_SAMPLE = "吾輩は猫である。"


def find_text_files(data_dir: Path) -> list[Path]:
    text_files = sorted(path for path in data_dir.rglob("*.txt") if path.is_file())
    if not text_files:
        raise FileNotFoundError(
            f"No processed text files found under {data_dir}. "
            "Run `uv run python scripts/prepare_aozora.py` first."
        )
    return text_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and inspect the character tokenizer vocabulary."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text_files = find_text_files(args.data_dir)

    tokenizer = CharTokenizer.from_files(text_files)
    tokenizer.save(args.vocab_path)
    reloaded = CharTokenizer.load(args.vocab_path)

    token_ids = reloaded.encode(args.sample, add_bos=True, add_eos=True)
    decoded = reloaded.decode(token_ids)

    print(f"text_files={len(text_files)}")
    print(f"vocab_path={args.vocab_path}")
    print(f"vocab_size={reloaded.vocab_size}")
    print(
        "special_ids="
        f"pad:{reloaded.pad_id},unk:{reloaded.unk_id},"
        f"bos:{reloaded.bos_id},eos:{reloaded.eos_id}"
    )
    print(f"sample={args.sample}")
    print(f"token_ids={token_ids}")
    print(f"decoded={decoded}")

    if decoded != args.sample:
        raise RuntimeError("Decoded sample did not match the original sample.")


if __name__ == "__main__":
    main()
