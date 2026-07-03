from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass(frozen=True)
class TextRecord:
    author_id: str
    path: Path
    text: str


def load_text_records(data_dir: Path) -> list[TextRecord]:
    records: list[TextRecord] = []
    for path in sorted(data_dir.rglob("*.txt")):
        if not path.is_file():
            continue
        author_id = path.parent.name
        records.append(
            TextRecord(
                author_id=author_id,
                path=path,
                text=path.read_text(encoding="utf-8"),
            )
        )
    if not records:
        raise FileNotFoundError(
            f"No processed text files found under {data_dir}. "
            "Run `uv run python scripts/prepare_aozora.py` first."
        )
    return records


class LanguageTokenizer(Protocol):
    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ...


@dataclass(frozen=True)
class LanguageModelingExample:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    author_id: str
    source_path: str
    start_index: int


class LanguageModelingDataset(Dataset[LanguageModelingExample]):
    def __init__(
        self,
        records: list[TextRecord],
        tokenizer: LanguageTokenizer,
        block_size: int,
        stride: int | None = None,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be at least 1")

        self.block_size = block_size
        self.stride = stride or block_size
        self.examples: list[LanguageModelingExample] = []

        for record in records:
            # Add BOS/EOS once per work so the future model can learn boundaries.
            token_ids = tokenizer.encode(record.text, add_bos=True, add_eos=True)
            max_start = len(token_ids) - block_size - 1
            for start_index in range(0, max_start + 1, self.stride):
                window = token_ids[start_index : start_index + block_size + 1]
                input_ids = torch.tensor(window[:-1], dtype=torch.long)
                target_ids = torch.tensor(window[1:], dtype=torch.long)
                self.examples.append(
                    LanguageModelingExample(
                        input_ids=input_ids,
                        target_ids=target_ids,
                        author_id=record.author_id,
                        source_path=str(record.path),
                        start_index=start_index,
                    )
                )

        if not self.examples:
            raise ValueError(
                "No examples were created. Use a smaller block_size or add more text."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> LanguageModelingExample:
        return self.examples[index]


def collate_language_modeling_examples(
    examples: list[LanguageModelingExample],
) -> dict[str, object]:
    return {
        "input_ids": torch.stack([example.input_ids for example in examples]),
        "target_ids": torch.stack([example.target_ids for example in examples]),
        "author_id": [example.author_id for example in examples],
        "source_path": [example.source_path for example in examples],
        "start_index": torch.tensor(
            [example.start_index for example in examples],
            dtype=torch.long,
        ),
    }


def create_train_validation_loaders(
    dataset: LanguageModelingDataset,
    batch_size: int,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    validation_size = max(1, int(len(dataset) * validation_fraction))
    train_size = len(dataset) - validation_size
    if train_size < 1:
        raise ValueError("Dataset is too small to create a train/validation split")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_language_modeling_examples,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_language_modeling_examples,
    )
    return train_loader, validation_loader
