from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_transformer.language_dataset import TextRecord, load_text_records


DEFAULT_MODEL_NAME = "rinna/japanese-gpt2-small"
DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_OUTPUT_DIR = Path("outputs/huggingface_rinna_smoke")


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str
    data_dir: Path
    output_dir: Path
    validation_work: str
    block_size: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float
    warmup_ratio: float
    max_train_batches: int | None
    max_validation_batches: int | None
    seed: int
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune rinna/japanese-gpt2-small with a plain PyTorch loop."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--validation-work",
        default="michikusa",
        help="Processed work filename stem reserved for validation (default: michikusa).",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit train batches per epoch for a quick smoke run.",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
        help="Limit validation batches per epoch for a quick smoke run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.block_size < 2:
        raise ValueError("block_size must be at least 2")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be at least 0 and less than 1")
    for option_name in ("max_train_batches", "max_validation_batches"):
        value = getattr(args, option_name)
        if value is not None and value < 1:
            raise ValueError(f"{option_name} must be at least 1 when provided")


def choose_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        device = torch.device(requested_device)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps is unavailable.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda is unavailable.")
        return device

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_records_by_work(
    records: list[TextRecord], validation_work: str
) -> tuple[list[TextRecord], list[TextRecord]]:
    """Keep one complete work out of training for a fairer validation check."""
    validation_stem = Path(validation_work).stem
    validation_records = [record for record in records if record.path.stem == validation_stem]
    train_records = [record for record in records if record.path.stem != validation_stem]
    if not validation_records:
        available_works = ", ".join(record.path.stem for record in records)
        raise ValueError(
            f"Validation work '{validation_stem}' was not found. Available works: {available_works}"
        )
    if not train_records:
        raise ValueError("At least one non-validation work is required for training.")
    return train_records, validation_records


def records_to_token_blocks(
    records: list[TextRecord], tokenizer: object, block_size: int
) -> torch.Tensor:
    """Tokenize each work separately and return fixed-length causal-LM blocks."""
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("The selected tokenizer must define eos_token_id.")

    blocks: list[list[int]] = []
    for record in records:
        token_ids = tokenizer.encode(record.text, add_special_tokens=False)
        token_ids.append(eos_token_id)
        # Each block has its own complete context. The model shifts labels inside
        # its loss function, so the final token in every block has no target.
        for start_index in range(0, len(token_ids) - block_size + 1, block_size):
            blocks.append(token_ids[start_index : start_index + block_size])

    if not blocks:
        raise ValueError(
            "No token blocks were created. Use a smaller block_size or add more text."
        )
    return torch.tensor(blocks, dtype=torch.long)


def create_loader(token_blocks: torch.Tensor, batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(token_blocks),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def limited_batch_count(loader: DataLoader, max_batches: int | None) -> int:
    return len(loader) if max_batches is None else min(len(loader), max_batches)


def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    warmup_steps = int(total_steps * warmup_ratio)

    def multiplier(step_index: int) -> float:
        if warmup_steps > 0 and step_index < warmup_steps:
            return (step_index + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step_index - warmup_steps) / decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), warmup_steps


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    gradient_accumulation_steps: int,
    max_batches: int | None,
) -> tuple[float, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    processed_batches = 0

    for batch_index, (input_ids,) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        input_ids = input_ids.to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("The causal language model did not return a loss.")

        total_loss += float(loss.detach().item())
        processed_batches += 1
        (loss / gradient_accumulation_steps).backward()

        if (
            processed_batches % gradient_accumulation_steps == 0
            or processed_batches == limited_batch_count(loader, max_batches)
        ):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    if processed_batches == 0:
        raise RuntimeError("No training batches were processed.")
    return total_loss / processed_batches, processed_batches


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    processed_batches = 0
    for batch_index, (input_ids,) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        outputs = model(input_ids=input_ids.to(device), labels=input_ids.to(device))
        if outputs.loss is None:
            raise RuntimeError("The causal language model did not return a loss.")
        total_loss += float(outputs.loss.item())
        processed_batches += 1
    if processed_batches == 0:
        raise RuntimeError("No validation batches were processed.")
    return total_loss / processed_batches, processed_batches


def json_ready_config(config: TrainingConfig) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def save_loss_curve(history: list[dict[str, float | int]], output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [float(row["train_loss"]) for row in history], marker="o", label="train")
    plt.plot(epochs, [float(row["validation_loss"]) for row in history], marker="o", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Hugging Face fine-tuning loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = TrainingConfig(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        validation_work=Path(args.validation_work).stem,
        block_size=args.block_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        seed=args.seed,
        device=args.device,
    )
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = choose_device(config.device)
    records = load_text_records(config.data_dir)
    train_records, validation_records = split_records_by_work(records, config.validation_work)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False)
    tokenizer.do_lower_case = True
    train_blocks = records_to_token_blocks(train_records, tokenizer, config.block_size)
    validation_blocks = records_to_token_blocks(validation_records, tokenizer, config.block_size)
    train_loader = create_loader(train_blocks, config.batch_size, shuffle=True, seed=config.seed)
    validation_loader = create_loader(validation_blocks, config.batch_size, shuffle=False, seed=config.seed)

    train_batches_per_epoch = limited_batch_count(train_loader, config.max_train_batches)
    total_steps = math.ceil(train_batches_per_epoch / config.gradient_accumulation_steps) * config.epochs
    model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler, warmup_steps = create_warmup_cosine_scheduler(
        optimizer, total_steps, config.warmup_ratio
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = config.output_dir / "best_model"
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0

    print(
        f"device={device.type} train_works={[record.path.stem for record in train_records]} "
        f"validation_works={[record.path.stem for record in validation_records]}"
    )
    print(
        f"train_blocks={len(train_blocks)} validation_blocks={len(validation_blocks)} "
        f"optimizer_steps={total_steps} warmup_steps={warmup_steps}"
    )

    for epoch in range(1, config.epochs + 1):
        train_loss, processed_train_batches = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            config.gradient_accumulation_steps, config.max_train_batches,
        )
        validation_loss, processed_validation_batches = evaluate(
            model, validation_loader, device, config.max_validation_batches
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "processed_train_batches": processed_train_batches,
                "processed_validation_batches": processed_validation_batches,
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"validation_loss={validation_loss:.4f}"
        )

    save_loss_curve(history, config.output_dir / "loss_curve.png")
    metrics = {
        "config": json_ready_config(config),
        "device": device.type,
        "train_work_ids": [record.path.stem for record in train_records],
        "validation_work_ids": [record.path.stem for record in validation_records],
        "train_blocks": len(train_blocks),
        "validation_blocks": len(validation_blocks),
        "train_batches_per_epoch": train_batches_per_epoch,
        "total_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "history": history,
    }
    (config.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"best_model={best_model_dir}")
    print(f"best_validation_loss={best_validation_loss:.4f} best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
