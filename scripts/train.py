from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_transformer.char_tokenizer import CharTokenizer
from mini_transformer.language_dataset import (
    LanguageModelingDataset,
    create_train_validation_loaders,
    load_text_records,
)
from mini_transformer.model import MiniTransformerConfig, MiniTransformerDecoder
from mini_transformer.sentencepiece_tokenizer import SentencePieceTokenizer


DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_VOCAB_PATH = Path("data/tokenizers/char_vocab.json")
DEFAULT_OUTPUT_DIR = Path("outputs/ticket6_smoke")


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path
    tokenizer_type: str
    vocab_path: Path
    output_dir: Path
    block_size: int
    stride: int
    batch_size: int
    validation_fraction: float
    embedding_dim: int
    num_layers: int
    num_heads: int
    feed_forward_dim: int
    dropout: float
    learning_rate: float
    scheduler: str
    warmup_steps: int | None
    warmup_ratio: float
    min_learning_rate: float
    epochs: int
    max_train_batches: int | None
    max_validation_batches: int | None
    early_stopping_patience: int | None
    early_stopping_min_delta: float
    use_wandb: bool
    wandb_project: str
    wandb_run_name: str | None
    seed: int
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the mini Transformer with a plain PyTorch loop."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--tokenizer-type",
        choices=("char", "sentencepiece"),
        default="char",
        help="Tokenizer format used by --vocab-path.",
    )
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--feed-forward-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--scheduler",
        choices=("none", "cosine"),
        default="none",
        help="Use 'cosine' for warmup followed by cosine decay.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Number of optimizer steps used for learning-rate warmup.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup fraction of total optimizer steps when --warmup-steps is omitted.",
    )
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit train batches for a quick smoke run.",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
        help="Limit validation batches for a quick smoke run.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop after this many epochs without validation improvement.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Required validation-loss improvement for early stopping.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log training metrics and best checkpoint to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", type=str, default="mini-transformer")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def validate_training_args(args: argparse.Namespace) -> None:
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if args.min_learning_rate < 0:
        raise ValueError("min_learning_rate must be 0 or greater")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("min_learning_rate must be <= learning_rate")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        raise ValueError("warmup_steps must be 0 or greater")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be at least 0 and less than 1")
    if args.early_stopping_patience is not None and args.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least 1")
    if args.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be 0 or greater")


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


def load_or_build_tokenizer(
    records: list,
    *,
    tokenizer_type: str,
    vocab_path: Path,
) -> CharTokenizer | SentencePieceTokenizer:
    if tokenizer_type == "sentencepiece":
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"SentencePiece model not found: {vocab_path}. "
                "Run `uv run python scripts/train_sentencepiece_tokenizer.py` first."
            )
        return SentencePieceTokenizer.load(vocab_path)

    if tokenizer_type == "char":
        if vocab_path.exists():
            return CharTokenizer.load(vocab_path)

        tokenizer = CharTokenizer.from_texts([record.text for record in records])
        tokenizer.save(vocab_path)
        return tokenizer

    raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")


def move_batch_to_device(
    batch: dict[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    target_ids = batch["target_ids"].to(device)
    return input_ids, target_ids


def language_modeling_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    # logits: [batch, block, vocab], targets: [batch, block].
    # Cross entropy expects [items, classes], so flatten batch and time together.
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
    )


def count_limited_batches(
    loader: torch.utils.data.DataLoader,
    max_batches: int | None,
) -> int:
    if max_batches is None:
        return len(loader)
    return min(max_batches, len(loader))


def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    total_training_steps: int,
    learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if scheduler_name == "none":
        return None
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    if total_training_steps < 1:
        raise ValueError("total_training_steps must be at least 1")

    minimum_ratio = min_learning_rate / learning_rate

    def lr_lambda(step_index: int) -> float:
        # LambdaLR calls this before the first optimizer step with step_index=0.
        current_step = step_index + 1
        if warmup_steps > 0 and current_step <= warmup_steps:
            return max(minimum_ratio, current_step / warmup_steps)

        decay_steps = max(1, total_training_steps - warmup_steps)
        decay_progress = min(
            1.0,
            max(0.0, (current_step - warmup_steps) / decay_steps),
        )
        cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: MiniTransformerDecoder,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, int]:
    model.train()
    losses: list[float] = []
    processed_batches = 0

    for batch_index, batch in enumerate(train_loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        input_ids, target_ids = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = language_modeling_loss(logits, target_ids)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(float(loss.detach().cpu()))
        processed_batches += 1

    if not losses:
        raise RuntimeError("No training batches were processed.")
    return sum(losses) / len(losses), processed_batches


@torch.no_grad()
def evaluate(
    model: MiniTransformerDecoder,
    validation_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.eval()
    losses: list[float] = []

    for batch_index, batch in enumerate(validation_loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        input_ids, target_ids = move_batch_to_device(batch, device)
        logits = model(input_ids)
        loss = language_modeling_loss(logits, target_ids)
        losses.append(float(loss.detach().cpu()))

    if not losses:
        raise RuntimeError("No validation batches were processed.")
    return sum(losses) / len(losses)


def save_loss_curve(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    validation_losses = [row["validation_loss"] for row in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_losses, marker="o", label="train")
    plt.plot(epochs, validation_losses, marker="o", label="validation")
    plt.xlabel("epoch")
    plt.ylabel("cross entropy loss")
    plt.title("Mini Transformer Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def json_ready_config(config: TrainingConfig) -> dict[str, object]:
    payload = asdict(config)
    for key in ("data_dir", "vocab_path", "output_dir"):
        payload[key] = str(payload[key])
    return payload


def save_checkpoint(
    path: Path,
    model: MiniTransformerDecoder,
    model_config: MiniTransformerConfig,
    training_config: TrainingConfig,
    history: list[dict[str, float]],
    vocab_path: Path,
    *,
    checkpoint_type: str,
    best_validation_loss: float,
    best_epoch: int | None,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": json_ready_config(training_config),
            "tokenizer_type": training_config.tokenizer_type,
            "vocab_path": str(vocab_path),
            "history": history,
            "checkpoint_type": checkpoint_type,
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
        },
        path,
    )


def build_run_config(
    training_config: TrainingConfig,
    model_config: MiniTransformerConfig,
    *,
    dataset_examples: int,
    train_batches: int,
    validation_batches: int,
    train_batches_per_epoch: int,
    total_training_steps: int,
    warmup_steps: int,
    vocab_size: int,
) -> dict[str, object]:
    return {
        "training_config": json_ready_config(training_config),
        "model_config": asdict(model_config),
        "dataset_examples": dataset_examples,
        "train_batches": train_batches,
        "validation_batches": validation_batches,
        "train_batches_per_epoch": train_batches_per_epoch,
        "total_training_steps": total_training_steps,
        "warmup_steps": warmup_steps,
        "vocab_size": vocab_size,
    }


def init_wandb_run(
    training_config: TrainingConfig,
    run_config: dict[str, object],
):
    if not training_config.use_wandb:
        return None

    import wandb

    return wandb.init(
        project=training_config.wandb_project,
        name=training_config.wandb_run_name,
        config=run_config,
    )


def log_wandb_epoch(
    wandb_run,
    *,
    epoch: int,
    train_loss: float,
    validation_loss: float,
    learning_rate: float,
    best_validation_loss: float,
    processed_train_batches: int,
) -> None:
    if wandb_run is None:
        return

    wandb_run.log(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": learning_rate,
            "best_validation_loss": best_validation_loss,
            "processed_train_batches": processed_train_batches,
        },
        step=epoch,
    )


def log_wandb_artifacts(
    wandb_run,
    *,
    loss_curve_path: Path,
    metrics_path: Path,
    best_checkpoint_path: Path,
) -> None:
    if wandb_run is None:
        return

    import wandb

    wandb_run.log({"loss_curve": wandb.Image(str(loss_curve_path))})
    wandb_run.save(str(metrics_path))

    artifact = wandb.Artifact(
        name=f"{wandb_run.name or 'mini-transformer'}-best-checkpoint",
        type="model",
        description="Best validation checkpoint for the mini Transformer run.",
    )
    artifact.add_file(str(best_checkpoint_path))
    wandb_run.log_artifact(artifact)


def main() -> None:
    args = parse_args()
    validate_training_args(args)
    device = choose_device(args.device)
    torch.manual_seed(args.seed)

    training_config = TrainingConfig(
        data_dir=args.data_dir,
        tokenizer_type=args.tokenizer_type,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        block_size=args.block_size,
        stride=args.stride,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feed_forward_dim=args.feed_forward_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        scheduler=args.scheduler,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        min_learning_rate=args.min_learning_rate,
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        device=device.type,
    )

    records = load_text_records(training_config.data_dir)
    tokenizer = load_or_build_tokenizer(
        records,
        tokenizer_type=training_config.tokenizer_type,
        vocab_path=training_config.vocab_path,
    )
    dataset = LanguageModelingDataset(
        records,
        tokenizer,
        block_size=training_config.block_size,
        stride=training_config.stride,
    )
    train_loader, validation_loader = create_train_validation_loaders(
        dataset,
        batch_size=training_config.batch_size,
        validation_fraction=training_config.validation_fraction,
        seed=training_config.seed,
    )

    model_config = MiniTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=training_config.block_size,
        embedding_dim=training_config.embedding_dim,
        num_layers=training_config.num_layers,
        num_heads=training_config.num_heads,
        feed_forward_dim=training_config.feed_forward_dim,
        dropout=training_config.dropout,
    )
    model = MiniTransformerDecoder(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    train_batches_per_epoch = count_limited_batches(
        train_loader,
        training_config.max_train_batches,
    )
    total_training_steps = train_batches_per_epoch * training_config.epochs
    if training_config.warmup_steps is None:
        warmup_steps = int(total_training_steps * training_config.warmup_ratio)
    else:
        warmup_steps = training_config.warmup_steps
    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        scheduler_name=training_config.scheduler,
        total_training_steps=total_training_steps,
        learning_rate=training_config.learning_rate,
        min_learning_rate=training_config.min_learning_rate,
        warmup_steps=warmup_steps,
    )
    run_config = build_run_config(
        training_config,
        model_config,
        dataset_examples=len(dataset),
        train_batches=len(train_loader),
        validation_batches=len(validation_loader),
        train_batches_per_epoch=train_batches_per_epoch,
        total_training_steps=total_training_steps,
        warmup_steps=warmup_steps,
        vocab_size=tokenizer.vocab_size,
    )
    wandb_run = init_wandb_run(training_config, run_config)

    output_dir = training_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        started_at = perf_counter()
        history: list[dict[str, float]] = []
        best_validation_loss = float("inf")
        best_epoch: int | None = None
        epochs_without_improvement = 0
        stopped_early = False
        best_checkpoint_path = output_dir / "best_checkpoint.pt"
        for epoch in range(1, training_config.epochs + 1):
            train_loss, processed_train_batches = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                training_config.max_train_batches,
            )
            validation_loss = evaluate(
                model,
                validation_loader,
                device,
                training_config.max_validation_batches,
            )
            current_learning_rate = optimizer.param_groups[0]["lr"]
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "learning_rate": current_learning_rate,
                    "processed_train_batches": processed_train_batches,
                }
            )

            improved = (
                validation_loss
                < best_validation_loss - training_config.early_stopping_min_delta
            )
            if improved:
                best_validation_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    best_checkpoint_path,
                    model,
                    model_config,
                    training_config,
                    history,
                    training_config.vocab_path,
                    checkpoint_type="best_validation",
                    best_validation_loss=best_validation_loss,
                    best_epoch=best_epoch,
                )
            else:
                epochs_without_improvement += 1

            log_wandb_epoch(
                wandb_run,
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learning_rate=current_learning_rate,
                best_validation_loss=best_validation_loss,
                processed_train_batches=processed_train_batches,
            )
            print(
                f"epoch={epoch} "
                f"train_loss={train_loss:.4f} "
                f"validation_loss={validation_loss:.4f} "
                f"lr={current_learning_rate:.6g}"
            )
            if (
                training_config.early_stopping_patience is not None
                and epochs_without_improvement
                >= training_config.early_stopping_patience
            ):
                stopped_early = True
                print(
                    "early_stopping="
                    f"epoch {epoch} stopped after "
                    f"{epochs_without_improvement} epochs without improvement"
                )
                break

        elapsed_seconds = perf_counter() - started_at
        checkpoint_path = output_dir / "checkpoint.pt"
        metrics_path = output_dir / "metrics.json"
        loss_curve_path = output_dir / "loss_curve.png"

        save_checkpoint(
            checkpoint_path,
            model,
            model_config,
            training_config,
            history,
            training_config.vocab_path,
            checkpoint_type="last",
            best_validation_loss=best_validation_loss,
            best_epoch=best_epoch,
        )
        save_loss_curve(history, loss_curve_path)

        metrics = {
            "config": json_ready_config(training_config),
            "model_config": asdict(model_config),
            "dataset_examples": len(dataset),
            "train_batches": len(train_loader),
            "validation_batches": len(validation_loader),
            "train_batches_per_epoch": train_batches_per_epoch,
            "total_training_steps": total_training_steps,
            "warmup_steps": warmup_steps,
            "vocab_size": tokenizer.vocab_size,
            "history": history,
            "final_train_loss": history[-1]["train_loss"],
            "final_validation_loss": history[-1]["validation_loss"],
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "elapsed_seconds": elapsed_seconds,
            "wandb": {
                "enabled": training_config.use_wandb,
                "project": training_config.wandb_project,
                "run_name": training_config.wandb_run_name,
                "run_id": wandb_run.id if wandb_run is not None else None,
            },
            "artifacts": {
                "checkpoint": str(checkpoint_path),
                "best_checkpoint": str(best_checkpoint_path),
                "metrics": str(metrics_path),
                "loss_curve": str(loss_curve_path),
            },
        }
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        log_wandb_artifacts(
            wandb_run,
            loss_curve_path=loss_curve_path,
            metrics_path=metrics_path,
            best_checkpoint_path=best_checkpoint_path,
        )

        print(f"device={device.type}")
        print(f"checkpoint={checkpoint_path}")
        print(f"best_checkpoint={best_checkpoint_path}")
        print(f"best_epoch={best_epoch}")
        print(f"best_validation_loss={best_validation_loss:.4f}")
        print(f"metrics={metrics_path}")
        print(f"loss_curve={loss_curve_path}")
        if wandb_run is not None:
            print(f"wandb_run={wandb_run.url}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
