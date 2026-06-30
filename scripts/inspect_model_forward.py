from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_transformer.model import MiniTransformerConfig, MiniTransformerDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a forward-pass smoke check for the mini Transformer."
    )
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--feed-forward-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    config = MiniTransformerConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feed_forward_dim=args.feed_forward_dim,
        dropout=args.dropout,
    )
    model = MiniTransformerDecoder(config)
    model.eval()

    input_ids = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.block_size),
        dtype=torch.long,
    )

    with torch.no_grad():
        logits = model(input_ids)

    expected_shape = (args.batch_size, args.block_size, args.vocab_size)
    print(f"input_ids.shape={tuple(input_ids.shape)}")
    print(f"logits.shape={tuple(logits.shape)}")
    print(f"expected_logits_shape={expected_shape}")
    print(f"parameter_count={count_parameters(model):,}")

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            f"Expected logits shape {expected_shape}, got {tuple(logits.shape)}"
        )


if __name__ == "__main__":
    main()
