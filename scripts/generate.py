from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_transformer.char_tokenizer import CharTokenizer
from mini_transformer.model import MiniTransformerConfig, MiniTransformerDecoder


DEFAULT_CHECKPOINT_PATH = Path("outputs/ticket6_smoke/checkpoint.pt")
DEFAULT_VOCAB_PATH = Path("data/tokenizers/char_vocab.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained mini Transformer checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--vocab-path", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default="吾輩は")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Higher values sample more randomly. Use 0 for greedy decoding.",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--stop-on-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


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


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[MiniTransformerDecoder, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = MiniTransformerConfig(**checkpoint["model_config"])
    model = MiniTransformerDecoder(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def resolve_vocab_path(
    requested_vocab_path: Path | None,
    checkpoint: dict[str, object],
) -> Path:
    if requested_vocab_path is not None:
        return requested_vocab_path
    if "vocab_path" in checkpoint:
        return Path(str(checkpoint["vocab_path"]))
    return DEFAULT_VOCAB_PATH


def filter_logits(
    logits: torch.Tensor,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    filtered_logits = logits.clone()

    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        kept_token_count = min(top_k, filtered_logits.numel())
        threshold = torch.topk(filtered_logits, kept_token_count).values[-1]
        filtered_logits = filtered_logits.masked_fill(
            filtered_logits < threshold,
            float("-inf"),
        )

    if top_p is not None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be greater than 0 and at most 1")
        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
        sorted_probabilities = F.softmax(sorted_logits, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

        # Remove tokens after the nucleus, but always keep the most likely token.
        remove_sorted = cumulative_probabilities > top_p
        remove_sorted[1:] = remove_sorted[:-1].clone()
        remove_sorted[0] = False

        remove_indices = sorted_indices[remove_sorted]
        filtered_logits[remove_indices] = float("-inf")

    return filtered_logits


def sample_next_token_id(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> int:
    if temperature < 0:
        raise ValueError("temperature must be 0 or greater")
    if temperature == 0:
        return int(torch.argmax(logits).item())

    scaled_logits = logits / temperature
    filtered_logits = filter_logits(scaled_logits, top_k=top_k, top_p=top_p)
    probabilities = F.softmax(filtered_logits, dim=-1)
    next_token_id = torch.multinomial(probabilities, num_samples=1)
    return int(next_token_id.item())


@torch.no_grad()
def generate_token_ids(
    model: MiniTransformerDecoder,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    eos_id: int,
    stop_on_eos: bool,
    device: torch.device,
) -> list[int]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    generated_token_ids = list(prompt_token_ids)
    for _ in range(max_new_tokens):
        # The decoder can only see the most recent block_size tokens.
        context_ids = generated_token_ids[-model.config.block_size :]
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        logits = model(input_ids)
        next_token_logits = logits[0, -1, :]
        next_token_id = sample_next_token_id(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated_token_ids.append(next_token_id)
        if stop_on_eos and next_token_id == eos_id:
            break

    return generated_token_ids


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    vocab_path = resolve_vocab_path(args.vocab_path, checkpoint)
    tokenizer = CharTokenizer.load(vocab_path)

    prompt_token_ids = tokenizer.encode(args.prompt, add_bos=True, add_eos=False)
    generated_token_ids = generate_token_ids(
        model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_id=tokenizer.eos_id,
        stop_on_eos=args.stop_on_eos,
        device=device,
    )
    generated_text = tokenizer.decode(generated_token_ids)

    lines = [
        f"device={device.type}",
        f"checkpoint={args.checkpoint}",
        f"vocab_path={vocab_path}",
        f"prompt={args.prompt}",
        f"temperature={args.temperature}",
        f"top_k={args.top_k}",
        f"top_p={args.top_p}",
        f"generated_token_count={len(generated_token_ids)}",
        "generated_text:",
        generated_text,
    ]
    output = "\n".join(lines) + "\n"
    print(output, end="")

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(output, encoding="utf-8")
        print(f"saved_output={args.output_path}")


if __name__ == "__main__":
    main()
