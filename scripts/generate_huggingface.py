from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODEL_DIR = Path("outputs/huggingface_rinna_smoke/best_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a Hugging Face fine-tuned Japanese GPT-2 model."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt", type=str, default="吾輩は")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Opt-in decoding control. Keep 1.0 for a fair baseline comparison.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
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


def validate_args(args: argparse.Namespace) -> None:
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if not args.prompt:
        raise ValueError("prompt must not be empty")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if args.temperature < 0:
        raise ValueError("temperature must be 0 or greater")
    if args.top_k < 0:
        raise ValueError("top_k must be 0 or greater")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("top_p must be greater than 0 and at most 1")
    if args.repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1.0")


@torch.no_grad()
def generate_text(args: argparse.Namespace) -> tuple[str, str]:
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    tokenizer.do_lower_case = True
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
    model.eval()
    encoded_prompt = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded_prompt["input_ids"].to(device)
    generation_args: dict[str, object] = {
        "input_ids": input_ids,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_args.update(
            {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p}
        )
    output_ids = model.generate(**generation_args)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text, device.type


def main() -> None:
    args = parse_args()
    validate_args(args)
    generated_text, device_name = generate_text(args)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(generated_text + "\n", encoding="utf-8")
    print(f"device={device_name}")
    print(generated_text)


if __name__ == "__main__":
    main()
