from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import CachedMultimodalDataset, Vocabulary
from src.engine import evaluate
from src.models import ModelSpec, SUPPORTED_MODELS, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained multimodal personality model.")
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-type", type=str, required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def _load_meta(cache_dir: Path) -> dict:
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run prepare_data.py first to generate cached tensors."
        )
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).resolve()
    ckpt_path = Path(args.checkpoint).resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    meta = _load_meta(cache_dir)
    image_size = args.image_size if args.image_size is not None else int(meta["image_size"])
    text_seq_len = int(meta["text_seq_len"])

    vocab = Vocabulary.load(cache_dir / "vocab.json")
    spec = ModelSpec(
        model_type=args.model_type,
        vocab_size=len(vocab.itos),
        text_seq_len=text_seq_len,
    )

    device = torch.device(args.device)
    model = build_model(spec=spec, image_size=image_size).to(device)

    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model_state"])

    valid_loader = DataLoader(
        CachedMultimodalDataset(cache_dir / "valid.npz"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        CachedMultimodalDataset(cache_dir / "test.npz"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    valid_metrics = evaluate(model=model, loader=valid_loader, device=device)
    test_metrics = evaluate(model=model, loader=test_loader, device=device)

    print(
        f"VALID  -> loss={valid_metrics['loss']:.4f}, "
        f"mae={valid_metrics['mae']:.4f}, accuracy={(1.0 - valid_metrics['mae']) * 100.0:.2f}"
    )
    print(
        f"TEST   -> loss={test_metrics['loss']:.4f}, "
        f"mae={test_metrics['mae']:.4f}, accuracy={(1.0 - test_metrics['mae']) * 100.0:.2f}"
    )

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"valid": valid_metrics, "test": test_metrics}, f, indent=2)
        print(f"Saved metrics file: {output_path}")


if __name__ == "__main__":
    main()
