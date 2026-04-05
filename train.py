from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import CachedMultimodalDataset, Vocabulary
from src.engine import evaluate, fit, save_history, set_seed
from src.models import ModelSpec, SUPPORTED_MODELS, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ouarka et al. (2024) multimodal personality model.")

    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--history-dir", type=str, default="histories")

    parser.add_argument(
        "--model-type",
        type=str,
        default="model_fusion",
        choices=sorted(SUPPORTED_MODELS),
        help="One of: scene, face, audio, text, early_fusion, model_fusion, self_attention, cross_attention",
    )

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--image-size", type=int, default=None, help="Override image size used at cache prep")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


def _load_meta(cache_dir: Path) -> dict:
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run prepare_data.py first to generate cached tensors."
        )
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metrics_to_print(metrics: dict, split_name: str) -> None:
    print(
        f"[{split_name}] loss={metrics['loss']:.4f} | "
        f"mae={metrics['mae']:.4f} | accuracy={(1.0 - metrics['mae']) * 100.0:.2f}"
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cache_dir = Path(args.cache_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    history_dir = Path(args.history_dir).resolve()

    run_name = args.run_name or f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_path = checkpoint_dir / f"{run_name}.pt"
    history_path = history_dir / f"{run_name}.json"
    metric_path = history_dir / f"{run_name}_metrics.json"

    meta = _load_meta(cache_dir)
    image_size = args.image_size if args.image_size is not None else int(meta["image_size"])
    text_seq_len = int(meta["text_seq_len"])

    vocab = Vocabulary.load(cache_dir / "vocab.json")

    train_ds = CachedMultimodalDataset(cache_dir / "train.npz")
    valid_ds = CachedMultimodalDataset(cache_dir / "valid.npz")
    test_ds = CachedMultimodalDataset(cache_dir / "test.npz")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    spec = ModelSpec(
        model_type=args.model_type,
        vocab_size=len(vocab.itos),
        text_seq_len=text_seq_len,
    )
    model = build_model(spec=spec, image_size=image_size)

    device = torch.device(args.device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if args.resume is not None:
        resume_path = Path(args.resume)
        if resume_path.exists():
            payload = torch.load(resume_path, map_location=device)
            model.load_state_dict(payload["model_state"])
            if "optimizer_state" in payload:
                optimizer.load_state_dict(payload["optimizer_state"])
            print(f"Loaded checkpoint: {resume_path}")

    history = fit(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        checkpoint_path=ckpt_path,
    )
    save_history(history, history_path)

    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state"])

    valid_metrics = evaluate(model=model, loader=valid_loader, device=device)
    test_metrics = evaluate(model=model, loader=test_loader, device=device)

    _metrics_to_print(valid_metrics, "VALID")
    _metrics_to_print(test_metrics, "TEST")

    with metric_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "model_type": args.model_type,
                "valid": valid_metrics,
                "test": test_metrics,
            },
            f,
            indent=2,
        )

    print(f"Saved best checkpoint: {ckpt_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved metrics: {metric_path}")


if __name__ == "__main__":
    main()
