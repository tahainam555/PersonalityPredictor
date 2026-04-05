from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from src.data import (
    Vocabulary,
    build_annotation_index,
    build_dataset_cache,
    build_transcription_index,
    collect_split_samples,
    safe_load_pickle,
)


def _resolve_path(root: Path, maybe_relative: Optional[str]) -> Optional[Path]:
    if maybe_relative is None:
        return None
    p = Path(maybe_relative)
    if not p.is_absolute():
        p = root / p
    return p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare cached multimodal tensors for CPU-friendly personality prediction training."
    )

    parser.add_argument("--root-dir", type=str, default=".")
    parser.add_argument("--train-dir", type=str, default="train_data")
    parser.add_argument("--valid-dir", type=str, default="valid_data")
    parser.add_argument("--test-dir", type=str, default="test_data")

    parser.add_argument("--annotation-train", type=str, default="annotation_training.pkl")
    parser.add_argument("--annotation-valid", type=str, default="annotation_validation.pkl")
    parser.add_argument("--annotation-test", type=str, default="annotation_test.pkl")

    parser.add_argument("--transcription-train", type=str, default="transcription_training.pkl")
    parser.add_argument("--transcription-valid", type=str, default="transcription_validation.pkl")
    parser.add_argument("--transcription-test", type=str, default="transcription_test.pkl")

    parser.add_argument("--cache-dir", type=str, default="cache")

    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--audio-steps", type=int, default=15)
    parser.add_argument("--audio-bins", type=int, default=128)
    parser.add_argument("--text-seq-len", type=int, default=50)
    parser.add_argument("--vocab-size", type=int, default=12000)

    parser.add_argument("--train-limit", type=int, default=400)
    parser.add_argument("--valid-limit", type=int, default=80)
    parser.add_argument("--test-limit", type=int, default=80)

    return parser.parse_args()


def _load_annotation_or_fallback(
    split_name: str,
    path: Optional[Path],
    fallback_raw: object,
) -> Dict[str, float]:
    raw = safe_load_pickle(path)
    if raw is None:
        print(f"[WARN] Missing {split_name} annotation file -> using training annotation fallback.")
        raw = fallback_raw
    return build_annotation_index(raw)


def _load_transcriptions(path: Optional[Path], split_name: str) -> Dict[str, str]:
    raw = safe_load_pickle(path)
    if raw is None:
        print(f"[WARN] Missing {split_name} transcription file -> using empty transcripts.")
    return build_transcription_index(raw)


def main() -> None:
    args = parse_args()
    root = Path(args.root_dir).resolve()

    train_dir = _resolve_path(root, args.train_dir)
    valid_dir = _resolve_path(root, args.valid_dir)
    test_dir = _resolve_path(root, args.test_dir)

    ann_train_path = _resolve_path(root, args.annotation_train)
    ann_valid_path = _resolve_path(root, args.annotation_valid)
    ann_test_path = _resolve_path(root, args.annotation_test)

    tr_train_path = _resolve_path(root, args.transcription_train)
    tr_valid_path = _resolve_path(root, args.transcription_valid)
    tr_test_path = _resolve_path(root, args.transcription_test)

    cache_dir = _resolve_path(root, args.cache_dir)
    assert cache_dir is not None
    cache_dir.mkdir(parents=True, exist_ok=True)

    if train_dir is None or not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if valid_dir is None or not valid_dir.exists():
        raise FileNotFoundError(f"Valid directory not found: {valid_dir}")
    if test_dir is None or not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    if ann_train_path is None or not ann_train_path.exists():
        raise FileNotFoundError(
            "Training annotation file is required. Expected annotation_training.pkl in root or pass --annotation-train"
        )

    raw_ann_train = safe_load_pickle(ann_train_path)
    assert raw_ann_train is not None

    ann_train = build_annotation_index(raw_ann_train)
    ann_valid = _load_annotation_or_fallback("validation", ann_valid_path, raw_ann_train)
    ann_test = _load_annotation_or_fallback("test", ann_test_path, raw_ann_train)

    tr_train = _load_transcriptions(tr_train_path, "training")
    tr_valid = _load_transcriptions(tr_valid_path, "validation")
    tr_test = _load_transcriptions(tr_test_path, "test")

    train_samples = collect_split_samples(train_dir, ann_train, tr_train, limit=args.train_limit)
    valid_samples = collect_split_samples(valid_dir, ann_valid, tr_valid, limit=args.valid_limit)
    test_samples = collect_split_samples(test_dir, ann_test, tr_test, limit=args.test_limit)

    print(
        f"Collected samples -> train: {len(train_samples)}, valid: {len(valid_samples)}, test: {len(test_samples)}"
    )

    if len(train_samples) == 0:
        raise ValueError(
            "No training samples matched annotation labels. "
            "Verify --annotation-train and video filenames in train_data/."
        )
    if len(valid_samples) == 0:
        raise ValueError(
            "No validation samples matched annotation labels. "
            "Provide --annotation-valid (annotation_validation.pkl) for valid_data/."
        )
    if len(test_samples) == 0:
        raise ValueError(
            "No test samples matched annotation labels. "
            "Provide --annotation-test (annotation_test.pkl) for test_data/."
        )

    vocab = Vocabulary.build((text for _, _, text in train_samples), vocab_size=args.vocab_size)
    vocab_path = cache_dir / "vocab.json"
    vocab.save(vocab_path)

    build_dataset_cache(
        samples=train_samples,
        vocab=vocab,
        output_path=cache_dir / "train.npz",
        image_size=args.image_size,
        num_frames=args.num_frames,
        text_seq_len=args.text_seq_len,
        audio_steps=args.audio_steps,
        audio_bins=args.audio_bins,
    )
    build_dataset_cache(
        samples=valid_samples,
        vocab=vocab,
        output_path=cache_dir / "valid.npz",
        image_size=args.image_size,
        num_frames=args.num_frames,
        text_seq_len=args.text_seq_len,
        audio_steps=args.audio_steps,
        audio_bins=args.audio_bins,
    )
    build_dataset_cache(
        samples=test_samples,
        vocab=vocab,
        output_path=cache_dir / "test.npz",
        image_size=args.image_size,
        num_frames=args.num_frames,
        text_seq_len=args.text_seq_len,
        audio_steps=args.audio_steps,
        audio_bins=args.audio_bins,
    )

    meta = {
        "train_count": len(train_samples),
        "valid_count": len(valid_samples),
        "test_count": len(test_samples),
        "image_size": args.image_size,
        "num_frames": args.num_frames,
        "audio_steps": args.audio_steps,
        "audio_bins": args.audio_bins,
        "text_seq_len": args.text_seq_len,
        "vocab_size": len(vocab.itos),
    }

    with (cache_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved cache to: {cache_dir}")


if __name__ == "__main__":
    main()
