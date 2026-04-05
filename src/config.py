from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

TRAIT_NAMES = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


@dataclass(frozen=True)
class PathConfig:
    root_dir: Path
    train_dir: Path
    valid_dir: Path
    test_dir: Path
    annotation_train: Path
    annotation_valid: Optional[Path]
    annotation_test: Optional[Path]
    transcription_train: Optional[Path]
    transcription_valid: Optional[Path]
    transcription_test: Optional[Path]
    cache_dir: Path
    checkpoint_dir: Path


@dataclass(frozen=True)
class FeatureConfig:
    image_size: int = 96
    num_frames: int = 10
    audio_steps: int = 15
    audio_bins: int = 128
    text_seq_len: int = 50
    vocab_size: int = 12000


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42
