from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.config import TRAIT_NAMES


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def load_pickle(path: Path) -> object:
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def normalize_video_key(video_key: str) -> str:
    video_key = str(video_key)
    if not video_key.endswith(".mp4"):
        video_key = f"{video_key}.mp4"
    return video_key


def build_annotation_index(raw_annotation: object) -> Dict[str, np.ndarray]:
    if not isinstance(raw_annotation, dict):
        raise TypeError("Annotation file must be a dict-like pickle object.")

    lower_to_raw = {str(k).lower(): k for k in raw_annotation.keys()}
    if all(trait in lower_to_raw for trait in TRAIT_NAMES):
        trait_maps = {trait: raw_annotation[lower_to_raw[trait]] for trait in TRAIT_NAMES}
        for trait, mapping in trait_maps.items():
            if not isinstance(mapping, dict):
                raise TypeError(f"Annotation trait map '{trait}' must be a dict.")

        common_keys = set.intersection(*[set(m.keys()) for m in trait_maps.values()])
        index: Dict[str, np.ndarray] = {}
        for key in common_keys:
            video_key = normalize_video_key(key)
            index[video_key] = np.asarray(
                [
                    trait_maps["openness"][key],
                    trait_maps["conscientiousness"][key],
                    trait_maps["extraversion"][key],
                    trait_maps["agreeableness"][key],
                    trait_maps["neuroticism"][key],
                ],
                dtype=np.float32,
            )
        return index

    # Alternate schema fallback: {video_name: [o, c, e, a, n]}
    fallback_index: Dict[str, np.ndarray] = {}
    for key, value in raw_annotation.items():
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 5:
            fallback_index[normalize_video_key(key)] = np.asarray(value[:5], dtype=np.float32)

    if fallback_index:
        return fallback_index

    raise ValueError(
        "Unsupported annotation schema. Expected trait dicts or video->5D score mapping."
    )


def build_transcription_index(raw_transcription: Optional[object]) -> Dict[str, str]:
    if raw_transcription is None:
        return {}

    if isinstance(raw_transcription, dict):
        transcription_index: Dict[str, str] = {}
        for key, value in raw_transcription.items():
            if value is None:
                text_value = ""
            else:
                text_value = str(value)
            transcription_index[normalize_video_key(key)] = text_value
        return transcription_index

    raise ValueError("Unsupported transcription schema. Expected dict mapping video->text.")


def safe_load_pickle(path: Optional[Path]) -> Optional[object]:
    if path is None:
        return None
    if not path.exists():
        return None
    return load_pickle(path)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return clean_text(text).split(" ")


@dataclass
class Vocabulary:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(cls, texts: Iterable[str], vocab_size: int) -> "Vocabulary":
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        special = [PAD_TOKEN, UNK_TOKEN]
        most_common = [word for word, _ in counter.most_common(max(vocab_size - len(special), 0))]
        itos = special + most_common
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str, seq_len: int) -> np.ndarray:
        tokens = tokenize(text)
        ids = [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in tokens[:seq_len]]
        if len(ids) < seq_len:
            ids.extend([self.stoi[PAD_TOKEN]] * (seq_len - len(ids)))
        return np.asarray(ids, dtype=np.int64)

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        itos = list(data["itos"])
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)


def _fallback_frame(image_size: int) -> np.ndarray:
    return np.zeros((image_size, image_size, 3), dtype=np.uint8)


def _largest_face_box(boxes: Sequence[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    return max(boxes, key=lambda b: b[2] * b[3])


def _center_square_crop(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0 : y0 + side, x0 : x0 + side]


def _extract_face_patch(frame: np.ndarray, face_detector: cv2.CascadeClassifier) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    if len(boxes) == 0:
        return _center_square_crop(frame)

    x, y, w, h = _largest_face_box(boxes)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return _center_square_crop(frame)
    return frame[y0:y1, x0:x1]


def _uniform_indices(total: int, count: int) -> np.ndarray:
    if total <= 1:
        return np.zeros((count,), dtype=np.int32)
    return np.linspace(0, max(total - 1, 0), num=count, dtype=np.int32)


def sample_video_frames(video_path: Path, num_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _uniform_indices(total_frames, num_frames)

    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)

    cap.release()
    return frames


def _resize_rgb(frame: np.ndarray, image_size: int) -> np.ndarray:
    resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)


def extract_audio_features(video_path: Path, audio_steps: int, audio_bins: int) -> np.ndarray:
    try:
        import librosa

        waveform, sample_rate = librosa.load(str(video_path), sr=16000, mono=True)
        if waveform is None or len(waveform) == 0:
            raise ValueError("Empty waveform")

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=audio_bins,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel + 1e-6).T.astype(np.float32)

    except Exception:
        return np.zeros((audio_steps, audio_bins), dtype=np.float32)

    if mel_db.shape[0] == 0:
        return np.zeros((audio_steps, audio_bins), dtype=np.float32)

    if mel_db.shape[0] < audio_steps:
        pad_count = audio_steps - mel_db.shape[0]
        pad_rows = np.repeat(mel_db[-1:, :], repeats=pad_count, axis=0)
        mel_db = np.vstack([mel_db, pad_rows])

    splits = np.array_split(mel_db, audio_steps)
    pooled = []
    for chunk in splits:
        if chunk.shape[0] == 0:
            pooled.append(pooled[-1] if pooled else np.zeros((audio_bins,), dtype=np.float32))
        else:
            pooled.append(chunk.mean(axis=0).astype(np.float32))

    return np.stack(pooled, axis=0)


def collect_split_samples(
    split_dir: Path,
    annotation_index: Dict[str, np.ndarray],
    transcription_index: Dict[str, str],
    limit: Optional[int] = None,
) -> List[Tuple[Path, np.ndarray, str]]:
    video_files = sorted(split_dir.glob("*.mp4"))
    samples: List[Tuple[Path, np.ndarray, str]] = []

    for video_path in video_files:
        key = normalize_video_key(video_path.name)
        if key not in annotation_index:
            continue
        label = annotation_index[key]
        transcript = transcription_index.get(key, "")
        samples.append((video_path, label, transcript))

    if limit is not None:
        samples = samples[:limit]
    return samples


def build_dataset_cache(
    samples: Sequence[Tuple[Path, np.ndarray, str]],
    vocab: Vocabulary,
    output_path: Path,
    image_size: int,
    num_frames: int,
    text_seq_len: int,
    audio_steps: int,
    audio_bins: int,
) -> None:
    if len(samples) == 0:
        raise ValueError(
            f"No samples found for split '{output_path.stem}'. "
            "Check that video filenames match the provided annotation file for this split."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    scene_data: List[np.ndarray] = []
    face_data: List[np.ndarray] = []
    audio_data: List[np.ndarray] = []
    text_data: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    video_ids: List[str] = []

    for video_path, target, transcript in tqdm(samples, desc=f"Preparing {output_path.stem}"):
        raw_frames = sample_video_frames(video_path, num_frames=num_frames)
        if len(raw_frames) == 0:
            raw_frames = [_fallback_frame(image_size) for _ in range(num_frames)]

        if len(raw_frames) < num_frames:
            raw_frames.extend([raw_frames[-1]] * (num_frames - len(raw_frames)))

        raw_frames = raw_frames[:num_frames]

        scene_frames = [_resize_rgb(frame, image_size=image_size) for frame in raw_frames]
        face_frames = [
            _resize_rgb(_extract_face_patch(frame, face_detector=face_detector), image_size=image_size)
            for frame in raw_frames
        ]

        scene_array = np.stack(scene_frames, axis=0).astype(np.uint8)
        face_array = np.stack(face_frames, axis=0).astype(np.uint8)
        audio_array = extract_audio_features(
            video_path=video_path,
            audio_steps=audio_steps,
            audio_bins=audio_bins,
        ).astype(np.float32)
        text_array = vocab.encode(transcript, seq_len=text_seq_len)

        scene_data.append(scene_array)
        face_data.append(face_array)
        audio_data.append(audio_array)
        text_data.append(text_array)
        targets.append(np.asarray(target, dtype=np.float32))
        video_ids.append(video_path.name)

    np.savez_compressed(
        output_path,
        video_ids=np.asarray(video_ids),
        scene=np.stack(scene_data, axis=0),
        face=np.stack(face_data, axis=0),
        audio=np.stack(audio_data, axis=0),
        text=np.stack(text_data, axis=0),
        targets=np.stack(targets, axis=0),
    )


class CachedMultimodalDataset(Dataset):
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.data = np.load(cache_path, allow_pickle=False, mmap_mode="r")

    def __len__(self) -> int:
        return int(self.data["targets"].shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene = torch.from_numpy(self.data["scene"][idx]).float().permute(0, 3, 1, 2) / 255.0
        face = torch.from_numpy(self.data["face"][idx]).float().permute(0, 3, 1, 2) / 255.0
        audio = torch.from_numpy(self.data["audio"][idx]).float()
        text = torch.from_numpy(self.data["text"][idx]).long()
        target = torch.from_numpy(self.data["targets"][idx]).float()

        return {
            "scene": scene,
            "face": face,
            "audio": audio,
            "text": text,
            "target": target,
        }
