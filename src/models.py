from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    vocab_size: int
    text_seq_len: int


class FrameCNNEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.proj(x)
        return x


class FrameTransformerEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 2,
        num_heads: int = 3,
        out_dim: int = 256,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.encoder(x)
        x = self.norm(x).mean(dim=1)
        x = self.proj(x)
        return x


class VisualBranch(nn.Module):
    """Dual-stream visual branch inspired by VGG16 + ViT fusion in the paper."""

    def __init__(self, image_size: int = 96):
        super().__init__()
        self.cnn_encoder = FrameCNNEncoder(out_dim=256)
        self.vit_encoder = FrameTransformerEncoder(image_size=image_size, out_dim=256)

        self.cnn_lstm1 = nn.LSTM(256, 128, batch_first=True)
        self.cnn_lstm2 = nn.LSTM(128, 64, batch_first=True)

        self.vit_lstm1 = nn.LSTM(256, 128, batch_first=True)
        self.vit_lstm2 = nn.LSTM(128, 64, batch_first=True)

        self.cnn_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )
        self.vit_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 5),
            nn.Sigmoid(),
        )

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # frames: [B, T, C, H, W]
        bsz, steps, channels, height, width = frames.shape
        flat = frames.reshape(bsz * steps, channels, height, width)

        cnn_feats = self.cnn_encoder(flat).reshape(bsz, steps, -1)
        vit_feats = self.vit_encoder(flat).reshape(bsz, steps, -1)

        cnn_seq, _ = self.cnn_lstm1(cnn_feats)
        cnn_seq, _ = self.cnn_lstm2(cnn_seq)

        vit_seq, _ = self.vit_lstm1(vit_feats)
        vit_seq, _ = self.vit_lstm2(vit_seq)

        cnn_last = self.cnn_fc(cnn_seq[:, -1, :])
        vit_last = self.vit_fc(vit_seq[:, -1, :])

        merged = 0.5 * (cnn_last + vit_last)
        pred = self.head(merged)

        seq = 0.5 * (cnn_seq + vit_seq)
        return pred, seq


class AudioBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(128, 32, kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2)
        self.drop = nn.Dropout(0.3)

        self.lstm1 = nn.LSTM(64, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 5),
            nn.Sigmoid(),
        )

    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # audio: [B, 15, 128]
        x = audio.transpose(1, 2)
        x = self.drop(F.relu(self.conv1(x)))
        x = self.drop(F.relu(self.conv2(x)))
        x = x.transpose(1, 2)

        seq, _ = self.lstm1(x)
        seq, _ = self.lstm2(seq)

        pred = self.head(seq[:, -1, :])
        return pred, seq


class TextBranch(nn.Module):
    def __init__(self, vocab_size: int, text_seq_len: int):
        super().__init__()
        self.text_seq_len = text_seq_len
        self.embedding = nn.Embedding(vocab_size, 100, padding_idx=0)

        self.conv_x1 = nn.Conv1d(100, 16, kernel_size=3)
        self.conv_x2 = nn.Conv1d(16, 8, kernel_size=3)

        self.conv_y1 = nn.Conv1d(100, 32, kernel_size=3)
        self.conv_y2 = nn.Conv1d(32, 16, kernel_size=3)

        x_flat_len = max(text_seq_len - 4, 1) * 8
        y_flat_len = max(text_seq_len - 4, 1) * 16

        self.x_fc = nn.Sequential(nn.Flatten(), nn.Linear(x_flat_len, 50), nn.ReLU(inplace=True))
        self.y_fc = nn.Sequential(nn.Flatten(), nn.Linear(y_flat_len, 50), nn.ReLU(inplace=True))

        self.head = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 5),
            nn.Sigmoid(),
        )

        self.seq_proj = nn.Linear(100, 64)

    def forward(self, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # text: [B, 50]
        emb = self.embedding(text)
        emb_c = emb.transpose(1, 2)

        x = F.relu(self.conv_x1(emb_c))
        x = F.relu(self.conv_x2(x))
        x = self.x_fc(x)

        y = F.relu(self.conv_y1(emb_c))
        y = F.relu(self.conv_y2(y))
        y = self.y_fc(y)

        merged = torch.cat([x, y], dim=1)
        pred = self.head(merged)

        seq = self.seq_proj(emb)
        return pred, seq


class SceneModel(nn.Module):
    def __init__(self, image_size: int):
        super().__init__()
        self.scene = VisualBranch(image_size=image_size)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred, _ = self.scene(batch["scene"])
        return pred


class FaceModel(nn.Module):
    def __init__(self, image_size: int):
        super().__init__()
        self.face = VisualBranch(image_size=image_size)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred, _ = self.face(batch["face"])
        return pred


class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio = AudioBranch()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred, _ = self.audio(batch["audio"])
        return pred


class TextModel(nn.Module):
    def __init__(self, vocab_size: int, text_seq_len: int):
        super().__init__()
        self.text = TextBranch(vocab_size=vocab_size, text_seq_len=text_seq_len)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred, _ = self.text(batch["text"])
        return pred


class EarlyFusionModel(nn.Module):
    def __init__(self, vocab_size: int, text_seq_len: int, image_size: int):
        super().__init__()
        self.scene = VisualBranch(image_size=image_size)
        self.face = VisualBranch(image_size=image_size)
        self.audio = AudioBranch()
        self.text = TextBranch(vocab_size=vocab_size, text_seq_len=text_seq_len)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        p_scene, _ = self.scene(batch["scene"])
        p_face, _ = self.face(batch["face"])
        p_audio, _ = self.audio(batch["audio"])
        p_text, _ = self.text(batch["text"])
        pred = (p_scene + p_face + p_audio + p_text) / 4.0
        return pred


class ModelFusionModel(nn.Module):
    def __init__(self, vocab_size: int, text_seq_len: int, image_size: int):
        super().__init__()
        self.scene = VisualBranch(image_size=image_size)
        self.face = VisualBranch(image_size=image_size)
        self.audio = AudioBranch()
        self.text = TextBranch(vocab_size=vocab_size, text_seq_len=text_seq_len)

        self.fusion = nn.Sequential(
            nn.Linear(20, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100, 5),
            nn.Sigmoid(),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        p_scene, _ = self.scene(batch["scene"])
        p_face, _ = self.face(batch["face"])
        p_audio, _ = self.audio(batch["audio"])
        p_text, _ = self.text(batch["text"])

        x = torch.cat([p_scene, p_face, p_audio, p_text], dim=1)
        pred = self.fusion(x)
        return pred


class SelfAttentionFusionModel(nn.Module):
    def __init__(self, vocab_size: int, text_seq_len: int, image_size: int):
        super().__init__()
        self.scene = VisualBranch(image_size=image_size)
        self.face = VisualBranch(image_size=image_size)
        self.audio = AudioBranch()
        self.text = TextBranch(vocab_size=vocab_size, text_seq_len=text_seq_len)

        self.scene_proj = nn.Linear(64, 64)
        self.face_proj = nn.Linear(64, 64)
        self.audio_proj = nn.Linear(64, 64)
        self.text_proj = nn.Linear(64, 64)

        self.scene_attn = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)
        self.face_attn = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)
        self.audio_attn = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)
        self.text_attn = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 5),
            nn.Sigmoid(),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        _, s_seq = self.scene(batch["scene"])
        _, f_seq = self.face(batch["face"])
        _, a_seq = self.audio(batch["audio"])
        _, t_seq = self.text(batch["text"])

        s = self.scene_proj(s_seq)
        f = self.face_proj(f_seq)
        a = self.audio_proj(a_seq)
        t = self.text_proj(t_seq)

        s, _ = self.scene_attn(s, s, s)
        f, _ = self.face_attn(f, f, f)
        a, _ = self.audio_attn(a, a, a)
        t, _ = self.text_attn(t, t, t)

        merged = torch.cat([s, f, a, t], dim=1)
        pooled = merged.mean(dim=1)
        pred = self.head(pooled)
        return pred


class CrossAttentionFusionModel(nn.Module):
    def __init__(self, vocab_size: int, text_seq_len: int, image_size: int):
        super().__init__()
        self.scene = VisualBranch(image_size=image_size)
        self.face = VisualBranch(image_size=image_size)
        self.audio = AudioBranch()
        self.text = TextBranch(vocab_size=vocab_size, text_seq_len=text_seq_len)

        self.video_proj = nn.Linear(64, 64)
        self.audio_proj = nn.Linear(64, 64)
        self.text_proj = nn.Linear(64, 64)

        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 5),
            nn.Sigmoid(),
        )

    def _ca(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(q, kv, kv)
        return out

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        _, s_seq = self.scene(batch["scene"])
        _, f_seq = self.face(batch["face"])
        _, a_seq = self.audio(batch["audio"])
        _, t_seq = self.text(batch["text"])

        v = 0.5 * (self.video_proj(s_seq) + self.video_proj(f_seq))
        a = self.audio_proj(a_seq)
        t = self.text_proj(t_seq)

        a1 = self._ca(v, a)  # video -> audio
        a2 = self._ca(a, v)  # audio -> video
        a3 = self._ca(v, t)  # video -> text
        a4 = self._ca(t, v)  # text -> video
        a5 = self._ca(a, t)  # audio -> text
        a6 = self._ca(t, a)  # text -> audio

        merged = torch.cat([a1, a2, a3, a4, a5, a6], dim=1)
        pooled = merged.mean(dim=1)
        pred = self.head(pooled)
        return pred


SUPPORTED_MODELS = {
    "scene",
    "face",
    "audio",
    "text",
    "early_fusion",
    "model_fusion",
    "self_attention",
    "cross_attention",
}


def build_model(spec: ModelSpec, image_size: int) -> nn.Module:
    model_type = spec.model_type.lower()
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model type: {spec.model_type}")

    if model_type == "scene":
        return SceneModel(image_size=image_size)
    if model_type == "face":
        return FaceModel(image_size=image_size)
    if model_type == "audio":
        return AudioModel()
    if model_type == "text":
        return TextModel(vocab_size=spec.vocab_size, text_seq_len=spec.text_seq_len)
    if model_type == "early_fusion":
        return EarlyFusionModel(
            vocab_size=spec.vocab_size,
            text_seq_len=spec.text_seq_len,
            image_size=image_size,
        )
    if model_type == "model_fusion":
        return ModelFusionModel(
            vocab_size=spec.vocab_size,
            text_seq_len=spec.text_seq_len,
            image_size=image_size,
        )
    if model_type == "self_attention":
        return SelfAttentionFusionModel(
            vocab_size=spec.vocab_size,
            text_seq_len=spec.text_seq_len,
            image_size=image_size,
        )

    return CrossAttentionFusionModel(
        vocab_size=spec.vocab_size,
        text_seq_len=spec.text_seq_len,
        image_size=image_size,
    )
