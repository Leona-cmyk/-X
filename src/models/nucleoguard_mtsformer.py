from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SensorEmbedding(nn.Module):
    """Fuse scalar sensor values with learnable sensor identity embeddings."""

    def __init__(self, num_sensors: int, sensor_dim: int) -> None:
        super().__init__()
        self.sensor_embedding = nn.Parameter(torch.randn(num_sensors, sensor_dim) * 0.02)
        self.value_projection = nn.Linear(1, sensor_dim)
        self.fusion = nn.Sequential(nn.LayerNorm(sensor_dim), nn.GELU(), nn.Linear(sensor_dim, sensor_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, sensors]
        value_token = self.value_projection(x.unsqueeze(-1))
        sensor_token = self.sensor_embedding.view(1, 1, x.shape[-1], -1)
        return self.fusion(value_token + sensor_token)


class SensorAwareGating(nn.Module):
    """Learn dynamic channel coupling weights from embedded sensor tokens."""

    def __init__(self, sensor_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(sensor_dim, sensor_dim // 2), nn.GELU(), nn.Linear(sensor_dim // 2, 1))

    def forward(self, sensor_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # sensor_tokens: [batch, time, sensors, dim]
        logits = self.gate(sensor_tokens).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        fused = torch.sum(sensor_tokens * weights.unsqueeze(-1), dim=2)
        return fused, weights


class TemporalPatchEmbedding(nn.Module):
    """Patchify temporal tokens with a learnable Conv1D projection."""

    def __init__(self, sensor_dim: int, model_dim: int, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.proj = nn.Conv1d(sensor_dim, model_dim, kernel_size=patch_size, stride=patch_stride)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, dim]
        if x.shape[1] < self.patch_size:
            pad = self.patch_size - x.shape[1]
            x = F.pad(x, (0, 0, 0, pad))
        z = self.proj(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(z)


class LocalTemporalConvBlock(nn.Module):
    """Depthwise separable temporal convolution with residual path."""

    def __init__(self, model_dim: int, kernel_size: int = 5, dropout: float = 0.15) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(model_dim)
        self.depthwise = nn.Conv1d(model_dim, model_dim, kernel_size, padding=padding, groups=model_dim)
        self.pointwise = nn.Conv1d(model_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.norm(x).transpose(1, 2)
        z = self.pointwise(self.act(self.depthwise(z))).transpose(1, 2)
        return residual + self.dropout(z)


class SpatioTemporalTransformerEncoder(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AttentionPooling(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(model_dim, model_dim // 2), nn.Tanh(), nn.Linear(model_dim // 2, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class MultiTaskHeads(nn.Module):
    def __init__(self, model_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.classification_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        )
        self.severity_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
        )

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.classification_head(embedding), self.severity_head(embedding).squeeze(-1)


class NucleoGuardMTSFormer(nn.Module):
    """Multi-task Spatio-Temporal Transformer for NPP accident diagnosis."""

    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        sensor_dim: int = 32,
        model_dim: int = 96,
        patch_size: int = 8,
        patch_stride: int = 4,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15,
        max_patches: int = 512,
    ) -> None:
        super().__init__()
        self.num_sensors = num_sensors
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.sensor_embedding = SensorEmbedding(num_sensors, sensor_dim)
        self.sensor_gate = SensorAwareGating(sensor_dim)
        self.patch_embedding = TemporalPatchEmbedding(sensor_dim, model_dim, patch_size, patch_stride)
        self.raw_patch_embedding = nn.Conv1d(num_sensors, model_dim, kernel_size=patch_size, stride=patch_stride)
        self.raw_norm = nn.LayerNorm(model_dim)
        self.local_conv = LocalTemporalConvBlock(model_dim, dropout=dropout)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_patches, model_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.encoder = SpatioTemporalTransformerEncoder(model_dim, num_heads, num_layers, dropout)
        self.pool = AttentionPooling(model_dim)
        self.heads = MultiTaskHeads(model_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        sensor_tokens = self.sensor_embedding(x)
        fused_time, sensor_gate = self.sensor_gate(sensor_tokens)
        patches = self.patch_embedding(fused_time)
        raw_patches = self.raw_norm(self.raw_patch_embedding(x.transpose(1, 2)).transpose(1, 2))
        min_len = min(patches.shape[1], raw_patches.shape[1])
        patches = patches[:, :min_len, :] + raw_patches[:, :min_len, :]
        patches = self.local_conv(patches)
        patches = patches + self.pos_embedding[:, : patches.shape[1], :]
        encoded = self.encoder(patches)
        embedding, temporal_attention = self.pool(encoded)
        logits, severity = self.heads(embedding)
        return {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=-1),
            "severity": severity,
            "temporal_attention": temporal_attention,
            "sensor_attention": sensor_gate.mean(dim=1),
            "embedding": embedding,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
