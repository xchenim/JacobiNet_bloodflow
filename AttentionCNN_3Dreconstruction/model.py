"""AttentionCNN maximum-pooling model for two-view coronary reconstruction.

This file contains only the architecture selected for release.  Module and
parameter names intentionally match the locked training checkpoint so that it
can be loaded with ``strict=True``.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torchvision import models


NUM_POINTS = 12
NUM_VIEWS = 2
XYZ_SCALE_M = 0.01


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(
            self.shared(self.avg_pool(value)) + self.shared(self.max_pool(value))
        )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        average = value.mean(dim=1, keepdim=True)
        maximum = value.amax(dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat((average, maximum), dim=1)))


class CBAM(nn.Module):
    """Residual CBAM used by the locked model."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(7)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + value * self.channel(value)
        return value + value * self.spatial(value)


class LegacyChannelAdapter(nn.Module):
    """Map one relative-distance-transform channel to three ResNet channels."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError(
                "AttentionCNN expects exactly one relative-DT input channel"
            )
        self.in_channels = in_channels
        self.legacy_channel = 0
        self.projection = nn.Conv2d(1, 3, kernel_size=1, bias=True)
        self.reset_to_legacy()

    def reset_to_legacy(self) -> None:
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.bias.zero_()
            self.projection.weight[:, 0, 0, 0] = 1.0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != 1:
            raise ValueError("each view must have shape (batch, 1, height, width)")
        return self.projection(value)


class SharedResNet50Encoder(nn.Module):
    """Shared ImageNet-V1 ResNet-50 with a layer-4 CBAM block."""

    def __init__(self, *, backbone_weights: Any = None) -> None:
        super().__init__()
        base = models.resnet50(weights=backbone_weights)
        self.input_adapter = LegacyChannelAdapter(1)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.cbam = CBAM(2048)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | int]:
        if images.ndim != 5:
            raise ValueError("images must have shape (batch, views, 1, height, width)")
        batch, views, channels, height, width = images.shape
        if views != NUM_VIEWS or channels != 1:
            raise ValueError("AttentionCNN expects two views with one channel per view")
        value = images.reshape(batch * views, channels, height, width)
        value = self.input_adapter(value)
        value = self.stem(value)
        value = self.layer1(value)
        layer2 = self.layer2(value)
        layer3 = self.layer3(layer2)
        layer4 = self.cbam(self.layer4(layer3))
        return {
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4,
            "batch": batch,
            "views": views,
        }


def _normalised_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, depth: int = 3
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(depth):
        layers.extend(
            (nn.Linear(current, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        )
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class MaxPoolRadiusHead(nn.Module):
    """Layer-2/3/4 spatial-maximum residual radius head.

    Disabled average and attention slots are retained as zero-valued context
    so the state dictionary and fixed-capacity ablation checkpoint remain
    exactly compatible with the selected model.
    """

    def __init__(self, base_head: nn.Module) -> None:
        super().__init__()
        self.base_head = base_head
        self.num_points = NUM_POINTS
        self.num_views = NUM_VIEWS
        self.d_model = 64
        self.hidden_dim = 256
        self.depth = 3
        self.use_layer2 = True
        self.head_init_seed = 20260805
        self.pooling_modes = ("spatial_max",)
        self.register_buffer(
            "pooling_mask", torch.tensor([0.0, 1.0, 0.0]), persistent=False
        )
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, 64, kernel_size=1) for channels in (512, 1024, 2048)
        )
        # Preserve checkpoint parameter slots and their zero-gradient paths.
        # The published pooling mask selects only the maximum statistic.
        self.attention_scores = nn.ModuleList(
            nn.Conv2d(64, 1, kernel_size=1) for _ in range(3)
        )
        self.residual = _normalised_mlp(1152, 256, NUM_POINTS, depth=3)
        final = self.residual[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("maximum-pooling residual must end in Linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _pool_level(
        self,
        value: torch.Tensor,
        projection: nn.Module,
        score_layer: nn.Module,
        batch: int,
        views: int,
    ) -> torch.Tensor:
        value = projection(value)
        channels = value.shape[1]
        flat = value.flatten(2)
        average = flat.mean(dim=2)
        maximum = flat.amax(dim=2)
        scores = score_layer(value).flatten(2).float()
        weights = torch.softmax(scores, dim=2).to(dtype=value.dtype)
        attentive = torch.sum(flat * weights, dim=2)
        statistics = (average, maximum, attentive)
        pooled = torch.cat(
            tuple(
                statistic * self.pooling_mask[index].to(dtype=value.dtype)
                for index, statistic in enumerate(statistics)
            ),
            dim=1,
        )
        return pooled.reshape(batch, views * 3 * channels)

    def forward(
        self, features: Mapping[str, torch.Tensor | int], pooled: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, views = int(features["batch"]), int(features["views"])
        levels = (features["layer2"], features["layer3"], features["layer4"])
        if any(not isinstance(level, torch.Tensor) for level in levels):
            raise TypeError("all configured feature levels must be tensors")
        contexts = [
            self._pool_level(level, projection, score, batch, views)
            for level, projection, score in zip(
                levels, self.projections, self.attention_scores
            )
        ]
        output = self.base_head(pooled) + self.residual(torch.cat(contexts, dim=1))
        return output[:, :1], output[:, 1:]


class ReconstructionDecoder(nn.Module):
    def __init__(self, *, advanced_radius_head: bool = True) -> None:
        super().__init__()
        input_dim = 2048 * NUM_VIEWS
        self.num_points = NUM_POINTS
        self.num_views = NUM_VIEWS
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.xyz_head = _normalised_mlp(input_dim, 512, NUM_POINTS * 3)
        base_radius_head = _normalised_mlp(input_dim, 256, NUM_POINTS)
        self.radius_head: nn.Module = (
            MaxPoolRadiusHead(base_radius_head)
            if advanced_radius_head
            else base_radius_head
        )

    def forward(
        self, features: Mapping[str, torch.Tensor | int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer4 = features["layer4"]
        if not isinstance(layer4, torch.Tensor):
            raise TypeError("layer4 must be a tensor")
        batch, views = int(features["batch"]), int(features["views"])
        pooled = self.pool(layer4).flatten(1).reshape(batch, views * 2048)
        xyz_normalised = self.xyz_head(pooled).reshape(batch, NUM_POINTS, 3)
        if isinstance(self.radius_head, MaxPoolRadiusHead):
            scale_standardised, ratio_tail = self.radius_head(features, pooled)
        else:
            radius_latent = self.radius_head(pooled)
            scale_standardised = radius_latent[:, :1]
            ratio_tail = radius_latent[:, 1:]
        return xyz_normalised, scale_standardised, ratio_tail


class AttentionCNNReconstruction(nn.Module):
    """Locked AttentionCNN model returning 12 paired ``(x, y, z, radius)`` points."""

    def __init__(
        self,
        *,
        log_r0_mean: float,
        log_r0_std: float,
        backbone_weights: Any = None,
        advanced_radius_head: bool = True,
    ) -> None:
        super().__init__()
        if log_r0_std <= 0:
            raise ValueError("log_r0_std must be positive")
        self.num_points = NUM_POINTS
        self.num_views = NUM_VIEWS
        self.decoder_type = "legacy_mlp"
        self.radius_parameterization = "log_r0_log_ratio"
        self.xyz_scale_m = XYZ_SCALE_M
        self.log_ratio_limit = None
        self.register_buffer("log_r0_mean", torch.tensor(float(log_r0_mean)))
        self.register_buffer("log_r0_std", torch.tensor(float(log_r0_std)))
        # Retained for strict compatibility with the published checkpoint.
        self.register_buffer("legacy_r_min_m", torch.tensor(0.0004))
        self.register_buffer("legacy_r_max_m", torch.tensor(0.0025))
        self.encoder = SharedResNet50Encoder(backbone_weights=backbone_weights)
        self.decoder = ReconstructionDecoder(advanced_radius_head=advanced_radius_head)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(images)
        xyz_normalised, scale_standardised, ratio_tail = self.decoder(features)
        log_r0 = (
            self.log_r0_mean.float()
            + self.log_r0_std.float() * scale_standardised.squeeze(-1).float()
        ).clamp(-20.0, 0.0)
        log_ratio = torch.cat(
            (torch.zeros_like(log_r0[:, None]), ratio_tail.float()), dim=1
        )
        log_radius = (log_r0[:, None] + log_ratio).clamp(-20.0, 0.0)
        radius = torch.exp(log_radius).unsqueeze(-1)
        xyz = xyz_normalised * self.xyz_scale_m
        radius_2d = radius[:, :, 0].float().clamp_min(1e-12)
        severity = 100.0 * (
            1.0 - radius_2d.amin(dim=1) / radius_2d[:, 0].clamp_min(1e-12)
        )
        grade = torch.bucketize(
            severity.detach(), severity.new_tensor((25.0, 50.0, 70.0)), right=True
        )
        return {
            "xyz": xyz,
            "radius": radius,
            "xyz_normalised": xyz_normalised,
            "radius_latent": torch.cat(
                (scale_standardised.float(), ratio_tail.float()), dim=1
            ),
            "log_radius_raw": log_r0[:, None] + log_ratio,
            "log_r0": log_r0,
            "log_ratio": log_ratio,
            "severity": severity,
            "grade": grade,
        }


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "radius_head": sum(
            parameter.numel() for parameter in model.decoder.radius_head.parameters()
        ),
    }


def install_max_pool_radius_head(
    model: AttentionCNNReconstruction, *, head_init_seed: int = 20260805
) -> MaxPoolRadiusHead:
    """Install the deterministic Stage-2 head while preserving global RNG state."""

    current = model.decoder.radius_head
    if isinstance(current, MaxPoolRadiusHead):
        return current
    if not isinstance(current, nn.Sequential):
        raise TypeError("Stage-2 installation requires the legacy Sequential head")
    reference = next(current.parameters(), None)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    try:
        torch.manual_seed(int(head_init_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(head_init_seed))
        installed = MaxPoolRadiusHead(current)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    if reference is not None:
        installed.to(device=reference.device, dtype=reference.dtype)
    model.decoder.radius_head = installed
    return installed


__all__ = [
    "AttentionCNNReconstruction",
    "MaxPoolRadiusHead",
    "install_max_pool_radius_head",
    "parameter_counts",
]
