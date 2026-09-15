"""Strict checkpoint loading and provenance checks for AttentionCNN maximum pooling."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    from .model import AttentionCNNReconstruction, parameter_counts
except ImportError:  # Direct script imports.
    from model import AttentionCNNReconstruction, parameter_counts


FINAL_CHECKPOINT_SHA256 = (
    "791abd3539ef98e4fb7c45514134045b9cfed3a4992023f809ffea4971d538ad"
)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_load(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    return payload


def _validate_metadata(metadata: Any) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint metadata is missing")
    if metadata.get("geometry_target") != "legacy12_min_preserving_pair":
        raise ValueError("checkpoint uses an unsupported geometry target")
    if list(metadata.get("input_channels", [])) != ["relative_distance_transform"]:
        raise ValueError("checkpoint is not the single-channel AttentionCNN model")
    config = metadata.get("code_config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint code_config is missing")
    model_config = config.get("model_config")
    if (
        not isinstance(model_config, Mapping)
        or int(model_config.get("input_channels", -1)) != 1
    ):
        raise ValueError("checkpoint model_config does not specify one input channel")
    head = model_config.get("radius_head_configuration")
    if not isinstance(head, Mapping):
        raise ValueError("checkpoint radius-head configuration is missing")
    expected = {
        "radius_head_type": "multiscale_attentive_stat_pool_radius_v1",
        "representation": "free_log_ratio",
        "num_points": 12,
        "num_views": 2,
        "d_model": 64,
        "hidden_dim": 256,
        "depth": 3,
        "use_fine_layer2_tokens": True,
    }
    for key, value in expected.items():
        if head.get(key) != value:
            raise ValueError(f"checkpoint radius-head mismatch for {key!r}")
    if list(head.get("pooling", [])) != ["spatial_max"]:
        raise ValueError("checkpoint is not the selected maximum-pooling model")
    stats = metadata.get("train_statistics")
    if not isinstance(stats, Mapping):
        raise ValueError("checkpoint training statistics are missing")
    if float(stats.get("log_r0_std", 0.0)) <= 0:
        raise ValueError("checkpoint log-radius statistics are invalid")
    return metadata


def load_reconstruction_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    channels_last: bool = False,
    expected_sha256: str | None = None,
) -> tuple[AttentionCNNReconstruction, dict[str, Any]]:
    """Load a compatible AttentionCNN checkpoint with a strict state-dict check."""

    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha.lower() != expected_sha256.lower():
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )
    payload = _torch_load(path)
    metadata = _validate_metadata(payload.get("metadata"))
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint model_state_dict is missing")
    stats = metadata["train_statistics"]
    model = AttentionCNNReconstruction(
        log_r0_mean=float(stats["log_r0_mean"]),
        log_r0_std=float(stats["log_r0_std"]),
    )
    model.load_state_dict(state, strict=True)
    model.eval().to(torch.device(device))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    audit = {
        "path": str(path),
        "sha256": actual_sha,
        "epoch": int(payload.get("epoch", -1)),
        "selection": payload.get("selection"),
        "model_version": metadata.get("model_version"),
        "strict_state_dict_load": True,
        "parameter_counts": parameter_counts(model),
        "metadata": dict(metadata),
    }
    return model, audit


__all__ = ["FINAL_CHECKPOINT_SHA256", "load_reconstruction_checkpoint", "sha256_file"]
