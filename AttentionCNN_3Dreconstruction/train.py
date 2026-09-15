"""Train the released AttentionCNN maximum-pooling model with the locked two-stage recipe.

Stage 1 optimizes the encoder, CBAM, input adapter, and XYZ head with scaled
centreline MSE and natural sampling. Stage 2 strict-loads Stage 1, installs the
maximum-pooling radius head deterministically, freezes encoder/XYZ parameters,
and optimizes physical-radius MSE with grade-by-tortuosity balanced sampling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)
from torchvision.models import ResNet50_Weights

try:
    from .checkpoint import sha256_file
    from .dataset import (
        SyntheticReconstructionDataset,
        severity_grade,
        severity_percent,
        tortuosity,
    )
    from .model import (
        AttentionCNNReconstruction,
        MaxPoolRadiusHead,
        install_max_pool_radius_head,
        parameter_counts,
    )
except ImportError:  # Direct script execution.
    from checkpoint import sha256_file
    from dataset import (
        SyntheticReconstructionDataset,
        severity_grade,
        severity_percent,
        tortuosity,
    )
    from model import (
        AttentionCNNReconstruction,
        MaxPoolRadiusHead,
        install_max_pool_radius_head,
        parameter_counts,
    )


SEED = 20260805
NUM_POINTS = 12
XYZ_SCALE_M = 0.01
RADIUS_SCALE_M = 0.001
GRADE_SELECTION_MAX_MAE_R_MM = 0.04269
GRADE_SELECTION_MAX_SEVERE_ERROR_RATE = 0.1598
STREAM_OFFSETS = {
    "xyz_natural": 1000,
    "radius_grade_tortuosity": 2100,
    "valid": 3000,
}
MODEL_VERSION = "attentioncnn-mscbam-v1"
GEOMETRY_TARGET = "legacy12_min_preserving_pair"


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return json_safe(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {key: json_safe(value) for key, value in row.items()} for row in rows
        )
    temporary.replace(path)


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seed_everything(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_output(path: Path) -> Path:
    output = path.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def load_statistics(
    path: Path | None, train_dataset: SyntheticReconstructionDataset
) -> dict[str, Any]:
    if path is not None:
        payload = json.loads(path.resolve().read_text(encoding="utf-8"))
        log_stats = payload.get("labels", {}).get("log_r0", {})
        mean, std = float(log_stats.get("mean", math.nan)), float(
            log_stats.get("std", math.nan)
        )
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError("statistics must contain valid labels.log_r0 mean/std")
        return payload
    values = np.asarray(
        [
            math.log(float(train_dataset.raw_label(case_id)[0, 3]))
            for case_id in train_dataset.case_ids
        ],
        dtype=np.float64,
    )
    return {
        "schema_version": 1,
        "labels": {
            "log_r0": {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "count": int(values.size),
            }
        },
        "computed_from": "training split inlet radii only",
    }


def _loader_options(batch_size: int, workers: int, worker_seed: int) -> dict[str, Any]:
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(worker_seed),
        "drop_last": False,
    }
    if workers > 0:
        options["prefetch_factor"] = 2
    return options


def grade_tortuosity_weights(
    dataset: SyntheticReconstructionDataset,
) -> tuple[torch.Tensor, dict[str, Any]]:
    grades: list[int] = []
    severities: list[float] = []
    tortuosities: list[float] = []
    for case_id in dataset.case_ids:
        label_path = dataset.split_root / case_id / f"{case_id}_stenosis.npy"
        label = np.asarray(np.load(label_path, allow_pickle=False), dtype=np.float64)
        if label.ndim == 3 and label.shape[0] == 1:
            label = label[0]
        severity = severity_percent(label[:, 3])
        grades.append(severity_grade(severity))
        severities.append(severity)
        tortuosities.append(tortuosity(label[:, :3]))
    grade_array = np.asarray(grades, dtype=np.int64)
    tortuosity_array = np.asarray(tortuosities, dtype=np.float64)
    grade_counts = np.bincount(grade_array, minlength=4)
    if grade_counts.shape != (4,) or np.any(grade_counts == 0):
        raise ValueError(
            "Stage 2 requires all four grades; got " + str(grade_counts.tolist())
        )
    raw_edges = np.quantile(tortuosity_array, [0.25, 0.5, 0.75], method="linear")
    edges = np.unique(raw_edges)
    edges = edges[(edges > tortuosity_array.min()) & (edges < tortuosity_array.max())]
    bins = np.searchsorted(edges, tortuosity_array, side="right").astype(np.int64)
    cells = np.zeros((4, len(edges) + 1), dtype=np.int64)
    np.add.at(cells, (grade_array, bins), 1)
    weights = np.empty(len(dataset), dtype=np.float64)
    for grade in range(4):
        nonempty = cells[grade] > 0
        kinds = int(nonempty.sum())
        mask = grade_array == grade
        selected = cells[grade, bins[mask]]
        weights[mask] = 1.0 / (kinds * selected)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise RuntimeError("invalid Stage-2 sampling weights")
    audit = {
        "name": "radius_grade_tortuosity",
        "kind": "grade_tortuosity",
        "replacement": True,
        "num_samples_per_epoch": len(dataset),
        "weights_min": float(weights.min()),
        "weights_max": float(weights.max()),
        "weights_sum": float(weights.sum()),
        "grade_counts": grade_counts.tolist(),
        "tortuosity_quartile_edges": edges.tolist(),
        "grade_by_tortuosity_cell_counts": cells.tolist(),
        "severity_range_percent": [
            float(np.min(severities)),
            float(np.max(severities)),
        ],
    }
    return torch.as_tensor(weights, dtype=torch.double), audit


def make_loaders(
    train_dataset: SyntheticReconstructionDataset,
    valid_dataset: SyntheticReconstructionDataset,
    *,
    stage: str,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    if stage == "xyz":
        order_seed = seed + STREAM_OFFSETS["xyz_natural"]
        sampler = RandomSampler(
            train_dataset,
            replacement=False,
            generator=torch.Generator().manual_seed(order_seed),
        )
        audit = {
            "name": "xyz_natural",
            "kind": "natural",
            "replacement": False,
            "num_samples_per_epoch": len(train_dataset),
        }
    elif stage == "radius":
        order_seed = seed + STREAM_OFFSETS["radius_grade_tortuosity"]
        weights, audit = grade_tortuosity_weights(train_dataset)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(order_seed),
        )
    else:
        raise ValueError(f"unknown stage {stage!r}")
    audit.update(order_seed=order_seed, worker_seed=order_seed + 50000)
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=False,
        **_loader_options(batch_size, workers, order_seed + 50000),
    )
    valid_order_seed = seed + STREAM_OFFSETS["valid"]
    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        shuffle=False,
        **_loader_options(batch_size, workers, valid_order_seed + 50000),
    )
    return train_loader, valid_loader, audit


def set_trainable(model: AttentionCNNReconstruction, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if stage == "xyz":
        for module in (model.encoder, model.decoder.xyz_head):
            for parameter in module.parameters():
                parameter.requires_grad = True
    elif stage == "radius":
        if not isinstance(model.decoder.radius_head, MaxPoolRadiusHead):
            raise TypeError("Stage 2 requires the installed maximum-pooling head")
        for parameter in model.decoder.radius_head.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"unknown stage {stage!r}")


def _group(name: str, parameters: Sequence[nn.Parameter], lr: float):
    selected = [parameter for parameter in parameters if parameter.requires_grad]
    if not selected:
        return None
    return {"name": name, "params": selected, "lr": lr, "base_lr": lr}


def make_optimizer(
    model: AttentionCNNReconstruction,
    *,
    stage: str,
    lr_head: float,
    lr_layer4: float,
    lr_backbone: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if stage == "radius":
        groups = [
            _group(
                "heads_and_input_adapter",
                list(model.decoder.radius_head.parameters()),
                lr_head,
            )
        ]
    else:
        groups = [
            _group(
                "heads_and_input_adapter",
                list(model.decoder.xyz_head.parameters())
                + list(model.encoder.input_adapter.parameters()),
                lr_head,
            ),
            _group(
                "layer4_cbam",
                list(model.encoder.layer4.parameters())
                + list(model.encoder.cbam.parameters()),
                lr_layer4,
            ),
            _group(
                "backbone_early",
                list(model.encoder.stem.parameters())
                + list(model.encoder.layer1.parameters())
                + list(model.encoder.layer2.parameters())
                + list(model.encoder.layer3.parameters()),
                lr_backbone,
            ),
        ]
    return torch.optim.AdamW(
        [group for group in groups if group is not None],
        weight_decay=weight_decay,
    )


def audit_trainability(
    model: AttentionCNNReconstruction,
    optimizer: torch.optim.Optimizer,
    stage: str,
) -> dict[str, Any]:
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_ids = {id(parameter) for _, parameter in trainable}
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    if len(optimizer_ids) != len(optimizer_parameters):
        raise RuntimeError("optimizer contains duplicate parameters")
    if optimizer_ids != trainable_ids:
        raise RuntimeError("optimizer does not exactly match trainable parameters")
    names = [name for name, _ in trainable]
    if stage == "radius" and any(
        not name.startswith("decoder.radius_head.") for name in names
    ):
        raise RuntimeError("Stage 2 exposed a parameter outside the radius head")
    return {
        "stage": stage,
        "optimizer_matches_trainable_parameters_exactly": True,
        "trainable_parameter_tensor_count": len(names),
        "trainable_scalar_parameter_count": sum(
            parameter.numel() for _, parameter in trainable
        ),
        "trainable_parameter_names_sha256": sha256_json(names),
        "encoder_and_xyz_frozen_in_radius_stage": (
            stage != "radius"
            or all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if name.startswith(("encoder.", "decoder.xyz_head."))
            )
        ),
    }


def freeze_batchnorm_stats(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


def prepare_images(
    images: torch.Tensor, device: torch.device, channels_last: bool
) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if channels_last:
        batch, views, channels, height, width = images.shape
        images = (
            images.reshape(batch * views, channels, height, width)
            .contiguous(memory_format=torch.channels_last)
            .view(batch, views, channels, height, width)
        )
    return images


def to_device_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def set_epoch_learning_rates(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    epochs: int,
    *,
    warmup_fraction: float,
    min_lr: float,
) -> None:
    warmup_epochs = max(1, int(math.ceil(epochs * warmup_fraction)))
    for group in optimizer.param_groups:
        base = float(group["base_lr"])
        floor = min(min_lr, base)
        if epoch < warmup_epochs:
            value = base * float(epoch + 1) / warmup_epochs
        elif epochs <= warmup_epochs + 1:
            value = base
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs - 1)
            value = floor + 0.5 * (base - floor) * (1.0 + math.cos(math.pi * progress))
        group["lr"] = max(floor, value)


def losses(
    prediction: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = F.mse_loss(
        prediction["xyz"].float() / XYZ_SCALE_M,
        batch["target_xyz"].float() / XYZ_SCALE_M,
    )
    radius = F.mse_loss(
        prediction["radius"].float() / RADIUS_SCALE_M,
        batch["target_radius"].float() / RADIUS_SCALE_M,
    )
    return xyz, radius


def train_epoch(
    model: AttentionCNNReconstruction,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    stage: str,
    device: torch.device,
    bf16: bool,
    channels_last: bool,
    gradient_clip: float,
    max_batches: int | None,
) -> dict[str, float | int]:
    model.train()
    freeze_batchnorm_stats(model.encoder)
    totals = {"objective": 0.0, "xyz": 0.0, "radius": 0.0}
    count = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        batch = to_device_batch(batch, device)
        images = prepare_images(batch["images"], device, channels_last)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, bf16):
            prediction = model(images)
            xyz_loss, radius_loss = losses(prediction, batch)
            objective = xyz_loss if stage == "xyz" else radius_loss
        objective.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                gradient_clip,
            )
        optimizer.step()
        batch_size = int(images.shape[0])
        count += batch_size
        totals["objective"] += float(objective.detach()) * batch_size
        totals["xyz"] += float(xyz_loss.detach()) * batch_size
        totals["radius"] += float(radius_loss.detach()) * batch_size
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    elapsed = time.perf_counter() - started
    return {
        "train_objective": totals["objective"] / count,
        "train_xyz_mse_scaled": totals["xyz"] / count,
        "train_radius_physical_mse_scaled": totals["radius"] / count,
        "train_cases": count,
        "train_seconds": elapsed,
    }


@torch.inference_mode()
def validate_epoch(
    model: AttentionCNNReconstruction,
    loader: DataLoader,
    *,
    device: torch.device,
    bf16: bool,
    channels_last: bool,
    max_batches: int | None,
) -> dict[str, float | int]:
    model.eval()
    xyz_total = radius_total = 0.0
    count = 0
    rmse_values: list[torch.Tensor] = []
    radius_values: list[torch.Tensor] = []
    severity_values: list[torch.Tensor] = []
    true_grades: list[torch.Tensor] = []
    pred_grades: list[torch.Tensor] = []
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        batch = to_device_batch(batch, device)
        images = prepare_images(batch["images"], device, channels_last)
        with autocast_context(device, bf16):
            prediction = model(images)
            xyz_loss, radius_loss = losses(prediction, batch)
        batch_size = int(images.shape[0])
        count += batch_size
        xyz_total += float(xyz_loss) * batch_size
        radius_total += float(radius_loss) * batch_size
        pred_xyz = prediction["xyz"].float()
        pred_radius = prediction["radius"].float()
        target_xyz = batch["target_xyz"].float()
        target_radius = batch["target_radius"].float()
        rmse_values.append(
            torch.sqrt(
                torch.mean(torch.sum((pred_xyz - target_xyz) ** 2, dim=-1), dim=-1)
            )
            .mul(1000.0)
            .cpu()
        )
        radius_values.append(
            torch.mean(torch.abs(pred_radius - target_radius), dim=(1, 2))
            .mul(1000.0)
            .cpu()
        )
        severity_values.append(
            torch.abs(
                prediction["severity"].float() - batch["full_severity"].float()
            ).cpu()
        )
        pred_grades.append(prediction["grade"].cpu())
        true_grades.append(batch["full_grade"].cpu())
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    rmse = torch.cat(rmse_values)
    radius_mae = torch.cat(radius_values)
    severity_error = torch.cat(severity_values)
    truth = torch.cat(true_grades)
    predicted = torch.cat(pred_grades)
    severe = truth == 3
    return {
        "val_xyz_mse_scaled": xyz_total / count,
        "val_radius_physical_mse_scaled": radius_total / count,
        "val_rmse_xyz_mm": float(rmse.mean()),
        "val_mae_r_mm": float(radius_mae.mean()),
        "val_severity_mae_percent": float(severity_error.mean()),
        "val_grade_error_rate": float((predicted != truth).float().mean()),
        "val_severe_grade_error_rate": (
            float((predicted[severe] != truth[severe]).float().mean())
            if bool(severe.any())
            else float("nan")
        ),
        "val_severe_count": int(severe.sum()),
        "val_cases": count,
        "val_seconds": time.perf_counter() - started,
    }


def model_metadata(
    *,
    model: AttentionCNNReconstruction,
    statistics: Mapping[str, Any],
    stage: str,
    args: argparse.Namespace,
    train_dataset: SyntheticReconstructionDataset,
    valid_dataset: SyntheticReconstructionDataset,
    sampling: Mapping[str, Any],
    trainability: Mapping[str, Any],
    stage1_source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    log_stats = statistics["labels"]["log_r0"]
    model_config: dict[str, Any] = {
        "model_version": "reconstruction_v2",
        "decoder_type": "legacy_mlp",
        "num_points": NUM_POINTS,
        "num_views": 2,
        "input_channels": 1,
        "input_channel_order": ["legacy_normalised_dt"],
        "xyz_scale_m": XYZ_SCALE_M,
        "radius_parameterisation": "log_r0_log_ratio",
        "log_r0_mean": float(log_stats["mean"]),
        "log_r0_std": float(log_stats["std"]),
        "log_ratio_limit": None,
        "cbam_enabled": True,
    }
    if stage == "radius":
        model_config["radius_head_configuration"] = {
            "radius_head_type": "multiscale_attentive_stat_pool_radius_v1",
            "representation": "free_log_ratio",
            "feature_sources": [
                "frozen_layer2",
                "frozen_layer3",
                "frozen_cbam_layer4",
            ],
            "pooling": ["spatial_max"],
            "residual_output_initialisation": "zeros",
            "base_head": {"radius_head_type": "legacy_global_mlp"},
            "num_points": 12,
            "num_views": 2,
            "global_dim": 4096,
            "d_model": 64,
            "hidden_dim": 256,
            "depth": 3,
            "use_fine_layer2_tokens": True,
            "head_init_seed": args.seed,
            "ground_truth_xyz_used": False,
            "pooling_available": [
                "spatial_average",
                "spatial_max",
                "learned_softmax_attention",
            ],
            "pooling_mask": [0, 1, 0],
            "pooling_ablation_policy": (
                "zero_disabled_statistics_keep_fixed_context_and_parameters"
            ),
            "pooling_context_dim": 1152,
        }
    manifest_hash = (
        sha256_file(args.manifest)
        if args.manifest is not None
        else sha256_json(
            {
                "train": train_dataset.case_ids,
                "valid": valid_dataset.case_ids,
            }
        )
    )
    split_hash = sha256_json(
        {
            "train": train_dataset.case_ids,
            "valid": valid_dataset.case_ids,
            "seed": args.seed,
        }
    )
    return {
        "schema_version": "reconstruction-v2",
        "model_version": MODEL_VERSION,
        "num_points": 12,
        "num_views": 2,
        "input_channels": ["relative_distance_transform"],
        "radius_parameterization": "log_r0_log_ratio",
        "severity_definition": "100*(1-hard_min(radius)/inlet_radius)",
        "dataset_manifest_sha256": manifest_hash,
        "split_manifest_sha256": split_hash,
        "geometry_target": GEOMETRY_TARGET,
        "train_statistics": {
            "log_r0_mean": float(log_stats["mean"]),
            "log_r0_std": float(log_stats["std"]),
            "num_cases": len(train_dataset),
            "case_id_sha256": hashlib.sha256(
                "\n".join(train_dataset.case_ids).encode("utf-8")
            ).hexdigest(),
        },
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "manifest_path": (
            str(Path(args.manifest).resolve()) if args.manifest is not None else None
        ),
        "code_config": {
            "model_alias": "AttentionCNN",
            "experiment_stage": stage,
            "target_xyz_key": "min_preserving_xyz",
            "target_radius_key": "min_preserving_radius",
            "target_index_policy": (
                "legacy12_nearest_slot_replaced_by_complete_raw_argmin_point"
            ),
            "target_pairing_invariant": (
                "xyz_and_radius_are_selected_by_one_shared_index_vector"
            ),
            "loss_family": "pure_mse",
            "model_config": model_config,
            "training_strategy": {
                "name": "strict_two_stage",
                "centreline_stage_epochs": args.epochs_stage1,
                "radius_stage_epochs": args.epochs_stage2,
                "stage_1_trainable": "encoder_and_xyz_head",
                "stage_2_trainable": "radius_head_only",
                "current_stage": stage,
            },
            "stage_sampling": {
                "mode": "stage_specific",
                "active_stage": stage,
                "base_seed": args.seed,
                "stage_stream": dict(sampling),
            },
            "optimizer": {
                "name": "AdamW",
                "lr_head": args.lr_head,
                "lr_layer4": args.lr_layer4,
                "lr_backbone": args.lr_backbone,
                "weight_decay": args.weight_decay,
                "warmup_fraction": args.warmup_fraction,
                "minimum_lr": args.min_lr,
                "gradient_clip": args.gradient_clip,
            },
            "radius_supervision": {
                "space": "physical",
                "formula": "MSE(pred_radius/0.001,target_radius/0.001)",
                "one_radius_mse_only": True,
                "point_weighting": False,
            },
            "freeze_contract": dict(trainability),
            "stage1_source": dict(stage1_source) if stage1_source else None,
            "formal_recipe_defaults_used": (
                args.seed == SEED
                and args.epochs_stage1 == 100
                and args.epochs_stage2 == 50
                and args.batch_size == 128
                and math.isclose(args.lr_head, 3e-4)
                and math.isclose(args.weight_decay, 1e-4)
            ),
        },
        "parameter_counts": parameter_counts(model),
    }


def save_checkpoint(
    path: Path,
    *,
    model: AttentionCNNReconstruction,
    metadata: Mapping[str, Any],
    row: Mapping[str, Any],
    stage: str,
    metric: str,
    value: float,
) -> None:
    payload = {
        "metadata": dict(metadata),
        "model_state_dict": model.state_dict(),
        "epoch": int(row["epoch"]),
        "metrics": dict(row),
        "selection": {
            "strategy": "strict_two_stage",
            "stage": stage,
            "metric": metric,
            "value": value,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_stage1(
    checkpoint: str | Path,
    model: AttentionCNNReconstruction,
) -> dict[str, Any]:
    path = Path(checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Stage-1 checkpoint must contain a mapping")
    metadata = payload.get("metadata")
    state = payload.get("model_state_dict")
    selection = payload.get("selection")
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
        raise ValueError("Stage-1 checkpoint lacks metadata or model_state_dict")
    if metadata.get("model_version") != MODEL_VERSION:
        raise ValueError("Stage-1 checkpoint model version differs")
    if metadata.get("geometry_target") != GEOMETRY_TARGET:
        raise ValueError("Stage-1 checkpoint geometry target differs")
    if not isinstance(selection, Mapping) or selection.get("stage") != "xyz":
        raise ValueError("checkpoint was not selected by Stage-1 validation")
    model.load_state_dict(state, strict=True)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "selection": dict(selection),
        "strict_state_dict_load": True,
    }


def _learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    values = {
        str(group["name"]): float(group["lr"]) for group in optimizer.param_groups
    }
    return {
        "lr_head": values.get("heads_and_input_adapter", 0.0),
        "lr_layer4": values.get("layer4_cbam", 0.0),
        "lr_backbone": values.get("backbone_early", 0.0),
    }


@dataclass(frozen=True)
class TrainingComponents:
    """Dependencies for one training run, isolated from other model variants."""

    model_factory: Callable[..., nn.Module] = AttentionCNNReconstruction
    dataset_factory: Callable[..., SyntheticReconstructionDataset] = (
        SyntheticReconstructionDataset
    )
    install_radius_head: Callable[..., Any] = install_max_pool_radius_head
    set_trainable: Callable[..., None] = set_trainable
    make_optimizer: Callable[..., torch.optim.Optimizer] = make_optimizer
    make_loaders: Callable[..., Any] = make_loaders
    model_metadata: Callable[..., dict[str, Any]] = model_metadata
    load_stage1: Callable[..., dict[str, Any]] = load_stage1


def run_stage(
    *,
    stage: str,
    output: Path,
    args: argparse.Namespace,
    statistics: Mapping[str, Any],
    train_dataset: SyntheticReconstructionDataset,
    valid_dataset: SyntheticReconstructionDataset,
    stage1_checkpoint: Path | None,
    components: TrainingComponents | None = None,
) -> tuple[Path, dict[str, Any]]:
    components = components or TrainingComponents()
    log_stats = statistics["labels"]["log_r0"]
    if stage == "xyz":
        backbone_weights = (
            ResNet50_Weights.IMAGENET1K_V1
            if args.backbone_init == "imagenet_v1"
            else None
        )
        model = components.model_factory(
            log_r0_mean=float(log_stats["mean"]),
            log_r0_std=float(log_stats["std"]),
            backbone_weights=backbone_weights,
            advanced_radius_head=False,
        )
        stage1_source = None
        epochs = args.epochs_stage1
    else:
        if stage1_checkpoint is None:
            raise ValueError("Stage 2 requires --stage1-checkpoint or --stage both")
        model = components.model_factory(
            log_r0_mean=float(log_stats["mean"]),
            log_r0_std=float(log_stats["std"]),
            backbone_weights=None,
            advanced_radius_head=False,
        )
        stage1_source = components.load_stage1(stage1_checkpoint, model)
        cpu_rng_before = torch.get_rng_state().clone()
        components.install_radius_head(model, head_init_seed=args.seed)
        if not torch.equal(cpu_rng_before, torch.get_rng_state()):
            raise RuntimeError(
                "Stage-2 head installation changed the global RNG stream"
            )
        epochs = args.epochs_stage2
    components.set_trainable(model, stage)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = components.make_optimizer(
        model,
        stage=stage,
        lr_head=args.lr_head,
        lr_layer4=args.lr_layer4,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
    )
    trainability = audit_trainability(model, optimizer, stage)
    train_loader, valid_loader, sampling = components.make_loaders(
        train_dataset,
        valid_dataset,
        stage=stage,
        batch_size=args.batch_size,
        workers=args.num_workers,
        seed=args.seed,
    )
    metadata = components.model_metadata(
        model=model,
        statistics=statistics,
        stage=stage,
        args=args,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        sampling=sampling,
        trainability=trainability,
        stage1_source=stage1_source,
    )
    run_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output),
        "metadata": metadata,
        "train_cases": len(train_dataset),
        "valid_cases": len(valid_dataset),
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "pytorch": torch.__version__,
        "smoke_test": args.smoke_test,
    }
    write_json_atomic(output / "run_config.json", run_config)

    best_objective = math.inf
    best_common = math.inf
    best_grade_key: tuple[int, float, float, int] | None = None
    rows: list[dict[str, Any]] = []
    max_batches = 1 if args.smoke_test else None
    effective_epochs = 1 if args.smoke_test else epochs
    objective_path = output / (
        "best_xyz_stage1.pt" if stage == "xyz" else "best_radius_objective_stage2.pt"
    )
    common_path = output / "best_radius_common_stage2.pt"
    grade_path = output / "best_radius_lexicographic_stage2.pt"
    started = time.perf_counter()
    for epoch in range(effective_epochs):
        set_epoch_learning_rates(
            optimizer,
            epoch,
            effective_epochs,
            warmup_fraction=args.warmup_fraction,
            min_lr=args.min_lr,
        )
        training = train_epoch(
            model,
            train_loader,
            optimizer,
            stage=stage,
            device=device,
            bf16=args.bf16,
            channels_last=args.channels_last,
            gradient_clip=args.gradient_clip,
            max_batches=max_batches,
        )
        validation = validate_epoch(
            model,
            valid_loader,
            device=device,
            bf16=args.bf16,
            channels_last=args.channels_last,
            max_batches=max_batches,
        )
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "stage": stage,
            "stage_sampler": sampling["name"],
            "selection_metric": (
                "val_xyz_mse_scaled"
                if stage == "xyz"
                else "val_radius_physical_mse_scaled"
            ),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            **_learning_rates(optimizer),
            **training,
            **validation,
        }
        objective_key = str(row["selection_metric"])
        objective_value = float(row[objective_key])
        objective_improved = objective_value < best_objective
        row["objective_best_so_far"] = objective_improved
        if objective_improved:
            best_objective = objective_value
            save_checkpoint(
                objective_path,
                model=model,
                metadata=metadata,
                row=row,
                stage=stage,
                metric=objective_key,
                value=objective_value,
            )
        if stage == "radius":
            common_value = float(row["val_mae_r_mm"])
            common_improved = common_value < best_common
            row["common_best_so_far"] = common_improved
            if common_improved:
                best_common = common_value
                save_checkpoint(
                    common_path,
                    model=model,
                    metadata=metadata,
                    row=row,
                    stage=stage,
                    metric="val_mae_r_mm",
                    value=common_value,
                )
            severe_error = float(row["val_severe_grade_error_rate"])
            eligible = (
                math.isfinite(severe_error)
                and float(row["val_mae_r_mm"]) <= GRADE_SELECTION_MAX_MAE_R_MM
                and severe_error <= GRADE_SELECTION_MAX_SEVERE_ERROR_RATE
            )
            grade_key = (
                int(round(float(row["val_grade_error_rate"]) * int(row["val_cases"]))),
                float(row["val_severity_mae_percent"]),
                float(row["val_mae_r_mm"]),
                int(row["epoch"]),
            )
            row["grade_selection_eligible"] = eligible
            row["grade_selection_best_so_far"] = False
            if eligible and (best_grade_key is None or grade_key < best_grade_key):
                best_grade_key = grade_key
                row["grade_selection_best_so_far"] = True
                save_checkpoint(
                    grade_path,
                    model=model,
                    metadata=metadata,
                    row=row,
                    stage=stage,
                    metric="constrained_validation_grade_lexicographic",
                    value=float(row["val_grade_error_rate"]),
                )
        rows.append(row)
        write_csv_atomic(output / "epochs.csv", rows)
        print(
            f"[{stage}] {epoch + 1:02d}/{effective_epochs} "
            f"objective={training['train_objective']:.6f} "
            f"RMSE={validation['val_rmse_xyz_mm']:.4f}mm "
            f"MAE_r={validation['val_mae_r_mm']:.4f}mm "
            f"grade={validation['val_grade_error_rate']:.2%}",
            flush=True,
        )
    if not objective_path.is_file():
        raise RuntimeError(f"{stage} produced no objective checkpoint")
    if stage == "xyz":
        selected = objective_path
    elif grade_path.is_file():
        selected = grade_path
    else:
        selected = common_path
    summary = {
        "status": "smoke_test_pass" if args.smoke_test else "complete",
        "stage": stage,
        "epochs_completed": effective_epochs,
        "total_seconds": time.perf_counter() - started,
        "objective_checkpoint": str(objective_path),
        "objective_best_value": best_objective,
        "common_checkpoint": str(common_path) if common_path.is_file() else None,
        "common_best_value": best_common if common_path.is_file() else None,
        "lexicographic_checkpoint": str(grade_path) if grade_path.is_file() else None,
        "lexicographic_best_key": (
            list(best_grade_key) if best_grade_key is not None else None
        ),
        "selected_checkpoint": str(selected),
        "stage1_source": stage1_source,
        "freeze_contract": trainability,
    }
    write_json_atomic(output / "summary.json", summary)
    return selected, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--statistics", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("both", "xyz", "radius"), default="both")
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--epochs-stage1", type=int, default=100)
    parser.add_argument("--epochs-stage2", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--lr-layer4", type=float, default=3e-5)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--backbone-init",
        choices=("imagenet_v1", "none"),
        default="imagenet_v1",
    )
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-valid", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--channels-last", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if min(args.epochs_stage1, args.epochs_stage2, args.batch_size) < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.num_workers < 0 or args.seed < 0:
        raise ValueError("num-workers and seed must be non-negative")
    if not 0 <= args.warmup_fraction <= 1:
        raise ValueError("warmup fraction must be in [0, 1]")
    if (
        min(
            args.lr_head,
            args.lr_layer4,
            args.lr_backbone,
            args.min_lr,
            args.gradient_clip,
        )
        < 0
    ):
        raise ValueError("learning rates and gradient clip must be non-negative")
    if not args.smoke_test and (
        args.limit_train is not None or args.limit_valid is not None
    ):
        raise ValueError("dataset limits are available only with --smoke-test")
    if args.stage == "radius" and args.stage1_checkpoint is None:
        raise ValueError("--stage radius requires --stage1-checkpoint")
    if args.stage != "radius" and args.stage1_checkpoint is not None:
        raise ValueError("--stage1-checkpoint is used only with --stage radius")
    root = args.dataset_root.resolve()
    for split in ("train", "valid"):
        if not (root / split).is_dir():
            raise FileNotFoundError(root / split)
    if args.manifest is not None and not args.manifest.resolve().is_file():
        raise FileNotFoundError(args.manifest)
    if args.statistics is not None and not args.statistics.resolve().is_file():
        raise FileNotFoundError(args.statistics)


def main(
    argv: Sequence[str] | None = None, *, components: TrainingComponents | None = None
) -> int:
    components = components or TrainingComponents()
    args = build_parser().parse_args(argv)
    validate_args(args)
    seed_everything(args.seed, args.deterministic)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    output = prepare_output(args.output_dir)
    train_dataset = components.dataset_factory(
        args.dataset_root,
        split="train",
        manifest=args.manifest,
        cache_root=args.cache_root,
        limit=args.limit_train,
    )
    valid_dataset = components.dataset_factory(
        args.dataset_root,
        split="valid",
        manifest=args.manifest,
        cache_root=args.cache_root,
        limit=args.limit_valid,
    )
    statistics = load_statistics(args.statistics, train_dataset)
    write_json_atomic(output / "training_statistics.json", statistics)
    summaries: dict[str, Any] = {}
    if args.stage in {"both", "xyz"}:
        stage1_dir = output / "stage1_xyz"
        stage1_dir.mkdir()
        stage1_checkpoint, summaries["stage1"] = run_stage(
            stage="xyz",
            output=stage1_dir,
            args=args,
            statistics=statistics,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            stage1_checkpoint=None,
            components=components,
        )
    else:
        stage1_checkpoint = args.stage1_checkpoint.resolve()
    if args.stage in {"both", "radius"}:
        stage2_dir = output / "stage2_radius"
        stage2_dir.mkdir()
        selected, summaries["stage2"] = run_stage(
            stage="radius",
            output=stage2_dir,
            args=args,
            statistics=statistics,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            stage1_checkpoint=stage1_checkpoint,
            components=components,
        )
    else:
        selected = stage1_checkpoint
    final = {
        "status": "smoke_test_pass" if args.smoke_test else "complete",
        "stage": args.stage,
        "selected_checkpoint": str(selected),
        "summaries": summaries,
    }
    write_json_atomic(output / "summary.json", final)
    print(json.dumps(json_safe(final), ensure_ascii=False, indent=2))
    train_dataset.close_cache()
    valid_dataset.close_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
