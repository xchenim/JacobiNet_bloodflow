"""Run strict AttentionCNN maximum-pooling inference and evaluate a dataset split."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .checkpoint import (
        FINAL_CHECKPOINT_SHA256,
        load_reconstruction_checkpoint,
        sha256_file,
    )
    from .dataset import SyntheticReconstructionDataset
    from .metrics import evaluate_predictions, write_evaluation
except ImportError:
    from checkpoint import (
        FINAL_CHECKPOINT_SHA256,
        load_reconstruction_checkpoint,
        sha256_file,
    )
    from dataset import SyntheticReconstructionDataset
    from metrics import evaluate_predictions, write_evaluation


def _prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _autocast(device: torch.device, bf16: bool):
    if device.type != "cuda" or not bf16:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _prepare_images(
    images: torch.Tensor, device: torch.device, channels_last: bool
) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if channels_last:
        batch, views, channels, height, width = images.shape
        flat = images.reshape(batch * views, channels, height, width).contiguous(
            memory_format=torch.channels_last
        )
        images = flat.view(batch, views, channels, height, width)
    return images


def _load_reference(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        case_ids = [str(value) for value in payload["case_ids"].tolist()]
        xyz = np.asarray(payload["pred_xyz_m"], dtype=np.float32)
        radius = np.asarray(payload["pred_r_m"], dtype=np.float32)
    if radius.ndim == 3 and radius.shape[-1] == 1:
        radius = radius[..., 0]
    return case_ids, xyz, radius


def _comparison(
    reference_path: Path,
    case_ids: Sequence[str],
    xyz: np.ndarray,
    radius: np.ndarray,
) -> dict[str, Any]:
    ref_cases, ref_xyz, ref_radius = _load_reference(reference_path)
    if len(ref_cases) != len(set(ref_cases)):
        raise ValueError("reference predictions contain duplicate case IDs")
    reference_row = {case_id: row for row, case_id in enumerate(ref_cases)}
    missing = [case_id for case_id in case_ids if case_id not in reference_row]
    if missing:
        raise ValueError(
            "reference predictions are missing release cases: "
            + ", ".join(missing[:10])
        )
    selected_rows = [reference_row[case_id] for case_id in case_ids]
    ref_xyz = ref_xyz[selected_rows]
    ref_radius = ref_radius[selected_rows]
    xyz_delta = np.abs(xyz.astype(np.float64) - ref_xyz.astype(np.float64))
    radius_delta = np.abs(radius.astype(np.float64) - ref_radius.astype(np.float64))
    return {
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": sha256_file(reference_path),
        "case_ids_aligned": True,
        "reference_case_count": len(ref_cases),
        "compared_case_count": len(case_ids),
        "xyz_shape": list(xyz.shape),
        "radius_shape": list(radius.shape),
        "xyz_bitwise_equal": bool(np.array_equal(xyz, ref_xyz)),
        "radius_bitwise_equal": bool(np.array_equal(radius, ref_radius)),
        "xyz_max_abs_difference_m": float(xyz_delta.max()),
        "radius_max_abs_difference_m": float(radius_delta.max()),
        "xyz_mean_abs_difference_m": float(xyz_delta.mean()),
        "radius_mean_abs_difference_m": float(radius_delta.mean()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = _prepare_output(args.output_dir, args.overwrite)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable; pass --no-bf16")
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    expected_sha = (
        None if args.allow_other_compatible_checkpoint else FINAL_CHECKPOINT_SHA256
    )
    model, checkpoint_audit = load_reconstruction_checkpoint(
        args.checkpoint,
        device=device,
        channels_last=args.channels_last,
        expected_sha256=expected_sha,
    )
    dataset = SyntheticReconstructionDataset(
        args.dataset_root,
        split=args.split,
        manifest=args.manifest,
        cache_root=args.cache_root,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    case_ids: list[str] = []
    xyz_batches: list[np.ndarray] = []
    radius_batches: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            images = _prepare_images(batch["images"], device, args.channels_last)
            with _autocast(device, args.bf16):
                prediction = model(images)
            xyz = prediction["xyz"].float()
            radius = prediction["radius"].squeeze(-1).float()
            if not torch.isfinite(xyz).all() or not torch.isfinite(radius).all():
                raise FloatingPointError("model produced NaN or infinity")
            if not torch.all(radius > 0):
                raise FloatingPointError("model produced non-positive radii")
            case_ids.extend(str(value) for value in batch["case_id"])
            xyz_batches.append(xyz.cpu().numpy())
            radius_batches.append(radius.cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if case_ids != dataset.case_ids:
        raise RuntimeError("DataLoader changed case order")
    xyz_array = np.concatenate(xyz_batches).astype(np.float32, copy=False)
    radius_array = np.concatenate(radius_batches).astype(np.float32, copy=False)
    raw_labels = [dataset.raw_label(case_id) for case_id in case_ids]
    summary, rows = evaluate_predictions(case_ids, xyz_array, radius_array, raw_labels)
    summary["checkpoint"] = {
        key: value for key, value in checkpoint_audit.items() if key != "metadata"
    }
    summary["dataset"] = {
        "root": str(dataset.dataset_root),
        "split": args.split,
        "num_cases": len(case_ids),
        "manifest": str(args.manifest.resolve()) if args.manifest else None,
        "cache_root": str(args.cache_root.resolve()) if args.cache_root else None,
    }
    summary["inference"] = {
        "device": str(device),
        "cuda_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "bf16": bool(args.bf16 and device.type == "cuda"),
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32 and device.type == "cuda"),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32 and device.type == "cuda"),
        "channels_last": bool(args.channels_last),
        "runtime_seconds": elapsed,
        "cases_per_second": len(case_ids) / max(elapsed, 1e-12),
    }
    np.savez_compressed(
        output / "predictions.npz",
        case_ids=np.asarray(case_ids),
        pred_xyz_m=xyz_array,
        pred_r_m=radius_array,
    )
    write_evaluation(output, _json_safe(summary), rows)
    if args.reference_predictions is not None:
        comparison = _comparison(
            args.reference_predictions, case_ids, xyz_array, radius_array
        )
        (output / "reproduction_check.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        comparison = None
    print(
        f"AttentionCNN: n={len(case_ids)}, "
        f"RMSE_xyz={summary['geometry']['rmse_xyz_mm']['mean']:.6f} mm, "
        f"MAE_r={summary['geometry']['mae_r_mm']['mean']:.6f} mm, "
        f"grade_error={summary['grading']['full_raw_truth_primary']['error_rate']:.2%}, "
        f"runtime={elapsed:.3f}s"
    )
    if comparison is not None:
        print(
            "Reference differences: "
            f"xyz={comparison['xyz_max_abs_difference_m']:.3e} m, "
            f"radius={comparison['radius_max_abs_difference_m']:.3e} m"
        )
    print(f"Output: {output}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--channels-last", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--allow-other-compatible-checkpoint", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        raise SystemExit("batch-size must be positive and num-workers non-negative")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("limit must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
