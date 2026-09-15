"""Evaluate the published AttentionCNN precision protocol on synthetic_100."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
from uuid import uuid4

from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from AttentionCNN_3Dreconstruction.checkpoint import (
    FINAL_CHECKPOINT_SHA256,
    load_reconstruction_checkpoint,
    sha256_file,
)
from AttentionCNN_3Dreconstruction.dataset import _fit_to_size
from AttentionCNN_3Dreconstruction.metrics import (
    evaluate_predictions,
    wilson,
    write_evaluation,
)
from AttentionCNN_3Dreconstruction.predict import _json_safe, _prepare_images


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CASES_ROOT = PROJECT_ROOT / "synthetic_100" / "cases"
DEFAULT_PREDS_ROOT = PROJECT_ROOT / "outputs" / "reconstruction_100"
DEFAULT_CHECKPOINT = SCRIPT_DIR.parent / "weights" / "attentioncnn.pt"
DEFAULT_SELECTION_MANIFEST = PROJECT_ROOT / "synthetic_100" / "manifest.json"
GRADE_NAMES = ("0-24%", "25-49%", "50-69%", "70-99%")
GRADE_THRESHOLDS = np.asarray((25.0, 50.0, 70.0), dtype=np.float64)


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_label_float64(case_dir: Path, case_id: str) -> np.ndarray:
    path = case_dir / "gt/stenosis.npy"
    points = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if points.ndim == 3 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 2 or points.shape[1] < 4 or points.shape[0] < 2:
        raise ValueError(f"invalid label shape {points.shape} in {path}")
    points = np.ascontiguousarray(points[:, :4], dtype=np.float64)
    if not np.isfinite(points).all() or np.any(points[:, 3] <= 0):
        raise ValueError(f"invalid label values in {path}")
    return points


class Synthetic100ProjectionDataset(Dataset):
    """Load source PNGs and preserve every numerical tensor as float64."""

    def __init__(self, cases_root: Path, case_ids: Sequence[str]) -> None:
        self.cases_root = cases_root.resolve()
        self.case_ids = list(case_ids)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case_id = self.case_ids[index]
        case_dir = self.cases_root / case_id / "projections"
        views = []
        for suffix in ("a", "b"):
            path = case_dir / "inputs" / f"view_{suffix}.png"
            with Image.open(path) as source:
                image = _fit_to_size(source)
                array = np.asarray(image, dtype=np.float64)[None, ...] / np.float64(
                    255.0
                )
            views.append(torch.from_numpy(np.ascontiguousarray(array)))
        return {"case_id": case_id, "images": torch.stack(views, dim=0)}

    def raw_label(self, case_id: str) -> np.ndarray:
        return load_label_float64(self.cases_root / case_id / "projections", case_id)

    def selection_label(self, case_id: str) -> np.ndarray:
        path = self.cases_root / case_id / "projections/gt/stenosis_raw.npy"
        points = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        if points.ndim == 3 and points.shape[0] == 1:
            points = points[0]
        if points.ndim != 2 or points.shape[1] < 4 or points.shape[0] < 2:
            raise ValueError(f"invalid selection label shape {points.shape} in {path}")
        points = np.ascontiguousarray(points[:, :4], dtype=np.float64)
        if not np.isfinite(points).all() or np.any(points[:, 3] <= 0):
            raise ValueError(f"invalid selection label values in {path}")
        return points


def load_selection_strata(
    path: Path, case_ids: Sequence[str]
) -> tuple[dict[str, int], dict[str, float], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows = [
            {
                "case_id": row["case_id"],
                "severity_stratum": GRADE_NAMES[
                    int(
                        np.searchsorted(
                            GRADE_THRESHOLDS,
                            row["label_stenosis_percent"],
                            side="right",
                        )
                    )
                ],
                "diameter_stenosis_percent": row["label_stenosis_percent"],
            }
            for row in manifest["cases"]
        ]
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    if len(rows) != len(case_ids):
        raise RuntimeError(
            f"selection manifest has {len(rows)} rows; expected {len(case_ids)}"
        )
    grade_by_name = {name: grade for grade, name in enumerate(GRADE_NAMES)}
    grade_by_case: dict[str, int] = {}
    severity_by_case: dict[str, float] = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in grade_by_case:
            raise RuntimeError(f"duplicate selection case: {case_id}")
        stratum = str(row["severity_stratum"])
        if stratum not in grade_by_name:
            raise ValueError(f"unknown severity stratum {stratum!r}")
        grade_by_case[case_id] = grade_by_name[stratum]
        severity_by_case[case_id] = float(row["diameter_stenosis_percent"])
    if set(grade_by_case) != set(case_ids):
        raise RuntimeError("selection manifest cases do not match labels cohort")
    counts = np.bincount(
        np.asarray([grade_by_case[case_id] for case_id in case_ids]), minlength=4
    )
    if counts.tolist() != [25, 25, 25, 25]:
        raise RuntimeError(
            "fixed cohort is not 25 cases per severity stratum: " + str(counts.tolist())
        )
    audit = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "cases": len(case_ids),
        "stratum_counts": counts.tolist(),
        "severity_formula": "100 * (1 - min(radius) / inlet_radius)",
        "truth_source": "fixed severity_stratum column",
    }
    return grade_by_case, severity_by_case, audit


def validate_recorded_severity(selection_labels, recorded):
    """Validate label severities against float64 or float32-stored source radii."""
    expected64 = np.asarray(
        [100.0 * (1.0 - label[:, 3].min() / label[0, 3]) for label in selection_labels],
        dtype=np.float64,
    )
    expected32 = np.asarray(
        [
            100.0
            * (
                1.0
                - float(label[:, 3].astype(np.float32).min())
                / float(np.float32(label[0, 3]))
            )
            for label in selection_labels
        ],
        dtype=np.float64,
    )
    recorded = np.asarray(recorded, dtype=np.float64)
    matches64 = np.abs(recorded - expected64) <= 5.1e-10
    matches32 = np.abs(recorded - expected32) <= 5.1e-10
    if not np.all(matches64 | matches32):
        raise RuntimeError("recorded selection severities do not match source labels")
    return {
        "float64_radius_matches": int(matches64.sum()),
        "float32_radius_only_matches": int((matches32 & ~matches64).sum()),
        "max_difference_from_float64_percentage_points": float(
            np.max(np.abs(recorded - expected64))
        ),
        "grading_radius_dtype": "float64",
    }


def fixed_stratified_grading(
    case_ids: Sequence[str],
    selection_labels: Sequence[np.ndarray],
    predicted_radius: np.ndarray,
    grade_by_case: dict[str, int],
    recorded_severity_by_case: dict[str, float],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    radius = np.asarray(predicted_radius, dtype=np.float64)
    true_severity = np.asarray(
        [100.0 * (1.0 - label[:, 3].min() / label[0, 3]) for label in selection_labels],
        dtype=np.float64,
    )
    predicted_severity = (1.0 - radius.min(axis=1) / radius[:, 0]) * 100.0
    true_grade = np.asarray(
        [grade_by_case[case_id] for case_id in case_ids], dtype=np.int64
    )
    recomputed_grade = np.searchsorted(
        GRADE_THRESHOLDS, true_severity, side="right"
    ).astype(np.int64)
    if not np.array_equal(true_grade, recomputed_grade):
        mismatches = [
            case_ids[index] for index in np.flatnonzero(true_grade != recomputed_grade)
        ]
        raise RuntimeError(
            "selection strata disagree with raw min/inlet severity: "
            + ", ".join(mismatches)
        )
    recorded = np.asarray(
        [recorded_severity_by_case[case_id] for case_id in case_ids],
        dtype=np.float64,
    )
    severity_identity = validate_recorded_severity(selection_labels, recorded)
    predicted_grade = np.searchsorted(
        GRADE_THRESHOLDS, predicted_severity, side="right"
    ).astype(np.int64)
    matrix = np.zeros((4, 4), dtype=np.int64)
    np.add.at(matrix, (true_grade, predicted_grade), 1)
    if matrix.sum(axis=1).tolist() != [25, 25, 25, 25]:
        raise RuntimeError("fixed grading denominator changed")
    correct = predicted_grade == true_grade
    under = predicted_grade < true_grade
    over = predicted_grade > true_grade
    total = len(case_ids)
    number_correct = int(correct.sum())
    interval = wilson(number_correct, total)
    per_grade = []
    for grade, name in enumerate(GRADE_NAMES):
        support = int(matrix[grade].sum())
        grade_correct = int(matrix[grade, grade])
        grade_interval = wilson(grade_correct, support)
        per_grade.append(
            {
                "grade": grade,
                "name": name,
                "support": support,
                "correct": grade_correct,
                "error_rate": 1.0 - grade_correct / support,
                "error_rate_wilson_ci": [
                    1.0 - grade_interval[1],
                    1.0 - grade_interval[0],
                ],
            }
        )
    residual = predicted_severity - true_severity
    summary = {
        "n": total,
        "truth_source": "synthetic_100 manifest severity strata",
        "true_severity_definition": "100 * (1 - min(raw_radius) / raw_inlet_radius)",
        "predicted_severity_definition": "100 * (1 - min(pred_radius) / pred_inlet_radius)",
        "grade_thresholds_percent": GRADE_THRESHOLDS.tolist(),
        "stratum_counts": matrix.sum(axis=1).tolist(),
        "selection_severity_identity": severity_identity,
        "severity_mae_percentage_points": float(np.mean(np.abs(residual))),
        "severity_rmse_percentage_points": float(np.sqrt(np.mean(residual**2))),
        "severity_bias_pred_minus_true_percentage_points": float(residual.mean()),
        "correct": number_correct,
        "incorrect": total - number_correct,
        "accuracy": number_correct / total,
        "error_rate": 1.0 - number_correct / total,
        "accuracy_wilson_ci": interval,
        "error_rate_wilson_ci": [1.0 - interval[1], 1.0 - interval[0]],
        "undergraded": int(under.sum()),
        "overgraded": int(over.sum()),
        "confusion_matrix_rows_true_cols_pred": matrix.tolist(),
        "per_grade": per_grade,
    }
    arrays = {
        "true_severity_percent": true_severity,
        "pred_severity_percent": predicted_severity,
        "true_grade": true_grade,
        "pred_grade": predicted_grade,
        "correct": correct,
        "undergraded": under,
        "overgraded": over,
    }
    return summary, arrays


def validate_inputs(
    cases_root: Path,
    checkpoint: Path,
    expected_cases: int,
    selected_case_ids: Sequence[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    if not cases_root.is_dir():
        raise FileNotFoundError(cases_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    case_ids = sorted(
        path.name
        for path in cases_root.iterdir()
        if path.is_dir() and len(path.name) == 5 and path.name.isdigit()
    )
    if selected_case_ids is not None:
        if not set(selected_case_ids).issubset(case_ids):
            raise ValueError("Selected synthetic cases are missing from cases-root")
        case_ids = sorted(selected_case_ids)
    if len(case_ids) != expected_cases:
        raise RuntimeError(
            f"expected {expected_cases} numeric case directories, got {len(case_ids)}"
        )
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        case_dir = cases_root / case_id / "projections"
        inputs = {
            "image_a": case_dir / "inputs/view_a.png",
            "image_b": case_dir / "inputs/view_b.png",
            "label": case_dir / "gt/stenosis.npy",
            "selection_label": case_dir / "gt/stenosis_raw.npy",
        }
        missing = [str(path) for path in inputs.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"case {case_id} is missing inputs: " + ", ".join(missing)
            )
        row: dict[str, Any] = {"case": case_id}
        for key, path in inputs.items():
            row[key] = relative_to_project(path)
            row[f"{key}_sha256"] = sha256_file(path)
        rows.append(row)
    return case_ids, rows


def prepare_output(path: Path) -> tuple[Path, Path]:
    output = path.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    staging = output / (".staging_" + uuid4().hex)
    staging.mkdir()
    return output, staging


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases_root = args.cases_root.resolve()
    checkpoint = args.checkpoint.resolve()
    selection_manifest = args.selection_manifest.resolve()
    selected_case_ids = None
    if selection_manifest.suffix.lower() == ".json":
        selected_case_ids = [
            row["case_id"]
            for row in json.loads(selection_manifest.read_text(encoding="utf-8"))[
                "cases"
            ]
        ]
    case_ids, input_manifest = validate_inputs(
        cases_root,
        checkpoint,
        args.expected_cases,
        selected_case_ids=selected_case_ids,
    )
    grade_by_case, recorded_severity_by_case, selection_audit = load_selection_strata(
        selection_manifest, case_ids
    )
    for protected in (cases_root, SCRIPT_DIR.parent, checkpoint.parent):
        candidate = args.output_root.resolve()
        if candidate.is_relative_to(protected) or protected.is_relative_to(candidate):
            raise ValueError("Output must be outside code, weights, and dataset inputs")
    output, staging = prepare_output(args.output_root)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    model, checkpoint_audit = load_reconstruction_checkpoint(
        checkpoint,
        device=device,
        channels_last=args.channels_last,
        expected_sha256=FINAL_CHECKPOINT_SHA256,
    )
    model = model.double()
    dataset = Synthetic100ProjectionDataset(cases_root, case_ids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    seen_cases: list[str] = []
    xyz_batches: list[np.ndarray] = []
    radius_batches: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    with torch.inference_mode():
        for batch in loader:
            images = _prepare_images(batch["images"], device, args.channels_last)
            prediction = model(images)
            xyz = prediction["xyz"].double()
            radius = prediction["radius"].squeeze(-1).double()
            if not torch.isfinite(xyz).all() or not torch.isfinite(radius).all():
                raise FloatingPointError("model produced NaN or infinity")
            if not torch.all(radius > 0):
                raise FloatingPointError("model produced non-positive radii")
            seen_cases.extend(str(value) for value in batch["case_id"])
            xyz_batches.append(xyz.cpu().numpy())
            radius_batches.append(radius.cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if seen_cases != case_ids:
        raise RuntimeError("DataLoader changed the validated case order")

    xyz_array = np.concatenate(xyz_batches).astype(np.float64, copy=False)
    radius_array = np.concatenate(radius_batches).astype(np.float64, copy=False)
    if xyz_array.shape != (len(case_ids), 12, 3):
        raise RuntimeError(f"unexpected XYZ shape: {xyz_array.shape}")
    if radius_array.shape != (len(case_ids), 12):
        raise RuntimeError(f"unexpected radius shape: {radius_array.shape}")
    raw_labels = [dataset.raw_label(case_id) for case_id in case_ids]
    selection_labels = [dataset.selection_label(case_id) for case_id in case_ids]
    summary, rows = evaluate_predictions(case_ids, xyz_array, radius_array, raw_labels)
    fixed_summary, fixed_arrays = fixed_stratified_grading(
        case_ids,
        selection_labels,
        radius_array,
        grade_by_case,
        recorded_severity_by_case,
    )
    summary["grading"]["fixed_stratified_primary"] = fixed_summary
    summary["fixed_stratified_is_primary_for_cfd_geo_cohort"] = True
    summary["full_raw_is_primary_for_grading"] = False
    summary["cfd_primary_predicted_grade_source"] = (
        "hard min/inlet predicted radius, matched to fixed selection design"
    )
    for index, row in enumerate(rows):
        row.update(
            {
                "fixed_true_severity_percent": float(
                    fixed_arrays["true_severity_percent"][index]
                ),
                "fixed_pred_severity_percent": float(
                    fixed_arrays["pred_severity_percent"][index]
                ),
                "fixed_true_grade": int(fixed_arrays["true_grade"][index]),
                "fixed_pred_grade": int(fixed_arrays["pred_grade"][index]),
                "fixed_correct": bool(fixed_arrays["correct"][index]),
                "fixed_undergraded": bool(fixed_arrays["undergraded"][index]),
                "fixed_overgraded": bool(fixed_arrays["overgraded"][index]),
            }
        )

    for index, case_id in enumerate(case_ids):
        case_output = staging / case_id
        case_output.mkdir()
        combined = np.concatenate(
            (xyz_array[index], radius_array[index, :, None]), axis=1
        ).astype(np.float64, copy=False)
        prediction_path = case_output / f"{case_id}_pred.npy"
        np.save(prediction_path, combined, allow_pickle=False)
        saved = np.load(prediction_path, allow_pickle=False)
        if saved.shape != (12, 4) or saved.dtype != np.float64:
            raise RuntimeError(f"invalid saved dtype/shape for {case_id}")
        if not np.isfinite(saved).all() or np.any(saved[:, 3] <= 0):
            raise RuntimeError(f"invalid saved values for {case_id}")
        rows[index]["prediction"] = f"{case_id}/{case_id}_pred.npy"

    write_csv(staging / "input_manifest.csv", input_manifest)
    manifest_sha = sha256_file(staging / "input_manifest.csv")
    np.savez_compressed(
        staging / "predictions.npz",
        case_ids=np.asarray(case_ids),
        pred_xyz_m=xyz_array,
        pred_r_m=radius_array,
    )
    summary.update(
        {
            "schema_version": 4,
            "training_performed": False,
            "model": {
                key: value
                for key, value in checkpoint_audit.items()
                if key != "metadata"
            },
            "dataset": {
                "root": str(cases_root),
                "cases_evaluated": len(case_ids),
                "source_image_dtype": "uint8",
                "normalised_input_dtype": "float64",
                "label_dtype": "float64",
                "input_manifest": str(output / "input_manifest.csv"),
                "input_manifest_sha256": manifest_sha,
                "selection_manifest": selection_audit,
            },
            "inference": {
                "device": str(device),
                "cuda_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                "torch_version": torch.__version__,
                "checkpoint_storage_dtype": "float32",
                "model_parameter_dtype": "float64",
                "xyz_output_dtype": "float64",
                "radius_decoder_dtype": "float32",
                "radius_output_dtype": "float32",
                "pooling_attention_dtype": "float32",
                "saved_prediction_dtype": "float64",
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "bf16": False,
                "tf32": False,
                "channels_last": bool(args.channels_last),
            },
            "prediction_format": {
                "pattern": "<case>/<case>_pred.npy",
                "shape": [12, 4],
                "columns": ["x_m", "y_m", "z_m", "radius_m"],
                "dtype": "float64",
                "postprocessing": "none; direct decoded AttentionCNN output",
            },
            "outputs": {
                "per_case_metrics": str(output / "per_case_metrics.csv"),
                "input_manifest": str(output / "input_manifest.csv"),
                "predictions_npz": str(output / "predictions.npz"),
            },
        }
    )
    write_evaluation(staging, _json_safe(summary), rows)
    confusion_path = staging / "evaluation_confusion.json"
    confusion_payload = json.loads(confusion_path.read_text(encoding="utf-8"))
    confusion_payload["fixed_stratified_primary"] = fixed_summary[
        "confusion_matrix_rows_true_cols_pred"
    ]
    confusion_payload["primary_for_cfd_geo_cohort"] = "fixed_stratified_primary"
    confusion_path.write_text(
        json.dumps(confusion_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (staging / "evaluation_per_case.csv").replace(staging / "per_case_metrics.csv")
    for artifact in list(staging.iterdir()):
        artifact.replace(output / artifact.name)
    staging.rmdir()
    return _json_safe(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", type=Path, default=DEFAULT_CASES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PREDS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--selection-manifest", type=Path, default=DEFAULT_SELECTION_MANIFEST
    )
    parser.add_argument("--expected-cases", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--channels-last", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.expected_cases, args.batch_size) < 1 or args.num_workers < 0:
        raise ValueError(
            "case count/batch size must be positive and workers non-negative"
        )
    summary = run(args)
    fixed = summary["grading"]["fixed_stratified_primary"]
    print(
        f"AttentionCNN synthetic cohort: n={summary['n']}, "
        f"RMSE_xyz={summary['geometry']['rmse_xyz_mm']['mean']:.6f} mm, "
        f"MAE_r={summary['geometry']['mae_r_mm']['mean']:.6f} mm, "
        f"fixed_stratified_grade_error={fixed['error_rate']:.2%} "
        f"({fixed['incorrect']}/{fixed['n']})"
    )
    print(f"Output: {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
