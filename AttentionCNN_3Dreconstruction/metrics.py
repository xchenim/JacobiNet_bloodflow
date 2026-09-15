"""Evaluation metrics used for the locked AttentionCNN model."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .dataset import min_preserving_sample
except ImportError:
    from dataset import min_preserving_sample


THRESHOLDS = np.asarray((25.0, 50.0, 70.0), dtype=np.float64)
GRADE_NAMES = ("0-24%", "25-49%", "50-69%", "70-99%")


def legacy_indices(num_raw_points: int, num_points: int = 12) -> np.ndarray:
    return np.linspace(0, num_raw_points - 1, num_points).astype(np.int64)


def severity_percent(radii: Any) -> np.ndarray:
    if isinstance(radii, np.ndarray) and radii.dtype != object:
        values = np.asarray(radii, dtype=np.float64)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or np.any(values <= 0):
            raise ValueError("radii must have shape (B, P) and be positive")
        return (1.0 - values.min(axis=1) / values[:, 0]) * 100.0
    return np.asarray(
        [severity_percent(np.asarray(row))[0] for row in radii], dtype=np.float64
    )


def grades(severity: Any) -> np.ndarray:
    return np.searchsorted(THRESHOLDS, np.asarray(severity), side="right").astype(
        np.int64
    )


def geometry_errors_mm(
    predicted_xyz: np.ndarray,
    true_xyz: np.ndarray,
    predicted_radius: np.ndarray,
    true_radius: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(predicted_xyz, dtype=np.float64)
    target_xyz = np.asarray(true_xyz, dtype=np.float64)
    radius = np.asarray(predicted_radius, dtype=np.float64)
    target_radius = np.asarray(true_radius, dtype=np.float64)
    rmse = np.sqrt(np.mean(np.sum((xyz - target_xyz) ** 2, axis=-1), axis=-1))
    mae = np.mean(np.abs(radius - target_radius), axis=-1)
    return rmse * 1000.0, mae * 1000.0


def summarize(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def confusion_matrix(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = np.zeros((4, 4), dtype=np.int64)
    np.add.at(result, (truth, prediction), 1)
    return result


def wilson(successes: int, total: int, confidence: float = 0.95) -> list[float]:
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, centre - half), min(1.0, centre + half)]


def grading_summary(
    true_radii: Sequence[np.ndarray] | np.ndarray, predicted_radii: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    true_severity = severity_percent(true_radii)
    predicted_severity = severity_percent(predicted_radii)
    true_grade, predicted_grade = grades(true_severity), grades(predicted_severity)
    matrix = confusion_matrix(true_grade, predicted_grade)
    residual = predicted_severity - true_severity
    correct = predicted_grade == true_grade
    under = predicted_grade < true_grade
    over = predicted_grade > true_grade
    total, number_correct = len(correct), int(correct.sum())
    interval = wilson(number_correct, total)
    per_grade = []
    for grade, name in enumerate(GRADE_NAMES):
        support = int(matrix[grade].sum())
        grade_correct = int(matrix[grade, grade])
        if support:
            accuracy = grade_correct / support
            grade_interval = wilson(grade_correct, support)
            error_rate = 1.0 - accuracy
            error_interval = [1.0 - grade_interval[1], 1.0 - grade_interval[0]]
        else:
            accuracy = float("nan")
            grade_interval = [float("nan"), float("nan")]
            error_rate = float("nan")
            error_interval = [float("nan"), float("nan")]
        per_grade.append(
            {
                "grade": grade,
                "name": name,
                "support": support,
                "correct": grade_correct,
                "accuracy": accuracy,
                "error_rate": error_rate,
                "accuracy_wilson_ci": grade_interval,
                "error_rate_wilson_ci": error_interval,
            }
        )
    summary = {
        "n": total,
        "severity_definition": "100 * (1 - hard_min(radius) / inlet_radius)",
        "grade_thresholds_percent": THRESHOLDS.tolist(),
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


def evaluate_predictions(
    case_ids: Sequence[str],
    predicted_xyz: np.ndarray,
    predicted_radius: np.ndarray,
    raw_labels: Sequence[np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    xyz = np.asarray(predicted_xyz, dtype=np.float64)
    radius = np.asarray(predicted_radius, dtype=np.float64)
    if radius.ndim == 3:
        radius = radius[..., 0]
    if xyz.shape != (len(case_ids), 12, 3) or radius.shape != (len(case_ids), 12):
        raise ValueError("predictions must have shapes (B,12,3) and (B,12)")
    targets = np.stack([min_preserving_sample(label) for label in raw_labels])
    rmse, mae = geometry_errors_mm(xyz, targets[:, :, :3], radius, targets[:, :, 3])
    legacy_targets = np.stack(
        [label[legacy_indices(label.shape[0]), :4] for label in raw_labels]
    )
    legacy_rmse, legacy_mae = geometry_errors_mm(
        xyz, legacy_targets[:, :, :3], radius, legacy_targets[:, :, 3]
    )
    legacy_summary, legacy_arrays = grading_summary(legacy_targets[:, :, 3], radius)
    full_summary, full_arrays = grading_summary(
        [label[:, 3] for label in raw_labels], radius
    )
    summary = {
        "n": len(case_ids),
        "units": {"coordinates_and_radius_input": "m", "geometry_metrics": "mm"},
        "geometry_target": "legacy12_min_preserving_pair",
        "geometry": {"rmse_xyz_mm": summarize(rmse), "mae_r_mm": summarize(mae)},
        "geometry_legacy12_compatibility": {
            "rmse_xyz_mm": summarize(legacy_rmse),
            "mae_r_mm": summarize(legacy_mae),
        },
        "grading": {
            "legacy12_truth": legacy_summary,
            "full_raw_truth_primary": full_summary,
        },
        "full_raw_is_primary_for_grading": True,
        "predicted_grade_source": "hard r_min / inlet from predicted radii",
    }
    rows = []
    for index, case_id in enumerate(case_ids):
        rows.append(
            {
                "case": case_id,
                "raw_point_count": int(raw_labels[index].shape[0]),
                "rmse_xyz_mm": float(rmse[index]),
                "mae_r_mm": float(mae[index]),
                "pred_severity_percent": float(
                    full_arrays["pred_severity_percent"][index]
                ),
                "pred_grade": int(full_arrays["pred_grade"][index]),
                "legacy12_true_severity_percent": float(
                    legacy_arrays["true_severity_percent"][index]
                ),
                "legacy12_true_grade": int(legacy_arrays["true_grade"][index]),
                "legacy12_correct": bool(legacy_arrays["correct"][index]),
                "full_raw_true_severity_percent": float(
                    full_arrays["true_severity_percent"][index]
                ),
                "full_raw_true_grade": int(full_arrays["true_grade"][index]),
                "full_raw_correct": bool(full_arrays["correct"][index]),
                "full_raw_undergraded": bool(full_arrays["undergraded"][index]),
                "full_raw_overgraded": bool(full_arrays["overgraded"][index]),
            }
        )
    return summary, rows


def write_evaluation(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    output = Path(output_dir)
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output / "evaluation_per_case.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    confusion = {
        "rows": "true_grade_0_to_3",
        "columns": "predicted_grade_0_to_3",
        "legacy12_truth": summary["grading"]["legacy12_truth"][
            "confusion_matrix_rows_true_cols_pred"
        ],
        "full_raw_truth_primary": summary["grading"]["full_raw_truth_primary"][
            "confusion_matrix_rows_true_cols_pred"
        ],
    }
    (output / "evaluation_confusion.json").write_text(
        json.dumps(confusion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


__all__ = ["evaluate_predictions", "write_evaluation"]
