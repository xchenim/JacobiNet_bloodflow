"""Evaluate the 100 synthetic cases: flow fields, pressure drop, and wall shear."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import trimesh

from JacobiNetPINN_flowsolver import evaluate as checkpoint_evaluation
from JacobiNetPINN_flowsolver import metrics
from JacobiNetPINN_flowsolver.runtime import apply_matmul_precision_policy
from JacobiNetPINN_flowsolver.wss import area_weighted_metrics, compute_wss

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE_ROOT.parent
REGIONS = {
    "global": "All CFD volume sample points.",
    "middle_third": "1/3 <= s <= 2/3; the interval used for the table throat error.",
    "post_stenotic": "All points with s > 2/3 in eligible cases.",
    "eligibility": "At least 50 CFD points with negative local axial velocity at s > 2/3; reverse-flow points select cases only.",
}
TABLE_METRICS = {
    "pressure_error_pa": (3, 1),
    "pressure_ape_percent": (4, 1),
    "global_velocity": (6, 0),
    "global_pressure": (7, 0),
    "middle_third_velocity": (8, 0),
    "middle_third_pressure": (9, 0),
    "post_stenotic_velocity": (10, 0),
    "post_stenotic_pressure": (11, 0),
    "wss_relative_l2": (13, 0),
    "wss_low_area_error_pp": (14, 0),
    "wss_high_area_error_pp": (15, 0),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def verify_file(path, expected):
    if checkpoint_evaluation.file_sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def compare_values(observed, expected):
    """Report numerical differences without altering predictions or CFD values."""
    actual = np.asarray(observed, dtype=np.float64)
    reference = np.asarray(expected, dtype=np.float64)
    if (
        actual.shape != reference.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Reference comparison requires aligned finite arrays")
    return {
        "bitwise_equal": bool(np.array_equal(actual, reference)),
        "max_abs_difference": float(np.max(np.abs(actual - reference))),
    }


def surface_pressure(model, scales, case, boundary):
    with np.load(case / "centerline_sections.npz", allow_pickle=False) as geometry:
        centers = geometry["centers_m"].astype(np.float64)
        radii = geometry["radii_m"].astype(np.float64)
    index = 0 if boundary == "inlet" else -1
    surface = trimesh.load_mesh(case / "geometry" / f"{boundary}.stl", process=False)
    points = np.asarray(surface.triangles_center, dtype=np.float64)
    areas = np.asarray(surface.area_faces, dtype=np.float64)
    if not len(points) or not np.all(areas > 0):
        raise ValueError("Invalid inlet/outlet surface")
    radial = np.linalg.norm(points - centers[index], axis=1)
    q = (radii[index] ** 2 - radial**2) / radii[index] ** 2
    inputs = torch.as_tensor(
        np.column_stack(
            (
                points / scales.length_scale_m,
                q,
                np.full(len(points), 0.0 if index == 0 else 1.0),
            )
        ),
        dtype=torch.float64,
    )
    return metrics._surface_model_pressure(model, inputs, areas, scales)


def prepare_inputs(case, weights, target, inventory, data_root, weight_records):
    """Stage verified inputs for the core checkpoint loader in a temporary directory."""
    (target / "train").mkdir()
    for name in (
        "case_metadata.json",
        "centerline_sections.npz",
        "export_0.288h.csv",
        "train_1e-4.xlsx",
        "post_5e-4.xlsx",
    ):
        source = case / name
        verify_file(source, inventory[source.relative_to(data_root.parent).as_posix()])
        shutil.copyfile(source, target / name)
    for local_name, archived_name in (
        ("jacobinet.pth", "jacobinet.pth"),
        ("pinn.pth", "best_model_weights_stenosis.pth"),
    ):
        source = weights / local_name
        expected = weight_records[Path(local_name).stem]
        if expected["path"] != f"{case.name}/{local_name}":
            raise ValueError(
                f"Checkpoint path does not match case identity: {case.name}"
            )
        if source.stat().st_size != expected["bytes"]:
            raise ValueError(f"Checkpoint size mismatch: {source}")
        verify_file(source, expected["sha256"])
        shutil.copyfile(source, target / "train" / archived_name)


def evaluate_case(record, args, reference_info, inventory, output):
    cid = record["case_id"]
    case = args.data_root / "cases" / cid
    result = {
        "case_id": cid,
        "label_stenosis_percent": record["label_stenosis_percent"],
        "horizon_steps": record["horizon_steps"],
    }
    values = {}
    for name in (
        "geometry/inlet.stl",
        "geometry/outlet.stl",
        "evaluation/regional_predictions.npz",
        "wss/cfd_wall_faces.npz",
        "wss/selected_wall_wss.npz",
    ):
        if (
            (name.startswith("geometry/") and "pressure" not in args.metrics)
            or (name.startswith("evaluation/") and "fields" not in args.metrics)
            or (name.startswith("wss/") and "wss" not in args.metrics)
        ):
            continue
        source = case / name
        verify_file(
            source, inventory[source.relative_to(args.data_root.parent).as_posix()]
        )

    # The temporary case preserves the loader's checkpoint-role filename checks.
    with TemporaryDirectory(prefix="case_", dir=output) as temporary:
        target = Path(temporary).resolve()
        if not target.is_relative_to(output.resolve()):
            raise ValueError("Temporary case escaped the output directory")
        prepare_inputs(
            case,
            args.weights_root / cid,
            target,
            inventory,
            args.data_root,
            record["weights"],
        )
        checkpoint_evaluation.configure_horizon(record["horizon_steps"])
        checkpoint = target / "train/best_model_weights_stenosis.pth"
        payload, identity = checkpoint_evaluation.load_validated_checkpoint(checkpoint)
        scales, cfd, _ = checkpoint_evaluation.prepare_case(target, payload, identity)
        result["checkpoint_sha256"] = checkpoint_evaluation.file_sha256(checkpoint)
        result["geometry_sha256"] = scales.geometry_sha256
        models = {}

        def model_for(reference_device):
            device = reference_device if args.device == "reference" else args.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is required by the selected precision protocol; use --device cpu for a CPU replay"
                )
            if device not in models:
                models[device] = checkpoint_evaluation.build_flow_model(
                    checkpoint, payload, scales, torch.device(device)
                )
            return models[device], device

        if "fields" in args.metrics:
            model, device = model_for(reference_info["field_device"])
            with torch.inference_mode():
                velocity, pressure = metrics.predict_physical(model, cfd, scales)
            with np.load(
                case / "evaluation/regional_predictions.npz", allow_pickle=False
            ) as archived:
                axial = np.asarray(archived["cfd_axial_velocity_m_s"], dtype=np.float64)
                comparison = {
                    "velocity": compare_values(
                        velocity, archived["selected_velocity_m_s"]
                    ),
                    "pressure": compare_values(
                        pressure, archived["selected_pressure_pa"]
                    ),
                }
            s = cfd.points_physical[:, 4]
            masks, eligible, reverse_points = metrics.published_region_masks(s, axial)
            result["post_stenotic_eligible"] = eligible
            result["reverse_flow_point_count"] = reverse_points
            regional = {}
            for name, mask in masks.items():
                if not mask.any():
                    raise ValueError(f"Required region is empty: {cid}/{name}")
                measured = {
                    "n": int(mask.sum()),
                    "velocity_relative_l2": metrics._vector_relative_l2(
                        velocity[mask], cfd.velocity_m_s[mask]
                    ),
                    "pressure_relative_l2": metrics.relative_l2(
                        pressure[mask], cfd.pressure_pa[mask]
                    ),
                }
                regional[name] = measured
                values[name + "_velocity"] = measured["velocity_relative_l2"]
                values[name + "_pressure"] = measured["pressure_relative_l2"]
            result["fields"] = {
                "device": device,
                "metrics": regional,
                "reference_comparison": comparison,
            }
            np.savez_compressed(
                output / "fields.npz", velocity_m_s=velocity, pressure_pa=pressure
            )

        if "pressure" in args.metrics:
            model, device = model_for("cuda")
            drop = surface_pressure(model, scales, case, "inlet") - surface_pressure(
                model, scales, case, "outlet"
            )
            truth = float(
                read_json(case / "case_metadata.json")["cfd"]["final_physical_sample"][
                    "pressure_drop_pa"
                ]
            )
            error = abs(drop - truth)
            result["pressure"] = {
                "device": device,
                "pinn_pa": drop,
                "cfd_pa": truth,
                "absolute_error_pa": error,
                "ape_percent": 100 * error / max(abs(truth), 1.0e-12),
                "difference_from_reference_pa": drop
                - reference_info["pressure_drop_pa"],
            }
            values["pressure_error_pa"] = error
            values["pressure_ape_percent"] = result["pressure"]["ape_percent"]

        if "wss" in args.metrics:
            model, device = model_for(record["wss"]["inference_device"])
            with np.load(case / "wss/cfd_wall_faces.npz", allow_pickle=False) as wall:
                points = wall["centroid_m"].astype(np.float64)
                normal = wall["unit_normal"].astype(np.float64)
                area = wall["area_m2"].astype(np.float64)
                truth = wall["cfd_wss_pa"].astype(np.float64)
            predicted, speed = compute_wss(
                model,
                points,
                normal,
                length_scale_m=scales.length_scale_m,
                velocity_scale_m_s=scales.velocity_scale_m_s,
                viscosity_pa_s=scales.dynamic_viscosity_pa_s,
                device=torch.device(device),
                batch_size=4096,
            )
            measured = area_weighted_metrics(predicted, truth, area)
            with np.load(
                case / "wss/selected_wall_wss.npz", allow_pickle=False
            ) as archived:
                if not np.array_equal(
                    truth, archived["cfd_wss_pa"]
                ) or not np.array_equal(area, archived["area_m2"]):
                    raise ValueError("Archived wall-face ordering differs from CFD")
                comparison = compare_values(predicted, archived["pinn_wss_pa"])
            result["wss"] = {
                "device": device,
                "face_count": len(area),
                "metrics": measured,
                "reference_comparison": comparison,
            }
            values.update(
                wss_relative_l2=measured["relative_l2"],
                wss_low_area_error_pp=100 * measured["low_area_fraction_difference"],
                wss_high_area_error_pp=100 * measured["high_area_fraction_difference"],
            )
            np.savez_compressed(
                output / "wss.npz", pinn_wss_pa=predicted, pinn_wall_speed_m_s=speed
            )
    result.update(status="computed", table_metrics=values)
    return result


def aggregate(results, table_rows, complete):
    """Compute unrounded statistics; manuscript numbers are comparison targets only."""
    columns = [None, (0, 25), (25, 50), (50, 70), (70, 100)]
    records = []
    keys = [
        key
        for key in TABLE_METRICS
        if any(key in row.get("table_metrics", {}) for row in results)
    ]
    for key in keys:
        table_row, ddof = TABLE_METRICS[key]
        for column, band in enumerate(columns, 2):
            selected = [
                row
                for row in results
                if row["status"] == "computed"
                and key in row["table_metrics"]
                and (band is None or band[0] <= row["label_stenosis_percent"] < band[1])
                and (
                    not key.startswith("post_stenotic") or row["post_stenotic_eligible"]
                )
            ]
            numbers = np.asarray(
                [row["table_metrics"][key] for row in selected], dtype=np.float64
            )
            observed = (
                [float(numbers.mean()), float(numbers.std(ddof=ddof))]
                if len(numbers) > ddof
                else None
            )
            expected = table_rows[table_row - 1][column - 1]
            tokens = re.findall(r"[-+]?\d+(?:\.\d+)?", expected)
            comparisons = []
            if observed is not None:
                if len(tokens) != 2:
                    status = "fail" if complete else "not_compared_subset"
                else:
                    for value, token in zip(observed, tokens):
                        target = float(token)
                        decimals = len(token.split(".")[1]) if "." in token else 0
                        displayed = float(format(value, f".{decimals}f")) == target
                        relative = abs(value - target) / abs(target) if target else None
                        within = (
                            (
                                relative <= 0.03
                                or math.isclose(relative, 0.03, rel_tol=1e-12)
                            )
                            if relative is not None
                            else value == 0
                        )
                        comparisons.append(
                            {
                                "observed": value,
                                "expected": target,
                                "relative_difference": relative,
                                "within_3_percent": within,
                                "display_match": displayed,
                                "accepted": displayed or within,
                            }
                        )
                    status = (
                        (
                            "pass"
                            if all(item["accepted"] for item in comparisons)
                            else "fail"
                        )
                        if complete
                        else "not_compared_subset"
                    )
            else:
                status = (
                    ("not_applicable" if not tokens else "fail")
                    if complete
                    else "not_compared_subset"
                )
            records.append(
                {
                    "metric": key,
                    "stratum": "all" if band is None else f"{band[0]}-{band[1]}",
                    "n": len(numbers),
                    "ddof": ddof,
                    "mean": observed[0] if observed else None,
                    "sd": observed[1] if observed else None,
                    "expected_display": expected,
                    "components": comparisons if complete else [],
                    "status": status,
                }
            )
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=WORKSPACE / "synthetic_100")
    parser.add_argument("--weights-root", type=Path, default=PACKAGE_ROOT / "weights")
    parser.add_argument(
        "--output-root", type=Path, default=WORKSPACE / "outputs/flow_100"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=("fields", "pressure", "wss"),
        default=["fields", "pressure", "wss"],
    )
    parser.add_argument(
        "--device", choices=("reference", "cpu", "cuda"), default="reference"
    )
    parser.add_argument(
        "--case-id",
        action="append",
        help="Repeat to evaluate a subset; manuscript comparisons require all 100 cases",
    )
    args = parser.parse_args(argv)
    args.data_root, args.weights_root = (
        args.data_root.resolve(),
        args.weights_root.resolve(),
    )
    output = args.output_root.resolve()
    for protected in (args.data_root, PACKAGE_ROOT, args.weights_root):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError(
                "Output must be outside the code, dataset, and weights directories"
            )
    manifest = read_json(args.data_root / "manifest.json")
    ids = [entry["case_id"] for entry in manifest["cases"]]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("Expected 100 unique synthetic cases")
    configuration = read_json(Path(__file__).with_name("reference.json"))
    if set(configuration["cases"]) != set(ids):
        raise ValueError("Reference protocol and dataset cohorts differ")
    inventory_path = args.data_root / "checksums.csv"
    verify_file(inventory_path, manifest["checksums_sha256"])
    with inventory_path.open(encoding="utf-8", newline="") as handle:
        inventory = {row["path"]: row["sha256"] for row in csv.DictReader(handle)}
    if args.case_id and not set(args.case_id).issubset(ids):
        raise ValueError("Requested case is not in the synthetic cohort")
    selected = [
        row
        for row in manifest["cases"]
        if not args.case_id or row["case_id"] in args.case_id
    ]
    output.mkdir(parents=True, exist_ok=False)
    apply_matmul_precision_policy("ieee")
    results = []
    for record in selected:
        destination = output / "cases" / record["case_id"]
        destination.mkdir(parents=True)
        try:
            result = evaluate_case(
                record,
                args,
                configuration["cases"][record["case_id"]],
                inventory,
                destination,
            )
        except Exception as error:
            result = {
                "case_id": record["case_id"],
                "label_stenosis_percent": record["label_stenosis_percent"],
                "status": "failed",
                "error": str(error),
            }
        results.append(result)
        write_json(destination / "result.json", result)
        print(
            json.dumps(
                {
                    "completed": len(results),
                    "total": len(selected),
                    "case_id": record["case_id"],
                    "status": result["status"],
                    "error": result.get("error"),
                }
            ),
            flush=True,
        )
    complete = len(results) == 100 and all(
        row["status"] == "computed" for row in results
    )
    comparison = aggregate(results, configuration["table_rows"], complete)
    write_json(output / "all_cases.json", results)
    write_json(output / "table_comparison.json", comparison)
    fields = [
        row["fields"]["reference_comparison"] for row in results if "fields" in row
    ]
    walls = [row["wss"]["reference_comparison"] for row in results if "wss" in row]
    summary = {
        "selected_cases": len(selected),
        "computed": sum(row["status"] == "computed" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "full_cohort": complete,
        "metrics": args.metrics,
        "network_dtype": "float32",
        "metric_dtype": "float64",
        "region_definitions": REGIONS,
        "field_arrays_bitwise_equal": sum(
            item["velocity"]["bitwise_equal"] and item["pressure"]["bitwise_equal"]
            for item in fields
        ),
        "wss_arrays_bitwise_equal": sum(item["bitwise_equal"] for item in walls),
        "table_cells": len(comparison),
        "table_cells_failed": sum(row["status"] == "fail" for row in comparison),
        "comparison_policy": "Each mean and SD must match manuscript display precision or differ by at most 3%; zero references use display precision or exact zero.",
        "manuscript_comparison_complete": complete and len(comparison) == 55,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return int(summary["failed"] > 0 or summary["table_cells_failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
