"""Float64 evaluation, regional errors, and pressure-drop metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

if __package__ and __package__.startswith("JacobiNetPINN_flowsolver"):
    from .dataset import CFDReference
    from .runtime import GeometryRegions
    from .physics import PhysicsScales
    from .runtime import model_dtype
else:
    from dataset import CFDReference
    from runtime import GeometryRegions
    from physics import PhysicsScales
    from runtime import model_dtype


INTERIOR_TOLERANCE = 1.0e-8
ENDPOINT_TOLERANCE = 1.0e-7


def relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    denominator = np.sqrt(np.mean(truth**2)) + 1.0e-12
    return float(np.sqrt(np.mean((prediction - truth) ** 2)) / denominator)


def _vector_relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    numerator = np.sqrt(np.mean(np.sum((prediction - truth) ** 2, axis=1)))
    denominator = np.sqrt(np.mean(np.sum(truth**2, axis=1))) + 1.0e-12
    return float(numerator / denominator)


def predict_physical(
    model: torch.nn.Module,
    reference: CFDReference,
    scales: PhysicsScales,
    *,
    inference_batch_size: int = 32768,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    points = reference.points_physical.copy()
    points[:, :3] /= scales.length_scale_m
    predictions = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(points), inference_batch_size):
            values = torch.as_tensor(
                points[start : start + inference_batch_size],
                dtype=model_dtype(model),
                device=device,
            )
            predictions.append(model(values).cpu().numpy().astype(np.float64))
    if was_training:
        model.train()
    nondimensional = np.concatenate(predictions, axis=0).astype(np.float64)
    velocity = nondimensional[:, :3] * scales.velocity_scale_m_s
    pressure = nondimensional[:, 3] * scales.pressure_scale_pa
    return velocity, pressure


def published_region_masks(s, axial_velocity_m_s):
    """Return paper regions and eligibility from CFD local axial velocity.

    Negative-velocity points select cases only. Eligible regional errors use
    every point beyond s=2/3, including points with forward flow.
    """
    s = np.asarray(s, dtype=np.float64)
    axial = np.asarray(axial_velocity_m_s, dtype=np.float64)
    if (
        s.ndim != 1
        or axial.shape != s.shape
        or not np.isfinite(s).all()
        or not np.isfinite(axial).all()
    ):
        raise ValueError(
            "CFD axial velocities and coordinates must be finite and aligned"
        )
    masks = {
        "global": np.ones(len(s), dtype=bool),
        "middle_third": (s >= 1 / 3) & (s <= 2 / 3),
        "post_stenotic": s > 2 / 3,
    }
    reverse_points = int(np.count_nonzero(masks["post_stenotic"] & (axial < 0)))
    return masks, reverse_points >= 50, reverse_points


def evaluate_model(
    model: torch.nn.Module,
    reference: CFDReference,
    scales: PhysicsScales,
    regions: GeometryRegions,
    *,
    prediction_callback=None,
    axial_velocity_m_s: np.ndarray | None = None,
) -> dict[str, float | int | bool | None]:
    """Evaluate global and geometry diagnostics; paper recirculation needs CFD axial velocities."""
    velocity, pressure = predict_physical(model, reference, scales)
    if prediction_callback is not None:
        prediction_callback(model, reference, scales, velocity, pressure)
    speed = np.linalg.norm(velocity, axis=1)
    output: dict[str, float | int | bool | None] = {
        "evaluation_points": int(len(reference.points_physical)),
        "velocity_vector_relative_l2": _vector_relative_l2(
            velocity, reference.velocity_m_s
        ),
        "speed_magnitude_relative_l2": relative_l2(speed, reference.speed_m_s),
        "u_relative_l2": relative_l2(velocity[:, 0], reference.velocity_m_s[:, 0]),
        "v_relative_l2": relative_l2(velocity[:, 1], reference.velocity_m_s[:, 1]),
        "w_relative_l2": relative_l2(velocity[:, 2], reference.velocity_m_s[:, 2]),
        "pressure_relative_l2": relative_l2(pressure, reference.pressure_pa),
    }
    z = reference.z_norm
    masks = {
        f"geometry_{name}": mask
        for name, mask in regions.numpy_masks(z).items()
        if name in ("upstream", "throat")
    }
    output["post_stenotic_eligible"] = None
    if axial_velocity_m_s is not None:
        paper_masks, eligible, reverse_points = published_region_masks(
            z, axial_velocity_m_s
        )
        output["post_stenotic_eligible"] = eligible
        output["reverse_flow_point_count"] = reverse_points
        if eligible:
            masks["post_stenotic"] = paper_masks["post_stenotic"]
    for name in masks:
        mask = masks[name]
        if not mask.any():
            continue
        output[f"{name}_point_count"] = int(mask.sum())
        output[f"{name}_velocity_vector_relative_l2"] = _vector_relative_l2(
            velocity[mask], reference.velocity_m_s[mask]
        )
        output[f"{name}_speed_magnitude_relative_l2"] = relative_l2(
            speed[mask], reference.speed_m_s[mask]
        )
        output[f"{name}_pressure_relative_l2"] = relative_l2(
            pressure[mask], reference.pressure_pa[mask]
        )
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _section_model_pressure(
    model: torch.nn.Module,
    points: torch.Tensor,
    scales: PhysicsScales,
) -> tuple[float, int]:
    selected = points[points[:, 3] > INTERIOR_TOLERANCE]
    if not len(selected):
        raise ValueError("Cross-section has no interior points")
    was_training = model.training
    model.eval()
    with torch.no_grad():
        pressure = model(selected)[:, 3].cpu().numpy().astype(np.float64)
    if was_training:
        model.train()
    return float(np.mean(pressure * scales.pressure_scale_pa)), int(len(selected))


def _surface_quadrature(
    workspace: Path,
    case_id: str,
    boundary: str,
    scales: PhysicsScales,
) -> tuple[torch.Tensor, np.ndarray, Path]:
    mesh_root = workspace / "synthetic_100" / "cases" / case_id
    with np.load(mesh_root / "centerline_sections.npz") as geometry:
        centers = np.asarray(geometry["centers_m"], dtype=np.float64)
        radii = np.asarray(geometry["radii_m"], dtype=np.float64)
    index = 0 if boundary == "inlet" else -1
    z_norm = 0.0 if boundary == "inlet" else 1.0
    surface_path = mesh_root / "geometry" / f"{boundary}.stl"
    surface = trimesh.load_mesh(surface_path, process=False)
    centroids = np.asarray(surface.triangles_center, dtype=np.float64)
    areas = np.asarray(surface.area_faces, dtype=np.float64)
    if not len(centroids) or not np.all(areas > 0.0):
        raise ValueError(f"Invalid surface mesh: {surface_path}")
    radial = np.linalg.norm(centroids - centers[index], axis=1)
    dist_ratio = (radii[index] ** 2 - radial**2) / radii[index] ** 2
    values = np.column_stack(
        (
            centroids / scales.length_scale_m,
            dist_ratio,
            np.full(len(centroids), z_norm),
        )
    )
    return torch.as_tensor(values, dtype=torch.float64), areas, surface_path


def _surface_model_pressure(
    model: torch.nn.Module,
    points: torch.Tensor,
    areas: np.ndarray,
    scales: PhysicsScales,
) -> float:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        pressure = (
            model(points.to(device=device, dtype=model_dtype(model)))[:, 3]
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    if was_training:
        model.train()
    return float(np.average(pressure * scales.pressure_scale_pa, weights=areas))


def pressure_drop_comparison(
    model: torch.nn.Module,
    inlet: torch.Tensor,
    outlet: torch.Tensor,
    reference: CFDReference,
    scales: PhysicsScales,
    workspace: Path,
    case_id: str,
) -> dict[str, Any]:
    sampled_inlet, sampled_inlet_count = _section_model_pressure(model, inlet, scales)
    sampled_outlet, sampled_outlet_count = _section_model_pressure(
        model, outlet, scales
    )
    inlet_points, inlet_areas, inlet_path = _surface_quadrature(
        workspace, case_id, "inlet", scales
    )
    outlet_points, outlet_areas, outlet_path = _surface_quadrature(
        workspace, case_id, "outlet", scales
    )
    pinn_inlet = _surface_model_pressure(model, inlet_points, inlet_areas, scales)
    pinn_outlet = _surface_model_pressure(model, outlet_points, outlet_areas, scales)
    pinn_drop = pinn_inlet - pinn_outlet

    reference_points = reference.points_physical
    interior = reference_points[:, 3] > INTERIOR_TOLERANCE
    inlet_mask = interior & np.isclose(
        reference_points[:, 4], 0.0, rtol=0.0, atol=ENDPOINT_TOLERANCE
    )
    outlet_mask = interior & np.isclose(
        reference_points[:, 4], 1.0, rtol=0.0, atol=ENDPOINT_TOLERANCE
    )
    sampled_cfd_inlet = float(np.mean(reference.pressure_pa[inlet_mask]))
    sampled_cfd_outlet = float(np.mean(reference.pressure_pa[outlet_mask]))
    sampled_cfd_drop = sampled_cfd_inlet - sampled_cfd_outlet

    formal_path = workspace / "synthetic_100" / "cases" / case_id / "case_metadata.json"
    formal_report = json.loads(formal_path.read_text(encoding="utf-8"))["cfd"]
    sample = formal_report.get("final_physical_sample") or formal_report.get(
        "convergence", {}
    ).get("latest_physical_sample", {})
    formal_drop = float(sample["pressure_drop_pa"])
    denominator = max(abs(formal_drop), 1.0e-12)
    return {
        "display_pinn_over_cfd_pa": f"{pinn_drop:.3f}/{formal_drop:.3f}",
        "pinn_pressure_drop_pa": pinn_drop,
        "cfd_pressure_drop_pa": formal_drop,
        "absolute_error_pa": abs(pinn_drop - formal_drop),
        "relative_error": abs(pinn_drop - formal_drop) / denominator,
        "pinn_inlet_mean_pressure_pa": pinn_inlet,
        "pinn_outlet_mean_pressure_pa": pinn_outlet,
        "sampled_pinn_pressure_drop_pa": sampled_inlet - sampled_outlet,
        "sampled_pinn_inlet_mean_pressure_pa": sampled_inlet,
        "sampled_pinn_outlet_mean_pressure_pa": sampled_outlet,
        "sampled_pinn_inlet_interior_count": sampled_inlet_count,
        "sampled_pinn_outlet_interior_count": sampled_outlet_count,
        "sampled_cfd_pressure_drop_pa": sampled_cfd_drop,
        "sampled_cfd_inlet_mean_pressure_pa": sampled_cfd_inlet,
        "sampled_cfd_outlet_mean_pressure_pa": sampled_cfd_outlet,
        "pinn_inlet_surface": str(inlet_path),
        "pinn_outlet_surface": str(outlet_path),
        "pinn_inlet_surface_sha256": _sha256(inlet_path),
        "pinn_outlet_surface_sha256": _sha256(outlet_path),
        "cfd_formal_solver_report": str(formal_path),
        "cfd_formal_solver_report_sha256": _sha256(formal_path),
        "definition": (
            "PINN inlet-minus-outlet STL triangle-area-weighted pressure; "
            "CFD Fluent inlet-minus-outlet area-average pressure, both in Pa"
        ),
        "relative_error_definition": "abs(PINN-CFD)/abs(CFD)",
        "display_convention": "PINN/CFD",
    }
