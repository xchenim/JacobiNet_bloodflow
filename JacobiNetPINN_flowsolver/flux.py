"""Integrate predicted flow across the supplied vessel cross-sections."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn

MONITOR_RADIAL_ORDER = 12
MONITOR_ANGULAR_COUNT = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
        raise ValueError("Frame vectors must have finite, nonzero norms")
    return values / norms


def load_centerline_geometry(path: str | Path) -> dict[str, object]:
    """Load and strictly validate the actual circular section geometry."""

    geometry_path = Path(path).expanduser().resolve()
    if not geometry_path.is_file():
        raise FileNotFoundError(geometry_path)
    with np.load(geometry_path) as data:
        required = {
            "centers_m",
            "radii_m",
            "tangents",
            "normals",
            "binormals",
            "rings_m",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"Centerline geometry is missing fields: {sorted(missing)}"
            )
        centers = np.asarray(data["centers_m"], dtype=np.float64)
        radii = np.asarray(data["radii_m"], dtype=np.float64)
        tangents = np.asarray(data["tangents"], dtype=np.float64)
        normals = np.asarray(data["normals"], dtype=np.float64)
        binormals = np.asarray(data["binormals"], dtype=np.float64)
        rings = np.asarray(data["rings_m"], dtype=np.float64)

    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) < 4:
        raise ValueError("centers_m must have shape (N, 3) with N >= 4")
    if radii.shape != (len(centers),):
        raise ValueError("radii_m must have shape (N,)")
    for name, values in (
        ("tangents", tangents),
        ("normals", normals),
        ("binormals", binormals),
    ):
        if values.shape != centers.shape:
            raise ValueError(f"{name} must have shape (N, 3)")
    if rings.ndim != 3 or rings.shape[0] != len(centers) or rings.shape[2] != 3:
        raise ValueError("rings_m must have shape (N, M, 3)")
    if rings.shape[1] < 16:
        raise ValueError("rings_m needs at least 16 boundary samples per section")
    arrays = (centers, radii, tangents, normals, binormals, rings)
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("Centerline section arrays must be finite")
    if np.any(radii <= 0.0):
        raise ValueError("Centerline radii must be positive")

    section_lengths = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    if np.any(~np.isfinite(section_lengths)) or np.any(section_lengths <= 0.0):
        raise ValueError("Centerline centers must define positive arc steps")
    arc_m = np.concatenate(([0.0], np.cumsum(section_lengths)))
    axial_nodes = arc_m / arc_m[-1]

    tangents = _unit_rows(tangents)
    normals = _unit_rows(normals)
    normals = normals - np.sum(normals * tangents, axis=1, keepdims=True) * tangents
    normals = _unit_rows(normals)
    binormals = _unit_rows(binormals)
    section_normals = _unit_rows(np.cross(normals, binormals))
    frame_alignment = np.sum(section_normals * tangents, axis=1)
    if float(np.min(frame_alignment)) <= 0.999:
        raise ValueError("Section normal/binormal frames disagree with tangents")

    chord_tangents = _unit_rows(np.gradient(centers, axis=0))
    tangent_alignment = np.sum(tangents * chord_tangents, axis=1)
    adjacent_alignment = np.sum(tangents[:-1] * tangents[1:], axis=1)
    if float(np.min(tangent_alignment)) <= 0.0:
        raise ValueError("Stored tangents are not consistently inlet-to-outlet")
    if float(np.min(adjacent_alignment)) <= 0.0:
        raise ValueError("Stored tangents contain a direction reversal")

    ring_offsets = rings - centers[:, None, :]
    ring_radius = np.linalg.norm(ring_offsets, axis=2)
    ring_radius_relative_error = float(
        np.max(np.abs(ring_radius - radii[:, None]) / radii[:, None])
    )
    ring_plane_relative_error = float(
        np.max(
            np.abs(np.sum(ring_offsets * section_normals[:, None, :], axis=2))
            / radii[:, None]
        )
    )
    if ring_radius_relative_error > 1.0e-8 or ring_plane_relative_error > 1.0e-8:
        raise ValueError("rings_m are inconsistent with the declared true sections")

    inlet_direction_error = float(
        np.linalg.norm(tangents[0] - np.array([0.0, 0.0, 1.0]))
    )
    if inlet_direction_error > 1.0e-6:
        raise ValueError(
            "Local base flow requires the centerline inlet tangent to be +z; "
            f"direction error is {inlet_direction_error:.6g}"
        )

    return {
        "path": str(geometry_path),
        "sha256": _sha256(geometry_path),
        "true_sections_sha256": _array_sha256(
            centers, radii, tangents, normals, binormals, rings
        ),
        "centers_m": centers,
        "radii_m": radii,
        "tangents": tangents,
        "normals": normals,
        "binormals": binormals,
        "section_normals": section_normals,
        "rings_m": rings,
        "arc_m": arc_m,
        "axial_nodes": axial_nodes,
        "minimum_tangent_chord_alignment": float(np.min(tangent_alignment)),
        "minimum_adjacent_tangent_alignment": float(np.min(adjacent_alignment)),
        "minimum_frame_tangent_alignment": float(np.min(frame_alignment)),
        "maximum_ring_radius_relative_error": ring_radius_relative_error,
        "maximum_ring_plane_relative_error": ring_plane_relative_error,
        "inlet_direction_error": inlet_direction_error,
    }


def _disk_quadrature(
    radial_order: int, angular_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if radial_order < 2 or angular_count < 8:
        raise ValueError("True-section quadrature grid is too small")
    abscissa, radial_weights = np.polynomial.legendre.leggauss(radial_order)
    radius = 0.5 * (abscissa + 1.0)
    radial_weights = 0.5 * radial_weights
    theta = 2.0 * np.pi * np.arange(angular_count, dtype=np.float64) / angular_count
    rr, tt = np.meshgrid(radius, theta, indexing="ij")
    ww, _ = np.meshgrid(radial_weights, theta, indexing="ij")
    weights = ww * rr * (2.0 * np.pi / angular_count)
    return rr.ravel(), tt.ravel(), weights.ravel()


def _section_points(
    geometry: dict[str, object],
    index: int,
    radial: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    centers = np.asarray(geometry["centers_m"])
    radii = np.asarray(geometry["radii_m"])
    normals = np.asarray(geometry["normals"])
    binormals = np.asarray(geometry["binormals"])
    return centers[index] + radii[index] * radial[:, None] * (
        np.cos(theta)[:, None] * normals[index]
        + np.sin(theta)[:, None] * binormals[index]
    )


def _flux_monitor_summary(ratios: np.ndarray) -> dict[str, Any]:
    median = float(np.median(ratios))
    return {
        "section_count": int(len(ratios)),
        "section_q_ratios": [float(value) for value in ratios],
        "minimum_q_ratio": float(np.min(ratios)),
        "median_q_ratio": median,
        "maximum_q_ratio": float(np.max(ratios)),
        "maximum_abs_q_ratio": float(np.max(np.abs(ratios))),
        "maximum_abs_relative_error_to_target": float(np.max(np.abs(ratios - 1.0))),
        "maximum_relative_spread_about_section_median": float(
            np.max(np.abs(ratios - median)) / max(abs(median), 1.0e-30)
        ),
    }


def true_section_flux_monitor(
    model: nn.Module,
    geometry_path: str | Path,
    *,
    length_scale_m: float,
    inlet_peak_velocity_nondim: float,
    radial_order: int = MONITOR_RADIAL_ORDER,
    angular_count: int = MONITOR_ANGULAR_COUNT,
) -> dict[str, Any]:
    """Forward-only actual-section Q monitor; never used in optimization."""

    geometry = load_centerline_geometry(geometry_path)
    radial, theta, weights = _disk_quadrature(radial_order, angular_count)
    radii = np.asarray(geometry["radii_m"])
    section_normals = np.asarray(geometry["section_normals"])
    reference_flow = (
        0.5 * np.pi * float(radii[0] ** 2) * float(inlet_peak_velocity_nondim)
    )
    if reference_flow <= 0.0:
        raise ValueError("Monitor reference flow must be positive")
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    total_ratios: list[float] = []
    was_training = model.training
    model.eval()
    rng_before = torch.get_rng_state().clone()
    with torch.inference_mode():
        for index in range(len(radii)):
            points_m = _section_points(geometry, index, radial, theta)
            values = torch.zeros((len(points_m), 5), device=device, dtype=dtype)
            values[:, :3] = torch.from_numpy(points_m / length_scale_m).to(
                device=device, dtype=dtype
            )
            total_velocity = model(values)[:, :3]
            normal = torch.from_numpy(section_normals[index]).to(
                device=device, dtype=dtype
            )
            total_density = (
                torch.sum(total_velocity * normal, dim=1).double().cpu().numpy()
            )
            total_flow = float(radii[index] ** 2) * float(
                np.dot(weights, total_density)
            )
            total_ratios.append(total_flow / reference_flow)
    if was_training:
        model.train()
    report: dict[str, Any] = {
        "mode": "forward_only_true_section_flux_monitor_v1",
        "forward_only": True,
        "included_in_loss": False,
        "geometry_sha256": str(geometry["sha256"]),
        "true_sections_sha256": str(geometry["true_sections_sha256"]),
        "quadrature": {
            "radial_rule": "gauss_legendre",
            "radial_order": radial_order,
            "angular_rule": "uniform_periodic",
            "angular_count": angular_count,
        },
        "normalization": "Q/(U_peak*pi*R_in^2/2)",
        "total_predicted_velocity": _flux_monitor_summary(
            np.asarray(total_ratios, dtype=np.float64)
        ),
        "rng_unchanged": bool(torch.equal(rng_before, torch.get_rng_state())),
    }
    report["report_payload_sha256"] = _canonical_sha256(report)
    return report
