"""Evaluate wall shear stress at native mesh-face centers."""

from __future__ import annotations

import numpy as np

import torch


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(sorted_weights)
    if cumulative[-1] <= 0.0:
        raise ValueError("weights must have positive sum")
    target = q * cumulative[-1]
    return float(
        sorted_values[min(np.searchsorted(cumulative, target), len(values) - 1)]
    )


def weighted_pearson(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    w64 = np.asarray(weights, dtype=np.float64)
    total = float(w64.sum())
    mx = float(np.sum(w64 * x64) / total)
    my = float(np.sum(w64 * y64) / total)
    dx = x64 - mx
    dy = y64 - my
    denominator = float(np.sqrt(np.sum(w64 * dx * dx) * np.sum(w64 * dy * dy)))
    if denominator == 0.0:
        return 1.0 if np.array_equal(x64, y64) else 0.0
    return float(np.sum(w64 * dx * dy) / denominator)


def rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = 0.5 * (position + end - 1)
        position = end
    return ranks


def area_weighted_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    area: np.ndarray,
) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    if predicted.shape != reference.shape or predicted.shape != area.shape:
        raise ValueError("WSS and area arrays must have identical shape")
    if not (
        np.isfinite(predicted).all()
        and np.isfinite(reference).all()
        and np.isfinite(area).all()
    ):
        raise ValueError("WSS metrics refuse non-finite arrays")
    if np.any(predicted < 0.0) or np.any(reference < 0.0) or np.any(area <= 0.0):
        raise ValueError("WSS must be non-negative and areas positive")
    total_area = float(area.sum())
    error = predicted - reference
    reference_rms = float(np.sqrt(np.sum(area * reference**2) / total_area))
    if reference_rms <= 0.0:
        raise ValueError("Reference WSS RMS must be positive")
    result = {
        "area_m2": total_area,
        "mae_pa": float(np.sum(area * np.abs(error)) / total_area),
        "bias_pa": float(np.sum(area * error) / total_area),
        "rmse_pa": float(np.sqrt(np.sum(area * error**2) / total_area)),
        "relative_l2": float(
            np.sqrt(np.sum(area * error**2) / total_area) / reference_rms
        ),
        "spatial_spearman_area_weighted": weighted_pearson(
            rank_average(predicted), rank_average(reference), area
        ),
    }
    for label, values in (("pinn", predicted), ("cfd", reference)):
        result[f"{label}_mean_pa"] = float(np.sum(area * values) / total_area)
        result[f"{label}_median_pa"] = weighted_quantile(values, area, 0.50)
        result[f"{label}_p25_pa"] = weighted_quantile(values, area, 0.25)
        result[f"{label}_p75_pa"] = weighted_quantile(values, area, 0.75)
        result[f"{label}_p95_pa"] = weighted_quantile(values, area, 0.95)
        result[f"{label}_low_area_fraction"] = float(
            np.sum(area[values < 0.4]) / total_area
        )
        result[f"{label}_high_area_fraction"] = float(
            np.sum(area[values > 4.0]) / total_area
        )
    result["low_area_fraction_difference"] = abs(
        result["pinn_low_area_fraction"] - result["cfd_low_area_fraction"]
    )
    result["high_area_fraction_difference"] = abs(
        result["pinn_high_area_fraction"] - result["cfd_high_area_fraction"]
    )
    return result


def compute_wss(
    model: torch.nn.Module,
    centroid_m: np.ndarray,
    unit_normal: np.ndarray,
    *,
    length_scale_m: float,
    velocity_scale_m_s: float,
    viscosity_pa_s: float,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.empty(len(centroid_m), dtype=np.float64)
    wall_speed = np.empty(len(centroid_m), dtype=np.float64)
    gradient_scale = float(velocity_scale_m_s / length_scale_m)
    model.eval()
    for start in range(0, len(centroid_m), batch_size):
        end = min(start + batch_size, len(centroid_m))
        xyz = (
            torch.as_tensor(
                centroid_m[start:end] / length_scale_m,
                dtype=torch.float32,
                device=device,
            )
            .detach()
            .clone()
            .requires_grad_(True)
        )
        output = model(xyz)
        velocity = output[:, :3]
        rows = []
        for component in range(3):
            gradient = torch.autograd.grad(
                velocity[:, component],
                xyz,
                grad_outputs=torch.ones_like(velocity[:, component]),
                retain_graph=component < 2,
                create_graph=False,
            )[0]
            rows.append(gradient)
        jacobian = torch.stack(rows, dim=1) * gradient_scale
        normal = torch.as_tensor(
            unit_normal[start:end], dtype=torch.float32, device=device
        )
        symmetric = jacobian + jacobian.transpose(1, 2)
        traction = torch.bmm(symmetric, normal.unsqueeze(2)).squeeze(2)
        tangential = traction - normal * torch.sum(
            traction * normal, dim=1, keepdim=True
        )
        tau = float(viscosity_pa_s) * tangential
        predicted[start:end] = (
            torch.linalg.vector_norm(tau, dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        wall_speed[start:end] = (
            torch.linalg.vector_norm(velocity * velocity_scale_m_s, dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    if not np.isfinite(predicted).all() or np.any(predicted < 0.0):
        raise RuntimeError("PINN WSS contains invalid values")
    return predicted, wall_speed
