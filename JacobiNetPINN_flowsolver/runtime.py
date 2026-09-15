"""Numerical runtime for JacobiNet-PINN training and evaluation.

Precision and sampling follow the supplied experiment protocol. Frozen JacobiNet
and Fourier features are cached with their spatial gradients and Laplacians.
The trainable heads propagate these derivatives explicitly; PyTorch backward
computes parameter gradients. CUDA Graph replay preserves the same operations,
while Adam updates and checkpoint state remain owned by the trainer.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.func import jacfwd, vmap
from torch.nn import functional as F

if __package__:
    from .physics import (
        PhysicsScales,
        weighted_component_sum,
        weighted_component_values,
    )
else:
    from physics import PhysicsScales, weighted_component_sum, weighted_component_values


# Floating-point precision

SUPPORTED_TRAINING_DTYPES = {
    "float32": (torch.float32, np.dtype(np.float32)),
    "float64": (torch.float64, np.dtype(np.float64)),
}


TRAINING_DTYPE_NAME = os.environ.get("PINN_TRAINING_DTYPE", "float32").lower()


if TRAINING_DTYPE_NAME not in SUPPORTED_TRAINING_DTYPES:
    raise ValueError(
        "PINN_TRAINING_DTYPE must be one of "
        f"{sorted(SUPPORTED_TRAINING_DTYPES)}, got {TRAINING_DTYPE_NAME!r}"
    )


TORCH_TRAINING_DTYPE, NUMPY_TRAINING_DTYPE = SUPPORTED_TRAINING_DTYPES[
    TRAINING_DTYPE_NAME
]


NUMPY_EVALUATION_DTYPE = np.dtype(np.float64)


EVALUATION_DTYPE_NAME = "float64"


PRECISION_POLICY_NAME = (
    "fp32_train_fp64_physics_eval_v1"
    if TRAINING_DTYPE_NAME == "float32"
    else "fp64_train_fp64_physics_eval_v1"
)


def torch_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    raise TypeError(f"Unsupported floating dtype: {dtype}")


def model_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration as error:
        raise ValueError("Model has no parameters from which to infer dtype") from error


def assert_model_training_dtype(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point() and parameter.dtype != TORCH_TRAINING_DTYPE:
            raise TypeError(
                f"Parameter {name} is {parameter.dtype}; expected {TORCH_TRAINING_DTYPE}"
            )
    for name, buffer in model.named_buffers():
        if buffer.is_floating_point() and buffer.dtype != TORCH_TRAINING_DTYPE:
            raise TypeError(
                f"Buffer {name} is {buffer.dtype}; expected {TORCH_TRAINING_DTYPE}"
            )


# Matrix-multiplication precision

MATMUL_PRECISION_MODES = ("ieee",)


PRECISION_POLICY_SCHEMA_VERSION = 1


def current_matmul_precision_policy(mode: str) -> dict[str, Any]:
    """Return the effective backend state without changing it."""

    if mode not in MATMUL_PRECISION_MODES:
        raise ValueError(f"Unsupported matmul precision mode: {mode!r}")
    return {
        "schema_version": PRECISION_POLICY_SCHEMA_VERSION,
        "mode": mode,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "torch_allow_tf32_cublas_override": os.environ.get(
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"
        ),
    }


def apply_matmul_precision_policy(mode: str) -> dict[str, Any]:
    """Apply a policy before CUDA Graph capture and verify every backend flag."""

    if mode not in MATMUL_PRECISION_MODES:
        raise ValueError(f"Unsupported matmul precision mode: {mode!r}")
    override = os.environ.get("NVIDIA_TF32_OVERRIDE")
    torch_override = os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
    if override is not None or torch_override is not None:
        raise RuntimeError(
            "The paper protocol rejects explicit TF32 environment overrides; unset "
            "NVIDIA_TF32_OVERRIDE and TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"
        )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = current_matmul_precision_policy(mode)
    expected_precision = "highest"
    expected_matmul = False
    if policy["float32_matmul_precision"] != expected_precision:
        raise RuntimeError("PyTorch float32 matmul precision did not apply")
    if policy["cuda_matmul_allow_tf32"] is not expected_matmul:
        raise RuntimeError("CUDA matmul TF32 flag did not apply")
    if policy["cudnn_allow_tf32"] is not False:
        raise RuntimeError("cuDNN TF32 must remain disabled for the paper protocol")
    return policy


def assert_matmul_precision_policy(expected: Mapping[str, Any]) -> None:
    """Fail if backend flags changed after a graph was captured."""

    mode = str(expected.get("mode"))
    actual = current_matmul_precision_policy(mode)
    for key in (
        "schema_version",
        "mode",
        "float32_matmul_precision",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "nvidia_tf32_override",
        "torch_allow_tf32_cublas_override",
    ):
        if actual.get(key) != expected.get(key):
            raise RuntimeError(f"Matmul precision policy changed after capture: {key}")


# Collocation sampling and evaluation regions


@dataclass(frozen=True)
class GeometryRegions:
    throat_start_s: float
    throat_end_s: float
    downstream_end_s: float
    throat_radius_ratio: float

    @classmethod
    def from_scales(
        cls,
        scales: PhysicsScales,
        *,
        throat_radius_ratio: float = 0.75,
        downstream_diameters: float = 4.0,
    ) -> "GeometryRegions":
        with np.load(scales.geometry_path) as geometry:
            centers = np.asarray(geometry["centers_m"], dtype=np.float64)
            radii = np.asarray(geometry["radii_m"], dtype=np.float64)
        arc = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
        )
        total_length = float(arc[-1])
        if total_length <= 0.0:
            raise ValueError("Centerline has zero arc length")
        s = arc / total_length
        lesion = radii / scales.inlet_radius_m <= float(throat_radius_ratio)
        if not lesion.any():
            minimum_index = int(np.argmin(radii))
            lesion[minimum_index] = True
        lesion_indices = np.flatnonzero(lesion)
        start = float(s[lesion_indices[0]])
        end = float(s[lesion_indices[-1]])
        downstream_span = (
            float(downstream_diameters) * 2.0 * scales.minimum_radius_m / total_length
        )
        downstream_end = min(1.0, end + downstream_span)
        if downstream_end <= end:
            downstream_end = min(1.0, end + max(0.05, 1.0 / len(s)))
        return cls(
            throat_start_s=start,
            throat_end_s=end,
            downstream_end_s=downstream_end,
            throat_radius_ratio=float(throat_radius_ratio),
        )

    def numpy_masks(self, z_norm: np.ndarray) -> dict[str, np.ndarray]:
        z = np.asarray(z_norm, dtype=np.float64)
        throat = (z >= self.throat_start_s) & (z <= self.throat_end_s)
        downstream = (z > self.throat_end_s) & (z <= self.downstream_end_s)
        upstream = z < self.throat_start_s
        far_downstream = z > self.downstream_end_s
        return {
            "upstream": upstream,
            "throat": throat,
            "downstream": downstream,
            "far_downstream": far_downstream,
        }

    def torch_masks(self, z_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            name: torch.as_tensor(mask, device=z_norm.device, dtype=torch.bool)
            for name, mask in self.numpy_masks(z_norm.detach().cpu().numpy()).items()
        }

    def to_dict(self) -> dict[str, float]:
        return {
            "throat_start_s": self.throat_start_s,
            "throat_end_s": self.throat_end_s,
            "downstream_end_s": self.downstream_end_s,
            "throat_radius_ratio": self.throat_radius_ratio,
        }


class CompleteUniformSampler:
    """Cycle through every point without dropping the final partial batch."""

    def __init__(self, points: torch.Tensor, seed: int):
        if not len(points):
            raise ValueError("Cannot sample an empty point tensor")
        self.points = points
        self.generator = torch.Generator(device=points.device)
        self.generator.manual_seed(int(seed))
        self.permutation = torch.randperm(
            len(points), device=points.device, generator=self.generator
        )
        self.cursor = 0
        self.cycles = 0

    def next_batch(self, batch_size: int) -> torch.Tensor:
        if batch_size <= 0:
            return self.points
        pieces = []
        needed = int(batch_size)
        while needed > 0:
            available = len(self.points) - self.cursor
            take = min(needed, available)
            chosen = self.permutation[self.cursor : self.cursor + take]
            pieces.append(chosen)
            self.cursor += take
            needed -= take
            if self.cursor == len(self.points):
                self.cycles += 1
                self.permutation = torch.randperm(
                    len(self.points),
                    device=self.points.device,
                    generator=self.generator,
                )
                self.cursor = 0
        return self.points[torch.cat(pieces)]


def deterministic_validation_points(
    points: torch.Tensor,
    *,
    count: int,
    seed: int,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError("Validation count must be positive")
    generator = torch.Generator(device=points.device)
    generator.manual_seed(int(seed))
    if count <= len(points):
        indices = torch.randperm(
            len(points), device=points.device, generator=generator
        )[:count]
    else:
        indices = torch.randint(
            len(points), (count,), device=points.device, generator=generator
        )
    return points[indices]


# Training constants

PDE_COMPONENT_NAMES = (
    "momentum_u",
    "momentum_v",
    "momentum_w",
    "continuity",
)


CUDA_GRAPH_WARMUP_STEPS = 3


# Frozen feature values and spatial derivatives

SPATIAL_DIM = 3


SHARED_FEATURE_DIM = 128


RS_DIM = 2


def _require_tensor(
    value: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    *,
    device: torch.device | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(value.shape)}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}; got {value.device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _validate_jet(jet: Any) -> None:
    n = len(jet.coordinates)
    _require_tensor(jet.coordinates, "coordinates", (n, SPATIAL_DIM))
    device = jet.coordinates.device
    _require_tensor(jet.q_value, "q_value", (n, SHARED_FEATURE_DIM), device=device)
    _require_tensor(
        jet.q_grad,
        "q_grad",
        (n, SHARED_FEATURE_DIM, SPATIAL_DIM),
        device=device,
    )
    _require_tensor(jet.q_lap, "q_lap", (n, SHARED_FEATURE_DIM), device=device)
    _require_tensor(jet.rs_value, "rs_value", (n, RS_DIM), device=device)
    _require_tensor(jet.rs_grad, "rs_grad", (n, RS_DIM, SPATIAL_DIM), device=device)
    _require_tensor(jet.rs_lap, "rs_lap", (n, RS_DIM), device=device)


@dataclass(frozen=True)
class FullFeatureJetPool:
    coordinates: torch.Tensor
    q_value: torch.Tensor
    q_grad: torch.Tensor
    q_lap: torch.Tensor
    rs_value: torch.Tensor
    rs_grad: torch.Tensor
    rs_lap: torch.Tensor

    def __post_init__(self) -> None:
        _validate_jet(self)


@dataclass(frozen=True)
class FullFeatureJetBatch:
    coordinates: torch.Tensor
    q_value: torch.Tensor
    q_grad: torch.Tensor
    q_lap: torch.Tensor
    rs_value: torch.Tensor
    rs_grad: torch.Tensor
    rs_lap: torch.Tensor
    pool_indices: torch.Tensor

    def __post_init__(self) -> None:
        _validate_jet(self)
        if not isinstance(self.pool_indices, torch.Tensor):
            raise TypeError("pool_indices must be a tensor")
        if self.pool_indices.dtype != torch.long or self.pool_indices.ndim != 1:
            raise TypeError("pool_indices must be a one-dimensional torch.long tensor")
        if len(self.pool_indices) != len(self.coordinates):
            raise ValueError("pool_indices length must match the selected batch")
        if self.pool_indices.device != self.coordinates.device:
            raise ValueError("pool_indices and feature jet must be on the same device")
        if not self.pool_indices.is_contiguous():
            raise ValueError("pool_indices must be contiguous")


def _validate_linear_silu(
    module: nn.Module, *, input_dim: int, output_dim: int, name: str
) -> None:
    if not isinstance(module, nn.Sequential):
        raise TypeError(f"{name} must be nn.Sequential")
    layers = list(module)
    if len(layers) < 3 or len(layers) % 2 == 0:
        raise ValueError(f"{name} must alternate Linear/SiLU and end in Linear")
    previous = input_dim
    for index, layer in enumerate(layers):
        if index % 2 == 0:
            if not isinstance(layer, nn.Linear) or layer.in_features != previous:
                raise ValueError(f"{name}[{index}] is not the expected Linear layer")
            previous = layer.out_features
        elif not isinstance(layer, nn.SiLU) or layer.inplace:
            raise ValueError(f"{name}[{index}] must be non-inplace SiLU")
    if previous != output_dim:
        raise ValueError(f"{name} must output {output_dim} channels")


def _validate_model(model: nn.Module) -> None:
    required = (
        "jacobinet",
        "rff_xyz",
        "rff_rs",
        "xyz_effective_B",
        "rs_effective_B",
        "net_velocity",
        "net_pressure",
        "inlet_scale",
        "reynolds_number",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise TypeError(f"model is missing required feature attributes: {missing}")
    if any((parameter.requires_grad for parameter in model.jacobinet.parameters())):
        raise ValueError(
            "JacobiNet must be frozen before building a feature-derivative cache"
        )
    _validate_linear_silu(
        model.jacobinet.net, input_dim=3, output_dim=2, name="jacobinet"
    )
    _validate_linear_silu(
        model.net_velocity,
        input_dim=SHARED_FEATURE_DIM,
        output_dim=3,
        name="net_velocity",
    )
    _validate_linear_silu(
        model.net_pressure,
        input_dim=SHARED_FEATURE_DIM,
        output_dim=1,
        name="net_pressure",
    )
    _require_tensor(model.xyz_effective_B, "xyz_effective_B", (32, 3))
    _require_tensor(
        model.rs_effective_B,
        "rs_effective_B",
        (32, 2),
        device=model.xyz_effective_B.device,
    )
    for name, parameter in model.named_parameters():
        if parameter.dtype != torch.float32:
            raise TypeError(f"model parameter {name} is not float32")
        if parameter.device != model.xyz_effective_B.device:
            raise ValueError(f"model parameter {name} is on the wrong device")
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError(f"model parameter {name} is not finite")


def _jacobinet_jet(
    model: nn.Module, coordinates: torch.Tensor, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def single(point: torch.Tensor) -> torch.Tensor:
        return model.jacobinet(point.unsqueeze(0)).squeeze(0)

    first = jacfwd(single)
    second = jacfwd(first)
    values: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    laplacians: list[torch.Tensor] = []
    for start in range(0, len(coordinates), chunk_size):
        chunk = coordinates[start : start + chunk_size]
        # Freeze value semantics per point.  A large GEMM and a vmapped
        # single-point Linear are mathematically identical but can differ by
        # a few FP32 ulps when the batch shape changes.  The pointwise form is
        # stable across cache chunk sizes and matches the independent oracle.
        value = vmap(single)(chunk)
        gradient = vmap(first)(chunk)
        # The transient chunk Hessian is reduced immediately; only its trace is
        # retained in the full-pool cache.
        hessian = vmap(second)(chunk)
        laplacian = torch.diagonal(hessian, dim1=-2, dim2=-1).sum(-1)
        values.append(value.detach())
        gradients.append(gradient.detach())
        laplacians.append(laplacian.detach())
    return tuple(
        torch.cat(parts, dim=0).contiguous()
        for parts in (values, gradients, laplacians)
    )


def _pointwise_rff_values(
    inputs: torch.Tensor, effective_B: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Evaluate fixed RFF values with the cache's frozen batch shape.

    FP32 GEMM results can differ by a few ulps when only the leading batch
    dimension changes.  Using the same bounded chunks as the derivative cache
    makes the value path reproducible against the pre-registered oracle while
    retaining vectorized GPU execution.
    """

    def single(values: torch.Tensor) -> torch.Tensor:
        projection = 2.0 * torch.pi * (values @ effective_B.T)
        return torch.cat((torch.cos(projection), torch.sin(projection)), dim=0)

    parts = [
        vmap(single)(inputs[start : start + chunk_size])
        for start in range(0, len(inputs), chunk_size)
    ]
    return torch.cat(parts, dim=0).contiguous()


def _direct_rff_jet(
    coordinates: torch.Tensor, effective_B: torch.Tensor, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    omega = (2.0 * torch.pi * effective_B).contiguous()
    value = _pointwise_rff_values(coordinates, effective_B, chunk_size)
    cosine, sine = value[:, :32], value[:, 32:]
    omega_norm2 = omega.square().sum(dim=1)
    gradient = torch.cat(
        (
            -cosine.new_ones(()) * sine[:, :, None] * omega[None, :, :],
            cosine[:, :, None] * omega[None, :, :],
        ),
        dim=1,
    )
    laplacian = torch.cat(
        (-cosine * omega_norm2[None, :], -sine * omega_norm2[None, :]), dim=1
    )
    return value.contiguous(), gradient.contiguous(), laplacian.contiguous()


def _composed_rff_jet(
    rs_value: torch.Tensor,
    rs_grad: torch.Tensor,
    rs_lap: torch.Tensor,
    effective_B: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    omega = (2.0 * torch.pi * effective_B).contiguous()
    value = _pointwise_rff_values(rs_value, effective_B, chunk_size)
    cosine, sine = value[:, :32], value[:, 32:]
    projection_grad = torch.einsum("ki,nij->nkj", omega, rs_grad)
    projection_lap = torch.einsum("ki,ni->nk", omega, rs_lap)
    grad_norm2 = projection_grad.square().sum(dim=2)
    gradient = torch.cat(
        (-sine[:, :, None] * projection_grad, cosine[:, :, None] * projection_grad),
        dim=1,
    )
    laplacian = torch.cat(
        (
            -cosine * grad_norm2 - sine * projection_lap,
            -sine * grad_norm2 + cosine * projection_lap,
        ),
        dim=1,
    )
    return value.contiguous(), gradient.contiguous(), laplacian.contiguous()


def build_full_feature_jet(
    model: nn.Module,
    pool_xyz: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> FullFeatureJetPool:
    """Build a detached FP32 value/gradient/Laplacian cache for a fixed pool."""
    _validate_model(model)
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(pool_xyz, torch.Tensor) or pool_xyz.ndim != 2:
        raise TypeError("pool_xyz must be a two-dimensional tensor")
    coordinates = pool_xyz.detach().contiguous()
    _require_tensor(coordinates, "pool_xyz", (len(coordinates), SPATIAL_DIM))
    if len(coordinates) == 0:
        raise ValueError("pool_xyz must contain at least one point")
    if coordinates.device != model.xyz_effective_B.device:
        raise ValueError("pool_xyz and model must be on the same device")

    rs_value, rs_grad, rs_lap = _jacobinet_jet(model, coordinates, chunk_size)
    xyz_value, xyz_grad, xyz_lap = _direct_rff_jet(
        coordinates, model.xyz_effective_B, chunk_size
    )
    rs_rff_value, rs_rff_grad, rs_rff_lap = _composed_rff_jet(
        rs_value, rs_grad, rs_lap, model.rs_effective_B, chunk_size
    )
    return FullFeatureJetPool(
        coordinates=coordinates,
        q_value=torch.cat((xyz_value, rs_rff_value), dim=1).detach().contiguous(),
        q_grad=torch.cat((xyz_grad, rs_rff_grad), dim=1).detach().contiguous(),
        q_lap=torch.cat((xyz_lap, rs_rff_lap), dim=1).detach().contiguous(),
        rs_value=rs_value,
        rs_grad=rs_grad,
        rs_lap=rs_lap,
    )


# Explicit derivative propagation and PDE residuals

EXPLICIT_JET_BACKEND = "A3-explicit-linear-silu-trial-jet-v1"


OUTPUT_DIM = 4


EXPECTED_HEAD_PARAMETER_TENSORS = 18


EXPECTED_HEAD_PARAMETER_COUNT = 116_100


FLOW_RFF_SPEC_SHA256 = (
    "86f56c55091bc2ab2abe46e6c4fd84efd86a7a5a477480067ce3b7728de6eb64"
)


@dataclass(frozen=True)
class ExplicitJet:
    """A batched tensor represented by value, spatial gradient, and trace Hessian."""

    value: torch.Tensor
    grad: torch.Tensor
    lap: torch.Tensor


def _head_layers(*, hidden_layers: int, output_dim: int) -> tuple[tuple[int, int], ...]:
    return ((SHARED_FEATURE_DIM, SHARED_FEATURE_DIM),) * hidden_layers + (
        (SHARED_FEATURE_DIM, output_dim),
    )


def _validate_exact_head(
    head: nn.Module,
    *,
    name: str,
    hidden_layers: int,
    output_dim: int,
) -> None:
    """Require the exact flow model alternating Linear/SiLU head, including biases."""

    if not isinstance(head, nn.Sequential):
        raise TypeError(f"{name} must be nn.Sequential")
    expected_linears = _head_layers(hidden_layers=hidden_layers, output_dim=output_dim)
    expected_module_count = 2 * len(expected_linears) - 1
    if len(head) != expected_module_count:
        raise ValueError(
            f"{name} is not the required architecture: expected "
            f"{expected_module_count} modules, got {len(head)}"
        )
    for index, (in_features, out_features) in enumerate(expected_linears):
        linear_index = 2 * index
        linear = head[linear_index]
        if type(linear) is not nn.Linear:
            raise TypeError(f"{name}[{linear_index}] must be exactly nn.Linear")
        if (
            linear.in_features != in_features
            or linear.out_features != out_features
            or linear.bias is None
        ):
            raise ValueError(
                f"{name}[{linear_index}] must be Linear({in_features}, "
                f"{out_features}, bias=True)"
            )
        if index < len(expected_linears) - 1:
            activation = head[linear_index + 1]
            if type(activation) is not nn.SiLU or activation.inplace:
                raise TypeError(
                    f"{name}[{linear_index + 1}] must be a non-inplace nn.SiLU"
                )


def _head_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    return [
        *model.net_velocity.parameters(),
        *model.net_pressure.parameters(),
    ]


def validate_explicit_flow_model(model: nn.Module) -> None:
    """Fail closed unless ``model`` is the trainable FP32 flow model/no-LBF model."""

    # Reuse the feature-jet contract for the frozen coordinate path, RFF dimensions,
    # devices, finiteness, and absence of local base flow.
    _validate_model(model)
    _validate_exact_head(
        model.net_velocity,
        name="net_velocity",
        hidden_layers=4,
        output_dim=3,
    )
    _validate_exact_head(
        model.net_pressure,
        name="net_pressure",
        hidden_layers=3,
        output_dim=1,
    )
    parameters = _head_parameters(model)
    if len(parameters) != EXPECTED_HEAD_PARAMETER_TENSORS:
        raise ValueError(
            "explicit derivative propagation requires exactly 18 trainable head parameter tensors"
        )
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError(
            "explicit derivative propagation does not permit tied head parameters"
        )
    if not all(parameter.requires_grad for parameter in parameters):
        raise ValueError(
            "all 18 flow model head parameter tensors must require gradients"
        )
    if (
        sum(parameter.numel() for parameter in parameters)
        != EXPECTED_HEAD_PARAMETER_COUNT
    ):
        raise ValueError(
            "explicit derivative propagation requires the 116,100-parameter flow model head allocation"
        )
    if not hasattr(model, "rff_spec") or not callable(
        getattr(model.rff_spec, "sha256", None)
    ):
        raise TypeError(
            "explicit derivative propagation requires the frozen flow model RFF specification"
        )
    if model.rff_spec.sha256() != FLOW_RFF_SPEC_SHA256:
        raise ValueError(
            "explicit derivative propagation is locked to flow model xyz sigma=10 and r/s sigma=2.5"
        )
    if (
        not math.isfinite(float(model.reynolds_number))
        or float(model.reynolds_number) <= 0.0
    ):
        raise ValueError("reynolds_number must be finite and positive")
    if not math.isfinite(float(model.inlet_scale)):
        raise ValueError("inlet_scale must be finite")


def _validate_feature_batch(model: nn.Module, batch: FullFeatureJetBatch) -> None:
    if not isinstance(batch, FullFeatureJetBatch):
        raise TypeError("feature_jet must be FullFeatureJetBatch")
    device = next(model.net_velocity.parameters()).device
    for name in (
        "coordinates",
        "q_value",
        "q_grad",
        "q_lap",
        "rs_value",
        "rs_grad",
        "rs_lap",
    ):
        tensor = getattr(batch, name)
        if tensor.dtype != torch.float32:
            raise TypeError(f"feature_jet.{name} must use torch.float32")
        if tensor.device != device:
            raise ValueError(
                f"feature_jet.{name} and prediction heads must share a device"
            )
        if tensor.requires_grad:
            raise ValueError(
                f"feature_jet.{name} must be a detached fixed-coordinate cache"
            )


def _feature_q_jet(batch: FullFeatureJetBatch) -> ExplicitJet:
    return ExplicitJet(batch.q_value, batch.q_grad, batch.q_lap)


def _feature_rs_jet(batch: FullFeatureJetBatch) -> ExplicitJet:
    return ExplicitJet(batch.rs_value, batch.rs_grad, batch.rs_lap)


def _linear_jet(layer: nn.Linear, jet: ExplicitJet) -> ExplicitJet:
    # Value uses the exact head module weights and bias.  Spatial derivatives
    # use the same weight object, so backward reaches it through every path.
    value = F.linear(jet.value, layer.weight, layer.bias)
    grad = torch.einsum("oi,nid->nod", layer.weight, jet.grad)
    lap = torch.einsum("oi,ni->no", layer.weight, jet.lap)
    return ExplicitJet(value, grad, lap)


def _silu_jet(layer: nn.SiLU, jet: ExplicitJet) -> ExplicitJet:
    x = jet.value
    sigmoid = torch.sigmoid(x)
    sigmoid_prime = sigmoid * (1.0 - sigmoid)
    first = sigmoid + x * sigmoid_prime
    second = 2.0 * sigmoid_prime + x * sigmoid_prime * (1.0 - 2.0 * sigmoid)
    value = layer(x)
    grad_norm2 = jet.grad.square().sum(dim=-1)
    grad = first.unsqueeze(-1) * jet.grad
    lap = second * grad_norm2 + first * jet.lap
    return ExplicitJet(value, grad, lap)


def _sequential_jet(head: nn.Sequential, jet: ExplicitJet) -> ExplicitJet:
    for layer in head:
        if type(layer) is nn.Linear:
            jet = _linear_jet(layer, jet)
        elif type(layer) is nn.SiLU and not layer.inplace:
            jet = _silu_jet(layer, jet)
        else:  # Validation makes this unreachable; retain a fail-closed hot path.
            raise TypeError("explicit-jet head only supports Linear/non-inplace SiLU")
    return jet


def _channel(jet: ExplicitJet, start: int, stop: int) -> ExplicitJet:
    return ExplicitJet(
        jet.value[:, start:stop],
        jet.grad[:, start:stop, :],
        jet.lap[:, start:stop],
    )


def _one_minus(jet: ExplicitJet) -> ExplicitJet:
    return ExplicitJet(1.0 - jet.value, -jet.grad, -jet.lap)


def _multiply(left: ExplicitJet, right: ExplicitJet) -> ExplicitJet:
    left_channels = left.value.shape[1]
    right_channels = right.value.shape[1]
    if left_channels != right_channels and left_channels != 1 and right_channels != 1:
        raise ValueError("jet product channels must match or be scalar-broadcastable")
    value = left.value * right.value
    grad = left.grad * right.value.unsqueeze(-1) + left.value.unsqueeze(-1) * right.grad
    lap = (
        left.lap * right.value
        + left.value * right.lap
        + 2.0 * (left.grad * right.grad).sum(dim=-1)
    )
    return ExplicitJet(value, grad, lap)


def _scale(jet: ExplicitJet, value: float) -> ExplicitJet:
    return ExplicitJet(jet.value * value, jet.grad * value, jet.lap * value)


def _cat(jets: tuple[ExplicitJet, ...]) -> ExplicitJet:
    return ExplicitJet(
        torch.cat(tuple(jet.value for jet in jets), dim=1),
        torch.cat(tuple(jet.grad for jet in jets), dim=1),
        torch.cat(tuple(jet.lap for jet in jets), dim=1),
    )


def _forward_unchecked(model: nn.Module, batch: FullFeatureJetBatch) -> ExplicitJet:
    q = _feature_q_jet(batch)
    rs = _feature_rs_jet(batch)
    radial = _channel(rs, 0, 1)
    axial = _channel(rs, 1, 2)

    velocity_raw = _sequential_jet(model.net_velocity, q)
    pressure_raw = _sequential_jet(model.net_pressure, q)

    # Exact flow model trial transform, propagated with the full product rule.
    radial_axial = _multiply(radial, axial)
    velocity_correction = _multiply(radial_axial, velocity_raw)
    inlet = _scale(_multiply(radial, _one_minus(axial)), float(model.inlet_scale))
    correction_w = _channel(velocity_correction, 2, 3)
    velocity = _cat(
        (
            _channel(velocity_correction, 0, 1),
            _channel(velocity_correction, 1, 2),
            ExplicitJet(
                correction_w.value + inlet.value,
                correction_w.grad + inlet.grad,
                correction_w.lap + inlet.lap,
            ),
        )
    )
    pressure = _multiply(_one_minus(axial), pressure_raw)
    return _cat((velocity, pressure))


def _pde_losses_unchecked(
    model: nn.Module, output: ExplicitJet
) -> dict[str, torch.Tensor]:
    value, grad, lap = output.value, output.grad, output.lap
    u, v, w = (value[:, index : index + 1] for index in range(3))
    grad_u, grad_v, grad_w = (grad[:, index, :] for index in range(3))
    grad_p = grad[:, 3, :]
    u_x, u_y, u_z = (grad_u[:, index : index + 1] for index in range(3))
    v_x, v_y, v_z = (grad_v[:, index : index + 1] for index in range(3))
    w_x, w_y, w_z = (grad_w[:, index : index + 1] for index in range(3))
    p_x, p_y, p_z = (grad_p[:, index : index + 1] for index in range(3))
    inverse_reynolds = 1.0 / float(model.reynolds_number)
    residual_u = u * u_x + v * u_y + w * u_z + p_x - inverse_reynolds * lap[:, 0:1]
    residual_v = u * v_x + v * v_y + w * v_z + p_y - inverse_reynolds * lap[:, 1:2]
    residual_w = u * w_x + v * w_y + w * w_z + p_z - inverse_reynolds * lap[:, 2:3]
    continuity = u_x + v_y + w_z
    return {
        "momentum_u": torch.mean(residual_u.square()),
        "momentum_v": torch.mean(residual_v.square()),
        "momentum_w": torch.mean(residual_w.square()),
        "continuity": torch.mean(continuity.square()),
    }


class ExplicitJetRuntime(nn.Module):
    """Validate flow model once, then expose the capture-friendly explicit-jet hot path."""

    def __init__(self, model: nn.Module):
        super().__init__()
        validate_explicit_flow_model(model)
        self.model = model

    def forward(self, feature_jet: FullFeatureJetBatch) -> ExplicitJet:
        _validate_feature_batch(self.model, feature_jet)
        return _forward_unchecked(self.model, feature_jet)

    def pde_losses(self, feature_jet: FullFeatureJetBatch) -> dict[str, torch.Tensor]:
        output = self(feature_jet)
        return _pde_losses_unchecked(self.model, output)


# CUDA Graph execution

EXPLICIT_BATCH_SIZE = 8192


EXPLICIT_CACHE_FIELDS = (
    "coordinates",
    "q_value",
    "q_grad",
    "q_lap",
    "rs_value",
    "rs_grad",
    "rs_lap",
)


EXPLICIT_GRAPH_CAPTURE_BOUNDARY = (
    "zero_grad_set_to_none_false",
    "analytic_linear_silu_trial_value_gradient_laplacian",
    "explicit_four_pde_residual_assembly",
    "weighted_pde_loss_assembly",
    "ordinary_parameter_backward",
)


EXPLICIT_EAGER_BOUNDARY = (
    "sampler_indices",
    "static_indices_copy",
    "seven_field_index_select_out",
    "finite_checks",
    "cosine_scheduler",
    "adam_step",
    "logging",
    "evaluation",
    "checkpointing",
)


EXPLICIT_REPLAY_ORDER = (
    "static_indices.copy_",
    "seven index_select(out=static_buffer)",
    "explicit_cuda_graph.replay",
)


@dataclass(frozen=True)
class ExplicitStaticBufferRecord:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    data_ptr: int


def _validate_weights(component_weights: Mapping[str, float]) -> dict[str, float]:
    if set(component_weights) != set(PDE_COMPONENT_NAMES):
        raise ValueError("explicit-jet runtime requires exactly four PDE weights")
    resolved: dict[str, float] = {}
    for name in PDE_COMPONENT_NAMES:
        value = float(component_weights[name])
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"Non-finite PDE weight: {name}")
        if value != 1.0:
            raise ValueError(
                "Frozen flow model explicit-jet PDE weights must all equal 1"
            )
        resolved[name] = value
    return resolved


def _trainable_named_parameters(
    model: nn.Module,
) -> tuple[tuple[str, nn.Parameter], ...]:
    return tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> tuple[int, ...]:
    return tuple(
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def _explicit_training_tensors(
    explicit_runtime: ExplicitJetRuntime,
    static_feature_jet: FullFeatureJetBatch,
    component_weights: Mapping[str, float],
) -> tuple[
    ExplicitJet,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    output_jet = explicit_runtime(static_feature_jet)
    components = _pde_losses_unchecked(explicit_runtime.model, output_jet)
    weighted = weighted_component_values(components, dict(component_weights))
    total = weighted_component_sum(components, dict(component_weights))
    return output_jet, components, weighted, total


class ExplicitJetCudaGraphTrainingCore:
    """Persistent true-explicit-jet graph and allocation-free dynamic jet buffers.

    ``replay(indices)`` updates the static feature batch and enqueues explicit
    PDE/backward.  The returned loss objects and ``named_gradient_tensors`` are
    persistent graph buffers; clone before the next replay when history is
    required.  The caller owns eager ``optimizer.step()``.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        full_cache: FullFeatureJetPool,
        capture_indices: torch.Tensor,
        component_weights: Mapping[str, float],
        warmup_steps: int = CUDA_GRAPH_WARMUP_STEPS,
        precision_policy: Mapping[str, Any] | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("explicit-jet CUDA Graph runtime requires CUDA")
        validate_explicit_flow_model(model)
        spec = getattr(model, "rff_spec", None)
        if spec is None or spec.sha256() != FLOW_RFF_SPEC_SHA256:
            raise ValueError("explicit-jet runtime requires flow model xyz10/rs2.5 RFF")
        if not isinstance(full_cache, FullFeatureJetPool):
            raise TypeError("full_cache must be FullFeatureJetPool")
        self.device = full_cache.coordinates.device
        if self.device.type != "cuda":
            raise ValueError("explicit-jet full feature cache must be on CUDA")
        if full_cache.coordinates.dtype != torch.float32:
            raise TypeError("explicit-jet full feature cache must use float32")
        if len(full_cache.coordinates) < 2:
            raise ValueError("explicit-jet cache must contain at least two points")
        if not isinstance(capture_indices, torch.Tensor):
            raise TypeError("capture_indices must be a tensor")
        if capture_indices.shape != (EXPLICIT_BATCH_SIZE,):
            raise ValueError(
                f"capture_indices must have shape ({EXPLICIT_BATCH_SIZE},)"
            )
        if capture_indices.dtype != torch.long:
            raise TypeError("capture_indices must use torch.long")
        if capture_indices.device != self.device:
            raise ValueError("capture_indices and cache must share a CUDA device")
        if not capture_indices.is_contiguous():
            raise ValueError("capture_indices must be contiguous")
        if int(capture_indices.min()) < 0 or int(capture_indices.max()) >= len(
            full_cache.coordinates
        ):
            raise IndexError("capture_indices are outside the full cache")
        if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool):
            raise TypeError("warmup_steps must be an integer")
        if warmup_steps < 2:
            raise ValueError("explicit-jet graph capture requires two warmups")
        if not isinstance(optimizer, torch.optim.Adam):
            raise TypeError("Frozen flow model explicit-jet runtime requires Adam")
        if precision_policy is None:
            raise ValueError("explicit-jet runtime requires an explicit IEEE policy")

        self.model = model
        self.optimizer = optimizer
        self.full_cache = full_cache
        self.component_weights = _validate_weights(component_weights)
        self.warmup_steps = int(warmup_steps)
        self.capture_precision_policy = dict(precision_policy)
        mode = str(self.capture_precision_policy.get("mode"))
        if mode != "ieee":
            raise ValueError("explicit-jet production runtime is fixed to IEEE FP32")
        if current_matmul_precision_policy(mode) != self.capture_precision_policy:
            raise RuntimeError("explicit-jet precision policy differs before capture")

        trainable = _trainable_named_parameters(model)
        if len(trainable) != EXPECTED_HEAD_PARAMETER_TENSORS:
            raise ValueError("explicit-jet runtime requires exactly 18 gradients")
        if sum(parameter.numel() for _, parameter in trainable) != (
            EXPECTED_HEAD_PARAMETER_COUNT
        ):
            raise ValueError("explicit-jet runtime requires 116,100 trainable values")
        trainable_ids = tuple(id(parameter) for _, parameter in trainable)
        optimizer_ids = _optimizer_parameter_ids(optimizer)
        if len(set(optimizer_ids)) != len(optimizer_ids):
            raise ValueError("explicit-jet Adam contains duplicate parameters")
        if set(trainable_ids) != set(optimizer_ids):
            raise ValueError(
                "explicit-jet Adam parameters differ from flow model heads"
            )

        setup_started = time.perf_counter()
        self.static_indices = torch.empty(
            (EXPLICIT_BATCH_SIZE,), dtype=torch.long, device=self.device
        )
        static_fields: dict[str, torch.Tensor] = {}
        for name in EXPLICIT_CACHE_FIELDS:
            source = getattr(full_cache, name)
            static_fields[name] = torch.empty(
                (EXPLICIT_BATCH_SIZE, *source.shape[1:]),
                dtype=source.dtype,
                device=source.device,
            )
        self._static_fields = static_fields
        self._gather_pairs = tuple(
            (getattr(full_cache, name), static_fields[name])
            for name in EXPLICIT_CACHE_FIELDS
        )
        self._copy_indices_and_gather(capture_indices)
        self.static_feature_jet = FullFeatureJetBatch(
            coordinates=static_fields["coordinates"],
            q_value=static_fields["q_value"],
            q_grad=static_fields["q_grad"],
            q_lap=static_fields["q_lap"],
            rs_value=static_fields["rs_value"],
            rs_grad=static_fields["rs_grad"],
            rs_lap=static_fields["rs_lap"],
            pool_indices=self.static_indices,
        )
        self.explicit_runtime = ExplicitJetRuntime(model)
        self.static_buffer_records = tuple(
            ExplicitStaticBufferRecord(
                name=name,
                shape=tuple(static_fields[name].shape),
                dtype=static_fields[name].dtype,
                device=static_fields[name].device,
                data_ptr=static_fields[name].data_ptr(),
            )
            for name in EXPLICIT_CACHE_FIELDS
        )
        self.static_buffer_data_ptrs = {
            record.name: record.data_ptr for record in self.static_buffer_records
        }
        self.static_indices_data_ptr = self.static_indices.data_ptr()
        self.graph = torch.cuda.CUDAGraph()
        self.output_jet: ExplicitJet
        self.component_tensors: dict[str, torch.Tensor]
        self.weighted_component_tensors: dict[str, torch.Tensor]
        self.total_loss: torch.Tensor

        optimizer_state_keys_before = tuple(id(key) for key in optimizer.state)
        self.explicit_runtime.train()
        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for index in range(self.warmup_steps):
                optimizer.zero_grad(set_to_none=index == 0)
                _, _, _, warmup_total = _explicit_training_tensors(
                    self.explicit_runtime,
                    self.static_feature_jet,
                    self.component_weights,
                )
                warmup_total.backward()
        current_stream.wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)

        missing = [name for name, parameter in trainable if parameter.grad is None]
        if missing:
            raise RuntimeError(
                "explicit-jet warmup did not materialize gradients: "
                + ", ".join(missing)
            )
        non_finite = [
            name
            for name, parameter in trainable
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        ]
        if non_finite:
            raise FloatingPointError(
                "Non-finite explicit-jet warmup gradients: " + ", ".join(non_finite)
            )
        if tuple(id(key) for key in optimizer.state) != optimizer_state_keys_before:
            raise RuntimeError("explicit-jet warmup changed eager Adam state")

        self._copy_indices_and_gather(capture_indices)
        with torch.cuda.graph(self.graph):
            optimizer.zero_grad(set_to_none=False)
            (
                self.output_jet,
                self.component_tensors,
                self.weighted_component_tensors,
                self.total_loss,
            ) = _explicit_training_tensors(
                self.explicit_runtime,
                self.static_feature_jet,
                self.component_weights,
            )
            self.total_loss.backward()
        torch.cuda.synchronize(self.device)
        self.setup_seconds = time.perf_counter() - setup_started
        self.named_gradient_tensors = tuple(
            (name, parameter.grad)
            for name, parameter in trainable
            if parameter.grad is not None
        )
        if len(self.named_gradient_tensors) != EXPECTED_HEAD_PARAMETER_TENSORS:
            raise RuntimeError("explicit-jet graph did not retain all 18 gradients")
        self.gradient_tensors = tuple(
            gradient for _, gradient in self.named_gradient_tensors
        )
        self._replay_result = (
            self.component_tensors,
            self.weighted_component_tensors,
            self.total_loss,
        )
        self._precision_verified_after_capture = False
        self.replay_count = 0

    def _copy_indices_and_gather(self, indices: torch.Tensor) -> None:
        self.static_indices.copy_(indices, non_blocking=True)
        for source, target in self._gather_pairs:
            torch.index_select(
                source,
                dim=0,
                index=self.static_indices,
                out=target,
            )

    def _validate_replay_indices(self, indices: torch.Tensor) -> None:
        if not isinstance(indices, torch.Tensor):
            raise TypeError("explicit-jet replay indices must be a tensor")
        if indices.shape != self.static_indices.shape:
            raise ValueError("explicit-jet replay index shape changed")
        if indices.dtype != torch.long:
            raise TypeError("explicit-jet replay indices must use torch.long")
        if indices.device != self.device:
            raise ValueError("explicit-jet replay index device changed")
        if not indices.is_contiguous():
            raise ValueError("explicit-jet replay indices must be contiguous")

    def replay(
        self, indices: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        """Enqueue dynamic gather then explicit PDE/backward on one stream."""
        self._validate_replay_indices(indices)
        if not self._precision_verified_after_capture:
            assert_matmul_precision_policy(self.capture_precision_policy)
            self._precision_verified_after_capture = True
        self._copy_indices_and_gather(indices)
        self.graph.replay()
        self.replay_count += 1
        return self._replay_result

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    @property
    def batch_size(self) -> int:
        return EXPLICIT_BATCH_SIZE

    def runtime_definition(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "backend": "C1-A3-explicit-jet-dynamic-CUDA-Graph",
            "explicit_jet_backend": EXPLICIT_JET_BACKEND,
            "batch_size": EXPLICIT_BATCH_SIZE,
            "cache_fields": list(EXPLICIT_CACHE_FIELDS),
            "dynamic_indices": True,
            "allocation_free_tensor_hot_path_after_init": True,
            "gather_api": "torch.index_select(out=preallocated_static_buffer)",
            "replay_order": list(EXPLICIT_REPLAY_ORDER),
            "same_current_stream_ordering": True,
            "capture_boundary": list(EXPLICIT_GRAPH_CAPTURE_BOUNDARY),
            "eager_boundary": list(EXPLICIT_EAGER_BOUNDARY),
            "explicit_linear_silu_trial_jet_propagation": True,
            "higher_order_coordinate_vjp_active": False,
            "coordinate_autograd_grad_calls": 0,
            "torch_func_transform_calls": 0,
            "ordinary_parameter_backward": True,
            "head_gradient_tensor_count": len(self.named_gradient_tensors),
            "optimizer": "torch.optim.Adam",
            "optimizer_captured": False,
            "optimizer_step_owner": "caller_eager_after_replay",
            "output_jet_tensors_persistent": True,
            "loss_tensors_persistent": True,
            "gradient_tensors_persistent": True,
            "warmup_steps": self.warmup_steps,
            "setup_seconds": self.setup_seconds,
            "static_buffer_data_ptrs": dict(self.static_buffer_data_ptrs),
            "static_indices_data_ptr": self.static_indices_data_ptr,
            "capture_precision_policy": dict(self.capture_precision_policy),
            "c1_rff_spec_sha256": FLOW_RFF_SPEC_SHA256,
        }


# Training integration and state validation

CODE_ROOT = Path(__file__).resolve().parent


SCHEMA_VERSION = 1


REVISION = "a3_production_training_backend_v1"


BACKEND_ID = "c1_a3_explicit_jet_production"


CACHE_CHUNK_SIZE = 256


SAMPLER_STATE_KEYS = frozenset(
    {"class", "generator_state", "permutation", "cursor", "cycles"}
)


PDE_COMPONENT_WEIGHTS = {
    "momentum_u": 1.0,
    "momentum_v": 1.0,
    "momentum_w": 1.0,
    "continuity": 1.0,
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_probe_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _hash_tree(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    elif isinstance(value, Mapping):
        digest.update(b"mapping{")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_tree(digest, key)
            _hash_tree(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii") + b"[")
        for item in value:
            _hash_tree(digest, item)
        digest.update(b"]")
    elif value is None or isinstance(value, (bool, int, float, str)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(repr(value).encode("utf-8"))
    else:
        raise TypeError(f"Unsupported optimizer identity value: {type(value).__name__}")


def optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    _hash_tree(digest, optimizer.state_dict())
    return digest.hexdigest()


def _dependency_paths() -> dict[str, Path]:
    return {
        "runtime.py": Path(__file__).resolve(),
        "physics.py": Path(__file__).with_name("physics.py"),
    }


def validate_backend_source_inventory() -> dict[str, str]:
    """Return content hashes for the numerical runtime and its physics dependency."""

    observed: dict[str, str] = {}
    for name, dependency in _dependency_paths().items():
        if not dependency.is_file():
            raise FileNotFoundError(f"Runtime dependency is missing: {dependency}")
        observed[name] = file_sha256(dependency)
    return observed


def _cpu_generator_state(value: torch.Tensor, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    state = value.detach().cpu().contiguous()
    if state.dtype != torch.uint8 or state.ndim != 1 or not len(state):
        raise TypeError(f"{label} must be a non-empty one-dimensional uint8 tensor")
    return state.clone()


class CompleteUniformIndexAdapter:
    """Return indices while advancing one official uniform sampler in place."""

    def __init__(self, sampler: CompleteUniformSampler) -> None:
        if not isinstance(sampler, CompleteUniformSampler):
            raise TypeError("adapter requires an official CompleteUniformSampler")
        if not isinstance(sampler.points, torch.Tensor) or not len(sampler.points):
            raise ValueError("official sampler must own a non-empty tensor pool")
        if sampler.permutation.dtype != torch.long:
            raise TypeError("official sampler permutation must use torch.long")
        if sampler.permutation.device != sampler.points.device:
            raise ValueError(
                "official sampler permutation and points must share a device"
            )
        self.sampler = sampler

    @property
    def pool_size(self) -> int:
        return int(len(self.sampler.points))

    @property
    def device(self) -> torch.device:
        return self.sampler.points.device

    def next_indices(self, batch_size: int) -> torch.Tensor:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            return torch.arange(self.pool_size, dtype=torch.long, device=self.device)
        pieces: list[torch.Tensor] = []
        needed = int(batch_size)
        while needed > 0:
            available = self.pool_size - int(self.sampler.cursor)
            take = min(needed, available)
            pieces.append(
                self.sampler.permutation[
                    self.sampler.cursor : self.sampler.cursor + take
                ]
            )
            self.sampler.cursor += take
            needed -= take
            if self.sampler.cursor == self.pool_size:
                self.sampler.cycles += 1
                self.sampler.permutation = torch.randperm(
                    self.pool_size,
                    device=self.device,
                    generator=self.sampler.generator,
                )
                self.sampler.cursor = 0
        result = torch.cat(pieces).contiguous()
        if result.dtype != torch.long or result.device != self.device:
            raise RuntimeError("official sampler index contract changed")
        return result

    def sampler_state_dict(self) -> dict[str, Any]:
        return {
            "class": "CompleteUniformSampler",
            "generator_state": _cpu_generator_state(
                self.sampler.generator.get_state(), "sampler.generator_state"
            ),
            "permutation": self.sampler.permutation.detach().cpu().contiguous().clone(),
            "cursor": int(self.sampler.cursor),
            "cycles": int(self.sampler.cycles),
        }

    def load_sampler_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != SAMPLER_STATE_KEYS:
            raise ValueError("uniform sampler state schema changed")
        if state.get("class") != "CompleteUniformSampler":
            raise ValueError("uniform sampler checkpoint class changed")
        permutation = state.get("permutation")
        if not isinstance(permutation, torch.Tensor):
            raise TypeError("sampler permutation must be a tensor")
        permutation = permutation.detach().cpu().contiguous()
        if permutation.dtype != torch.long or tuple(permutation.shape) != (
            self.pool_size,
        ):
            raise ValueError("sampler permutation shape or dtype changed")
        expected = torch.arange(self.pool_size, dtype=torch.long)
        if not torch.equal(torch.sort(permutation).values, expected):
            raise ValueError("sampler permutation is not a complete permutation")
        cursor = state.get("cursor")
        cycles = state.get("cycles")
        if (
            not isinstance(cursor, int)
            or isinstance(cursor, bool)
            or not 0 <= cursor < self.pool_size
        ):
            raise ValueError("sampler cursor is outside the pool")
        if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 0:
            raise ValueError("sampler cycles must be a non-negative integer")
        generator_state = _cpu_generator_state(
            state.get("generator_state"), "sampler_state.generator_state"
        )
        self.sampler.generator.set_state(generator_state)
        self.sampler.permutation = permutation.to(self.device).contiguous()
        self.sampler.cursor = int(cursor)
        self.sampler.cycles = int(cycles)

    def audit(self) -> dict[str, Any]:
        state = self.sampler_state_dict()
        payload = {
            "class": state["class"],
            "pool_size": self.pool_size,
            "device": str(self.device),
            "generator_state_sha256": tensor_probe_sha256(state["generator_state"]),
            "permutation_sha256": tensor_probe_sha256(state["permutation"]),
            "cursor": state["cursor"],
            "cycles": state["cycles"],
        }
        payload["sha256"] = canonical_sha256(payload)
        return payload


def build_independent_capture_indices(
    pool_size: int, device: torch.device | str
) -> torch.Tensor:
    """Build a graph-capture batch without touching sampler or global RNG."""

    if not isinstance(pool_size, int) or isinstance(pool_size, bool) or pool_size <= 0:
        raise ValueError("pool_size must be a positive integer")
    return (
        torch.arange(EXPLICIT_BATCH_SIZE, dtype=torch.long, device=device)
        .remainder(pool_size)
        .contiguous()
    )


def production_checkpoint_metadata() -> dict[str, Any]:
    """Small backend marker; never contains cache or CUDA Graph tensors."""

    return {
        "schema_version": SCHEMA_VERSION,
        "training_backend": BACKEND_ID,
        "backend_revision": REVISION,
        "rff_spec_sha256": FLOW_RFF_SPEC_SHA256,
        "dtype": "float32",
        "matmul_precision_mode": "ieee",
        "optimizer": "torch.optim.Adam",
        "optimizer_parameter_policy": "18_trainable_heads_only_116100_values",
        "cache_checkpointed": False,
        "cuda_graph_checkpointed": False,
    }


def _trainable_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    return tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )


def validate_model_optimizer_contract(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[nn.Parameter, ...]:
    validate_explicit_flow_model(model)
    parameters = _trainable_parameters(model)
    if len(parameters) != EXPECTED_HEAD_PARAMETER_TENSORS:
        raise ValueError(
            "explicit-jet production requires exactly 18 trainable tensors"
        )
    if (
        sum(parameter.numel() for parameter in parameters)
        != EXPECTED_HEAD_PARAMETER_COUNT
    ):
        raise ValueError(
            "explicit-jet production requires exactly 116,100 trainable values"
        )
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise TypeError("explicit-jet production trainable parameters must use float32")
    if not model.training:
        raise ValueError("explicit-jet production model must be in training mode")
    if any(parameter.grad is not None for parameter in parameters):
        raise ValueError(
            "explicit-jet production backend must be built between optimizer steps"
        )
    if type(optimizer) is not torch.optim.Adam:
        raise TypeError(
            "explicit-jet production optimizer must be exactly torch.optim.Adam"
        )
    optimizer_parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    )
    if len(set(map(id, optimizer_parameters))) != len(optimizer_parameters):
        raise ValueError("explicit-jet production Adam contains duplicate parameters")
    if set(map(id, optimizer_parameters)) != set(map(id, parameters)):
        raise ValueError(
            "explicit-jet production Adam parameters differ from flow model trainables"
        )
    if model.rff_spec.sha256() != FLOW_RFF_SPEC_SHA256:
        raise ValueError(
            "explicit-jet production is locked to flow model xyz10/rs2.5 RFF"
        )
    return parameters


def _validate_pool(
    full_pool: torch.Tensor, model: nn.Module, source_pool_identity: Mapping[str, Any]
) -> tuple[torch.device, str]:
    if not isinstance(full_pool, torch.Tensor) or full_pool.ndim != 2:
        raise TypeError("full_pool must be a two-dimensional tensor")
    if tuple(full_pool.shape[1:]) != (5,) or len(full_pool) < 2:
        raise ValueError("full_pool must contain at least two five-column points")
    if full_pool.dtype != torch.float32 or not full_pool.is_contiguous():
        raise TypeError("full_pool must be contiguous float32")
    device = next(model.parameters()).device
    if device.type != "cuda" or full_pool.device != device:
        raise ValueError(
            "explicit-jet production model and full_pool must share a CUDA device"
        )
    if not isinstance(source_pool_identity, Mapping):
        raise TypeError("source_pool_identity must be a mapping")
    observed_sha = tensor_probe_sha256(full_pool)
    if (
        source_pool_identity.get("pool_size") != len(full_pool)
        or source_pool_identity.get("pool_probe_sha256") != observed_sha
    ):
        raise ValueError("explicit-jet production source pool identity mismatch")
    return device, observed_sha


def _cache_field_identity(cache: FullFeatureJetPool) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    total_bytes = 0
    for name in EXPLICIT_CACHE_FIELDS:
        value = getattr(cache, name)
        byte_count = int(value.numel() * value.element_size())
        total_bytes += byte_count
        if (
            value.requires_grad
            or not value.is_contiguous()
            or value.dtype != torch.float32
        ):
            raise ValueError(f"Feature-cache contract changed: {name}")
        fields[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "bytes": byte_count,
            "requires_grad": False,
        }
    return {"fields": fields, "total_bytes": total_bytes}


class ExplicitJetTrainingBackend:
    """Cold cache plus persistent explicit-jet CUDA Graph training core."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        full_pool: torch.Tensor,
        source_pool_identity: Mapping[str, Any],
        precision_policy: Mapping[str, Any],
        component_weights: Mapping[str, float] = PDE_COMPONENT_WEIGHTS,
        capture_indices: torch.Tensor | None = None,
        cache_chunk_size: int = CACHE_CHUNK_SIZE,
        rebuilt_after_resume: bool = False,
    ) -> None:
        if cache_chunk_size != CACHE_CHUNK_SIZE:
            raise ValueError(
                "explicit-jet production cache chunk size is frozen to 256"
            )
        if dict(component_weights) != PDE_COMPONENT_WEIGHTS:
            raise ValueError("explicit-jet production requires four unit PDE weights")
        if (
            not isinstance(precision_policy, Mapping)
            or precision_policy.get("mode") != "ieee"
        ):
            raise ValueError("explicit-jet production is locked to IEEE FP32")
        if not isinstance(rebuilt_after_resume, bool):
            raise TypeError("rebuilt_after_resume must be boolean")
        self.source_file_sha256 = validate_backend_source_inventory()
        self.model = model
        self.optimizer = optimizer
        self.trainable_parameters = validate_model_optimizer_contract(model, optimizer)
        self.device, pool_sha = _validate_pool(full_pool, model, source_pool_identity)
        self.full_pool = full_pool
        self.source_pool_identity = dict(source_pool_identity)
        self.precision_policy = dict(precision_policy)
        self.component_weights = dict(component_weights)
        self.cache_chunk_size = CACHE_CHUNK_SIZE
        self.rebuilt_after_resume = rebuilt_after_resume
        model_sha_before = model_state_sha256(model)
        optimizer_sha_before = optimizer_state_sha256(optimizer)
        cpu_rng_before = torch.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state(self.device).clone()

        if capture_indices is None:
            capture_indices = build_independent_capture_indices(
                len(full_pool), self.device
            )
        if (
            not isinstance(capture_indices, torch.Tensor)
            or capture_indices.dtype != torch.long
            or capture_indices.device != self.device
            or tuple(capture_indices.shape) != (EXPLICIT_BATCH_SIZE,)
            or not capture_indices.is_contiguous()
        ):
            raise ValueError("explicit-jet production capture indices contract changed")
        if int(capture_indices.min()) < 0 or int(capture_indices.max()) >= len(
            full_pool
        ):
            raise IndexError(
                "explicit-jet production capture indices are outside the pool"
            )
        self.capture_indices = capture_indices

        torch.cuda.synchronize(self.device)
        cache_started = time.perf_counter()
        self.full_cache = build_full_feature_jet(
            model,
            full_pool[:, :3].contiguous(),
            chunk_size=CACHE_CHUNK_SIZE,
        )
        torch.cuda.synchronize(self.device)
        self.cache_build_seconds = time.perf_counter() - cache_started
        cache_layout = _cache_field_identity(self.full_cache)

        torch.cuda.synchronize(self.device)
        graph_started = time.perf_counter()
        self.core = ExplicitJetCudaGraphTrainingCore(
            model=model,
            optimizer=optimizer,
            full_cache=self.full_cache,
            capture_indices=capture_indices,
            component_weights=self.component_weights,
            precision_policy=self.precision_policy,
        )
        torch.cuda.synchronize(self.device)
        self.graph_setup_seconds = time.perf_counter() - graph_started
        self.total_setup_seconds = self.cache_build_seconds + self.graph_setup_seconds

        model_sha_after = model_state_sha256(model)
        optimizer_sha_after = optimizer_state_sha256(optimizer)
        if model_sha_after != model_sha_before:
            raise RuntimeError("explicit-jet cache/Graph setup changed model state")
        if optimizer_sha_after != optimizer_sha_before:
            raise RuntimeError(
                "explicit-jet cache/Graph setup changed eager Adam state"
            )
        if not torch.equal(torch.get_rng_state(), cpu_rng_before):
            raise RuntimeError("explicit-jet cache/Graph setup changed CPU RNG state")
        if not torch.equal(torch.cuda.get_rng_state(self.device), cuda_rng_before):
            raise RuntimeError("explicit-jet cache/Graph setup changed CUDA RNG state")

        backend_path = Path(__file__).resolve()
        self.cache_identity = {
            "schema_version": SCHEMA_VERSION,
            "backend_revision": REVISION,
            "backend_source_sha256": file_sha256(backend_path),
            "dependency_source_sha256": dict(self.source_file_sha256),
            "case_id": source_pool_identity.get("case_id"),
            "pool_size": int(len(full_pool)),
            "pool_probe_sha256": pool_sha,
            "initial_runtime_model_state_sha256": model_sha_before,
            "initial_runtime_optimizer_state_sha256": optimizer_sha_before,
            "cache_chunk_size": CACHE_CHUNK_SIZE,
            "cache_fields": cache_layout["fields"],
            "cache_total_bytes": cache_layout["total_bytes"],
            "cache_build_seconds": self.cache_build_seconds,
            "graph_setup_seconds": self.graph_setup_seconds,
            "total_setup_seconds": self.total_setup_seconds,
            "cache_checkpointed": False,
            "cuda_graph_checkpointed": False,
            "recreated_after_resume": rebuilt_after_resume,
        }

    def replay(
        self, indices: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        """Replay PDE/backward only; the caller owns eager Adam.step()."""

        return self.core.replay(indices)

    def synchronize(self) -> None:
        self.core.synchronize()

    @property
    def replay_count(self) -> int:
        return int(self.core.replay_count)

    def runtime_definition(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "revision": REVISION,
            "backend": "C1-A3-explicit-jet-production",
            "backend_id": BACKEND_ID,
            "dtype": "float32",
            "matmul_precision_mode": "ieee",
            "rff_spec_sha256": FLOW_RFF_SPEC_SHA256,
            "trainable_parameter_tensors": EXPECTED_HEAD_PARAMETER_TENSORS,
            "trainable_parameter_count": EXPECTED_HEAD_PARAMETER_COUNT,
            "cache_chunk_size": CACHE_CHUNK_SIZE,
            "cache_fields": list(EXPLICIT_CACHE_FIELDS),
            "cold_cache_build": True,
            "coordinate_autodiff_active_during_training": False,
            "optimizer": "torch.optim.Adam",
            "optimizer_captured": False,
            "optimizer_step_owner": "caller_eager_after_replay",
            "cache_checkpointed": False,
            "cuda_graph_checkpointed": False,
            "resume_action": "rebuild_cache_and_graph_from_restored_model",
            "recreated_after_resume": self.rebuilt_after_resume,
            "checkpoint_metadata": production_checkpoint_metadata(),
            "cache_identity": deepcopy(self.cache_identity),
            "core": self.core.runtime_definition(),
        }

    def checkpoint_policy(self) -> dict[str, Any]:
        return {
            "backend_state_in_checkpoint": False,
            "cache_in_checkpoint": False,
            "cuda_graph_in_checkpoint": False,
            "checkpoint_owners": ["model", "optimizer", "sampler", "rng"],
            "resume_action": "rebuild_runtime_from_restored_state",
            "resume_compatibility": "same_backend_only",
            "checkpoint_metadata": production_checkpoint_metadata(),
        }
