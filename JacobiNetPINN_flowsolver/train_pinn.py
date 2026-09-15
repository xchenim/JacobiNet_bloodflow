"""Train the fixed JacobiNet–flow model/explicit-jet PINN at a validated endpoint budget.

Training is label-free, uses no early stopping, and supports the validated
10k/20k/30k/40k cosine horizons.
"""

from __future__ import annotations

if __package__ and __package__.startswith("JacobiNetPINN_flowsolver"):
    from .flux import true_section_flux_monitor
else:
    from flux import true_section_flux_monitor

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import platform
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

if __package__:
    from .checkpoint import archived_flow_protocol
else:
    from checkpoint import archived_flow_protocol


if __package__ and __package__.startswith("JacobiNetPINN_flowsolver"):
    from . import runtime as explicit_jet_backend
    from .runtime import (
        ExplicitJetTrainingBackend,
        CompleteUniformIndexAdapter,
        build_independent_capture_indices,
    )
    from .runtime import CUDA_GRAPH_WARMUP_STEPS
    from .checkpoint import (
        load_jacobinet_checkpoint,
        load_pinn_checkpoint,
        save_pinn_checkpoint,
        prepare_training_output,
    )
    from .dataset import load_training_points
    from .model import (
        JacobiNet,
        NetPINN,
        base_rff_matrices_from_state,
        consume_reference_rff_rng_draws,
        pde_losses,
    )
    from .runtime import (
        CompleteUniformSampler,
        GeometryRegions,
        deterministic_validation_points,
    )
else:
    import runtime as explicit_jet_backend
    from runtime import (
        ExplicitJetTrainingBackend,
        CompleteUniformIndexAdapter,
        build_independent_capture_indices,
    )
    from runtime import CUDA_GRAPH_WARMUP_STEPS
    from checkpoint import (
        load_jacobinet_checkpoint,
        load_pinn_checkpoint,
        save_pinn_checkpoint,
        prepare_training_output,
    )
    from dataset import load_training_points
    from model import (
        JacobiNet,
        NetPINN,
        base_rff_matrices_from_state,
        consume_reference_rff_rng_draws,
        pde_losses,
    )
    from runtime import (
        CompleteUniformSampler,
        GeometryRegions,
        deterministic_validation_points,
    )
CONFIGURATION_IDENTITY_REVISION = "semantic_v2"
OPTIMIZATION_PINN_METHOD_REVISION = "pinn_c1_speed_search_float32_xyz3"
RESUME_STATE_SCHEMA_VERSION = 2
if __package__ and __package__.startswith("JacobiNetPINN_flowsolver"):
    from .physics import (
        RAW_EQUAL_LOSS,
        THROAT_MODE,
        build_loss_balance,
        build_physics_scales,
        weighted_component_values,
    )
    from .runtime import (
        EVALUATION_DTYPE_NAME,
        PRECISION_POLICY_NAME,
        TORCH_TRAINING_DTYPE,
        TRAINING_DTYPE_NAME,
        assert_model_training_dtype,
    )
    from .rff import load_spec
    from .runtime import apply_matmul_precision_policy
else:
    from physics import (
        RAW_EQUAL_LOSS,
        THROAT_MODE,
        build_loss_balance,
        build_physics_scales,
        weighted_component_values,
    )
    from runtime import (
        EVALUATION_DTYPE_NAME,
        PRECISION_POLICY_NAME,
        TORCH_TRAINING_DTYPE,
        TRAINING_DTYPE_NAME,
        assert_model_training_dtype,
    )
    from rff import load_spec
    from runtime import apply_matmul_precision_policy


_ARCHIVE_DEFAULTS = archived_flow_protocol(cohort=False)
EXPLICIT_JET_RUNTIME_MODE = _ARCHIVE_DEFAULTS["runtime_mode"]
EXPLICIT_JET_METHOD_SUFFIX = _ARCHIVE_DEFAULTS["method_suffix"]
METHOD_REVISION = _ARCHIVE_DEFAULTS["method_revision"]
TRAINING_PROTOCOL_ID = _ARCHIVE_DEFAULTS["protocol_id"]
SCHEDULER_HORIZON_STEPS = 20_000
ALLOWED_HORIZONS = frozenset({10_000, 20_000, 30_000, 40_000})


def configure_horizon(horizon: int) -> None:
    """Configure one validated fixed-endpoint budget before a run."""

    global EXPLICIT_JET_METHOD_SUFFIX
    global METHOD_REVISION
    global TRAINING_PROTOCOL_ID
    global SCHEDULER_HORIZON_STEPS
    global LOSS_ONLY_CONFIGURATION_IDENTITY_REVISION
    if horizon not in ALLOWED_HORIZONS:
        raise ValueError("horizon must be 10000, 20000, 30000, or 40000")
    identifiers = archived_flow_protocol(horizon)
    EXPLICIT_JET_METHOD_SUFFIX = identifiers["method_suffix"]
    METHOD_REVISION = identifiers["method_revision"]
    TRAINING_PROTOCOL_ID = identifiers["protocol_id"]
    SCHEDULER_HORIZON_STEPS = horizon
    LOSS_ONLY_CONFIGURATION_IDENTITY_REVISION = identifiers["configuration_revision"]


LOSS_ONLY_RESUME_STATE_SCHEMA_VERSION = RESUME_STATE_SCHEMA_VERSION + 1
LOSS_ONLY_CONFIGURATION_IDENTITY_REVISION = (
    f"{CONFIGURATION_IDENTITY_REVISION}+h20k_no_early_stop_v1"
)
EARLY_STOP_ENABLED = False
LOSS_STOP_RULE_ID = "weighted_pde_loss_conservative_v1"
LOSS_STOP_MIN_STEP = 12_000
LOSS_TOTAL_THRESHOLD = 3.0e-6
LOSS_COMPONENT_THRESHOLD = 1.0e-6
LOSS_PATIENCE_EVALS = 20
LOSS_EVAL_FREQUENCY_STEPS = 25
LOSS_COMPONENTS = ("momentum_u", "momentum_v", "momentum_w", "continuity")
LOSS_CONVERGED_STATUS = "loss-converged-unverified"
MAX_STEPS_STATUS = "max-steps-reached/unverified"
SEGMENT_END_STATUS = "segment-budget-reached/unverified"
if TRAINING_DTYPE_NAME != "float32":
    raise RuntimeError(
        "The fixed loss-only protocol is frozen to float32 training; "
        f"got {TRAINING_DTYPE_NAME!r}"
    )
BOUNDARY_MODES = ("soft_penalty", "monitor_only")
VARIANT = _ARCHIVE_DEFAULTS["variant"]
FAST_RUNTIME_MODES = (EXPLICIT_JET_RUNTIME_MODE,)


def validate_schedule_arguments(
    *,
    run_steps: int,
    scheduler_horizon_steps: int,
    lr_start: float,
    lr_end: float,
) -> None:
    """Validate the configured training budget and closed-form cosine LR contract."""

    if isinstance(run_steps, bool) or not isinstance(run_steps, int):
        raise TypeError("run_steps must be an integer")
    if isinstance(scheduler_horizon_steps, bool) or not isinstance(
        scheduler_horizon_steps, int
    ):
        raise TypeError("scheduler_horizon_steps must be an integer")
    if scheduler_horizon_steps != SCHEDULER_HORIZON_STEPS:
        raise ValueError(
            f"scheduler horizon must equal the configured value {SCHEDULER_HORIZON_STEPS}"
        )
    if not 1 <= run_steps <= SCHEDULER_HORIZON_STEPS:
        raise ValueError(f"run_steps must be in [1, {SCHEDULER_HORIZON_STEPS}]")
    learning_rates: dict[str, float] = {}
    for label, value in (("lr_start", lr_start), ("lr_end", lr_end)):
        if isinstance(value, bool):
            raise TypeError(f"{label} must be a real number")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label} must be a real number") from exc
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
        learning_rates[label] = parsed
    if learning_rates["lr_start"] < learning_rates["lr_end"]:
        raise ValueError("lr_start must be greater than or equal to lr_end")


def validate_loss_stop_arguments(
    *,
    min_step: int,
    total_threshold: float,
    component_threshold: float,
    patience_evals: int,
    eval_frequency_steps: int,
) -> None:
    """Reject any drift from the calibrated conservative stopping rule."""

    expected = {
        "min_step": LOSS_STOP_MIN_STEP,
        "total_threshold": LOSS_TOTAL_THRESHOLD,
        "component_threshold": LOSS_COMPONENT_THRESHOLD,
        "patience_evals": LOSS_PATIENCE_EVALS,
        "eval_frequency_steps": LOSS_EVAL_FREQUENCY_STEPS,
    }
    observed = {
        "min_step": min_step,
        "total_threshold": total_threshold,
        "component_threshold": component_threshold,
        "patience_evals": patience_evals,
        "eval_frequency_steps": eval_frequency_steps,
    }
    if isinstance(min_step, bool) or not isinstance(min_step, int):
        raise TypeError("loss-stop-min-step must be an integer")
    if isinstance(patience_evals, bool) or not isinstance(patience_evals, int):
        raise TypeError("loss-patience-evals must be an integer")
    if isinstance(eval_frequency_steps, bool) or not isinstance(
        eval_frequency_steps, int
    ):
        raise TypeError("eval-frequency-steps must be an integer")
    for label, value in (
        ("loss-total-threshold", total_threshold),
        ("loss-component-threshold", component_threshold),
    ):
        if isinstance(value, bool):
            raise TypeError(f"{label} must be a real number")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{label} must be a real number") from error
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
    if observed != expected:
        raise ValueError(
            "Loss-only stopping rule is frozen; "
            f"expected={expected}, observed={observed}"
        )


def loss_stop_rule() -> dict[str, Any]:
    return {
        "rule_id": LOSS_STOP_RULE_ID,
        "min_step": LOSS_STOP_MIN_STEP,
        "evaluation_frequency_steps": LOSS_EVAL_FREQUENCY_STEPS,
        "total_threshold": LOSS_TOTAL_THRESHOLD,
        "component_threshold": LOSS_COMPONENT_THRESHOLD,
        "patience_evaluations": LOSS_PATIENCE_EVALS,
        "total_fields": ["validation_score", "ema_total", "train_mean_total"],
        "weighted_component_names": list(LOSS_COMPONENTS),
        "weighted_component_sources": ["validation", "ema", "train_mean"],
        "inclusive_comparison": True,
        "labels_used": False,
        "enabled": EARLY_STOP_ENABLED,
        "decision_use": "monitor_only",
    }


def _finite_nonnegative_loss(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be numeric") from error
    if not math.isfinite(parsed):
        raise FloatingPointError(f"Non-finite loss-stop value: {label}={parsed}")
    if parsed < 0.0:
        raise ValueError(f"Loss-stop value must be non-negative: {label}={parsed}")
    return parsed


def loss_stop_snapshot(
    *,
    global_step: int,
    validation_score: Any,
    validation_weighted: Mapping[str, Any],
    ema: Mapping[str, Any],
    training_means: Mapping[str, Any],
    min_step: int = LOSS_STOP_MIN_STEP,
    total_threshold: float = LOSS_TOTAL_THRESHOLD,
    component_threshold: float = LOSS_COMPONENT_THRESHOLD,
) -> dict[str, Any]:
    """Return the complete fail-closed predicate state for one evaluation."""

    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise TypeError("global_step must be an integer")
    if global_step < 1:
        raise ValueError("global_step must be positive")
    totals = {
        "validation_score": _finite_nonnegative_loss(
            validation_score, "validation_score"
        ),
        "ema_total": _finite_nonnegative_loss(ema.get("total"), "ema_total"),
        "train_mean_total": _finite_nonnegative_loss(
            training_means.get("train_mean_total"), "train_mean_total"
        ),
    }
    components: dict[str, float] = {}
    for name in LOSS_COMPONENTS:
        components[f"weighted_validation_{name}"] = _finite_nonnegative_loss(
            validation_weighted.get(name), f"weighted_validation_{name}"
        )
        components[f"ema_weighted_{name}"] = _finite_nonnegative_loss(
            ema.get(f"weighted_{name}"), f"ema_weighted_{name}"
        )
        components[f"train_mean_weighted_{name}"] = _finite_nonnegative_loss(
            training_means.get(f"train_mean_weighted_{name}"),
            f"train_mean_weighted_{name}",
        )
    eligible = bool(
        global_step >= min_step and global_step % LOSS_EVAL_FREQUENCY_STEPS == 0
    )
    qualified = bool(
        eligible
        and all(value <= total_threshold for value in totals.values())
        and all(value <= component_threshold for value in components.values())
    )
    return {
        "rule_id": LOSS_STOP_RULE_ID,
        "global_step": global_step,
        "eligible": eligible,
        "qualified": qualified,
        "totals": totals,
        "weighted_components": components,
        "limits": {
            "min_step": min_step,
            "evaluation_frequency_steps": LOSS_EVAL_FREQUENCY_STEPS,
            "total_threshold": total_threshold,
            "component_threshold": component_threshold,
        },
    }


def validate_loss_stop_snapshot(
    snapshot: Any, *, expected_step: int | None = None
) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping):
        raise TypeError("loss-stop snapshot must be a mapping")
    required = {
        "rule_id",
        "global_step",
        "eligible",
        "qualified",
        "totals",
        "weighted_components",
        "limits",
    }
    _require_exact_keys(snapshot, required, "loss_stop_snapshot")
    if snapshot["rule_id"] != LOSS_STOP_RULE_ID:
        raise ValueError("loss-stop snapshot rule mismatch")
    step = _strict_int(
        snapshot["global_step"], "loss_stop_snapshot.global_step", minimum=1
    )
    if expected_step is not None and step != expected_step:
        raise ValueError("loss-stop snapshot global_step mismatch")
    if not isinstance(snapshot["eligible"], bool) or not isinstance(
        snapshot["qualified"], bool
    ):
        raise TypeError("loss-stop eligibility/qualification must be boolean")
    totals = snapshot["totals"]
    _require_exact_keys(
        totals,
        {"validation_score", "ema_total", "train_mean_total"},
        "loss_stop_snapshot.totals",
    )
    parsed_totals = {
        name: _finite_nonnegative_loss(value, f"loss_stop_snapshot.totals.{name}")
        for name, value in totals.items()
    }
    expected_component_keys = {
        f"{prefix}_{name}"
        for prefix in ("weighted_validation", "ema_weighted", "train_mean_weighted")
        for name in LOSS_COMPONENTS
    }
    components = snapshot["weighted_components"]
    _require_exact_keys(
        components, expected_component_keys, "loss_stop_snapshot.weighted_components"
    )
    parsed_components = {
        name: _finite_nonnegative_loss(
            value, f"loss_stop_snapshot.weighted_components.{name}"
        )
        for name, value in components.items()
    }
    expected_limits = {
        "min_step": LOSS_STOP_MIN_STEP,
        "evaluation_frequency_steps": LOSS_EVAL_FREQUENCY_STEPS,
        "total_threshold": LOSS_TOTAL_THRESHOLD,
        "component_threshold": LOSS_COMPONENT_THRESHOLD,
    }
    if snapshot["limits"] != expected_limits:
        raise ValueError("loss-stop snapshot limits mismatch")
    eligible = bool(
        step >= LOSS_STOP_MIN_STEP and step % LOSS_EVAL_FREQUENCY_STEPS == 0
    )
    qualified = bool(
        eligible
        and all(value <= LOSS_TOTAL_THRESHOLD for value in parsed_totals.values())
        and all(
            value <= LOSS_COMPONENT_THRESHOLD for value in parsed_components.values()
        )
    )
    if snapshot["eligible"] != eligible or snapshot["qualified"] != qualified:
        raise ValueError("loss-stop snapshot decision is inconsistent with its values")
    return dict(snapshot)


def advance_loss_stop_counter(
    previous_count: int, snapshot: Mapping[str, Any]
) -> tuple[int, bool]:
    if isinstance(previous_count, bool) or not isinstance(previous_count, int):
        raise TypeError("previous_count must be an integer")
    if previous_count < 0:
        raise ValueError("previous_count is invalid for the fixed protocol")
    validated = validate_loss_stop_snapshot(snapshot)
    if not validated["eligible"]:
        return previous_count, False
    count = previous_count + 1 if validated["qualified"] else 0
    # Keep the checkpoint monitoring counter; this protocol never stops early.
    return count, False


def loss_only_terminal_status(*, converged: bool, completed_steps: int) -> str:
    if not isinstance(converged, bool):
        raise TypeError("converged must be boolean")
    if isinstance(completed_steps, bool) or not isinstance(completed_steps, int):
        raise TypeError("completed_steps must be an integer")
    if not 1 <= completed_steps <= SCHEDULER_HORIZON_STEPS:
        raise ValueError(f"completed_steps must be in [1, {SCHEDULER_HORIZON_STEPS}]")
    if converged:
        return LOSS_CONVERGED_STATUS
    if completed_steps == SCHEDULER_HORIZON_STEPS:
        return MAX_STEPS_STATUS
    return SEGMENT_END_STATUS


def release_source_identity() -> dict[str, str]:
    """Record the public trainer and final explicit-jet backend used by this run."""

    trainer_source = Path(__file__).resolve()
    backend_source = Path(explicit_jet_backend.__file__).resolve()
    return {
        "release_trainer_path": trainer_source.name,
        "release_trainer_sha256": sha256(trainer_source),
        "a3_backend_path": backend_source.name,
        "a3_backend_sha256": sha256(backend_source),
    }


def trainable_head_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if len(parameters) != 18:
        raise ValueError(
            "explicit-jet flow model optimizer requires exactly 18 trainable tensors"
        )
    if sum(parameter.numel() for parameter in parameters) != 116_100:
        raise ValueError(
            "explicit-jet flow model optimizer requires exactly 116,100 values"
        )
    return parameters


def nested_state_equal(reference: Any, candidate: Any) -> bool:
    """Exact recursive equality for tensors and RNG/optimizer state trees."""

    if torch.is_tensor(reference) or torch.is_tensor(candidate):
        return (
            torch.is_tensor(reference)
            and torch.is_tensor(candidate)
            and torch.equal(reference, candidate)
        )
    if isinstance(reference, Mapping) or isinstance(candidate, Mapping):
        if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
            return False
        return set(reference) == set(candidate) and all(
            nested_state_equal(reference[key], candidate[key]) for key in reference
        )
    if isinstance(reference, (list, tuple)) or isinstance(candidate, (list, tuple)):
        if type(reference) is not type(candidate) or len(reference) != len(candidate):
            return False
        return all(
            nested_state_equal(left, right) for left, right in zip(reference, candidate)
        )
    return bool(reference == candidate)


def nested_state_sha256(value: Any) -> str:
    """Stable audit hash for nested sampler/RNG/optimizer state."""

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=lambda key: repr(key)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            for child in item:
                visit(child)
        else:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


ACCUMULATOR_NAMES = (
    "momentum_u",
    "momentum_v",
    "momentum_w",
    "continuity",
    "weighted_momentum_u",
    "weighted_momentum_v",
    "weighted_momentum_w",
    "weighted_continuity",
    "bc_inlet",
    "bc_outlet",
    "bc_wall",
    "weighted_bc_inlet",
    "weighted_bc_outlet",
    "weighted_bc_wall",
    "total",
)
FULL_RESUME_REQUIRED_KEYS = frozenset(
    {
        "full_resume_state",
        "resume_state_schema_version",
        "configuration_identity_revision",
        "global_step",
        "terminal_loss_snapshot",
        "configuration_sha256",
        "terminal_step_checkpoint",
        "label_diagnostics_disabled",
        "loss_stop_rule",
        "stop_reason",
        "train_seed",
        "rff_seed",
        "rff_spec_sha256",
        "optimizer_state_dict",
        "scheduler_state",
        "sampler_state",
        "rng_state",
        "accumulators",
        "ema",
        "accumulated_steps",
        "pde_points_processed",
        "boundary_points_processed",
        "best_physics_score",
        "best_physics_step",
        "best_physics_snapshot",
        "first_loss_qualified_step",
        "first_loss_qualified_seconds",
        "loss_converged_step",
        "loss_converged_seconds",
        "consecutive_loss_stop_evals",
        "evaluation_count",
        "training_seconds",
        "resumed_from",
    }
)
RNG_STATE_REQUIRED_KEYS = frozenset(
    {"python", "numpy", "torch_cpu", "torch_cuda", "boundary_generator"}
)
NUMPY_RNG_REQUIRED_KEYS = frozenset(
    {"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"}
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def cosine_lr(index: int, maximum: int, start: float, end: float) -> float:
    if maximum <= 1:
        return float(end)
    fraction = min(max(index / (maximum - 1), 0.0), 1.0)
    return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * fraction)))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(values: torch.Tensor) -> str:
    array = values.detach().cpu().numpy().astype(np.float64, copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str] | set[str], label: str
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    actual = set(value)
    missing = sorted(set(expected) - actual)
    unexpected = sorted(actual - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"{label} keys mismatch: missing={missing}, unexpected={unexpected}"
        )


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _finite_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value.lower()


def cpu_contiguous_generator_state(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.dtype != torch.uint8:
        raise TypeError(f"{label} must have dtype torch.uint8")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional tensor")
    return value.detach().to(device="cpu").contiguous()


def _strip_runtime_geometry_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_runtime_geometry_paths(item)
            for key, item in value.items()
            if key not in {"geometry_path", "release_trainer_path", "a3_backend_path"}
        }
    if isinstance(value, list):
        return [_strip_runtime_geometry_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_runtime_geometry_paths(item) for item in value)
    return value


def resume_configuration_identity(configuration: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(configuration, Mapping):
        raise TypeError("configuration must be a mapping")
    if (
        configuration.get("configuration_identity_revision")
        != LOSS_ONLY_CONFIGURATION_IDENTITY_REVISION
    ):
        raise ValueError("configuration identity revision mismatch")
    identity = _strip_runtime_geometry_paths(deepcopy(dict(configuration)))
    identity.pop("max_steps", None)
    identity.pop("run_steps", None)
    physics_scales = identity.get("physics_scales")
    if not isinstance(physics_scales, dict):
        raise TypeError("configuration.physics_scales must be a dictionary")
    _validate_sha256(
        physics_scales.get("geometry_sha256"),
        "configuration.physics_scales.geometry_sha256",
    )
    return identity


def resume_configuration_sha256(configuration: Mapping[str, Any]) -> str:
    identity = resume_configuration_identity(configuration)
    payload = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def clone_model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in model.state_dict().items()
    }


def capture_best_snapshot(
    model: torch.nn.Module, checkpoint_state: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "model_state_dict": clone_model_state_dict(model),
        "checkpoint_state": deepcopy(dict(checkpoint_state)),
    }


def _validate_best_snapshot(
    snapshot: Any,
    model: torch.nn.Module,
    *,
    label: str,
    expected_step: int | None,
    expected_score: float,
    expected_configuration_sha256: str,
) -> None:
    if expected_step is None:
        if snapshot is not None:
            raise ValueError(f"{label} must be null when its best step is null")
        return
    if not isinstance(snapshot, Mapping):
        raise TypeError(f"{label} must be a mapping")
    _require_exact_keys(snapshot, {"model_state_dict", "checkpoint_state"}, label)
    checkpoint_state = snapshot["checkpoint_state"]
    if not isinstance(checkpoint_state, Mapping):
        raise TypeError(f"{label}.checkpoint_state must be a mapping")
    required_checkpoint_keys = {
        "global_step",
        "validation_score",
        "configuration_sha256",
    }
    missing = sorted(required_checkpoint_keys - set(checkpoint_state))
    if missing:
        raise ValueError(f"{label}.checkpoint_state is missing {missing}")
    snapshot_step = _strict_int(
        checkpoint_state["global_step"], f"{label}.global_step", minimum=1
    )
    if snapshot_step != expected_step:
        raise ValueError(f"{label} global_step does not match its recorded best step")
    snapshot_score = _finite_float(
        checkpoint_state["validation_score"],
        f"{label}.validation_score",
        minimum=0.0,
    )
    if snapshot_score != expected_score:
        raise ValueError(
            f"{label} validation score does not match the recorded best score"
        )
    if checkpoint_state["configuration_sha256"] != expected_configuration_sha256:
        raise ValueError(f"{label} configuration hash mismatch")
    model_state = snapshot["model_state_dict"]
    if not isinstance(model_state, Mapping):
        raise TypeError(f"{label}.model_state_dict must be a mapping")
    active_state = model.state_dict()
    _require_exact_keys(model_state, set(active_state), f"{label}.model_state_dict")
    for name, expected in active_state.items():
        observed = model_state[name]
        if not isinstance(observed, torch.Tensor):
            raise TypeError(f"{label}.model_state_dict[{name!r}] must be a tensor")
        if observed.shape != expected.shape or observed.dtype != expected.dtype:
            raise ValueError(f"{label}.model_state_dict[{name!r}] is incompatible")


def normalize_best_snapshot(
    snapshot: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "model_state_dict": {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in snapshot["model_state_dict"].items()
        },
        "checkpoint_state": deepcopy(dict(snapshot["checkpoint_state"])),
    }


def materialize_best_snapshot_checkpoint(
    path: Path,
    model: torch.nn.Module,
    scales: Any,
    snapshot: Mapping[str, Any],
) -> None:
    active_state = clone_model_state_dict(model)
    try:
        model.load_state_dict(snapshot["model_state_dict"], strict=True)
        save_pinn_checkpoint(
            path, model, scales, deepcopy(dict(snapshot["checkpoint_state"]))
        )
    finally:
        model.load_state_dict(active_state, strict=True)


def _validate_sampler_state(sampler: object, state: Any) -> None:
    if not isinstance(state, Mapping):
        raise TypeError("sampler_state must be a mapping")
    expected = {"class", "generator_state"}
    if isinstance(sampler, CompleteUniformSampler):
        expected.update({"permutation", "cursor", "cycles"})
    else:
        raise TypeError(f"Unsupported sampler for resume: {type(sampler).__name__}")
    _require_exact_keys(state, expected, "sampler_state")
    if state["class"] != type(sampler).__name__:
        raise ValueError("Checkpoint sampler class does not match the active sampler")
    cpu_contiguous_generator_state(
        state["generator_state"], "sampler_state.generator_state"
    )
    if isinstance(sampler, CompleteUniformSampler):
        permutation = state["permutation"]
        if not isinstance(permutation, torch.Tensor):
            raise TypeError("sampler_state.permutation must be a tensor")
        if permutation.dtype != sampler.permutation.dtype:
            raise TypeError("Checkpoint sampler permutation dtype mismatch")
        if permutation.shape != sampler.permutation.shape:
            raise ValueError("Checkpoint sampler permutation shape mismatch")
        cursor = _strict_int(state["cursor"], "sampler_state.cursor")
        _strict_int(state["cycles"], "sampler_state.cycles")
        if not 0 <= cursor < len(sampler.points):
            raise ValueError("Checkpoint sampler cursor is invalid")


def _validate_rng_state(state: Any) -> None:
    if not isinstance(state, Mapping):
        raise TypeError("rng_state must be a mapping")
    _require_exact_keys(state, RNG_STATE_REQUIRED_KEYS, "rng_state")
    try:
        random.Random().setstate(state["python"])
    except (TypeError, ValueError) as error:
        raise ValueError("rng_state.python is invalid") from error
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise TypeError("rng_state.numpy must be a mapping")
    _require_exact_keys(numpy_state, NUMPY_RNG_REQUIRED_KEYS, "rng_state.numpy")
    if not isinstance(numpy_state["bit_generator"], str):
        raise TypeError("rng_state.numpy.bit_generator must be a string")
    numpy_keys = numpy_state["keys"]
    if not isinstance(numpy_keys, torch.Tensor) or numpy_keys.dtype != torch.int64:
        raise TypeError("rng_state.numpy.keys must be an int64 tensor")
    _strict_int(numpy_state["position"], "rng_state.numpy.position")
    _strict_int(numpy_state["has_gauss"], "rng_state.numpy.has_gauss")
    _finite_float(numpy_state["cached_gaussian"], "rng_state.numpy.cached_gaussian")
    try:
        np.random.RandomState().set_state(
            (
                numpy_state["bit_generator"],
                numpy_keys.detach().cpu().numpy().astype(np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError("rng_state.numpy is invalid") from error
    cpu_state = cpu_contiguous_generator_state(
        state["torch_cpu"], "rng_state.torch_cpu"
    )
    try:
        torch.Generator(device="cpu").set_state(cpu_state)
    except RuntimeError as error:
        raise ValueError("rng_state.torch_cpu is invalid") from error
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, (list, tuple)):
        raise TypeError("rng_state.torch_cuda must be a sequence")
    if len(cuda_states) != torch.cuda.device_count():
        raise ValueError("Checkpoint CUDA RNG device count mismatch")
    for index, cuda_state in enumerate(cuda_states):
        cpu_contiguous_generator_state(cuda_state, f"rng_state.torch_cuda[{index}]")
    cpu_contiguous_generator_state(
        state["boundary_generator"], "rng_state.boundary_generator"
    )


def _validate_optional_step_and_seconds(
    state: Mapping[str, Any], step_key: str, seconds_key: str, global_step: int
) -> None:
    step = state[step_key]
    seconds = state[seconds_key]
    if step is None:
        if seconds is not None:
            raise ValueError(f"{seconds_key} must be null when {step_key} is null")
        return
    parsed_step = _strict_int(step, step_key, minimum=1)
    if parsed_step > global_step:
        raise ValueError(f"{step_key} exceeds global_step")
    _finite_float(seconds, seconds_key, minimum=0.0)


def validate_full_resume_state(
    state: Any,
    *,
    model: torch.nn.Module,
    sampler: object,
    expected_configuration_sha256: str,
    expected_rff_spec_sha256: str,
    expected_train_seed: int,
    expected_rff_seed: int,
    expected_batch_size: int,
    expected_scheduler_horizon: int,
    expected_lr_start: float,
    expected_lr_end: float,
) -> None:
    if not isinstance(state, Mapping):
        raise TypeError("Resume checkpoint training_state must be a mapping")
    _require_exact_keys(state, FULL_RESUME_REQUIRED_KEYS, "training_state")
    if state["full_resume_state"] is not True:
        raise ValueError("Resume checkpoint is not marked as a full resume state")
    if state["resume_state_schema_version"] != LOSS_ONLY_RESUME_STATE_SCHEMA_VERSION:
        raise ValueError("Resume state schema version mismatch")
    if (
        state["configuration_identity_revision"]
        != LOSS_ONLY_CONFIGURATION_IDENTITY_REVISION
    ):
        raise ValueError("Resume configuration identity revision mismatch")
    if state["terminal_step_checkpoint"] is not True:
        raise ValueError("Resume source is not a terminal-step checkpoint")
    global_step = _strict_int(state["global_step"], "global_step", minimum=1)
    configuration_sha256 = _validate_sha256(
        state["configuration_sha256"], "configuration_sha256"
    )
    if configuration_sha256 != _validate_sha256(
        expected_configuration_sha256, "expected_configuration_sha256"
    ):
        raise ValueError("Resume checkpoint configuration hash mismatch")
    rff_spec_sha256 = _validate_sha256(state["rff_spec_sha256"], "rff_spec_sha256")
    if rff_spec_sha256 != _validate_sha256(
        expected_rff_spec_sha256, "expected_rff_spec_sha256"
    ):
        raise ValueError("Resume checkpoint RFF specification mismatch")
    if state["train_seed"] != expected_train_seed:
        raise ValueError("Resume checkpoint training seed mismatch")
    if state["rff_seed"] != expected_rff_seed:
        raise ValueError("Resume checkpoint RFF seed mismatch")
    if state["label_diagnostics_disabled"] is not True:
        raise ValueError("Resume checkpoint permits label diagnostics")
    if state["loss_stop_rule"] != loss_stop_rule():
        raise ValueError("Resume loss-stop rule mismatch")
    if state["stop_reason"] not in {
        LOSS_CONVERGED_STATUS,
        MAX_STEPS_STATUS,
        SEGMENT_END_STATUS,
    }:
        raise ValueError("Resume stop reason is unsupported")
    terminal_snapshot = validate_loss_stop_snapshot(
        state["terminal_loss_snapshot"], expected_step=global_step
    )
    if (
        state["stop_reason"] == MAX_STEPS_STATUS
        and global_step != SCHEDULER_HORIZON_STEPS
    ):
        raise ValueError(
            f"max-steps status requires global_step={SCHEDULER_HORIZON_STEPS}"
        )
    if (
        state["stop_reason"] == SEGMENT_END_STATUS
        and global_step >= SCHEDULER_HORIZON_STEPS
    ):
        raise ValueError("segment status is invalid at the scheduler horizon")

    optimizer_state = state["optimizer_state_dict"]
    if not isinstance(optimizer_state, Mapping):
        raise TypeError("optimizer_state_dict must be a mapping")
    _require_exact_keys(
        optimizer_state, {"state", "param_groups"}, "optimizer_state_dict"
    )
    if (
        not isinstance(optimizer_state["param_groups"], list)
        or not optimizer_state["param_groups"]
    ):
        raise ValueError("optimizer_state_dict.param_groups must be a non-empty list")

    scheduler_state = state["scheduler_state"]
    if not isinstance(scheduler_state, Mapping):
        raise TypeError("scheduler_state must be a mapping")
    _require_exact_keys(
        scheduler_state,
        {"kind", "horizon_steps", "last_step", "last_lr"},
        "scheduler_state",
    )
    if scheduler_state["kind"] != "cosine_lr_closed_form_v1":
        raise ValueError("Unsupported scheduler state kind")
    scheduler_horizon = _strict_int(
        scheduler_state["horizon_steps"], "scheduler_state.horizon_steps", minimum=1
    )
    if scheduler_horizon != expected_scheduler_horizon:
        raise ValueError("Resume scheduler horizon mismatch")
    scheduler_last_step = _strict_int(
        scheduler_state["last_step"], "scheduler_state.last_step", minimum=1
    )
    if scheduler_last_step != global_step:
        raise ValueError("Resume scheduler last_step mismatch")
    expected_last_lr = cosine_lr(
        global_step - 1,
        expected_scheduler_horizon,
        expected_lr_start,
        expected_lr_end,
    )
    observed_last_lr = _finite_float(
        scheduler_state["last_lr"], "scheduler_state.last_lr", minimum=0.0
    )
    if observed_last_lr != expected_last_lr:
        raise ValueError("Resume scheduler last_lr mismatch")
    for group in optimizer_state["param_groups"]:
        if not isinstance(group, Mapping) or group.get("lr") != expected_last_lr:
            raise ValueError("Optimizer learning rate does not match scheduler state")

    _validate_sampler_state(sampler, state["sampler_state"])
    _validate_rng_state(state["rng_state"])
    for name in ("accumulators", "ema"):
        values = state[name]
        if not isinstance(values, Mapping):
            raise TypeError(f"{name} must be a mapping")
        _require_exact_keys(values, set(ACCUMULATOR_NAMES), name)
        for component, value in values.items():
            _finite_float(value, f"{name}.{component}")
    accumulated_steps = _strict_int(state["accumulated_steps"], "accumulated_steps")
    if accumulated_steps != 0:
        raise ValueError(
            "Terminal resume state must not contain a partial accumulation window"
        )
    if any(float(value) != 0.0 for value in state["accumulators"].values()):
        raise ValueError("Terminal resume accumulators must be reset after evaluation")
    pde_points_processed = _strict_int(
        state["pde_points_processed"], "pde_points_processed"
    )
    if pde_points_processed != global_step * expected_batch_size:
        raise ValueError(
            "Resume PDE point count does not match global_step and batch size"
        )
    boundary_counts = state["boundary_points_processed"]
    if not isinstance(boundary_counts, Mapping):
        raise TypeError("boundary_points_processed must be a mapping")
    _require_exact_keys(
        boundary_counts, {"inlet", "outlet", "wall"}, "boundary_points_processed"
    )
    for name, value in boundary_counts.items():
        _strict_int(value, f"boundary_points_processed.{name}")

    physics_step = _strict_int(
        state["best_physics_step"], "best_physics_step", minimum=1
    )
    if physics_step > global_step:
        raise ValueError("best_physics_step exceeds global_step")
    physics_score = _finite_float(
        state["best_physics_score"], "best_physics_score", minimum=0.0
    )
    _validate_best_snapshot(
        state["best_physics_snapshot"],
        model,
        label="best_physics_snapshot",
        expected_step=physics_step,
        expected_score=physics_score,
        expected_configuration_sha256=expected_configuration_sha256,
    )
    _validate_optional_step_and_seconds(
        state,
        "first_loss_qualified_step",
        "first_loss_qualified_seconds",
        global_step,
    )
    _validate_optional_step_and_seconds(
        state, "loss_converged_step", "loss_converged_seconds", global_step
    )
    if (
        state["loss_converged_step"] is not None
        and state["first_loss_qualified_step"] is None
    ):
        raise ValueError("loss convergence cannot precede first qualification")
    first_step = state["first_loss_qualified_step"]
    if first_step is not None and (
        int(first_step) < LOSS_STOP_MIN_STEP
        or int(first_step) % LOSS_EVAL_FREQUENCY_STEPS
    ):
        raise ValueError("first loss-qualified step violates min-step/cadence")
    consecutive = _strict_int(
        state["consecutive_loss_stop_evals"], "consecutive_loss_stop_evals"
    )
    evaluation_count = _strict_int(
        state["evaluation_count"], "evaluation_count", minimum=1
    )
    if consecutive > evaluation_count:
        raise ValueError("consecutive_loss_stop_evals exceeds evaluation_count")
    converged = state["loss_converged_step"] is not None
    if converged != (state["stop_reason"] == LOSS_CONVERGED_STATUS):
        raise ValueError("loss convergence fields disagree with stop reason")
    if converged:
        if consecutive != LOSS_PATIENCE_EVALS:
            raise ValueError("converged checkpoint must stop at exact loss patience")
        step = int(state["loss_converged_step"])
        if step < LOSS_STOP_MIN_STEP or step % LOSS_EVAL_FREQUENCY_STEPS:
            raise ValueError("loss convergence step violates min-step/cadence")
    if first_step is None and consecutive != 0:
        raise ValueError("loss counter requires a first qualified step")
    if terminal_snapshot["eligible"]:
        if terminal_snapshot["qualified"] and consecutive == 0:
            raise ValueError("qualified terminal snapshot cannot have a zero counter")
        if not terminal_snapshot["qualified"] and consecutive != 0:
            raise ValueError("failed terminal snapshot must reset the loss counter")
    _finite_float(state["training_seconds"], "training_seconds", minimum=0.0)
    if state["resumed_from"] is not None and not isinstance(state["resumed_from"], str):
        raise TypeError("resumed_from must be null or a string")


def sampler_state_dict(sampler: object) -> dict:
    state = {
        "class": type(sampler).__name__,
        "generator_state": cpu_contiguous_generator_state(
            sampler.generator.get_state(), "sampler.generator_state"
        ),
    }
    if isinstance(sampler, CompleteUniformSampler):
        state.update(
            {
                "permutation": sampler.permutation.detach().cpu(),
                "cursor": int(sampler.cursor),
                "cycles": int(sampler.cycles),
            }
        )
    else:
        raise TypeError(f"Unsupported sampler for resume: {type(sampler).__name__}")
    return state


def load_sampler_state(sampler: object, state: dict) -> None:
    _validate_sampler_state(sampler, state)
    sampler.generator.set_state(
        cpu_contiguous_generator_state(
            state["generator_state"], "sampler_state.generator_state"
        )
    )
    if isinstance(sampler, CompleteUniformSampler):
        permutation = state["permutation"].to(sampler.points.device)
        sampler.permutation = permutation
        sampler.cursor = int(state["cursor"])
        sampler.cycles = int(state["cycles"])


def rng_state_dict(boundary_generator: torch.Generator) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": cpu_contiguous_generator_state(
            torch.get_rng_state(), "torch_cpu_rng_state"
        ),
        "torch_cuda": [
            cpu_contiguous_generator_state(state, f"torch_cuda_rng_state[{index}]")
            for index, state in enumerate(torch.cuda.get_rng_state_all())
        ],
        "boundary_generator": cpu_contiguous_generator_state(
            boundary_generator.get_state(), "boundary_generator_state"
        ),
    }


def restore_rng_state(state: dict, boundary_generator: torch.Generator) -> None:
    _validate_rng_state(state)
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["keys"].cpu().numpy().astype(np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(
        cpu_contiguous_generator_state(state["torch_cpu"], "rng_state.torch_cpu")
    )
    cuda_states = [
        cpu_contiguous_generator_state(value, f"rng_state.torch_cuda[{index}]")
        for index, value in enumerate(state["torch_cuda"])
    ]
    torch.cuda.set_rng_state_all(cuda_states)
    boundary_generator.set_state(
        cpu_contiguous_generator_state(
            state["boundary_generator"], "rng_state.boundary_generator"
        )
    )


def average_pde_losses(
    model: NetPINN,
    points: torch.Tensor,
    chunk_size: int,
    *,
    build_parameter_graph: bool = True,
    spatial_input_only: bool = False,
    derivative_mode: str = "scalar_vjp",
) -> dict[str, float]:
    totals = {
        name: 0.0 for name in ("momentum_u", "momentum_v", "momentum_w", "continuity")
    }
    count = 0
    was_training = model.training
    model.eval()
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        losses = pde_losses(
            model,
            chunk,
            build_parameter_graph=build_parameter_graph,
            spatial_input_only=spatial_input_only,
            derivative_mode=derivative_mode,
        )
        for name, value in losses.items():
            totals[name] += float(value.detach()) * len(chunk)
        count += len(chunk)
        del losses
    if was_training:
        model.train()
    return {name: value / max(count, 1) for name, value in totals.items()}


def gradient_norms(
    model: NetPINN,
    points: torch.Tensor,
    component_weights: dict[str, float] | None = None,
    *,
    spatial_input_only: bool = False,
    derivative_mode: str = "scalar_vjp",
) -> dict[str, float]:
    was_training = model.training
    model.train()
    losses = pde_losses(
        model,
        points,
        spatial_input_only=spatial_input_only,
        derivative_mode=derivative_mode,
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    output = {}
    names = list(losses)
    weights = component_weights or {name: 1.0 for name in names}
    if set(weights) != set(names):
        raise ValueError("Gradient component weights do not match PDE losses")
    for index, name in enumerate(names):
        gradients = torch.autograd.grad(
            losses[name] * float(weights[name]),
            parameters,
            retain_graph=index < len(names) - 1,
            allow_unused=True,
        )
        squared = torch.zeros((), dtype=points.dtype, device=points.device)
        for gradient in gradients:
            if gradient is not None:
                squared = squared + gradient.detach().square().sum()
        output[f"gradient_norm_{name}"] = float(torch.sqrt(squared).cpu())
    if not was_training:
        model.eval()
    return output


def should_measure_gradient_norms(
    evaluation_count: int, gradient_frequency_evals: int
) -> bool:
    """Return a horizon-independent gradient diagnostic schedule."""
    return gradient_frequency_evals > 0 and (
        evaluation_count == 1 or evaluation_count % gradient_frequency_evals == 0
    )


def boundary_monitor(
    model: NetPINN,
    inlet: torch.Tensor,
    outlet: torch.Tensor,
    wall: torch.Tensor,
    *,
    velocity_scale_m_s: float,
    pressure_scale_pa: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Evaluate fluid BC leakage and Jacobi coordinate error without gradients."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        inlet_output = model(inlet)
        inlet_target = torch.zeros_like(inlet_output[:, :3])
        inlet_target[:, 2] = inlet[:, 3] * model.inlet_scale
        inlet_error = (
            (inlet_output[:, :3] - inlet_target).cpu().numpy().astype(np.float64)
        )
        outlet_pressure = model(outlet)[:, 3].cpu().numpy().astype(np.float64)
        wall_velocity = model(wall)[:, :3].cpu().numpy().astype(np.float64)
        inlet_rs = model.jacobinet(inlet[:, :3]).cpu().numpy().astype(np.float64)
        outlet_rs = model.jacobinet(outlet[:, :3]).cpu().numpy().astype(np.float64)
        wall_rs = model.jacobinet(wall[:, :3]).cpu().numpy().astype(np.float64)
    if was_training:
        model.train()
    inlet_norm = np.linalg.norm(inlet_error, axis=1)
    wall_speed = np.linalg.norm(wall_velocity, axis=1)
    components = {
        "inlet": float(np.mean(inlet_error**2, dtype=np.float64)),
        "outlet": float(np.mean(outlet_pressure**2, dtype=np.float64)),
        "wall": float(np.mean(wall_velocity**2, dtype=np.float64)),
    }
    physical = {
        "inlet_velocity_vector_rms_m_s": float(
            np.sqrt(np.mean(inlet_norm**2, dtype=np.float64)) * velocity_scale_m_s
        ),
        "inlet_velocity_vector_max_m_s": float(np.max(inlet_norm) * velocity_scale_m_s),
        "outlet_pressure_rms_pa": float(
            np.sqrt(np.mean(outlet_pressure**2, dtype=np.float64)) * pressure_scale_pa
        ),
        "outlet_pressure_max_abs_pa": float(
            np.max(np.abs(outlet_pressure)) * pressure_scale_pa
        ),
        "wall_speed_rms_m_s": float(
            np.sqrt(np.mean(wall_speed**2, dtype=np.float64)) * velocity_scale_m_s
        ),
        "wall_speed_max_m_s": float(np.max(wall_speed) * velocity_scale_m_s),
    }

    def rms_max(values: np.ndarray) -> tuple[float, float]:
        values = np.asarray(values, dtype=np.float64)
        return (
            float(np.sqrt(np.mean(values**2, dtype=np.float64))),
            float(np.max(np.abs(values))),
        )

    inlet_rms, inlet_max = rms_max(inlet_rs[:, 1])
    outlet_rms, outlet_max = rms_max(outlet_rs[:, 1] - 1.0)
    wall_rms, wall_max = rms_max(wall_rs[:, 0])
    jacobi = {
        "inlet_axial_rms": inlet_rms,
        "inlet_axial_max_abs": inlet_max,
        "outlet_axial_rms": outlet_rms,
        "outlet_axial_max_abs": outlet_max,
        "wall_distance_rms": wall_rms,
        "wall_distance_max_abs": wall_max,
    }
    return components, physical, jacobi


def effective_boundary_weight(boundary_mode: str, declared_weight: float) -> float:
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(f"Unsupported boundary mode: {boundary_mode}")
    return float(declared_weight) if boundary_mode == "soft_penalty" else 0.0


def validation_objective(
    weighted_pde: dict[str, float],
    weighted_boundary: dict[str, float],
    *,
    boundary_mode: str,
    boundary_weight: float,
) -> float:
    return float(sum(weighted_pde.values())) + effective_boundary_weight(
        boundary_mode, boundary_weight
    ) * float(sum(weighted_boundary.values()))


def parse_training_arguments() -> argparse.Namespace:
    """Read CLI inputs and apply the published numerical protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jacobinet-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument(
        "--rff-spec-file",
        type=Path,
        help="Optional RFF JSON; defaults to the published flow configuration",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-history", type=Path)
    parser.add_argument(
        "--run-steps",
        "--max-steps",
        dest="run_steps",
        type=int,
        default=None,
        help="Defaults to the selected scheduler horizon.",
    )
    parser.add_argument(
        "--scheduler-horizon-steps",
        type=int,
        choices=sorted(ALLOWED_HORIZONS),
        required=True,
    )
    args = parser.parse_args()
    fixed = {
        "variant": VARIANT,
        "runtime_mode": EXPLICIT_JET_RUNTIME_MODE,
        "rff_seed": 99,
        "scaling_mode": THROAT_MODE,
        "loss_weighting": RAW_EQUAL_LOSS,
        "continuity_weight": 1.0,
        "batch_size": 8192,
        "eval_frequency_steps": LOSS_EVAL_FREQUENCY_STEPS,
        "loss_stop_min_step": LOSS_STOP_MIN_STEP,
        "loss_total_threshold": LOSS_TOTAL_THRESHOLD,
        "loss_component_threshold": LOSS_COMPONENT_THRESHOLD,
        "loss_patience_evals": LOSS_PATIENCE_EVALS,
        "pde_point_mode": "internal",
        "sampler": "uniform",
        "boundary_mode": "monitor_only",
        "boundary_weight": 0.0,
        "boundary_batch_size": 2048,
        "validation_count": 2048,
        "validation_batch_size": 2048,
        "gradient_frequency_evals": 10,
        "gradient_point_count": 256,
        "timing_warmup_steps": 0,
        "matmul_precision_mode": "ieee",
        "lr_start": 0.001,
        "lr_end": 1e-05,
        "ema_beta": 0.95,
    }
    for name, value in fixed.items():
        setattr(args, name, value)
    configure_horizon(args.scheduler_horizon_steps)
    if args.run_steps is None:
        args.run_steps = args.scheduler_horizon_steps
    validate_schedule_arguments(
        run_steps=args.run_steps,
        scheduler_horizon_steps=args.scheduler_horizon_steps,
        lr_start=args.lr_start,
        lr_end=args.lr_end,
    )
    validate_loss_stop_arguments(
        min_step=args.loss_stop_min_step,
        total_threshold=args.loss_total_threshold,
        component_threshold=args.loss_component_threshold,
        patience_evals=args.loss_patience_evals,
        eval_frequency_steps=args.eval_frequency_steps,
    )
    return args


def validate_training_arguments(args: argparse.Namespace) -> tuple[Path, str, int]:
    """Validate the fixed schedule, seed, CUDA device, and resume inputs."""
    if args.continuity_weight != 1.0:
        raise ValueError(
            "The published RFF configuration fixes continuity_weight=1 and baseflow off"
        )
    if args.scaling_mode != THROAT_MODE or args.loss_weighting != RAW_EQUAL_LOSS:
        raise ValueError(
            "The fixed protocol requires throat_mass_conservation and raw_equal"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    positive = (
        args.run_steps,
        args.scheduler_horizon_steps,
        args.batch_size,
        args.eval_frequency_steps,
        args.loss_patience_evals,
        args.validation_count,
        args.validation_batch_size,
        args.gradient_point_count,
    )
    if any((value < 1 for value in positive)):
        raise ValueError(
            "Step, batch, patience, validation, and gradient counts must be positive"
        )
    if args.run_steps > args.scheduler_horizon_steps:
        raise ValueError("run-steps cannot exceed scheduler-horizon-steps")
    if not 0 <= args.timing_warmup_steps < args.run_steps:
        raise ValueError("timing-warmup-steps must be in [0, run_steps)")
    if not 0.0 <= args.ema_beta < 1.0:
        raise ValueError("EMA beta must be in [0, 1)")
    case_root = args.case_root.resolve()
    case_id = args.case_id
    seed = args.seed
    if seed != 99 or args.rff_seed != 99:
        raise ValueError(
            "The published training protocol fixes train/RFF seeds to 99/99"
        )
    if args.batch_size != 8192:
        raise ValueError("The published training protocol fixes batch_size=8192")
    if args.scheduler_horizon_steps != SCHEDULER_HORIZON_STEPS:
        raise ValueError(
            f"scheduler horizon must equal the configured value {SCHEDULER_HORIZON_STEPS}"
        )
    if seed < 0 or args.rff_seed < 0:
        raise ValueError("Training and RFF seeds must be non-negative")
    if (args.resume_checkpoint is None) != (args.resume_history is None):
        raise ValueError(
            "--resume-checkpoint and --resume-history must be supplied together"
        )
    return case_root, case_id, seed


@dataclass(frozen=True, slots=True)
class BackendSetup:
    core: ExplicitJetTrainingBackend
    source_identity: dict
    audit: dict
    copy_seconds: float
    constructor_seconds: float


def prepare_training_backend(
    *,
    model,
    optimizer,
    sampler,
    boundary_generator,
    device,
    pde_pool,
    pde_pool_identity,
    matmul_precision_policy,
    loss_balance,
    resume_state,
) -> BackendSetup:
    """Build feature caches and CUDA Graphs while verifying training state is unchanged."""
    setup_state_before = {
        "model_sha256": model_state_sha256(model),
        "optimizer_sha256": explicit_jet_backend.optimizer_state_sha256(optimizer),
        "sampler_sha256": nested_state_sha256(sampler_state_dict(sampler)),
        "rng_sha256": nested_state_sha256(rng_state_dict(boundary_generator)),
    }
    setup_optimizer_before = deepcopy(optimizer.state_dict())
    setup_sampler_before = deepcopy(sampler_state_dict(sampler))
    setup_rng_before = deepcopy(rng_state_dict(boundary_generator))
    torch.cuda.synchronize(device)
    pde_cache_source_copy_started = time.perf_counter()
    pde_cache_source = pde_pool.contiguous()
    torch.cuda.synchronize(device)
    pde_cache_source_copy_seconds = time.perf_counter() - pde_cache_source_copy_started
    pde_cache_source_identity = {
        "construction": "pde_pool.contiguous()",
        "case_id": pde_pool_identity["case_id"],
        "pool_size": int(len(pde_cache_source)),
        "pool_probe_sha256": explicit_jet_backend.tensor_probe_sha256(pde_cache_source),
        "source_pool_is_contiguous": bool(pde_pool.is_contiguous()),
        "cache_source_is_contiguous": bool(pde_cache_source.is_contiguous()),
        "storage_independent_from_official_pool": pde_cache_source.data_ptr()
        != pde_pool.data_ptr(),
        "values_bitwise_equal_to_official_pool": bool(
            torch.equal(pde_cache_source, pde_pool)
        ),
        "official_sampler_retains_original_pool_object": sampler.points is pde_pool,
    }
    pde_cache_source_checks = {
        "cache_source_is_contiguous": pde_cache_source_identity[
            "cache_source_is_contiguous"
        ],
        "cache_source_storage_is_independent": pde_cache_source_identity[
            "storage_independent_from_official_pool"
        ],
        "cache_source_values_bitwise_equal": pde_cache_source_identity[
            "values_bitwise_equal_to_official_pool"
        ],
        "cache_source_probe_matches_official_pool": pde_cache_source_identity[
            "pool_probe_sha256"
        ]
        == pde_pool_identity["pool_probe_sha256"],
        "official_sampler_retains_original_pool_object": pde_cache_source_identity[
            "official_sampler_retains_original_pool_object"
        ],
    }
    if not all(pde_cache_source_checks.values()):
        raise RuntimeError(
            "explicit-jet contiguous cache source contract failed: "
            + json.dumps(pde_cache_source_checks, sort_keys=True)
        )
    capture_indices = build_independent_capture_indices(len(pde_pool), device)
    torch.cuda.synchronize(device)
    explicit_jet_backend_constructor_started = time.perf_counter()
    explicit_jet_training_core = ExplicitJetTrainingBackend(
        model=model,
        optimizer=optimizer,
        full_pool=pde_cache_source,
        source_pool_identity=pde_cache_source_identity,
        precision_policy=matmul_precision_policy,
        component_weights=loss_balance.pde_weights,
        capture_indices=capture_indices,
        cache_chunk_size=explicit_jet_backend.CACHE_CHUNK_SIZE,
        rebuilt_after_resume=resume_state is not None,
    )
    torch.cuda.synchronize(device)
    explicit_jet_backend_constructor_wall_seconds = (
        time.perf_counter() - explicit_jet_backend_constructor_started
    )
    setup_optimizer_after = optimizer.state_dict()
    setup_sampler_after = sampler_state_dict(sampler)
    setup_rng_after = rng_state_dict(boundary_generator)
    setup_state_after = {
        "model_sha256": model_state_sha256(model),
        "optimizer_sha256": explicit_jet_backend.optimizer_state_sha256(optimizer),
        "sampler_sha256": nested_state_sha256(setup_sampler_after),
        "rng_sha256": nested_state_sha256(setup_rng_after),
    }
    explicit_jet_setup_checks = {
        **pde_cache_source_checks,
        "model_state_bitwise_unchanged": setup_state_before["model_sha256"]
        == setup_state_after["model_sha256"],
        "optimizer_state_bitwise_unchanged": nested_state_equal(
            setup_optimizer_before, setup_optimizer_after
        ),
        "official_sampler_state_bitwise_unchanged": nested_state_equal(
            setup_sampler_before, setup_sampler_after
        ),
        "all_rng_state_bitwise_unchanged": nested_state_equal(
            setup_rng_before, setup_rng_after
        ),
        "capture_indices_did_not_advance_sampler": setup_state_before["sampler_sha256"]
        == setup_state_after["sampler_sha256"],
    }
    if not all(explicit_jet_setup_checks.values()):
        raise RuntimeError(
            "explicit-jet cold cache/Graph construction changed formal training state: "
            + json.dumps(explicit_jet_setup_checks, sort_keys=True)
        )
    explicit_jet_setup_audit = {
        "passed": True,
        "checks": explicit_jet_setup_checks,
        "audit_boundary": "before pde_pool.contiguous() through A3 backend construction",
        "pde_cache_source_identity": pde_cache_source_identity,
        "pde_cache_source_copy_seconds": pde_cache_source_copy_seconds,
        "state_before_sha256": setup_state_before,
        "state_after_sha256": setup_state_after,
        "rng_scope": [
            "python",
            "numpy",
            "torch_cpu",
            "all_torch_cuda_devices",
            "boundary_generator",
        ],
        "sampler_scope": "official CompleteUniformSampler full state",
        "backend_checkpoint_policy": explicit_jet_training_core.checkpoint_policy(),
        "full_backend_constructor_wall_seconds": explicit_jet_backend_constructor_wall_seconds,
    }
    return BackendSetup(
        explicit_jet_training_core,
        pde_cache_source_identity,
        explicit_jet_setup_audit,
        pde_cache_source_copy_seconds,
        explicit_jet_backend_constructor_wall_seconds,
    )


@dataclass(frozen=True, slots=True)
class StepResult:
    components: dict
    weighted_components: dict
    loss: torch.Tensor
    should_evaluate: bool
    batch_size: int
    learning_rate: float


def optimizer_step(
    model,
    optimizer,
    sampler_adapter,
    explicit_jet_training_core,
    args,
    global_step: int,
) -> StepResult:
    """Run one sampled PDE-gradient replay and Adam update."""
    model.train()
    lr = cosine_lr(
        global_step - 1, args.scheduler_horizon_steps, args.lr_start, args.lr_end
    )
    for group in optimizer.param_groups:
        group["lr"] = lr
    if sampler_adapter is None:
        raise RuntimeError("explicit-jet uniform index adapter was not initialized")
    batch = sampler_adapter.next_indices(args.batch_size)
    component_tensors, weighted_component_tensors, pde_total = (
        explicit_jet_training_core.replay(batch)
    )
    should_evaluate = (
        global_step % args.eval_frequency_steps == 0 or global_step == args.run_steps
    )
    total_loss = pde_total
    check_finite_now = global_step <= 3 or should_evaluate
    if check_finite_now and (not torch.isfinite(total_loss)):
        raise FloatingPointError(f"Non-finite loss at step {global_step}")
    if global_step <= 3 or should_evaluate:
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and (
                not torch.isfinite(parameter.grad).all()
            ):
                raise FloatingPointError(
                    f"Non-finite gradient for {name} at step {global_step}"
                )
    optimizer.step()
    return StepResult(
        component_tensors,
        weighted_component_tensors,
        total_loss,
        should_evaluate,
        len(batch),
        lr,
    )


def loss_logging_vector(step: StepResult, zeros: torch.Tensor) -> torch.Tensor:
    """Pack PDE, weighted PDE, monitor-only zeros, and total loss in fixed order."""
    components = tuple(step.components[name] for name in LOSS_COMPONENTS)
    weighted = tuple(step.weighted_components[name] for name in LOSS_COMPONENTS)
    return torch.cat(
        (torch.stack(components + weighted), zeros, step.loss.detach().reshape(1))
    ).detach()


def write_training_report(
    report: dict, output_root: Path, trainer_process_started: float
) -> None:
    """Write the completed run report and emit its concise location/status record."""
    report_path = output_root / "pinn_report.json"
    report_write_started = time.perf_counter()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_write_seconds = time.perf_counter() - report_write_started
    trainer_process_wall_seconds = time.perf_counter() - trainer_process_started
    print(
        json.dumps(
            {
                "event": "completed",
                "report": str(report_path),
                "stop_reason": report["stop_reason"],
                "completed_steps": report["completed_steps"],
                "report_write_seconds": report_write_seconds,
                "trainer_process_wall_seconds": trainer_process_wall_seconds,
            }
        ),
        flush=True,
    )


def main() -> int:
    trainer_process_started = time.perf_counter()
    runtime_source_identity = release_source_identity()
    args = parse_training_arguments()
    spatial_input_only = True
    derivative_mode = "batched_second_vjp"
    validation_frequency_steps = args.eval_frequency_steps
    boundary_frequency_steps = args.eval_frequency_steps
    case_root, case_id, seed = validate_training_arguments(args)
    rff_spec = load_spec(args.rff_spec_file)
    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    input_model_setup_started = time.perf_counter()
    scales = build_physics_scales(case_root, args.scaling_mode, case_id=case_id)
    loss_balance = build_loss_balance(
        scales, args.loss_weighting, continuity_weight=args.continuity_weight
    )
    regions = GeometryRegions.from_scales(scales)
    data_path = case_root / "train_1e-4.xlsx"
    jacobi_path = args.jacobinet_checkpoint.resolve()
    if not jacobi_path.is_file():
        raise FileNotFoundError(jacobi_path)
    input_paths = [jacobi_path]
    if args.resume_checkpoint is not None:
        input_paths.extend((args.resume_checkpoint, args.resume_history))
    output_root = prepare_training_output(
        args.output_root, case_root, input_paths=input_paths
    )
    final_checkpoint = output_root / "best_model_weights_stenosis.pth"
    physics_checkpoint = output_root / "best_physics_checkpoint.pth"
    terminal_checkpoint = output_root / "terminal_step_checkpoint.pth"
    set_seed(seed)
    matmul_precision_policy = apply_matmul_precision_policy("ieee")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    points = load_training_points(data_path, scales, device)
    jacobi = JacobiNet().to(dtype=TORCH_TRAINING_DTYPE)
    load_jacobinet_checkpoint(jacobi_path, jacobi, scales, map_location="cpu")
    for parameter in jacobi.parameters():
        parameter.requires_grad = False
    jacobi.eval()
    rff_rng_state = torch.get_rng_state().clone()
    base_B_xyz, base_B_rs = base_rff_matrices_from_state(
        legacy_cpu_rng_state=rff_rng_state,
        train_seed=seed,
        rff_seed=args.rff_seed,
    )
    consume_reference_rff_rng_draws()
    jacobi = jacobi.to(device=device)
    model = NetPINN(
        jacobi,
        scales,
        rff_spec=rff_spec,
        base_B_xyz=base_B_xyz,
        base_B_rs=base_B_rs,
        train_seed=seed,
        rff_seed=args.rff_seed,
    ).to(device=device, dtype=TORCH_TRAINING_DTYPE)
    assert_model_training_dtype(model)
    initial_model_state_sha256 = model_state_sha256(model)
    trainable_parameter_count = sum(
        (
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )
    rff_xyz_sha256 = tensor_sha256(model.rff_xyz.B)
    rff_rs_sha256 = tensor_sha256(model.rff_rs.B)
    trainable_parameters = trainable_head_parameters(model)
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr_start)
    pde_pool = points.internal
    pde_pool_probe_sha256 = explicit_jet_backend.tensor_probe_sha256(pde_pool)
    pde_pool_identity = {
        "case_id": case_id,
        "pool_size": int(len(pde_pool)),
        "pool_probe_sha256": pde_pool_probe_sha256,
        "is_contiguous": bool(pde_pool.is_contiguous()),
    }
    pde_cache_source_definition = {
        "construction": "pde_pool.contiguous()",
        "owner": "A3 feature-cache construction only",
        "official_sampler_uses_original_pde_pool": True,
        "source_pool_is_contiguous": bool(pde_pool.is_contiguous()),
        "cache_source_required_contiguous": True,
        "expected_pool_size": int(len(pde_pool)),
        "expected_pool_probe_sha256": pde_pool_probe_sha256,
    }
    sampler_seed = seed + 211
    sampler = CompleteUniformSampler(pde_pool, sampler_seed)
    sampler_adapter = CompleteUniformIndexAdapter(sampler)
    sampling_definition = {
        "mode": "complete_uniform_without_dropped_remainder",
        "seed": sampler_seed,
        "initial_permutation_sha256": tensor_sha256(sampler.permutation),
        "training_payload": "CUDA long indices into immutable A3 feature cache",
        "adapter": type(sampler_adapter).__name__,
        "official_sampler_state_schema_preserved": True,
    }
    validation = deterministic_validation_points(
        points.internal, count=args.validation_count, seed=seed + 307
    )
    gradient_points = deterministic_validation_points(
        validation,
        count=min(args.gradient_point_count, len(validation)),
        seed=seed + 401,
    )
    boundary_generator = torch.Generator(device=device)
    boundary_generator.manual_seed(seed + 503)
    effective_bc_weight = effective_boundary_weight("monitor_only", 0.0)
    configuration = {
        "method_revision": METHOD_REVISION,
        "training_protocol_id": TRAINING_PROTOCOL_ID,
        "configuration_identity_revision": LOSS_ONLY_CONFIGURATION_IDENTITY_REVISION,
        "variant": args.variant,
        "runtime_mode": args.runtime_mode,
        "runtime_definition": {
            "logging": "device_vector_single_transfer_per_evaluation",
            "monitor_only_fast_path": True,
            "validation_parameter_graph": False,
            "loss_stop_evaluation_frequency_steps": args.eval_frequency_steps,
            "validation_frequency_steps": validation_frequency_steps,
            "boundary_frequency_steps": boundary_frequency_steps,
            "gradient_frequency_steps": None,
            "deferred_best_checkpoint_write": True,
            "terminal_final_evaluation_reuse": False,
            "pde_autodiff_input": (
                "cached_explicit_xyz_r_s_feature_jet_no_coordinate_autodiff"
                if spatial_input_only
                else "full_loaded_point_tensor"
            ),
            "pde_derivative_mode": derivative_mode,
            "finite_check_frequency_steps": args.eval_frequency_steps,
            "cuda_graph": {
                "enabled": True,
                "api": "torch.cuda.CUDAGraph_raw_A3_explicit_jet",
                "warmup_stream": "dedicated_side_stream",
                "warmup_steps": CUDA_GRAPH_WARMUP_STEPS,
                "static_batch_copy": False,
                "dynamic_index_gather": True,
                "capture_boundary": [
                    "seven_cached_feature_field_gathers",
                    "explicit_linear_silu_trial_jet_propagation",
                    "four_PDE_residuals_and_parameter_backward",
                ],
                "eager_boundary": [
                    "closed_form_cosine_scheduler",
                    "complete_uniform_index_sampler",
                    "adam_step",
                    "diagnostics_history_checkpoint_report",
                ],
                "optimizer_captured": False,
                "checkpointed": False,
                "recreated_after_resume": True,
                "dual_batch_gate_requested": False,
                "coordinate_autodiff_active_during_training": False,
                "feature_cache_fields": list(
                    explicit_jet_backend.EXPLICIT_CACHE_FIELDS
                ),
                "feature_cache_chunk_size": explicit_jet_backend.CACHE_CHUNK_SIZE,
                "feature_cache_checkpointed": False,
            },
        },
        "runtime_source_identity": runtime_source_identity,
        "a3_backend_revision": explicit_jet_backend.REVISION,
        "a3_checkpoint_compatibility": "A3_explicit_only",
        "a3_checkpoint_metadata": explicit_jet_backend.production_checkpoint_metadata(),
        "optimizer_parameter_scope": "18_trainable_C1_head_tensors_only",
        "optimizer_trainable_tensor_count": len(trainable_parameters),
        "pde_pool_identity": pde_pool_identity,
        "pde_cache_source_definition": pde_cache_source_definition,
        "dtype": TRAINING_DTYPE_NAME,
        "autodiff_dtype": TRAINING_DTYPE_NAME,
        "evaluation_dtype": EVALUATION_DTYPE_NAME,
        "precision_policy": PRECISION_POLICY_NAME,
        "scaling_mode": args.scaling_mode,
        "loss_balance": loss_balance.to_dict(),
        "continuity_weight": args.continuity_weight,
        "rff_spec_id": rff_spec.spec_id,
        "rff_spec_sha256": rff_spec.sha256(),
        "rff_spec": rff_spec.to_dict(),
        "train_seed": seed,
        "rff_seed": args.rff_seed,
        "model_config": model.model_config,
        "trainable_parameter_count": trainable_parameter_count,
        "rff_xyz_sha256": rff_xyz_sha256,
        "rff_rs_sha256": rff_rs_sha256,
        "true_section_flux_monitor": {
            "mode": "forward_only",
            "included_in_loss": False,
            "stages": ["initial", "terminal", "final_selected"],
            "components": [
                "total_predicted_velocity",
            ],
        },
        "max_steps": args.run_steps,
        "run_steps": args.run_steps,
        "scheduler_horizon_steps": args.scheduler_horizon_steps,
        "batch_size": args.batch_size,
        "eval_frequency_steps": args.eval_frequency_steps,
        "label_diagnostics_disabled": True,
        "loss_stop_rule": loss_stop_rule(),
        "pde_point_mode": "internal",
        "sampler": "uniform",
        "boundary_mode": "monitor_only",
        "boundary_weight": 0.0,
        "effective_boundary_weight": effective_bc_weight,
        "fluid_bc_loss_active": False,
        "boundary_batch_size": args.boundary_batch_size,
        "validation_count": len(validation),
        "validation_batch_size": args.validation_batch_size,
        "validation_sha256": tensor_sha256(validation),
        "gradient_frequency_evals": args.gradient_frequency_evals,
        "gradient_point_count": len(gradient_points),
        "timing_warmup_steps": args.timing_warmup_steps,
        "learning_rate": {
            "start": args.lr_start,
            "end": args.lr_end,
            "basis": "optimizer_step",
        },
        "ema_beta": args.ema_beta,
        "sampling_definition": sampling_definition,
        "region_definition": regions.to_dict(),
        "physics_scales": scales.to_dict(),
        "jacobinet_sha256": sha256(jacobi_path),
        "initial_model_state_sha256": initial_model_state_sha256,
    }
    if matmul_precision_policy is not None:
        configuration["matmul_precision_policy"] = dict(matmul_precision_policy)
    configuration_sha256 = resume_configuration_sha256(configuration)
    torch.cuda.synchronize(device)
    input_model_setup_seconds = time.perf_counter() - input_model_setup_started
    model.checkpoint_method_revision = METHOD_REVISION
    model.checkpoint_variant = args.variant
    model.checkpoint_configuration_sha256 = configuration_sha256
    resume_state: dict | None = None
    resume_payload: dict | None = None
    resume_checkpoint_load_seconds = 0.0
    if args.resume_checkpoint is not None:
        torch.cuda.synchronize(device)
        resume_checkpoint_load_started = time.perf_counter()
        resume_payload = load_pinn_checkpoint(
            args.resume_checkpoint, model, scales, map_location=device
        )
        candidate = resume_payload.get("training_state")
        if not isinstance(candidate, Mapping):
            raise TypeError(
                "explicit-jet resume checkpoint lacks a full training_state"
            )
        if candidate.get("configuration_sha256") != configuration_sha256:
            raise ValueError(
                "explicit-jet-only resume rejected a checkpoint from another backend/configuration"
            )
        validate_full_resume_state(
            candidate,
            model=model,
            sampler=sampler,
            expected_configuration_sha256=configuration_sha256,
            expected_rff_spec_sha256=rff_spec.sha256(),
            expected_train_seed=seed,
            expected_rff_seed=args.rff_seed,
            expected_batch_size=args.batch_size,
            expected_scheduler_horizon=args.scheduler_horizon_steps,
            expected_lr_start=args.lr_start,
            expected_lr_end=args.lr_end,
        )
        resume_state = candidate
        if resume_state["stop_reason"] == LOSS_CONVERGED_STATUS:
            raise ValueError("Loss-converged checkpoint must not be resumed")
        if int(resume_state["global_step"]) >= args.run_steps:
            raise ValueError(
                "Resume checkpoint has already reached the requested run_steps"
            )
        serialized_groups = resume_state["optimizer_state_dict"]["param_groups"]
        if sum((len(group.get("params", ())) for group in serialized_groups)) != 18:
            raise ValueError(
                "explicit-jet resume Adam must contain exactly 18 parameter tensors"
            )
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        load_sampler_state(sampler, resume_state["sampler_state"])
        torch.cuda.synchronize(device)
        resume_checkpoint_load_seconds = (
            time.perf_counter() - resume_checkpoint_load_started
        )
    print(
        json.dumps(
            {
                "event": "start",
                "device": torch.cuda.get_device_name(device),
                "case_id": case_id,
                "configuration_sha256": configuration_sha256,
                "point_counts": points.counts(),
                "physics_scales": scales.to_dict(),
                "configuration": configuration,
                "resume_checkpoint": (
                    str(args.resume_checkpoint.resolve())
                    if args.resume_checkpoint is not None
                    else None
                ),
            }
        ),
        flush=True,
    )
    torch.cuda.synchronize(device)
    initial_diagnostics_started = time.perf_counter()
    initial_true_section_flux = true_section_flux_monitor(
        model,
        scales.geometry_path,
        length_scale_m=scales.length_scale_m,
        inlet_peak_velocity_nondim=scales.inlet_peak_velocity_nondim,
    )
    initial_validation = average_pde_losses(
        model,
        validation,
        args.validation_batch_size,
        build_parameter_graph=False,
        spatial_input_only=spatial_input_only,
        derivative_mode=derivative_mode,
    )
    initial_validation_weighted = weighted_component_values(
        initial_validation, loss_balance.pde_weights
    )
    initial_boundary, initial_boundary_physical, initial_jacobinet_boundary = (
        boundary_monitor(
            model,
            points.inlet,
            points.outlet,
            points.wall,
            velocity_scale_m_s=scales.velocity_scale_m_s,
            pressure_scale_pa=scales.pressure_scale_pa,
        )
    )
    initial_boundary_weighted = weighted_component_values(
        initial_boundary, loss_balance.boundary_weights
    )
    initial_validation_score = validation_objective(
        initial_validation_weighted,
        initial_boundary_weighted,
        boundary_mode="monitor_only",
        boundary_weight=0.0,
    )
    print(
        json.dumps(
            {
                "event": "initial",
                **{f"raw_{name}": value for name, value in initial_validation.items()},
                **{
                    f"weighted_{name}": value
                    for name, value in initial_validation_weighted.items()
                },
                **{
                    f"raw_boundary_{name}": value
                    for name, value in initial_boundary.items()
                },
                **{
                    f"weighted_boundary_{name}": value
                    for name, value in initial_boundary_weighted.items()
                },
                **{
                    f"boundary_{name}": value
                    for name, value in initial_boundary_physical.items()
                },
                **{
                    f"jacobinet_{name}": value
                    for name, value in initial_jacobinet_boundary.items()
                },
                "validation_score": initial_validation_score,
                "legacy_equivalent_validation_score": initial_validation_score
                * loss_balance.legacy_equivalent_multiplier,
            }
        ),
        flush=True,
    )
    torch.cuda.synchronize(device)
    initial_diagnostics_seconds = time.perf_counter() - initial_diagnostics_started
    accumulator_names = ACCUMULATOR_NAMES
    resume_state_restore_started = time.perf_counter()
    if resume_state is None:
        rows: list[dict] = []
        accumulators = {name: 0.0 for name in accumulator_names}
        ema: dict[str, float] = {}
        accumulated_steps = 0
        pde_points_processed = 0
        boundary_points_processed = {"inlet": 0, "outlet": 0, "wall": 0}
        best_physics_score = float("inf")
        best_physics_step = None
        best_physics_snapshot = None
        first_loss_qualified_step = None
        first_loss_qualified_seconds = None
        loss_converged_step = None
        loss_converged_seconds = None
        consecutive_loss_stop_evals = 0
        terminal_loss_snapshot: dict[str, Any] = {}
        evaluation_count = 0
        completed_steps = 0
        training_seconds_before = 0.0
    else:
        prior = pd.read_csv(args.resume_history, float_precision="round_trip")
        rows = prior.to_dict(orient="records")
        completed_steps = int(resume_state["global_step"])
        if not rows or int(rows[-1]["global_step"]) != completed_steps:
            raise ValueError(
                "Resume history does not end at the checkpoint global_step"
            )
        history_steps = [int(row["global_step"]) for row in rows]
        if history_steps != sorted(set(history_steps)):
            raise ValueError(
                "Resume history global_step values must be unique and increasing"
            )
        if len(rows) != int(resume_state["evaluation_count"]):
            raise ValueError("Resume history row count does not match evaluation_count")
        required_history_fields = {
            "loss_stop_eligible",
            "loss_stop_qualified",
            "consecutive_loss_stop_evals",
        }
        if not required_history_fields <= set(rows[-1]):
            raise ValueError("Resume history lacks loss-stop state columns")
        if int(rows[-1]["consecutive_loss_stop_evals"]) != int(
            resume_state["consecutive_loss_stop_evals"]
        ):
            raise ValueError("Resume history/checkpoint loss counter mismatch")
        history_decisions: dict[str, bool] = {}
        for name in ("loss_stop_eligible", "loss_stop_qualified"):
            value = rows[-1][name]
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"Resume history {name} must be boolean")
            history_decisions[name] = bool(value)
        if history_decisions["loss_stop_eligible"] != bool(
            resume_state["terminal_loss_snapshot"]["eligible"]
        ) or history_decisions["loss_stop_qualified"] != bool(
            resume_state["terminal_loss_snapshot"]["qualified"]
        ):
            raise ValueError("Resume history/checkpoint loss decision mismatch")
        accumulators = {
            name: float(resume_state["accumulators"].get(name, 0.0))
            for name in accumulator_names
        }
        ema = {name: float(value) for name, value in resume_state["ema"].items()}
        accumulated_steps = int(resume_state["accumulated_steps"])
        pde_points_processed = int(resume_state["pde_points_processed"])
        boundary_points_processed = {
            name: int(value)
            for name, value in resume_state["boundary_points_processed"].items()
        }
        best_physics_score = float(resume_state["best_physics_score"])
        best_physics_step = resume_state["best_physics_step"]
        best_physics_snapshot = normalize_best_snapshot(
            resume_state["best_physics_snapshot"]
        )
        first_loss_qualified_step = resume_state["first_loss_qualified_step"]
        first_loss_qualified_seconds = resume_state["first_loss_qualified_seconds"]
        loss_converged_step = resume_state["loss_converged_step"]
        loss_converged_seconds = resume_state["loss_converged_seconds"]
        consecutive_loss_stop_evals = int(resume_state["consecutive_loss_stop_evals"])
        terminal_loss_snapshot = dict(resume_state["terminal_loss_snapshot"])
        evaluation_count = int(resume_state["evaluation_count"])
        training_seconds_before = float(resume_state["training_seconds"])
        restore_rng_state(resume_state["rng_state"], boundary_generator)
    resume_state_restore_seconds = time.perf_counter() - resume_state_restore_started
    setup = prepare_training_backend(
        model=model,
        optimizer=optimizer,
        sampler=sampler,
        boundary_generator=boundary_generator,
        device=device,
        pde_pool=pde_pool,
        pde_pool_identity=pde_pool_identity,
        matmul_precision_policy=matmul_precision_policy,
        loss_balance=loss_balance,
        resume_state=resume_state,
    )
    explicit_jet_training_core = setup.core
    pde_cache_source_identity = setup.source_identity
    explicit_jet_setup_audit = setup.audit
    pde_cache_source_copy_seconds = setup.copy_seconds
    explicit_jet_backend_constructor_wall_seconds = setup.constructor_seconds
    cuda_graph_setup_seconds = setup.core.graph_setup_seconds
    ema_initialized = bool(ema)
    monitor_logging_zeros = torch.zeros(6, dtype=points.internal.dtype, device=device)
    accumulator_vector = torch.tensor(
        [accumulators[name] for name in accumulator_names],
        dtype=torch.float64,
        device=device,
    )
    ema_vector = torch.tensor(
        [ema.get(name, 0.0) for name in accumulator_names],
        dtype=torch.float64,
        device=device,
    )
    validation_components = dict(initial_validation)
    validation_weighted = dict(initial_validation_weighted)
    validation_score = float(initial_validation_score)
    legacy_equivalent_validation_score = (
        validation_score * loss_balance.legacy_equivalent_multiplier
    )
    boundary_values = dict(initial_boundary)
    boundary_weighted = dict(initial_boundary_weighted)
    boundary_physical = dict(initial_boundary_physical)
    jacobinet_boundary = dict(initial_jacobinet_boundary)
    start = time.perf_counter()
    timing_window_started_at = start if args.timing_warmup_steps == 0 else None
    timing_window_seconds: float | None = None
    timing_window_closed_at_step: int | None = None
    scheduled_evaluation_seconds = 0.0

    def elapsed_training_seconds() -> float:
        return training_seconds_before + time.perf_counter() - start

    for global_step in range(completed_steps + 1, args.run_steps + 1):
        step_result = optimizer_step(
            model,
            optimizer,
            sampler_adapter,
            explicit_jet_training_core,
            args,
            global_step,
        )
        should_evaluate = step_result.should_evaluate
        lr = step_result.learning_rate
        completed_steps = global_step
        pde_points_processed += step_result.batch_size
        accumulated_steps += 1
        if monitor_logging_zeros is None:
            raise RuntimeError("Training logging buffers are not initialized")
        current_vector = loss_logging_vector(step_result, monitor_logging_zeros)
        if accumulator_vector is None or ema_vector is None:
            raise RuntimeError("Training vector buffers are not initialized")
        current_vector = current_vector.to(dtype=torch.float64)
        accumulator_vector.add_(current_vector)
        if ema_initialized:
            ema_vector.mul_(args.ema_beta).add_(
                current_vector, alpha=1.0 - args.ema_beta
            )
        else:
            ema_vector.copy_(current_vector)
            ema_initialized = True
        if not should_evaluate:
            if global_step == args.timing_warmup_steps:
                torch.cuda.synchronize(device)
                timing_window_started_at = time.perf_counter()
            continue
        torch.cuda.synchronize(device)
        scheduled_evaluation_started = time.perf_counter()
        evaluation_count += 1
        validation_fresh = (
            global_step % validation_frequency_steps == 0
            or global_step == args.run_steps
        )
        boundary_fresh = (
            global_step % boundary_frequency_steps == 0 or global_step == args.run_steps
        )
        if validation_fresh:
            validation_components = average_pde_losses(
                model,
                validation,
                args.validation_batch_size,
                build_parameter_graph=False,
                spatial_input_only=spatial_input_only,
                derivative_mode=derivative_mode,
            )
            validation_weighted = weighted_component_values(
                validation_components, loss_balance.pde_weights
            )
        if boundary_fresh:
            boundary_values, boundary_physical, jacobinet_boundary = boundary_monitor(
                model,
                points.inlet,
                points.outlet,
                points.wall,
                velocity_scale_m_s=scales.velocity_scale_m_s,
                pressure_scale_pa=scales.pressure_scale_pa,
            )
            boundary_weighted = weighted_component_values(
                boundary_values, loss_balance.boundary_weights
            )
        if validation_fresh:
            validation_score = validation_objective(
                validation_weighted,
                boundary_weighted,
                boundary_mode="monitor_only",
                boundary_weight=0.0,
            )
            legacy_equivalent_validation_score = (
                validation_score * loss_balance.legacy_equivalent_multiplier
            )
        gradient_values = {}
        gradient_due = should_measure_gradient_norms(
            evaluation_count, args.gradient_frequency_evals
        )
        if gradient_due:
            gradient_values = gradient_norms(
                model,
                gradient_points,
                loss_balance.pde_weights,
                spatial_input_only=spatial_input_only,
                derivative_mode=derivative_mode,
            )
        packed_logging = (
            torch.cat((accumulator_vector, ema_vector)).detach().cpu().tolist()
        )
        count = len(accumulator_names)
        accumulators = dict(zip(accumulator_names, packed_logging[:count]))
        ema = dict(zip(accumulator_names, packed_logging[count:]))
        training_means = {
            f"train_mean_{name}": value / max(accumulated_steps, 1)
            for name, value in accumulators.items()
        }
        terminal_loss_snapshot = loss_stop_snapshot(
            global_step=global_step,
            validation_score=validation_score,
            validation_weighted=validation_weighted,
            ema=ema,
            training_means=training_means,
            min_step=args.loss_stop_min_step,
            total_threshold=args.loss_total_threshold,
            component_threshold=args.loss_component_threshold,
        )
        consecutive_loss_stop_evals, _ = advance_loss_stop_counter(
            consecutive_loss_stop_evals, terminal_loss_snapshot
        )
        if terminal_loss_snapshot["qualified"] and first_loss_qualified_step is None:
            first_loss_qualified_step = global_step
            first_loss_qualified_seconds = elapsed_training_seconds()
        row = {
            "global_step": global_step,
            "optimizer_steps": global_step,
            "equivalent_epochs": global_step * args.batch_size / max(len(pde_pool), 1),
            "pde_points_processed": pde_points_processed,
            "learning_rate": lr,
            "elapsed_seconds": elapsed_training_seconds(),
            "validation_score": validation_score,
            "legacy_equivalent_validation_score": legacy_equivalent_validation_score,
            "validation_fresh": validation_fresh,
            "boundary_fresh": boundary_fresh,
            "gradient_fresh": gradient_due,
            "loss_stop_eligible": terminal_loss_snapshot["eligible"],
            "loss_stop_qualified": terminal_loss_snapshot["qualified"],
            "consecutive_loss_stop_evals": consecutive_loss_stop_evals,
            **training_means,
            **{f"ema_{name}": value for name, value in ema.items()},
            **{
                f"validation_{name}": value
                for name, value in validation_components.items()
            },
            **{
                f"weighted_validation_{name}": value
                for name, value in validation_weighted.items()
            },
            **{f"boundary_{name}": value for name, value in boundary_values.items()},
            **{
                f"weighted_boundary_{name}": value
                for name, value in boundary_weighted.items()
            },
            **{f"boundary_{name}": value for name, value in boundary_physical.items()},
            **{
                f"jacobinet_{name}": value for name, value in jacobinet_boundary.items()
            },
            **gradient_values,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if accumulator_vector is None:
            raise RuntimeError("Training accumulator is not initialized")
        accumulator_vector.zero_()
        accumulators = {name: 0.0 for name in accumulators}
        accumulated_steps = 0
        checkpoint_state = {
            "global_step": global_step,
            "validation_score": validation_score,
            "legacy_equivalent_validation_score": legacy_equivalent_validation_score,
            "label_diagnostics_disabled": True,
            "loss_stop_snapshot": terminal_loss_snapshot,
            "configuration_sha256": configuration_sha256,
        }
        if validation_fresh and validation_score < best_physics_score:
            best_physics_score = validation_score
            best_physics_step = global_step
            best_physics_snapshot = capture_best_snapshot(model, checkpoint_state)
        torch.cuda.synchronize(device)
        scheduled_evaluation_seconds += (
            time.perf_counter() - scheduled_evaluation_started
        )
        if global_step == args.timing_warmup_steps:
            torch.cuda.synchronize(device)
            timing_window_started_at = time.perf_counter()
    torch.cuda.synchronize(device)
    if timing_window_seconds is None and timing_window_started_at is not None:
        timing_window_seconds = time.perf_counter() - timing_window_started_at
        timing_window_closed_at_step = completed_steps
    training_seconds = elapsed_training_seconds()
    current_process_training_seconds = training_seconds - training_seconds_before
    if explicit_jet_training_core.replay_count != completed_steps - (
        int(resume_state["global_step"]) if resume_state is not None else 0
    ):
        raise RuntimeError(
            "explicit-jet replay count differs from current-process optimizer steps"
        )
    expected_points = completed_steps * args.batch_size
    if pde_points_processed != expected_points:
        raise RuntimeError("PDE point count does not match completed optimizer steps")
    stop_reason = loss_only_terminal_status(
        converged=False, completed_steps=completed_steps
    )
    if any(boundary_points_processed.values()):
        raise RuntimeError(
            "monitor_only run processed boundary points in the training objective"
        )
    torch.cuda.synchronize(device)
    terminal_diagnostics_started = time.perf_counter()
    terminal_true_section_flux = true_section_flux_monitor(
        model,
        scales.geometry_path,
        length_scale_m=scales.length_scale_m,
        inlet_peak_velocity_nondim=scales.inlet_peak_velocity_nondim,
    )
    terminal_boundary, terminal_boundary_physical, terminal_jacobinet_boundary = (
        boundary_monitor(
            model,
            points.inlet,
            points.outlet,
            points.wall,
            velocity_scale_m_s=scales.velocity_scale_m_s,
            pressure_scale_pa=scales.pressure_scale_pa,
        )
    )
    torch.cuda.synchronize(device)
    terminal_diagnostics_seconds = time.perf_counter() - terminal_diagnostics_started
    checkpoint_selection_io_started = time.perf_counter()
    if best_physics_step is None or best_physics_snapshot is None:
        raise RuntimeError(
            "A terminal resume checkpoint requires a best physics snapshot"
        )
    terminal_resume_state = {
        "full_resume_state": True,
        "resume_state_schema_version": LOSS_ONLY_RESUME_STATE_SCHEMA_VERSION,
        "configuration_identity_revision": LOSS_ONLY_CONFIGURATION_IDENTITY_REVISION,
        "global_step": completed_steps,
        "terminal_loss_snapshot": terminal_loss_snapshot,
        "label_diagnostics_disabled": True,
        "loss_stop_rule": loss_stop_rule(),
        "stop_reason": stop_reason,
        "configuration_sha256": configuration_sha256,
        "terminal_step_checkpoint": True,
        "train_seed": seed,
        "rff_seed": args.rff_seed,
        "rff_spec_sha256": rff_spec.sha256(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state": {
            "kind": "cosine_lr_closed_form_v1",
            "horizon_steps": args.scheduler_horizon_steps,
            "last_step": completed_steps,
            "last_lr": float(optimizer.param_groups[0]["lr"]),
        },
        "sampler_state": sampler_state_dict(sampler),
        "rng_state": rng_state_dict(boundary_generator),
        "accumulators": accumulators,
        "ema": ema,
        "accumulated_steps": accumulated_steps,
        "pde_points_processed": pde_points_processed,
        "boundary_points_processed": boundary_points_processed,
        "best_physics_score": best_physics_score,
        "best_physics_step": best_physics_step,
        "best_physics_snapshot": best_physics_snapshot,
        "first_loss_qualified_step": first_loss_qualified_step,
        "first_loss_qualified_seconds": first_loss_qualified_seconds,
        "loss_converged_step": loss_converged_step,
        "loss_converged_seconds": loss_converged_seconds,
        "consecutive_loss_stop_evals": consecutive_loss_stop_evals,
        "evaluation_count": evaluation_count,
        "training_seconds": training_seconds,
        "resumed_from": (
            str(args.resume_checkpoint.resolve())
            if args.resume_checkpoint is not None
            else None
        ),
    }
    validate_full_resume_state(
        terminal_resume_state,
        model=model,
        sampler=sampler,
        expected_configuration_sha256=configuration_sha256,
        expected_rff_spec_sha256=rff_spec.sha256(),
        expected_train_seed=seed,
        expected_rff_seed=args.rff_seed,
        expected_batch_size=args.batch_size,
        expected_scheduler_horizon=args.scheduler_horizon_steps,
        expected_lr_start=args.lr_start,
        expected_lr_end=args.lr_end,
    )
    save_pinn_checkpoint(terminal_checkpoint, model, scales, terminal_resume_state)
    materialize_best_snapshot_checkpoint(
        physics_checkpoint, model, scales, best_physics_snapshot
    )
    if stop_reason == MAX_STEPS_STATUS:
        selected_path = physics_checkpoint
        selected_kind = "best_physics_validation_score_at_max_steps"
    else:
        selected_path = terminal_checkpoint
        selected_kind = "segment_terminal_for_exact_resume"
    selected_payload = load_pinn_checkpoint(
        selected_path, model, scales, map_location=device
    )
    if selected_payload.get("model_config") != model.model_config:
        raise ValueError(
            "Selected checkpoint model_config does not match the active RFF mode"
        )
    save_pinn_checkpoint(
        final_checkpoint,
        model,
        scales,
        {
            **selected_payload.get("training_state", {}),
            "label_diagnostics_disabled": True,
            "loss_stop_rule": loss_stop_rule(),
            "stop_reason": stop_reason,
            "run_terminal_step": completed_steps,
            "terminal_loss_snapshot": terminal_loss_snapshot,
            "selected_from": selected_path.name,
            "selected_kind": selected_kind,
        },
    )
    selected_checkpoint_step = selected_payload.get("training_state", {}).get(
        "global_step"
    )
    checkpoint_selection_io_seconds = (
        time.perf_counter() - checkpoint_selection_io_started
    )
    torch.cuda.synchronize(device)
    final_selected_diagnostics_started = time.perf_counter()
    if selected_checkpoint_step == completed_steps:
        final_true_section_flux = terminal_true_section_flux
        final_boundary = terminal_boundary
        final_boundary_physical = terminal_boundary_physical
        final_jacobinet_boundary = terminal_jacobinet_boundary
    else:
        final_true_section_flux = true_section_flux_monitor(
            model,
            scales.geometry_path,
            length_scale_m=scales.length_scale_m,
            inlet_peak_velocity_nondim=scales.inlet_peak_velocity_nondim,
        )
        final_boundary, final_boundary_physical, final_jacobinet_boundary = (
            boundary_monitor(
                model,
                points.inlet,
                points.outlet,
                points.wall,
                velocity_scale_m_s=scales.velocity_scale_m_s,
                pressure_scale_pa=scales.pressure_scale_pa,
            )
        )
    torch.cuda.synchronize(device)
    final_selected_diagnostics_seconds = (
        time.perf_counter() - final_selected_diagnostics_started
    )
    history_path = output_root / "training_history.csv"
    history_write_started = time.perf_counter()
    pd.DataFrame(rows).to_csv(history_path, index=False)
    history_write_seconds = time.perf_counter() - history_write_started
    nominal_steps_per_epoch = math.ceil(len(pde_pool) / args.batch_size)
    pde_pool_passes = pde_points_processed / len(pde_pool)
    sampling_source_draws = {"complete_uniform": pde_points_processed}
    report_assembly_started = time.perf_counter()
    report = {
        "status": "completed",
        "method_revision": METHOD_REVISION,
        "training_protocol_id": TRAINING_PROTOCOL_ID,
        "configuration_sha256": configuration_sha256,
        "case_id": case_id,
        "variant": args.variant,
        "dtype": TRAINING_DTYPE_NAME,
        "autodiff_dtype": TRAINING_DTYPE_NAME,
        "physics_scale_dtype": EVALUATION_DTYPE_NAME,
        "evaluation_dtype": EVALUATION_DTYPE_NAME,
        "precision_policy": PRECISION_POLICY_NAME,
        "device": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "seed": seed,
        "train_seed": seed,
        "rff_seed": args.rff_seed,
        "rff_spec_id": rff_spec.spec_id,
        "rff_spec_sha256": rff_spec.sha256(),
        "initial_model_state_sha256": initial_model_state_sha256,
        "model_config": model.model_config,
        "trainable_parameter_count": trainable_parameter_count,
        "rff_xyz_sha256": rff_xyz_sha256,
        "rff_rs_sha256": rff_rs_sha256,
        "pde_pool_identity": pde_pool_identity,
        "pde_cache_source_identity": pde_cache_source_identity,
        "max_steps": args.run_steps,
        "run_steps": args.run_steps,
        "scheduler_horizon_steps": args.scheduler_horizon_steps,
        "completed_steps": completed_steps,
        "optimizer_steps": completed_steps,
        "pde_points_processed": pde_points_processed,
        "boundary_points_processed": boundary_points_processed,
        "sampling_source_draws": sampling_source_draws,
        "pde_pool_passes": pde_pool_passes,
        "nominal_steps_per_epoch": nominal_steps_per_epoch,
        "equivalent_completed_epochs": pde_pool_passes,
        "stop_reason": stop_reason,
        "resumed": args.resume_checkpoint is not None,
        "resume_checkpoint": (
            str(args.resume_checkpoint.resolve())
            if args.resume_checkpoint is not None
            else None
        ),
        "resume_history": (
            str(args.resume_history.resolve())
            if args.resume_history is not None
            else None
        ),
        "resume_start_step": (
            int(resume_state["global_step"]) if resume_state is not None else 0
        ),
        "label_diagnostics_disabled": True,
        "loss_stop_rule": loss_stop_rule(),
        "terminal_loss_snapshot": terminal_loss_snapshot,
        "first_loss_qualified_step": first_loss_qualified_step,
        "first_loss_qualified_seconds": first_loss_qualified_seconds,
        "loss_converged_step": loss_converged_step,
        "loss_converged_seconds": loss_converged_seconds,
        "consecutive_loss_stop_evals": consecutive_loss_stop_evals,
        "best_physics_step": best_physics_step,
        "best_physics_validation_score": best_physics_score,
        "selected_checkpoint_kind": selected_kind,
        "selected_checkpoint_step": selected_checkpoint_step,
        "selected_checkpoint_source": str(selected_path),
        "training_seconds": training_seconds,
        "current_process_training_seconds": current_process_training_seconds,
        "seconds_per_optimizer_step": training_seconds / max(completed_steps, 1),
        "optimizer_steps_per_second": completed_steps / max(training_seconds, 1e-12),
        "timing_warmup_steps": args.timing_warmup_steps,
        "timing_window_steps": completed_steps - args.timing_warmup_steps,
        "timing_window_seconds": timing_window_seconds,
        "timing_window_closed_at_step": timing_window_closed_at_step,
        "timing_window_seconds_per_step": (
            None
            if timing_window_seconds is None
            else timing_window_seconds
            / max(completed_steps - args.timing_warmup_steps, 1)
        ),
        "cuda_graph_runtime": {
            **explicit_jet_training_core.runtime_definition(),
            "setup_seconds": cuda_graph_setup_seconds,
        },
        "explicit_runtime": {
            **explicit_jet_training_core.runtime_definition(),
            "setup_no_mutation_audit": explicit_jet_setup_audit,
            "checkpoint_policy": explicit_jet_training_core.checkpoint_policy(),
            "backend_source_identity": runtime_source_identity,
        },
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "initial_true_section_flux_monitor": initial_true_section_flux,
        "initial_validation_components": initial_validation,
        "initial_weighted_validation_components": initial_validation_weighted,
        "initial_raw_boundary_components": initial_boundary,
        "initial_weighted_boundary_components": initial_boundary_weighted,
        "initial_boundary_physical_metrics": initial_boundary_physical,
        "initial_jacobinet_boundary_metrics": initial_jacobinet_boundary,
        "initial_validation_score": initial_validation_score,
        "initial_legacy_equivalent_validation_score": initial_validation_score
        * loss_balance.legacy_equivalent_multiplier,
        "terminal_step": completed_steps,
        "terminal_true_section_flux_monitor": terminal_true_section_flux,
        "terminal_boundary_components": terminal_boundary,
        "terminal_boundary_physical_metrics": terminal_boundary_physical,
        "terminal_jacobinet_boundary_metrics": terminal_jacobinet_boundary,
        "final_true_section_flux_monitor": final_true_section_flux,
        "final_boundary_components": final_boundary,
        "final_boundary_physical_metrics": final_boundary_physical,
        "final_jacobinet_boundary_metrics": final_jacobinet_boundary,
        "point_counts": points.counts(),
        "configuration": configuration,
        "physics_scales": scales.to_dict(),
        "inputs": {
            "training_xlsx": str(data_path),
            "jacobinet_checkpoint": str(jacobi_path),
            "jacobinet_sha256": sha256(jacobi_path),
        },
        "outputs": {
            "pinn_checkpoint": str(final_checkpoint),
            "best_physics_checkpoint": str(physics_checkpoint),
            "terminal_checkpoint": str(terminal_checkpoint),
            "training_history": str(history_path),
        },
        "method": {
            "loss": "one PDE-only objective: weighted momentum_u + momentum_v + momentum_w + continuity; continuity_weight is fixed at 1; no boundary, data, flux, or auxiliary loss is active",
            "representation": "one configurable xyz RFF projection and one configurable rs RFF projection are shared by velocity and pressure; q=10/sigma scales the frozen B0 directions without changing the 128-D feature budget",
            "boundary": "full inlet/outlet/wall sets evaluated under no_grad at the recorded boundary cadence and terminal step; fluid BC MSE, physical RMS/max, and Jacobi coordinate leakage are reported",
            "gradient_norms": "weighted objective components",
            "true_section_flux": "Forward-only initial, terminal, and selected-model monitoring of total predicted flow; excluded from the loss.",
            "evaluation": "label-free fixed validation PDE residuals only",
            "checkpoint": "At the configured full horizon, select the lowest fixed-validation PDE score. A partial run selects its terminal state for exact resume. Early stopping is disabled.",
            "early_stop": loss_stop_rule(),
            "precision": f"{TRAINING_DTYPE_NAME} model, collocation data, and autodiff; {EVALUATION_DTYPE_NAME} physics scales and label-free monitors",
        },
    }
    report_assembly_seconds = time.perf_counter() - report_assembly_started
    report["timing_breakdown"] = {
        "clock": "time.perf_counter with CUDA phase-boundary synchronization",
        "input_model_configuration_setup_seconds": input_model_setup_seconds,
        "resume_checkpoint_load_seconds": resume_checkpoint_load_seconds,
        "resume_state_restore_seconds": resume_state_restore_seconds,
        "initial_diagnostics_seconds": initial_diagnostics_seconds,
        "pde_cache_source_copy_seconds": pde_cache_source_copy_seconds,
        "feature_cache_cold_build_seconds": explicit_jet_training_core.cache_build_seconds,
        "cuda_graph_setup_seconds": explicit_jet_training_core.graph_setup_seconds,
        "backend_total_setup_seconds": explicit_jet_training_core.total_setup_seconds,
        "backend_constructor_wall_seconds": explicit_jet_backend_constructor_wall_seconds,
        "backend_setup_including_pool_copy_seconds": pde_cache_source_copy_seconds
        + explicit_jet_backend_constructor_wall_seconds,
        "current_process_training_loop_seconds": current_process_training_seconds,
        "scheduled_evaluation_seconds_current_process": scheduled_evaluation_seconds,
        "non_evaluation_training_core_seconds_current_process": max(
            current_process_training_seconds - scheduled_evaluation_seconds, 0.0
        ),
        "cumulative_training_seconds": training_seconds,
        "terminal_diagnostics_seconds": terminal_diagnostics_seconds,
        "checkpoint_selection_io_seconds": checkpoint_selection_io_seconds,
        "final_selected_diagnostics_seconds": final_selected_diagnostics_seconds,
        "history_write_seconds": history_write_seconds,
        "report_payload_assembly_seconds": report_assembly_seconds,
        "trainer_process_seconds_before_report_write": time.perf_counter()
        - trainer_process_started,
        "outer_runner_wall_seconds": None,
        "outer_runner_wall_owner": "isolated subprocess runner",
    }
    report.update(
        {
            "matmul_precision_policy": dict(matmul_precision_policy),
            "fixed_batch_sha256": None,
            "initial_sampler_audit": None,
            "terminal_sampler_audit": None,
        }
    )
    write_training_report(report, output_root, trainer_process_started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
