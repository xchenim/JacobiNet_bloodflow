"""Versioned checkpoint validation, numerical state loading, and protected outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

if __package__ and __package__.startswith("JacobiNetPINN_flowsolver"):
    from .physics import PhysicsScales, assert_compatible
    from .runtime import (
        EVALUATION_DTYPE_NAME,
        PRECISION_POLICY_NAME,
        TRAINING_DTYPE_NAME,
        assert_model_training_dtype,
        model_dtype,
        torch_dtype_name,
    )
else:
    from physics import PhysicsScales, assert_compatible
    from runtime import (
        EVALUATION_DTYPE_NAME,
        PRECISION_POLICY_NAME,
        TRAINING_DTYPE_NAME,
        assert_model_training_dtype,
        model_dtype,
        torch_dtype_name,
    )


SCHEMA_VERSION = 3


def archived_flow_protocol(
    horizon: int = 20000, *, cohort: bool = True
) -> dict[str, str]:
    """Resolve serialized identifiers used by the supplied flow checkpoints.

    Historical tokens are file-format identifiers, not selectable model variants.
    """
    if horizon not in (10000, 20000, 30000, 40000):
        raise ValueError("horizon must be 10000, 20000, 30000, or 40000")
    scope = "_all100" if cohort else ""
    tag = f"h{horizon // 1000}k_no_early_stop{scope}_v1"
    protocol_scope = "-all100" if cohort else ""
    return {
        "method_suffix": f"a3_explicit_jet_{tag}",
        "method_revision": f"pinn_c1_speed_search_float32_xyz3+a3_explicit_jet_{tag}",
        "protocol_id": f"C1-A3-explicit-jet-H{horizon // 1000}k-no-early-stop{protocol_scope}-v1",
        "configuration_revision": f"semantic_v2+{tag}",
        "variant": "C1_SPEED_SEARCH",
        "runtime_mode": "a3_explicit_jet_cuda_graph",
    }


def _atomic_torch_save(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def _assert_state_dict_dtype(
    state_dict: dict[str, torch.Tensor], expected: torch.dtype
) -> None:
    mismatches = [
        f"{name}={value.dtype}"
        for name, value in state_dict.items()
        if value.is_floating_point() and value.dtype != expected
    ]
    if mismatches:
        raise TypeError(
            f"Checkpoint floating tensors must be {expected}; got "
            + ", ".join(mismatches[:8])
        )


def _precision_metadata() -> dict[str, str]:
    return {
        "model_dtype": TRAINING_DTYPE_NAME,
        "autodiff_dtype": TRAINING_DTYPE_NAME,
        "physics_scale_dtype": EVALUATION_DTYPE_NAME,
        "evaluation_dtype": EVALUATION_DTYPE_NAME,
        "precision_policy": PRECISION_POLICY_NAME,
    }


def save_jacobinet_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    scales: PhysicsScales,
    training_state: dict[str, Any],
) -> None:
    assert_model_training_dtype(model)
    state_dict = model.state_dict()
    _assert_state_dict_dtype(state_dict, model_dtype(model))
    _atomic_torch_save(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "jacobinet",
            "method_revision": f"jacobinet_{TRAINING_DTYPE_NAME}_v2",
            "dtype": torch_dtype_name(model_dtype(model)),
            **_precision_metadata(),
            "model_state_dict": state_dict,
            "case_id": scales.case_id,
            "geometry_sha256": scales.geometry_sha256,
            "training_state": training_state,
        },
        Path(path),
    )


def load_jacobinet_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    scales: PhysicsScales,
    *,
    map_location: str | torch.device | None = None,
) -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("kind") != "jacobinet":
        raise ValueError("Expected a structured JacobiNet checkpoint")
    expected_dtype = torch_dtype_name(model_dtype(model))
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("dtype") != expected_dtype
    ):
        raise ValueError(
            f"JacobiNet checkpoint is not schema-{SCHEMA_VERSION} {expected_dtype}"
        )
    if payload.get("case_id") != scales.case_id:
        raise ValueError("JacobiNet checkpoint case_id mismatch")
    if payload.get("geometry_sha256") != scales.geometry_sha256:
        raise ValueError("JacobiNet checkpoint geometry mismatch")
    if payload.get("precision_policy") != PRECISION_POLICY_NAME:
        raise ValueError("JacobiNet checkpoint precision policy mismatch")
    _assert_state_dict_dtype(payload["model_state_dict"], model_dtype(model))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    assert_model_training_dtype(model)
    return payload


def save_pinn_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    scales: PhysicsScales,
    training_state: dict[str, Any],
) -> None:
    assert_model_training_dtype(model)
    state_dict = model.state_dict()
    _assert_state_dict_dtype(state_dict, model_dtype(model))
    _atomic_torch_save(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "optimized_pinn",
            "method_revision": getattr(
                model, "checkpoint_method_revision", "pinn_high_stenosis_v2"
            ),
            "variant": getattr(model, "checkpoint_variant", None),
            "configuration_sha256": getattr(
                model, "checkpoint_configuration_sha256", None
            ),
            "dtype": torch_dtype_name(model_dtype(model)),
            **_precision_metadata(),
            "model_state_dict": state_dict,
            "model_config": getattr(model, "model_config", {}),
            "physics_scales": scales.to_dict(),
            "training_state": training_state,
        },
        Path(path),
    )


def load_pinn_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    scales: PhysicsScales,
    *,
    map_location: str | torch.device | None = None,
) -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("kind") != "optimized_pinn":
        raise ValueError("Expected a structured PINN checkpoint")
    expected_dtype = torch_dtype_name(model_dtype(model))
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("dtype") != expected_dtype
    ):
        raise ValueError(
            f"PINN checkpoint is not schema-{SCHEMA_VERSION} {expected_dtype}"
        )
    if payload.get("precision_policy") != PRECISION_POLICY_NAME:
        raise ValueError("PINN checkpoint precision policy mismatch")
    expected_revision = getattr(model, "checkpoint_method_revision", None)
    expected_variant = getattr(model, "checkpoint_variant", None)
    expected_configuration = getattr(model, "checkpoint_configuration_sha256", None)
    if (
        expected_revision is not None
        and payload.get("method_revision") != expected_revision
    ):
        raise ValueError("PINN checkpoint method revision mismatch")
    if expected_variant is not None and payload.get("variant") != expected_variant:
        raise ValueError("PINN checkpoint variant mismatch")
    if (
        expected_configuration is not None
        and payload.get("configuration_sha256") != expected_configuration
    ):
        raise ValueError("PINN checkpoint configuration mismatch")
    _assert_state_dict_dtype(payload["model_state_dict"], model_dtype(model))
    expected_config = getattr(model, "model_config", None)
    if expected_config is None or payload.get("model_config") != expected_config:
        raise ValueError("PINN checkpoint model configuration differs from the model")
    actual = PhysicsScales.from_dict(payload["physics_scales"])
    assert_compatible(scales, actual)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    assert_model_training_dtype(model)
    return payload


def prepare_training_output(
    output_root: str | Path, case_root: str | Path, *, input_paths=()
) -> Path:
    """Create an empty output directory outside source code and input data."""
    output = Path(output_root).resolve()
    case = Path(case_root).resolve()
    workspace = Path(__file__).resolve().parent.parent
    protected = [case, *(Path(p).resolve() for p in input_paths)]
    protected.extend(
        workspace / name
        for name in (
            "AttentionCNN_3Dreconstruction",
            "JacobiNetPINN_flowsolver",
            "RCA_generator",
            "synthetic_100",
        )
    )
    protected.extend(p for p in case.parents if p.name == "synthetic_100")
    for source in protected:
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Training output must be separate from code and inputs")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Training output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    return output
