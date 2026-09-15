"""Evaluate a fixed-budget PINN checkpoint against one case's CFD labels.

Supports the published 10k/20k/30k/40k training horizons. Checkpoint selection
uses physics validation; CFD labels are used only for this separate evaluation.
Results are written to a new JSON file.

CLI::

    python JacobiNetPINN_flowsolver/evaluate.py \
        --case-root PATH \
        --checkpoint PATH/terminal_step_checkpoint.pth \
        --output-json PATH/audit.json \
        --expected-horizon 20000 \
        [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch

if __package__:
    from .checkpoint import archived_flow_protocol
else:
    from checkpoint import archived_flow_protocol


if __package__ and __package__.startswith("JacobiNetPINN_flowsolver"):
    from .checkpoint import (
        SCHEMA_VERSION as CHECKPOINT_SCHEMA_VERSION,
        load_pinn_checkpoint,
    )
    from .dataset import load_cfd_reference
    from .metrics import evaluate_model
    from .model import JacobiNet, NetPINN
    from .runtime import GeometryRegions
    from .physics import (
        PhysicsScales,
        THROAT_MODE,
        assert_compatible,
        build_physics_scales,
    )
    from .runtime import (
        PRECISION_POLICY_NAME,
        TORCH_TRAINING_DTYPE,
        TRAINING_DTYPE_NAME,
        assert_model_training_dtype,
    )
    from .rff import RFFSpec
else:
    from checkpoint import (
        SCHEMA_VERSION as CHECKPOINT_SCHEMA_VERSION,
        load_pinn_checkpoint,
    )
    from dataset import load_cfd_reference
    from metrics import evaluate_model
    from model import JacobiNet, NetPINN
    from runtime import GeometryRegions
    from physics import (
        PhysicsScales,
        THROAT_MODE,
        assert_compatible,
        build_physics_scales,
    )
    from runtime import (
        PRECISION_POLICY_NAME,
        TORCH_TRAINING_DTYPE,
        TRAINING_DTYPE_NAME,
        assert_model_training_dtype,
    )
    from rff import RFFSpec


AUDIT_SCHEMA_VERSION = 1
AUDIT_REVISION = "h20k_no_early_stop_posthoc_label_audit_v1"
_ARCHIVE_DEFAULTS = archived_flow_protocol(cohort=False)
EXPECTED_METHOD_REVISION = _ARCHIVE_DEFAULTS["method_revision"]
EXPECTED_PROTOCOL_ID = _ARCHIVE_DEFAULTS["protocol_id"]
EXPECTED_VARIANT = _ARCHIVE_DEFAULTS["variant"]
EXPECTED_MAX_STEPS = 20_000
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_HORIZONS = frozenset({10_000, 20_000, 30_000, 40_000})


def configure_horizon(horizon: int) -> None:
    """Configure audit identities for one validated fixed-endpoint budget."""

    global AUDIT_REVISION
    global EXPECTED_METHOD_REVISION
    global EXPECTED_PROTOCOL_ID
    global EXPECTED_MAX_STEPS
    global EXPECTED_CONFIGURATION_IDENTITY_REVISION
    if horizon not in ALLOWED_HORIZONS:
        raise ValueError("expected horizon must be 10000, 20000, 30000, or 40000")
    identifiers = archived_flow_protocol(horizon)
    AUDIT_REVISION = f"flow_{horizon}_posthoc_label_audit_v1"
    EXPECTED_METHOD_REVISION = identifiers["method_revision"]
    EXPECTED_PROTOCOL_ID = identifiers["protocol_id"]
    EXPECTED_MAX_STEPS = horizon
    EXPECTED_CONFIGURATION_IDENTITY_REVISION = identifiers["configuration_revision"]


# These values mirror the frozen loss-only trainer contract. They are audit
# predicates, not evaluator options.
EXPECTED_RESUME_STATE_SCHEMA_VERSION = 3
EXPECTED_CONFIGURATION_IDENTITY_REVISION = "semantic_v2+h20k_no_early_stop_v1"
EXPECTED_LOSS_STOP_RULE = {
    "rule_id": "weighted_pde_loss_conservative_v1",
    "min_step": 12_000,
    "evaluation_frequency_steps": 25,
    "total_threshold": 3.0e-6,
    "component_threshold": 1.0e-6,
    "patience_evaluations": 20,
    "total_fields": ["validation_score", "ema_total", "train_mean_total"],
    "weighted_component_names": [
        "momentum_u",
        "momentum_v",
        "momentum_w",
        "continuity",
    ],
    "weighted_component_sources": ["validation", "ema", "train_mean"],
    "inclusive_comparison": True,
    "labels_used": False,
    "enabled": False,
    "decision_use": "monitor_only",
}
ALLOWED_STOP_REASONS = {
    "loss-converged-unverified",
    "max-steps-reached/unverified",
    "segment-budget-reached/unverified",
}
REQUIRED_CASE_FILES = (
    "case_metadata.json",
    "export_0.288h.csv",
    "train_1e-4.xlsx",
    "post_5e-4.xlsx",
    "train/jacobinet.pth",
)
TERMINAL_CHECKPOINT_NAME = "terminal_step_checkpoint.pth"
FINAL_CHECKPOINT_NAME = "best_model_weights_stenosis.pth"
ALLOWED_FINAL_SELECTION_KINDS = {
    "loss_convergence_confirmation_terminal",
    "best_physics_validation_score_at_max_steps",
}
PROTECTED_OUTPUT_NAMES = {
    "attempt_manifest.json",
    "experiment_manifest.json",
    "experiment_status.json",
    "method_freeze.json",
    "pinn_report.json",
    "run_result.json",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value).lower()
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _strict_positive_step(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if not 1 <= value <= EXPECTED_MAX_STEPS:
        raise ValueError(f"{label} must be in [1, {EXPECTED_MAX_STEPS}]")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _checkpoint_role(path: Path, training_state: Mapping[str, Any]) -> str:
    selected_kind = training_state.get("selected_kind")
    if selected_kind is not None:
        if path.name != FINAL_CHECKPOINT_NAME:
            raise ValueError(
                "A final loss-only checkpoint must retain the frozen "
                f"{FINAL_CHECKPOINT_NAME!r} filename"
            )
        if selected_kind not in ALLOWED_FINAL_SELECTION_KINDS:
            raise ValueError(f"Unsupported final selected_kind: {selected_kind!r}")
        selected_from = training_state.get("selected_from")
        if not isinstance(selected_from, str) or not selected_from.strip():
            raise ValueError("Final checkpoint selected_from must be a non-empty path")
        expected_source_name = (
            TERMINAL_CHECKPOINT_NAME
            if selected_kind == "loss_convergence_confirmation_terminal"
            else "best_physics_checkpoint.pth"
        )
        if selected_from.replace("\\", "/").rsplit("/", 1)[-1] != expected_source_name:
            raise ValueError(
                "Final checkpoint selected_from does not match selected_kind"
            )
        return "final"
    if training_state.get("terminal_step_checkpoint") is True:
        if path.name != TERMINAL_CHECKPOINT_NAME:
            raise ValueError(
                "A terminal loss-only checkpoint must retain the frozen "
                f"{TERMINAL_CHECKPOINT_NAME!r} filename"
            )
        return "terminal"
    raise ValueError("Checkpoint is neither a frozen terminal nor final checkpoint")


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return parsed


def _validate_loss_snapshot(
    value: Any,
    label: str,
    *,
    expected_step: int,
) -> dict[str, Any]:
    snapshot = _require_mapping(value, label)
    required = {
        "rule_id",
        "global_step",
        "eligible",
        "qualified",
        "totals",
        "weighted_components",
        "limits",
    }
    if set(snapshot) != required:
        raise ValueError(f"{label} keys do not match the frozen loss snapshot schema")
    if snapshot["rule_id"] != EXPECTED_LOSS_STOP_RULE["rule_id"]:
        raise ValueError(f"{label} rule mismatch")
    if snapshot["global_step"] != expected_step:
        raise ValueError(f"{label} global_step mismatch")
    if not isinstance(snapshot["eligible"], bool) or not isinstance(
        snapshot["qualified"], bool
    ):
        raise TypeError(f"{label} eligibility fields must be boolean")

    totals = _require_mapping(snapshot["totals"], f"{label}.totals")
    expected_totals = set(EXPECTED_LOSS_STOP_RULE["total_fields"])
    if set(totals) != expected_totals:
        raise ValueError(f"{label}.totals keys mismatch")
    for name, value_item in totals.items():
        _finite_nonnegative(value_item, f"{label}.totals.{name}")

    components = _require_mapping(
        snapshot["weighted_components"], f"{label}.weighted_components"
    )
    expected_components = (
        {
            f"weighted_validation_{name}"
            for name in EXPECTED_LOSS_STOP_RULE["weighted_component_names"]
        }
        | {
            f"ema_weighted_{name}"
            for name in EXPECTED_LOSS_STOP_RULE["weighted_component_names"]
        }
        | {
            f"train_mean_weighted_{name}"
            for name in EXPECTED_LOSS_STOP_RULE["weighted_component_names"]
        }
    )
    if set(components) != expected_components:
        raise ValueError(f"{label}.weighted_components keys mismatch")
    for name, value_item in components.items():
        _finite_nonnegative(value_item, f"{label}.weighted_components.{name}")

    limits = _require_mapping(snapshot["limits"], f"{label}.limits")
    expected_limits = {
        "min_step": EXPECTED_LOSS_STOP_RULE["min_step"],
        "evaluation_frequency_steps": EXPECTED_LOSS_STOP_RULE[
            "evaluation_frequency_steps"
        ],
        "total_threshold": EXPECTED_LOSS_STOP_RULE["total_threshold"],
        "component_threshold": EXPECTED_LOSS_STOP_RULE["component_threshold"],
    }
    if _canonical_json(limits) != _canonical_json(expected_limits):
        raise ValueError(f"{label}.limits mismatch")
    return dict(snapshot)


def _validate_full_resume_identity(
    training_state: Mapping[str, Any],
    *,
    global_step: int,
    model_config: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if training_state.get("full_resume_state") is not True:
        raise ValueError("Checkpoint is not marked as a full resume state")
    if (
        training_state.get("resume_state_schema_version")
        != EXPECTED_RESUME_STATE_SCHEMA_VERSION
    ):
        raise ValueError("Checkpoint resume-state schema mismatch")
    if (
        training_state.get("configuration_identity_revision")
        != EXPECTED_CONFIGURATION_IDENTITY_REVISION
    ):
        raise ValueError("Checkpoint configuration identity revision mismatch")
    if training_state.get("terminal_step_checkpoint") is not True:
        raise ValueError("Checkpoint is not marked terminal_step_checkpoint=True")
    observed_rule = _require_mapping(
        training_state.get("loss_stop_rule"),
        "training_state.loss_stop_rule",
    )
    if _canonical_json(observed_rule) != _canonical_json(EXPECTED_LOSS_STOP_RULE):
        raise ValueError("Checkpoint loss-stop rule mismatch")

    stop_reason = training_state.get("stop_reason")
    if stop_reason not in ALLOWED_STOP_REASONS:
        raise ValueError("Checkpoint stop_reason is unsupported")
    snapshot = _validate_loss_snapshot(
        training_state.get("terminal_loss_snapshot"),
        "training_state.terminal_loss_snapshot",
        expected_step=global_step,
    )

    if training_state.get("train_seed") != 99:
        raise ValueError("Checkpoint train_seed mismatch")
    if training_state.get("rff_seed") != 99:
        raise ValueError("Checkpoint rff_seed mismatch")
    observed_rff_sha = _require_sha256(
        training_state.get("rff_spec_sha256"),
        "training_state.rff_spec_sha256",
    )
    if observed_rff_sha != model_config.get("rff_spec_sha256"):
        raise ValueError("Checkpoint training/model RFF specification mismatch")

    scheduler = _require_mapping(
        training_state.get("scheduler_state"),
        "training_state.scheduler_state",
    )
    if scheduler.get("kind") != "cosine_lr_closed_form_v1":
        raise ValueError("Checkpoint scheduler kind mismatch")
    if scheduler.get("horizon_steps") != EXPECTED_MAX_STEPS:
        raise ValueError("Checkpoint scheduler horizon mismatch")
    if scheduler.get("last_step") != global_step:
        raise ValueError("Checkpoint scheduler last_step mismatch")

    if stop_reason == "max-steps-reached/unverified" and (
        global_step != EXPECTED_MAX_STEPS
    ):
        raise ValueError(f"max-steps status requires global_step={EXPECTED_MAX_STEPS}")
    if stop_reason == "segment-budget-reached/unverified" and (
        global_step >= EXPECTED_MAX_STEPS
    ):
        raise ValueError("segment status is invalid at the scheduler horizon")
    if stop_reason == "loss-converged-unverified":
        if global_step < EXPECTED_LOSS_STOP_RULE["min_step"]:
            raise ValueError("loss convergence occurred before the frozen minimum step")
        if not snapshot["qualified"]:
            raise ValueError("loss-converged checkpoint must have qualified snapshot")
    return dict(observed_rule), str(stop_reason), snapshot


def validate_loss_only_checkpoint_payload(
    payload: Any, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Validate identity and return non-tensor checkpoint audit metadata."""

    checkpoint = Path(checkpoint_path).resolve()
    root = _require_mapping(payload, "checkpoint")
    if root.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Checkpoint must use schema {CHECKPOINT_SCHEMA_VERSION}")
    if root.get("kind") != "optimized_pinn":
        raise ValueError("Checkpoint kind must be 'optimized_pinn'")
    if root.get("method_revision") != EXPECTED_METHOD_REVISION:
        raise ValueError(
            f"Checkpoint method does not match the {EXPECTED_MAX_STEPS}-step no-early-stop protocol"
        )
    if root.get("variant") != EXPECTED_VARIANT:
        raise ValueError("Checkpoint is not the frozen flow model variant")
    if root.get("dtype") != TRAINING_DTYPE_NAME:
        raise ValueError("Checkpoint training dtype is not float32")
    if root.get("precision_policy") != PRECISION_POLICY_NAME:
        raise ValueError("Checkpoint precision policy mismatch")

    configuration_sha = _require_sha256(
        root.get("configuration_sha256"), "configuration_sha256"
    )
    model_state = _require_mapping(root.get("model_state_dict"), "model_state_dict")
    if not model_state:
        raise ValueError("model_state_dict must not be empty")
    model_config = _require_mapping(root.get("model_config"), "model_config")
    physics_scales = _require_mapping(root.get("physics_scales"), "physics_scales")
    training_state = _require_mapping(root.get("training_state"), "training_state")

    if training_state.get("label_diagnostics_disabled") is not True:
        raise ValueError("Checkpoint is not marked label_diagnostics_disabled=True")
    if training_state.get("configuration_sha256") != configuration_sha:
        raise ValueError("Checkpoint top/training-state configuration hash mismatch")

    global_step = _strict_positive_step(
        training_state.get("global_step"), "training_state.global_step"
    )
    role = _checkpoint_role(checkpoint, training_state)
    selected_kind = training_state.get("selected_kind")
    stop_reason: str | None = None
    observed_rule: dict[str, Any] | None = None
    snapshot: dict[str, Any]
    configuration_identity_revision: str | None = None

    if role == "terminal" or (
        selected_kind == "loss_convergence_confirmation_terminal"
    ):
        observed_rule, stop_reason, snapshot = _validate_full_resume_identity(
            training_state,
            global_step=global_step,
            model_config=model_config,
        )
        configuration_identity_revision = str(
            training_state["configuration_identity_revision"]
        )
        if selected_kind == "loss_convergence_confirmation_terminal" and (
            stop_reason != "loss-converged-unverified"
        ):
            raise ValueError(
                "loss-convergence final checkpoint has inconsistent stop_reason"
            )
    else:
        snapshot = _validate_loss_snapshot(
            training_state.get("loss_stop_snapshot"),
            "training_state.loss_stop_snapshot",
            expected_step=global_step,
        )
        _finite_nonnegative(
            training_state.get("validation_score"),
            "training_state.validation_score",
        )

    case_id = str(physics_scales.get("case_id", ""))
    if not re.fullmatch(r"[0-9]{5}", case_id):
        raise ValueError("Checkpoint physics_scales.case_id must be five digits")
    geometry_sha = _require_sha256(
        physics_scales.get("geometry_sha256"),
        "checkpoint physics_scales.geometry_sha256",
    )
    if physics_scales.get("mode") != THROAT_MODE:
        raise ValueError("Checkpoint scaling mode is not throat_mass_conservation")

    return {
        "role": role,
        "case_id": case_id,
        "global_step": global_step,
        "configuration_sha256": configuration_sha,
        "configuration_identity_revision": configuration_identity_revision,
        "geometry_sha256": geometry_sha,
        "stop_reason": stop_reason,
        "loss_stop_rule": observed_rule,
        "loss_stop_snapshot": snapshot,
        "selected_kind": selected_kind,
        "selected_from": training_state.get("selected_from"),
        "method_revision": root["method_revision"],
        "training_protocol_id": EXPECTED_PROTOCOL_ID,
        "variant": root["variant"],
        "model_config": dict(model_config),
    }


def load_validated_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = validate_loss_only_checkpoint_payload(payload, path)
    return dict(payload), metadata


def prepare_case(
    case_root: str | Path,
    checkpoint_payload: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
) -> tuple[PhysicsScales, Any, dict[str, Any]]:
    root = Path(case_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    missing = [name for name in REQUIRED_CASE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Staged case is missing: {missing}")

    metadata_path = root / "case_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("case_metadata.json must contain an object")
    case_id = str(metadata.get("case_id", ""))
    if case_id != checkpoint_metadata["case_id"]:
        raise ValueError("Case metadata/checkpoint case_id mismatch")

    cfd = metadata.get("cfd", {})
    export_report = cfd.get("export")
    if not isinstance(export_report, dict):
        raise ValueError("case_metadata.cfd.export must contain an object")
    if export_report.get("status") != "completed":
        raise ValueError("CFD export report is not completed")
    if cfd.get("status") != "converged":
        raise ValueError("CFD solver source is not converged")

    stored_scales = PhysicsScales.from_dict(dict(checkpoint_payload["physics_scales"]))
    scales = build_physics_scales(
        root,
        stored_scales.mode,
        case_id=case_id,
        inlet_peak_velocity_m_s=stored_scales.inlet_peak_velocity_m_s,
        rho_kg_m3=stored_scales.rho_kg_m3,
        dynamic_viscosity_pa_s=stored_scales.dynamic_viscosity_pa_s,
    )
    assert_compatible(scales, stored_scales)
    if scales.geometry_sha256 != checkpoint_metadata["geometry_sha256"]:
        raise ValueError("Case geometry/checkpoint hash mismatch")
    declared_geometry_sha = metadata.get("source_sha256")
    if (
        declared_geometry_sha is not None
        and _require_sha256(declared_geometry_sha, "case_metadata.source_sha256")
        != scales.geometry_sha256
    ):
        raise ValueError("Case metadata source geometry hash mismatch")

    reference_path = root / "export_0.288h.csv"
    if file_sha256(reference_path) != export_report["training_csv_sha256"]:
        raise ValueError("CFD reference hash differs from case metadata")
    reference = load_cfd_reference(reference_path)
    input_hashes = {name: file_sha256(root / name) for name in REQUIRED_CASE_FILES}
    return (
        scales,
        reference,
        {
            "case_root": str(root),
            "case_id": case_id,
            "case_metadata_sha256": input_hashes["case_metadata.json"],
            "cfd_reference_sha256": input_hashes["export_0.288h.csv"],
            "jacobinet_checkpoint_sha256": input_hashes["train/jacobinet.pth"],
            "geometry_path": scales.geometry_path,
            "geometry_sha256": scales.geometry_sha256,
            "input_sha256": input_hashes,
        },
    )


def _validated_checkpoint_rff_base_matrix(
    model_state: Mapping[str, Any],
    key: str,
    expected_shape: tuple[int, int],
) -> torch.Tensor:
    if key not in model_state:
        raise KeyError(f"model_state_dict is missing required {key}")
    matrix = model_state[key]
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"model_state_dict.{key} must be a tensor")
    if matrix.layout != torch.strided:
        raise TypeError(f"model_state_dict.{key} must be a strided tensor")
    if tuple(matrix.shape) != expected_shape:
        raise ValueError(
            f"model_state_dict.{key} must have shape {expected_shape}; "
            f"got {tuple(matrix.shape)}"
        )
    if matrix.dtype != TORCH_TRAINING_DTYPE:
        raise TypeError(
            f"model_state_dict.{key} must use {TORCH_TRAINING_DTYPE}; "
            f"got {matrix.dtype}"
        )
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError(f"model_state_dict.{key} must contain only finite values")
    return matrix.detach().to(device="cpu").contiguous().clone()


def build_flow_model(
    checkpoint_path: str | Path,
    checkpoint_payload: Mapping[str, Any],
    scales: PhysicsScales,
    device: torch.device,
) -> NetPINN:
    config = _require_mapping(checkpoint_payload.get("model_config"), "model_config")
    expected_fixed = {
        "architecture": "shared_configurable_rff_v2",
        "hidden_dim_velocity": 128,
        "hidden_dim_pressure": 128,
        "rff_dim_xyz": 64,
        "rff_dim_rs": 64,
        "train_seed": 99,
        "rff_seed": 99,
    }
    for key, expected in expected_fixed.items():
        if config.get(key) != expected:
            raise ValueError(f"Published model_config mismatch for {key}")
    if "velocity_trial" in config:
        raise ValueError("Checkpoint must use the paper velocity trial function")

    spec_payload = _require_mapping(config.get("rff_spec"), "model_config.rff_spec")
    rff_spec = RFFSpec.from_dict(dict(spec_payload))
    if config.get("rff_spec_id") != rff_spec.spec_id:
        raise ValueError("RFF spec ID mismatch")
    if config.get("rff_spec_sha256") != rff_spec.sha256():
        raise ValueError("RFF spec SHA mismatch")

    model_state = _require_mapping(
        checkpoint_payload.get("model_state_dict"), "model_state_dict"
    )
    base_B_xyz = _validated_checkpoint_rff_base_matrix(
        model_state,
        "rff_xyz.B",
        (32, 3),
    )
    base_B_rs = _validated_checkpoint_rff_base_matrix(
        model_state,
        "rff_rs.B",
        (32, 2),
    )
    jacobi = JacobiNet().to(dtype=TORCH_TRAINING_DTYPE)
    model = NetPINN(
        jacobi,
        scales,
        hidden_dim_velocity=128,
        hidden_dim_pressure=128,
        rff_dim_xyz=64,
        rff_dim_rs=64,
        rff_spec=rff_spec,
        base_B_xyz=base_B_xyz,
        base_B_rs=base_B_rs,
        train_seed=99,
        rff_seed=99,
    ).to(device=device, dtype=TORCH_TRAINING_DTYPE)
    model.checkpoint_method_revision = EXPECTED_METHOD_REVISION
    model.checkpoint_variant = EXPECTED_VARIANT
    model.checkpoint_configuration_sha256 = checkpoint_payload["configuration_sha256"]
    loaded = load_pinn_checkpoint(
        checkpoint_path,
        model,
        scales,
        map_location=device,
    )
    loaded_base_matrices = (
        ("rff_xyz.B", model.rff_xyz.B, base_B_xyz),
        ("rff_rs.B", model.rff_rs.B, base_B_rs),
    )
    for label, observed, expected in loaded_base_matrices:
        if not torch.equal(observed.detach().cpu(), expected):
            raise RuntimeError(
                f"Loaded {label} differs from validated checkpoint state"
            )
    effective_matrices = (
        (
            "xyz_effective_B",
            model.xyz_effective_B,
            model.rff_xyz.B * model.xyz_frequency_multipliers,
        ),
        (
            "rs_effective_B",
            model.rs_effective_B,
            model.rff_rs.B * model.rs_frequency_multipliers,
        ),
    )
    for label, observed, expected in effective_matrices:
        if not torch.equal(observed, expected):
            raise RuntimeError(f"{label} was not reconstructed from checkpoint base B")
    if loaded.get("model_config") != model.model_config:
        raise ValueError(
            "Checkpoint model_config does not match reconstructed flow model"
        )
    assert_model_training_dtype(model)
    model.eval()
    return model


def primary_l2_metrics(raw_metrics: Mapping[str, Any]) -> dict[str, float | int]:
    required = (
        "evaluation_points",
        "speed_magnitude_relative_l2",
        "velocity_vector_relative_l2",
        "pressure_relative_l2",
    )
    missing = [name for name in required if name not in raw_metrics]
    if missing:
        raise ValueError(f"evaluate_model omitted required metrics: {missing}")
    evaluation_points = raw_metrics["evaluation_points"]
    if isinstance(evaluation_points, bool) or not isinstance(evaluation_points, int):
        raise TypeError("evaluation_points must be an integer")
    if evaluation_points <= 0:
        raise ValueError("evaluation_points must be positive")
    values: dict[str, float] = {}
    for name in required[1:]:
        value = float(raw_metrics[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        values[name] = value
    return {
        "evaluation_points": evaluation_points,
        **values,
        "joint_l2": max(
            values["speed_magnitude_relative_l2"],
            values["pressure_relative_l2"],
        ),
    }


def evaluate_checkpoint(
    *,
    case_root: str | Path,
    checkpoint_path: str | Path,
    device_name: str,
    prediction_callback=None,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).resolve()
    payload, checkpoint_metadata = load_validated_checkpoint(checkpoint)
    scales, reference, case_metadata = prepare_case(
        case_root, payload, checkpoint_metadata
    )
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda:0")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("device must be 'cpu' or 'cuda'")

    model = build_flow_model(checkpoint, payload, scales, device)
    with torch.inference_mode():
        raw_metrics = evaluate_model(
            model,
            reference,
            scales,
            GeometryRegions.from_scales(scales),
            prediction_callback=prediction_callback,
        )
    metrics = primary_l2_metrics(raw_metrics)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_revision": AUDIT_REVISION,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "posthoc_cfd_label_report_only",
        "training_or_selection_side_effects": False,
        "decision_use": (
            "report_only; forbidden for training selection, ranking, " "or continuation"
        ),
        "case": case_metadata,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            **{
                key: checkpoint_metadata[key]
                for key in (
                    "role",
                    "case_id",
                    "global_step",
                    "configuration_sha256",
                    "geometry_sha256",
                    "method_revision",
                    "training_protocol_id",
                    "variant",
                    "configuration_identity_revision",
                    "stop_reason",
                    "loss_stop_rule",
                    "loss_stop_snapshot",
                    "selected_kind",
                    "selected_from",
                )
            },
        },
        "evaluation": {
            "function": "metrics.evaluate_model",
            "training_dtype": TRAINING_DTYPE_NAME,
            "evaluation_dtype": "float64",
            "device": str(device),
            "joint_l2_definition": (
                "max(speed_magnitude_relative_l2, pressure_relative_l2)"
            ),
            "metrics": metrics,
        },
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def write_new_audit_json(
    output_path: str | Path,
    payload: Mapping[str, Any],
    *,
    case_root: str | Path,
) -> Path:
    target = Path(output_path).resolve()
    root = Path(case_root).resolve()
    if target.suffix.lower() != ".json":
        raise ValueError("--output-json must end in .json")
    if target.name in PROTECTED_OUTPUT_NAMES:
        raise ValueError("Refusing a training/runner-owned output filename")
    if _is_relative_to(target, root / "train"):
        raise ValueError("Audit JSON must not be written into the training directory")
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                dict(payload),
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise FileExistsError(target)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-root",
        type=Path,
        required=True,
        help="Read-only staged case root containing frozen CFD-label inputs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Frozen loss-only terminal_step_checkpoint.pth or "
            "best_model_weights_stenosis.pth."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="New audit JSON path; existing files are never overwritten.",
    )
    parser.add_argument(
        "--expected-horizon",
        type=int,
        choices=sorted(ALLOWED_HORIZONS),
        required=True,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Inference device only; default: cpu.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_horizon(args.expected_horizon)
    payload = evaluate_checkpoint(
        case_root=args.case_root,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
    )
    output = write_new_audit_json(
        args.output_json,
        payload,
        case_root=args.case_root,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output_json": str(output),
                "output_sha256": file_sha256(output),
                "case_id": payload["case"]["case_id"],
                "checkpoint_role": payload["checkpoint"]["role"],
                "metrics": payload["evaluation"]["metrics"],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
