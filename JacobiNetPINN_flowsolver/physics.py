"""Per-case physical nondimensionalization for the JacobiNet-PINN solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


LEGACY_MODE = "legacy_inlet_mean"
THROAT_MODE = "throat_mass_conservation"
SUPPORTED_MODES = (LEGACY_MODE, THROAT_MODE)
RAW_EQUAL_LOSS = "raw_equal"
VELOCITY_SCALE_RELATIVE_LOSS = "velocity_scale_relative_v1"
SUPPORTED_LOSS_WEIGHTING = (RAW_EQUAL_LOSS, VELOCITY_SCALE_RELATIVE_LOSS)
SUPPORTED_CONTINUITY_WEIGHTS = (1.0, 0.75, 0.5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PhysicsScales:
    schema_version: int
    mode: str
    case_id: str
    geometry_path: str
    geometry_sha256: str
    rho_kg_m3: float
    dynamic_viscosity_pa_s: float
    kinematic_viscosity_m2_s: float
    inlet_peak_velocity_m_s: float
    inlet_mean_velocity_m_s: float
    inlet_radius_m: float
    minimum_radius_m: float
    inlet_to_minimum_area_ratio: float
    stenosis_percent: float
    length_scale_m: float
    velocity_scale_m_s: float
    pressure_scale_pa: float
    reynolds_number: float
    inlet_peak_velocity_nondim: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "PhysicsScales":
        return cls(**payload)


@dataclass(frozen=True)
class LossBalance:
    """Constant weights that preserve residual ratios across velocity scales.

    For ``alpha = U_scale / U_legacy``, the throat representation gives
    momentum MSEs proportional to ``alpha**-4`` and continuity/velocity-BC
    MSEs proportional to ``alpha**-2``.  Multiplying momentum and outlet
    pressure MSEs by ``alpha**2`` makes every objective component share the
    same ``alpha**-2`` factor without introducing very large alpha**4
    multipliers into float32 backpropagation.
    """

    revision: str
    policy: str
    alpha: float
    alpha_squared: float
    common_mse_factor_vs_legacy: float
    legacy_equivalent_multiplier: float
    pde_weights: dict[str, float]
    boundary_weights: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def build_loss_balance(
    scales: PhysicsScales,
    policy: str,
    *,
    continuity_weight: float = 1.0,
) -> LossBalance:
    if policy not in SUPPORTED_LOSS_WEIGHTING:
        raise ValueError(
            f"Unsupported loss weighting {policy!r}; choose {SUPPORTED_LOSS_WEIGHTING}"
        )
    continuity_weight = float(continuity_weight)
    if continuity_weight not in SUPPORTED_CONTINUITY_WEIGHTS:
        raise ValueError(
            "Unsupported continuity weight "
            f"{continuity_weight!r}; choose {SUPPORTED_CONTINUITY_WEIGHTS}"
        )
    alpha = float(scales.velocity_scale_m_s / scales.inlet_mean_velocity_m_s)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"Invalid velocity-scale ratio alpha={alpha!r}")
    alpha_squared = alpha**2
    if policy == RAW_EQUAL_LOSS:
        pde_weights = {
            "momentum_u": 1.0,
            "momentum_v": 1.0,
            "momentum_w": 1.0,
            "continuity": continuity_weight,
        }
        boundary_weights = {"inlet": 1.0, "outlet": 1.0, "wall": 1.0}
        common_factor = 1.0
        legacy_equivalent_multiplier = 1.0
    else:
        pde_weights = {
            "momentum_u": alpha_squared,
            "momentum_v": alpha_squared,
            "momentum_w": alpha_squared,
            "continuity": continuity_weight,
        }
        boundary_weights = {
            "inlet": 1.0,
            "outlet": alpha_squared,
            "wall": 1.0,
        }
        common_factor = 1.0 / alpha_squared
        legacy_equivalent_multiplier = alpha_squared
    base_revision = (
        "raw_equal_v1" if policy == RAW_EQUAL_LOSS else VELOCITY_SCALE_RELATIVE_LOSS
    )
    continuity_tag = format(continuity_weight, "g").replace(".", "p")
    revision = (
        base_revision
        if continuity_weight == 1.0
        else f"{base_revision}_continuity_{continuity_tag}"
    )
    return LossBalance(
        revision=revision,
        policy=policy,
        alpha=alpha,
        alpha_squared=alpha_squared,
        common_mse_factor_vs_legacy=common_factor,
        legacy_equivalent_multiplier=legacy_equivalent_multiplier,
        pde_weights=pde_weights,
        boundary_weights=boundary_weights,
    )


def weighted_component_values(components: dict, weights: dict[str, float]) -> dict:
    if set(components) != set(weights):
        raise ValueError(
            f"Component/weight mismatch: components={sorted(components)}, "
            f"weights={sorted(weights)}"
        )
    return {name: value * float(weights[name]) for name, value in components.items()}


def weighted_component_sum(components: dict, weights: dict[str, float]):
    weighted = weighted_component_values(components, weights)
    iterator = iter(weighted.values())
    total = next(iterator)
    for value in iterator:
        total = total + value
    return total


def _expected_geometry_hashes(case_root: Path, metadata: dict) -> dict[str, str]:
    expected: dict[str, str] = {}
    metadata_sha = metadata.get("source_sha256")
    if metadata_sha:
        expected["case_metadata.source_sha256"] = str(metadata_sha).lower()
    attempt_manifest_path = case_root / "attempt_manifest.json"
    if attempt_manifest_path.is_file():
        manifest = json.loads(attempt_manifest_path.read_text(encoding="utf-8"))
        formal_sha = manifest.get("source_inventory", {}).get("geometry_sha256")
        if not formal_sha:
            raise ValueError(
                "attempt_manifest source_inventory.geometry_sha256 is required"
            )
        expected["attempt_manifest formal geometry_sha256"] = str(formal_sha).lower()
    return expected


def _validate_geometry_hash(path: Path, expected: dict[str, str]) -> None:
    actual = _sha256(path).lower()
    for source, expected_sha in expected.items():
        if actual != expected_sha:
            raise ValueError(
                f"Geometry hash mismatch for {source}: "
                f"expected={expected_sha}, actual={actual}, path={path}"
            )


def _resolve_geometry(case_root: Path, metadata: dict, case_id: str) -> Path:
    expected_hashes = _expected_geometry_hashes(case_root, metadata)
    frozen_geometry = case_root / "centerline_sections.npz"
    if frozen_geometry.is_file():
        if "case_metadata.source_sha256" not in expected_hashes:
            raise ValueError(
                "Frozen centerline_sections.npz requires " "case_metadata.source_sha256"
            )
        resolved = frozen_geometry.resolve()
        _validate_geometry_hash(resolved, expected_hashes)
        return resolved

    candidates = []
    if metadata.get("source_geometry"):
        candidate = Path(metadata["source_geometry"]).expanduser()
        candidates.append(
            candidate if candidate.is_absolute() else case_root / candidate
        )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            _validate_geometry_hash(resolved, expected_hashes)
            return resolved
    raise FileNotFoundError(
        "No centerline_sections.npz found; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def build_physics_scales(
    case_root: str | Path,
    mode: str = LEGACY_MODE,
    *,
    case_id: str | None = None,
    inlet_radius_m: float | None = None,
    inlet_peak_velocity_m_s: float = 0.288,
    rho_kg_m3: float = 1060.0,
    dynamic_viscosity_pa_s: float = 0.0040,
) -> PhysicsScales:
    """Build a complete, physically consistent set of per-case scales.

    ``legacy_inlet_mean`` uses the inlet mean scale. The throat mode uses
    incompressible mass conservation to make throat velocity and pressure O(1).
    """

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported scaling mode {mode!r}; choose {SUPPORTED_MODES}")
    root = Path(case_root).resolve()
    metadata_path = root / "case_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    resolved_case_id = str(case_id or metadata.get("case_id") or root.name)
    geometry_path = _resolve_geometry(root, metadata, resolved_case_id)
    with np.load(geometry_path) as geometry:
        radii = np.asarray(geometry["radii_m"], dtype=np.float64)
    if radii.ndim != 1 or not len(radii):
        raise ValueError(f"Invalid radii_m in {geometry_path}")
    if not np.isfinite(radii).all() or np.any(radii <= 0.0):
        raise ValueError(f"radii_m must be positive and finite in {geometry_path}")

    radius_in = float(inlet_radius_m or metadata.get("inlet_radius_m") or radii[0])
    radius_min = float(metadata.get("minimum_radius_m") or np.min(radii))
    if not np.isfinite(radius_in) or not np.isfinite(radius_min):
        raise ValueError("Inlet and minimum radii must be finite")
    if radius_in <= 0.0 or radius_min <= 0.0 or radius_min > radius_in:
        raise ValueError(
            f"Invalid radii: inlet={radius_in:.12g}, minimum={radius_min:.12g}"
        )
    if inlet_peak_velocity_m_s <= 0.0:
        raise ValueError("Inlet peak velocity must be positive")
    if rho_kg_m3 <= 0.0 or dynamic_viscosity_pa_s <= 0.0:
        raise ValueError("Density and dynamic viscosity must be positive")

    inlet_mean = float(inlet_peak_velocity_m_s) / 2.0
    area_ratio = (radius_in / radius_min) ** 2
    velocity_multiplier = area_ratio if mode == THROAT_MODE else 1.0
    length_scale = radius_in / 2.0
    velocity_scale = inlet_mean * velocity_multiplier
    pressure_scale = float(rho_kg_m3) * velocity_scale**2
    kinematic_viscosity = float(dynamic_viscosity_pa_s) / float(rho_kg_m3)
    reynolds = velocity_scale * length_scale / kinematic_viscosity
    inlet_nondim = float(inlet_peak_velocity_m_s) / velocity_scale
    stenosis = 100.0 * (1.0 - radius_min / radius_in)

    values = (
        length_scale,
        velocity_scale,
        pressure_scale,
        reynolds,
        inlet_nondim,
        area_ratio,
        stenosis,
    )
    if not np.isfinite(values).all():
        raise ValueError("Computed nondimensional scales are not finite")

    return PhysicsScales(
        schema_version=2,
        mode=mode,
        case_id=resolved_case_id,
        geometry_path=str(geometry_path),
        geometry_sha256=_sha256(geometry_path),
        rho_kg_m3=float(rho_kg_m3),
        dynamic_viscosity_pa_s=float(dynamic_viscosity_pa_s),
        kinematic_viscosity_m2_s=kinematic_viscosity,
        inlet_peak_velocity_m_s=float(inlet_peak_velocity_m_s),
        inlet_mean_velocity_m_s=inlet_mean,
        inlet_radius_m=radius_in,
        minimum_radius_m=radius_min,
        inlet_to_minimum_area_ratio=area_ratio,
        stenosis_percent=stenosis,
        length_scale_m=length_scale,
        velocity_scale_m_s=velocity_scale,
        pressure_scale_pa=pressure_scale,
        reynolds_number=reynolds,
        inlet_peak_velocity_nondim=inlet_nondim,
    )


def assert_compatible(expected: PhysicsScales, actual: PhysicsScales) -> None:
    """Fail loudly when a checkpoint is paired with different physical scales."""

    identity_fields = ("mode", "case_id", "geometry_sha256")
    for field in identity_fields:
        if getattr(expected, field) != getattr(actual, field):
            raise ValueError(
                f"Physics scale mismatch for {field}: "
                f"expected={getattr(expected, field)!r}, actual={getattr(actual, field)!r}"
            )
    numeric_fields = (
        "length_scale_m",
        "velocity_scale_m_s",
        "pressure_scale_pa",
        "reynolds_number",
        "inlet_peak_velocity_nondim",
    )
    for field in numeric_fields:
        if not np.isclose(
            getattr(expected, field), getattr(actual, field), rtol=1.0e-12, atol=0.0
        ):
            raise ValueError(
                f"Physics scale mismatch for {field}: "
                f"expected={getattr(expected, field):.16g}, "
                f"actual={getattr(actual, field):.16g}"
            )
