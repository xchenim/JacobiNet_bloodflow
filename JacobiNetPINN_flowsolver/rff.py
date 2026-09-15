"""Fixed random Fourier feature specifications for the published flow models.

The Fourier matrix uses ``B ~ N(0, 1) / 10``.  This module keeps
those frozen directions and reports frequency through the unambiguous ratio
``q = 10 / sigma_project``.  Frequency changes therefore never redraw B and
never change the 32 xyz + 32 rs direction budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


SCHEMA_VERSION = 1
BASE_SIGMA = 10.0
DIRECTIONS_PER_DOMAIN = 32
ROUTING = "shared_velocity_pressure"
DIRECTION_POLICY = "reuse_b0_sigma10_directions"
ROW_ASSIGNMENT = "deterministic_interleaved"
_SPEC_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def _positive_finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive; got {value!r}")
    return result


def _positive_rows(value: int, label: str) -> int:
    result = int(value)
    if result != value or result <= 0:
        raise ValueError(f"{label} must be a positive integer; got {value!r}")
    return result


@dataclass(frozen=True)
class XYZBand:
    q: float
    rows: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "q", _positive_finite(self.q, "xyz q"))
        object.__setattr__(self, "rows", _positive_rows(self.rows, "xyz rows"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "rows": self.rows,
            "effective_sigma": BASE_SIGMA / self.q,
        }


@dataclass(frozen=True)
class RSBand:
    q_r: float
    q_s: float
    rows: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "q_r", _positive_finite(self.q_r, "r q"))
        object.__setattr__(self, "q_s", _positive_finite(self.q_s, "s q"))
        object.__setattr__(self, "rows", _positive_rows(self.rows, "rs rows"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "q_r": self.q_r,
            "q_s": self.q_s,
            "rows": self.rows,
            "effective_sigma_r": BASE_SIGMA / self.q_r,
            "effective_sigma_s": BASE_SIGMA / self.q_s,
        }


def _interleaved_labels(counts: Iterable[int]) -> tuple[int, ...]:
    """Spread every band through the row bank using weighted round robin."""
    totals = tuple(_positive_rows(value, "band rows") for value in counts)
    target = sum(totals)
    assigned = [0 for _ in totals]
    labels: list[int] = []
    for position in range(target):
        candidates = [
            index for index, total in enumerate(totals) if assigned[index] < total
        ]
        # Largest cumulative deficit wins.  The stable index tie-break makes
        # assignments deterministic across Python versions.
        chosen = max(
            candidates,
            key=lambda index: (
                totals[index] * (position + 1) / target - assigned[index],
                -index,
            ),
        )
        labels.append(chosen)
        assigned[chosen] += 1
    if tuple(assigned) != totals:
        raise AssertionError("interleaved band allocation is incomplete")
    return tuple(labels)


@dataclass(frozen=True)
class RFFSpec:
    spec_id: str
    xyz_bands: tuple[XYZBand, ...]
    rs_bands: tuple[RSBand, ...]
    condition: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not _SPEC_ID.fullmatch(self.spec_id):
            raise ValueError(f"Invalid RFF spec_id: {self.spec_id!r}")
        object.__setattr__(self, "xyz_bands", tuple(self.xyz_bands))
        object.__setattr__(self, "rs_bands", tuple(self.rs_bands))
        if not self.xyz_bands or not self.rs_bands:
            raise ValueError("xyz and rs must each declare at least one frequency band")
        if sum(band.rows for band in self.xyz_bands) != DIRECTIONS_PER_DOMAIN:
            raise ValueError("xyz bands must contain exactly 32 directions")
        if sum(band.rows for band in self.rs_bands) != DIRECTIONS_PER_DOMAIN:
            raise ValueError("rs bands must contain exactly 32 directions")
        if self.condition is not None:
            # Canonical JSON validation also rejects unserialisable metadata.
            json.dumps(self.condition, sort_keys=True, allow_nan=False)

    def xyz_q_rows(self) -> tuple[float, ...]:
        labels = _interleaved_labels(band.rows for band in self.xyz_bands)
        return tuple(self.xyz_bands[index].q for index in labels)

    def rs_q_rows(self) -> tuple[tuple[float, float], ...]:
        labels = _interleaved_labels(band.rows for band in self.rs_bands)
        return tuple(
            (self.rs_bands[index].q_r, self.rs_bands[index].q_s) for index in labels
        )

    def is_base_frequency(self) -> bool:
        return all(value == 1.0 for value in self.xyz_q_rows()) and all(
            q_r == 1.0 and q_s == 1.0 for q_r, q_s in self.rs_q_rows()
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "spec_id": self.spec_id,
            "direction_policy": DIRECTION_POLICY,
            "routing": ROUTING,
            "row_assignment": ROW_ASSIGNMENT,
            "base_sigma": {"xyz": BASE_SIGMA, "rs": BASE_SIGMA},
            "directions": {
                "xyz": DIRECTIONS_PER_DOMAIN,
                "rs": DIRECTIONS_PER_DOMAIN,
            },
            "bands": {
                "xyz": [band.to_dict() for band in self.xyz_bands],
                "rs": [band.to_dict() for band in self.rs_bands],
            },
            "resolved_rows": {
                "xyz_q": list(self.xyz_q_rows()),
                "rs_q": [list(values) for values in self.rs_q_rows()],
            },
            "condition": self.condition,
        }
        return payload

    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RFFSpec":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported RFF specification schema")
        if payload.get("direction_policy") not in (None, DIRECTION_POLICY):
            raise ValueError("RFF direction policy is not supported")
        if payload.get("routing") not in (None, ROUTING):
            raise ValueError("Only shared velocity/pressure routing is supported in v1")
        bands = payload.get("bands")
        if not isinstance(bands, dict):
            raise ValueError("RFF specification lacks bands")
        xyz = tuple(
            XYZBand(q=item["q"], rows=item["rows"]) for item in bands.get("xyz", [])
        )
        rs = tuple(
            RSBand(q_r=item["q_r"], q_s=item["q_s"], rows=item["rows"])
            for item in bands.get("rs", [])
        )
        return cls(
            spec_id=str(payload["spec_id"]),
            xyz_bands=xyz,
            rs_bands=rs,
            condition=payload.get("condition"),
        )


def single_frequency_spec(
    spec_id: str,
    *,
    q_xyz: float = 1.0,
    q_r: float = 1.0,
    q_s: float = 1.0,
    condition: dict[str, Any] | None = None,
) -> RFFSpec:
    return RFFSpec(
        spec_id=spec_id,
        xyz_bands=(XYZBand(q_xyz, DIRECTIONS_PER_DOMAIN),),
        rs_bands=(RSBand(q_r, q_s, DIRECTIONS_PER_DOMAIN),),
        condition=condition,
    )


def load_spec(path: str | Path | None = None) -> RFFSpec:
    """Load an explicit configuration, or use the fixed published flow model."""
    if path is None:
        return FLOW_RFF_SPEC
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RFF specification must be a JSON object")
    return RFFSpec.from_dict(payload)


BASE_RFF_SPEC = single_frequency_spec("b0_qx1_qr1_qs1")
FLOW_RFF_SPEC = single_frequency_spec("xyz_q1__rs_q4", q_r=4.0, q_s=4.0)
REGISTERED_SPECS = {spec.spec_id: spec for spec in (BASE_RFF_SPEC, FLOW_RFF_SPEC)}


def registered_spec(spec_id: str) -> RFFSpec:
    try:
        return REGISTERED_SPECS[spec_id]
    except KeyError as error:
        raise ValueError(f"Unknown registered RFF spec: {spec_id}") from error
