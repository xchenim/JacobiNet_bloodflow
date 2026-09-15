"""Training-precision collocation loading with float64 CFD references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ and __package__.startswith("JacobiNetPINN_flowsolver"):
    from .physics import PhysicsScales
    from .runtime import NUMPY_TRAINING_DTYPE, TORCH_TRAINING_DTYPE
else:
    from physics import PhysicsScales
    from runtime import NUMPY_TRAINING_DTYPE, TORCH_TRAINING_DTYPE


TRAIN_COLUMNS = ["x", "y", "z", "dist_ratio", "z_norm"]
REFERENCE_COLUMNS = [
    " X [ m ]",
    " Y [ m ]",
    " Z [ m ]",
    " dist_ratio",
    " z_norm",
    " Velocity [ m s^-1 ]",
    " Velocity u [ m s^-1 ]",
    " Velocity v [ m s^-1 ]",
    " Velocity w [ m s^-1 ]",
    " Pressure [ Pa ]",
]


@dataclass(frozen=True)
class TrainingPoints:
    internal: torch.Tensor
    inlet: torch.Tensor
    outlet: torch.Tensor
    wall: torch.Tensor

    @property
    def all_points(self) -> torch.Tensor:
        return torch.cat((self.internal, self.inlet, self.wall, self.outlet), dim=0)

    def counts(self) -> dict[str, int]:
        return {
            "internal": len(self.internal),
            "inlet": len(self.inlet),
            "outlet": len(self.outlet),
            "wall": len(self.wall),
            "all": len(self.internal)
            + len(self.inlet)
            + len(self.outlet)
            + len(self.wall),
        }


@dataclass(frozen=True)
class CFDReference:
    points_physical: np.ndarray
    speed_m_s: np.ndarray
    velocity_m_s: np.ndarray
    pressure_pa: np.ndarray

    @property
    def z_norm(self) -> np.ndarray:
        return self.points_physical[:, 4]


def _sheet_tensor(
    frame: pd.DataFrame,
    scales: PhysicsScales,
    device: torch.device,
) -> torch.Tensor:
    values = frame[TRAIN_COLUMNS].to_numpy(dtype=NUMPY_TRAINING_DTYPE, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Training workbook contains non-finite coordinates")
    values[:, :3] /= scales.length_scale_m
    return torch.as_tensor(values, dtype=TORCH_TRAINING_DTYPE, device=device)


def load_training_points(
    path: str | Path,
    scales: PhysicsScales,
    device: torch.device,
) -> TrainingPoints:
    sheets = pd.read_excel(path, sheet_name=None)
    required = {"internal", "inlet", "outlet", "bd"}
    missing = required - set(sheets)
    if missing:
        raise ValueError(f"Training workbook is missing sheets: {sorted(missing)}")
    return TrainingPoints(
        internal=_sheet_tensor(sheets["internal"], scales, device),
        inlet=_sheet_tensor(sheets["inlet"], scales, device),
        outlet=_sheet_tensor(sheets["outlet"], scales, device),
        wall=_sheet_tensor(sheets["bd"], scales, device),
    )


def load_cfd_reference(path: str | Path) -> CFDReference:
    frame = pd.read_csv(path, skiprows=5)
    missing = set(REFERENCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing CFD columns in {path}: {sorted(missing)}")
    numeric = frame[REFERENCE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    bad = ~np.isfinite(values)
    if bad.any():
        rows, columns = np.nonzero(bad)
        details = [
            (int(row) + 7, REFERENCE_COLUMNS[int(col)])
            for row, col in zip(rows[:10], columns[:10])
        ]
        raise ValueError(
            f"Invalid CFD numeric values in {path}; (CSV line, column): {details}"
        )
    if not len(values):
        raise ValueError(f"No CFD reference rows in {path}")
    return CFDReference(
        points_physical=values[:, :5],
        speed_m_s=values[:, 5],
        velocity_m_s=values[:, 6:9],
        pressure_pa=values[:, 9],
    )
