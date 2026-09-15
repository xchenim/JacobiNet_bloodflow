"""Final AttentionCNN data contract for source PNG/NPY files or immutable mmap caches."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


GRADE_THRESHOLDS = (25.0, 50.0, 70.0)


def severity_percent(radius: np.ndarray) -> float:
    values = np.asarray(radius, dtype=np.float64).reshape(-1)
    if values.size < 1 or values[0] <= 0 or np.any(values <= 0):
        raise ValueError("radius must be positive and begin at the inlet")
    return float(100.0 * (1.0 - values.min() / values[0]))


def severity_grade(severity: float) -> int:
    return int(np.searchsorted(np.asarray(GRADE_THRESHOLDS), severity, side="right"))


def legacy_sample_indices(num_raw_points: int, num_points: int = 12) -> np.ndarray:
    if num_raw_points < 1 or num_points < 1:
        raise ValueError("point counts must be positive")
    if num_raw_points == num_points:
        return np.arange(num_raw_points, dtype=np.int64)
    return torch.linspace(0, num_raw_points - 1, num_points).long().numpy()


def min_preserving_sample_indices(
    radius: np.ndarray, num_points: int = 12
) -> np.ndarray:
    values = np.asarray(radius, dtype=np.float64).reshape(-1)
    indices = legacy_sample_indices(values.size, num_points).copy()
    raw_minimum = int(np.argmin(values))
    nearest_slot = int(np.argmin(np.abs(indices - raw_minimum)))
    indices[nearest_slot] = raw_minimum
    if np.any(np.diff(indices) < 0):
        raise RuntimeError("minimum-preserving replacement broke point order")
    return indices


def min_preserving_sample(points: np.ndarray, num_points: int = 12) -> np.ndarray:
    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError("label must have shape (N, >=4)")
    return np.ascontiguousarray(
        values[min_preserving_sample_indices(values[:, 3], num_points), :4]
    )


def load_label(case_dir: Path, case_id: str) -> np.ndarray:
    path = case_dir / f"{case_id}_stenosis.npy"
    points = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if points.ndim == 3 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 2 or points.shape[1] < 4 or points.shape[0] < 2:
        raise ValueError(f"invalid label shape {points.shape} in {path}")
    points = np.ascontiguousarray(points[:, :4])
    if not np.isfinite(points).all() or np.any(points[:, 3] <= 0):
        raise ValueError(f"invalid label values in {path}")
    return points


def _fit_to_size(image: Image.Image, size: int = 128) -> Image.Image:
    image = image.convert("L")
    width, height = image.size
    pad_w, pad_h = max(size - width, 0), max(size - height, 0)
    if pad_w or pad_h:
        image = ImageOps.expand(
            image,
            border=(
                pad_w // 2,
                pad_h // 2,
                pad_w - pad_w // 2,
                pad_h - pad_h // 2,
            ),
            fill=0,
        )
        width, height = image.size
    left, top = (width - size) // 2, (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def _manifest_case_ids(path: Path, split: str) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        return sorted(
            str(row["case_id"]) for row in rows if str(row.get("split", split)) == split
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("cases"), list):
        entries = [
            entry
            for entry in payload["cases"]
            if not isinstance(entry, Mapping) or str(entry.get("split", split)) == split
        ]
    elif isinstance(payload, Mapping) and isinstance(payload.get(split), list):
        entries = payload[split]
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError(f"unrecognised manifest structure: {path}")
    case_ids = []
    for entry in entries:
        if isinstance(entry, Mapping):
            entry = entry.get("case_id", entry.get("case", entry.get("id")))
        case_ids.append(str(entry))
    return sorted(dict.fromkeys(case_ids))


class SyntheticReconstructionDataset(Dataset):
    """Two relative-DT views and paired 12-point geometry targets."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str = "test",
        manifest: str | Path | None = None,
        cache_root: str | Path | None = None,
        case_ids: Sequence[str] | None = None,
        limit: int | None = None,
        tensor_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if tensor_dtype not in (torch.float32, torch.float64):
            raise ValueError("tensor_dtype must be torch.float32 or torch.float64")
        self.tensor_dtype = tensor_dtype
        self.numpy_dtype = np.float64 if tensor_dtype == torch.float64 else np.float32
        root = Path(dataset_root).resolve()
        if (root / split).is_dir():
            self.dataset_root, self.split_root = root, (root / split).resolve()
        elif root.name == split and root.is_dir():
            self.dataset_root, self.split_root = root.parent, root
        else:
            raise FileNotFoundError(f"cannot resolve split {split!r} under {root}")
        self.split = split
        if case_ids is not None:
            ids = sorted(dict.fromkeys(str(value) for value in case_ids))
        elif manifest is not None:
            ids = _manifest_case_ids(Path(manifest), split)
        else:
            ids = sorted(
                path.name for path in self.split_root.iterdir() if path.is_dir()
            )
        if limit is not None:
            ids = ids[:limit]
        if not ids:
            raise ValueError("no cases selected")
        self.case_ids = ids
        self._row_by_case: dict[str, int] | None = None
        self._cache_dir: Path | None = None
        self._cache_metadata: dict[str, Any] | None = None
        self._cache_arrays: dict[str, np.ndarray] = {}
        if cache_root is not None:
            self._configure_cache(Path(cache_root))
        else:
            self._validate_sources()

    def _validate_sources(self) -> None:
        for case_id in self.case_ids:
            case_dir = self.split_root / case_id
            required = [
                case_dir / f"{case_id}_stenosis.npy",
                case_dir / f"image{case_id}a_stenosis.png",
                case_dir / f"image{case_id}b_stenosis.png",
            ]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError("missing sample files: " + ", ".join(missing))

    def _configure_cache(self, cache_root: Path) -> None:
        cache_dir = cache_root.resolve()
        if not (cache_dir / "metadata.json").is_file():
            cache_dir = cache_dir / self.split
        metadata_path = cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1:
            raise ValueError("unsupported cache schema")
        if metadata.get("format") != "reconstruction_v2_split_mmap":
            raise ValueError("unsupported cache format")
        if metadata.get("split") != self.split:
            raise ValueError("cache split mismatch")
        if int(metadata.get("image_size", -1)) != 128:
            raise ValueError("cache image size must be 128")
        cache_ids = [str(value) for value in metadata.get("case_ids", [])]
        row_by_case = {case: row for row, case in enumerate(cache_ids)}
        missing = [case for case in self.case_ids if case not in row_by_case]
        if missing:
            raise ValueError(
                "selected cases are missing from cache: " + ", ".join(missing[:10])
            )
        artifacts = metadata.get("artifacts", {})
        for key in ("normalised_dt", "labels", "label_offsets"):
            record = artifacts.get(key)
            if not isinstance(record, Mapping):
                raise ValueError(f"cache artifact {key!r} is missing")
            path = cache_dir / str(record.get("filename", ""))
            if not path.is_file() or path.stat().st_size != int(
                record.get("bytes", -1)
            ):
                raise ValueError(f"cache artifact {key!r} is missing or truncated")
        self._cache_dir = cache_dir.resolve()
        self._cache_metadata = metadata
        self._row_by_case = row_by_case

    def _open_cache(self) -> dict[str, np.ndarray]:
        if self._cache_arrays:
            return self._cache_arrays
        if self._cache_dir is None or self._cache_metadata is None:
            raise RuntimeError("cache is not configured")
        artifacts = self._cache_metadata["artifacts"]
        self._cache_arrays = {
            key: np.load(
                self._cache_dir / artifacts[key]["filename"],
                mmap_mode="r",
                allow_pickle=False,
            )
            for key in ("normalised_dt", "labels", "label_offsets")
        }
        return self._cache_arrays

    def close_cache(self) -> None:
        for array in self._cache_arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._cache_arrays = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache_arrays"] = {}
        return state

    def __del__(self) -> None:
        try:
            self.close_cache()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.case_ids)

    def _cached_label(self, case_id: str) -> np.ndarray:
        if self._row_by_case is None:
            raise RuntimeError("cache is not configured")
        row = self._row_by_case[case_id]
        cache = self._open_cache()
        offsets = cache["label_offsets"]
        start, end = int(offsets[row]), int(offsets[row + 1])
        return np.array(cache["labels"][start:end], dtype=np.float64, copy=True)

    def raw_label(self, case_id: str) -> np.ndarray:
        if self._cache_metadata is not None:
            return self._cached_label(case_id)
        return load_label(self.split_root / case_id, case_id)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case_id = self.case_ids[index]
        if self._cache_metadata is not None:
            if self._row_by_case is None:
                raise RuntimeError("cache row map is unavailable")
            row = self._row_by_case[case_id]
            normalised = (
                np.asarray(
                    self._open_cache()["normalised_dt"][row], dtype=self.numpy_dtype
                )
                / 255.0
            )
            images = torch.from_numpy(
                np.ascontiguousarray(normalised[:, None, :, :])
            ).to(dtype=self.tensor_dtype)
            full = self._cached_label(case_id)
        else:
            case_dir = self.split_root / case_id
            views = []
            for suffix in ("a", "b"):
                image = _fit_to_size(
                    Image.open(case_dir / f"image{case_id}{suffix}_stenosis.png")
                )
                array = np.asarray(image, dtype=self.numpy_dtype)[None, ...] / 255.0
                views.append(
                    torch.from_numpy(np.ascontiguousarray(array)).to(
                        dtype=self.tensor_dtype
                    )
                )
            images = torch.stack(views, dim=0)
            full = load_label(case_dir, case_id)
        target = min_preserving_sample(full)
        return {
            "case_id": case_id,
            "images": images,
            "target_xyz": torch.from_numpy(target[:, :3].copy()).to(
                dtype=self.tensor_dtype
            ),
            "target_radius": torch.from_numpy(target[:, 3:4].copy()).to(
                dtype=self.tensor_dtype
            ),
            "full_severity": torch.tensor(
                severity_percent(full[:, 3]), dtype=self.tensor_dtype
            ),
            "full_grade": torch.tensor(
                severity_grade(severity_percent(full[:, 3])), dtype=torch.long
            ),
        }


def tortuosity(points_xyz: np.ndarray) -> float:
    points = np.asarray(points_xyz, dtype=np.float64)
    arc = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    chord = np.linalg.norm(points[-1] - points[0])
    return float(arc / max(chord, 1e-12))


__all__ = [
    "SyntheticReconstructionDataset",
    "legacy_sample_indices",
    "min_preserving_sample",
    "min_preserving_sample_indices",
    "severity_grade",
    "severity_percent",
    "tortuosity",
]
