"""Generate seeded RCA geometry and paired projection images."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--layout", choices=("dataset", "legacy"), default="dataset")
    p.add_argument("--split", choices=("train", "valid", "test"), default="train")
    p.add_argument("--config", type=Path, help="JSON overrides for names in configs.py")
    a = p.parse_args()
    if __package__:
        from . import configs
    else:
        import configs
    if a.config:
        overrides = json.loads(a.config.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict) or set(overrides) - set(
            configs.PARAMETER_NAMES
        ):
            raise ValueError("config must contain only known parameter names")
        defaults = configs.parameters()
        for name, value in overrides.items():
            default = defaults[name]
            if isinstance(default, bool) and not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
            if (
                not isinstance(default, bool)
                and isinstance(default, (int, float))
                and (isinstance(value, bool) or not isinstance(value, (int, float)))
            ):
                raise ValueError(f"{name} must be numeric")
            if type(default) is int and type(value) is not int:
                raise ValueError(f"{name} must be an integer")
            if isinstance(default, str) and not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
            setattr(configs, name, value)
    configs.validate()
    if not 1 <= a.count <= 100000:
        raise ValueError("count must be between 1 and 100000 for five-digit IDs")
    if not 0 <= a.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    output = a.output_root.resolve()
    package_root = Path(__file__).resolve().parent
    frozen = package_root.parent / "synthetic_100"
    for protected in (frozen, package_root):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("new samples must be outside the code and frozen dataset")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("output-root must be an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    configs.save_path, configs.num_trees = str(output), a.count
    configs.save_visualization = a.visualize
    configs.output_layout, configs.split = a.layout, a.split
    if __package__:
        from .generator import generate
    else:
        from generator import generate

    generate(a.seed)
    cases = []
    for i in range(a.count):
        case_id = f"{i:05d}"
        folder = (
            output / a.split / case_id if a.layout == "dataset" else output / "labels"
        )
        required = [
            folder / f"{case_id}{suffix}.npy"
            for suffix in ("", "_stenosis", "_stenosis_raw")
        ]
        images = folder if a.layout == "dataset" else output / "images"
        for view in "ab":
            full_name = (
                f"image{case_id}{view}.png"
                if a.layout == "dataset"
                else f"image{i:04d}{view}.png"
            )
            required.append(images / full_name)
            required.extend(
                images / f"image{case_id}{view}{suffix}.png"
                for suffix in ("_stenosis", "_stenosis_centercrop")
            )
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"Incomplete generated case {case_id}")
        points = np.load(folder / f"{case_id}_stenosis.npy", allow_pickle=False)
        if (
            points.ndim != 2
            or points.shape[1] != 4
            or len(points) < 2
            or not np.isfinite(points).all()
            or np.any(points[:, 3] <= 0)
        ):
            raise ValueError(f"Invalid generated geometry {case_id}")
        radius = points[:, 3].astype(np.float32).astype(np.float64)
        severity = float(100 * (1 - radius.min() / radius[0]))
        cases.append(
            {
                "case_id": case_id,
                "split": a.split,
                "n_points": len(points),
                "r_inlet_m": float(radius[0]),
                "r_min_m": float(radius.min()),
                "severity_percent": severity,
                "grade": int(
                    np.searchsorted([25.0, 50.0, 70.0], severity, side="right")
                ),
            }
        )
    if a.layout == "dataset":
        manifest = {
            "schema_version": 1,
            "source_root": ".",
            "grade_formula": "100 * (1 - r_min / r_inlet)",
            "grade_thresholds_percent": [25.0, 50.0, 70.0],
            "cases": cases,
        }
        (output / "manifest_v2.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    files = [
        {"path": path.relative_to(output).as_posix(), "sha256": sha256(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    module = Path(__file__).resolve().parent
    sources = {
        path.name: sha256(path)
        for path in sorted(module.iterdir())
        if path.suffix in (".py", ".txt") and path.is_file()
    }
    record = {
        "seed": a.seed,
        "count_requested": a.count,
        "count_generated": len(cases),
        "layout": a.layout,
        "split": a.split,
        "schema_version": 1,
        "parameters": configs.parameters(),
        "source_sha256": sources,
        "control_points_sha256": sha256(
            Path(configs.control_point_path) / "RCA_ctrl_points.npy"
        ),
        "files": files,
    }
    (output / "generation_manifest.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "cases": len(cases), "files": len(files)}))


if __name__ == "__main__":
    main()
