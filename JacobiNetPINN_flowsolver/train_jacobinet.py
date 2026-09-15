"""Train the fixed case-specific JacobiNet used by the released PINN.

The public protocol uses FP32, a fixed 50% internal subset, all boundaries,
CUDA Graph acceleration, hard_rmse < 1e-3, and a 100k-step cap.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import platform
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


if __package__ and __package__.startswith("JacobiNetPINN_flowsolver"):
    from .checkpoint import save_jacobinet_checkpoint, prepare_training_output
    from .dataset import load_training_points
    from .model import JacobiNet
    from .physics import LEGACY_MODE, build_physics_scales
    from .runtime import (
        EVALUATION_DTYPE_NAME,
        TORCH_TRAINING_DTYPE,
        TRAINING_DTYPE_NAME,
        assert_model_training_dtype,
    )
else:
    from checkpoint import save_jacobinet_checkpoint, prepare_training_output
    from dataset import load_training_points
    from model import JacobiNet
    from physics import LEGACY_MODE, build_physics_scales
    from runtime import (
        EVALUATION_DTYPE_NAME,
        TORCH_TRAINING_DTYPE,
        TRAINING_DTYPE_NAME,
        assert_model_training_dtype,
    )


METHOD_REVISION = "jacobinet_fp32_blocksync_cuda_graph_experimental_v1"
LR_START = 1.0e-3
LR_END = 1.0e-5


def cosine_lr(index: int, maximum: int, start: float, end: float) -> float:
    if maximum <= 0:
        return float(end)
    fraction = min(max(index / maximum, 0.0), 1.0)
    return float(end + 0.5 * (start - end) * (1.0 + np.cos(np.pi * fraction)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class JacobiTrainingCore:
    """CUDA Graph for the fixed-data JacobiNet training loss and gradients."""

    def __init__(
        self,
        *,
        model: JacobiNet,
        optimizer: torch.optim.Optimizer,
        internal_xyz: torch.Tensor,
        internal_target: torch.Tensor,
        boundary_xyz: torch.Tensor,
        boundary_target: torch.Tensor,
        inlet_slice: slice,
        outlet_slice: slice,
        wall_slice: slice,
        runtime: str,
    ) -> None:
        if runtime != "cuda_graph":
            raise ValueError(f"Unsupported runtime: {runtime}")
        self.model = model
        self.optimizer = optimizer
        self.internal_xyz = internal_xyz
        self.internal_target = internal_target
        self.boundary_xyz = boundary_xyz
        self.boundary_target = boundary_target
        self.inlet_slice = inlet_slice
        self.outlet_slice = outlet_slice
        self.wall_slice = wall_slice
        self.runtime = runtime
        # Use the specified reduction kernel; spelling this as
        # mean(square(error)) changes FP32 gradient rounding by ~1e-7.
        self.mse = nn.MSELoss()
        self.graph: torch.cuda.CUDAGraph | None = None
        self.outputs: dict[str, torch.Tensor] | None = None
        self.setup_seconds = 0.0
        self._capture()

    def _tensors(self) -> dict[str, torch.Tensor]:
        prediction_internal = self.model(self.internal_xyz)
        prediction_boundary = self.model(self.boundary_xyz)
        loss_internal = self.mse(prediction_internal, self.internal_target)
        loss_boundary = self.mse(prediction_boundary, self.boundary_target)
        radial, axial = prediction_boundary[:, 0], prediction_boundary[:, 1]
        loss_inlet = torch.mean(axial[self.inlet_slice].square())
        loss_outlet = torch.mean((axial[self.outlet_slice] - 1.0).square())
        loss_wall = torch.mean(radial[self.wall_slice].square())
        loss_hard = loss_inlet + loss_outlet + loss_wall
        loss = loss_internal + 10.0 * loss_boundary
        return {
            "loss": loss,
            "internal_mse": loss_internal,
            "boundary_mse": loss_boundary,
            "hard_mse": loss_hard,
        }

    def _capture(self) -> None:
        started = time.perf_counter()
        self.model.train()
        warmup_stream = torch.cuda.Stream(device=self.internal_xyz.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.internal_xyz.device))
        with torch.cuda.stream(warmup_stream):
            for index in range(3):
                self.optimizer.zero_grad(set_to_none=index == 0)
                outputs = self._tensors()
                outputs["loss"].backward()
        torch.cuda.current_stream(self.internal_xyz.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(self.internal_xyz.device)
        missing = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        if missing:
            raise RuntimeError(
                "CUDA Graph warmup missed gradients: " + ", ".join(missing)
            )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.optimizer.zero_grad(set_to_none=False)
            outputs = self._tensors()
            outputs["loss"].backward()
        torch.cuda.synchronize(self.internal_xyz.device)
        self.graph = graph
        self.outputs = outputs
        self.setup_seconds = time.perf_counter() - started

    def forward_backward(self) -> dict[str, torch.Tensor]:
        assert self.graph is not None and self.outputs is not None
        self.graph.replay()
        return self.outputs


class AdamRollback:
    """Preallocated block-start model and Adam tensor snapshot."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Adam) -> None:
        self.parameters = [parameter for parameter in model.parameters()]
        self.parameter_values = [
            parameter.detach().clone() for parameter in self.parameters
        ]
        self.optimizer = optimizer
        self.state_values: list[dict[str, Any]] = []
        for parameter in self.parameters:
            state = optimizer.state.get(parameter)
            if not state:
                raise RuntimeError(
                    "Adam state must be initialized before block snapshots"
                )
            self.state_values.append(
                {
                    name: (
                        value.detach().clone()
                        if torch.is_tensor(value)
                        else deepcopy(value)
                    )
                    for name, value in state.items()
                }
            )

    @torch.no_grad()
    def save(self) -> None:
        for target, parameter in zip(self.parameter_values, self.parameters):
            target.copy_(parameter)
        for parameter, saved in zip(self.parameters, self.state_values):
            state = self.optimizer.state[parameter]
            for name, target in saved.items():
                value = state[name]
                if torch.is_tensor(value):
                    target.copy_(value)
                else:
                    saved[name] = deepcopy(value)

    @torch.no_grad()
    def restore(self) -> None:
        for parameter, source in zip(self.parameters, self.parameter_values):
            parameter.copy_(source)
        for parameter, saved in zip(self.parameters, self.state_values):
            state = self.optimizer.state[parameter]
            for name, source in saved.items():
                value = state[name]
                if torch.is_tensor(value):
                    value.copy_(source)
                else:
                    state[name] = deepcopy(source)


@torch.no_grad()
def copy_model(target: nn.Module, source: nn.Module) -> None:
    target_state = target.state_dict()
    source_state = source.state_dict()
    for name in target_state:
        target_state[name].copy_(source_state[name])


def scalar_metrics(outputs: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        "total_loss": float(outputs["loss"].detach()),
        "internal_rmse": float(torch.sqrt(outputs["internal_mse"]).detach()),
        "boundary_rmse": float(torch.sqrt(outputs["boundary_mse"]).detach()),
        "hard_rmse": float(torch.sqrt(outputs["hard_mse"]).detach()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()
    args.max_steps = 100_000
    args.hard_tol = 1.0e-3
    args.history_frequency = 100
    args.internal_fraction = 0.5
    args.block_size = 100
    args.runtime = "cuda_graph"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.max_steps < 1 or args.history_frequency < 1 or args.block_size < 1:
        raise ValueError(
            "Step budgets, history frequency and block size must be positive"
        )
    if args.block_size != args.history_frequency:
        raise ValueError("Formal exact mode requires block_size == history_frequency")
    if not 0.0 < args.internal_fraction <= 1.0:
        raise ValueError("--internal-fraction must be in (0, 1]")

    case_root = args.case_root.resolve()
    case_id = args.case_id
    seed = args.seed
    device = torch.device("cuda:0")
    scales = build_physics_scales(case_root, LEGACY_MODE, case_id=case_id)
    data_path = case_root / "train_1e-4.xlsx"
    output_root = prepare_training_output(args.output_root, case_root)
    checkpoint = output_root / "jacobinet.pth"
    set_seed(seed)
    torch.cuda.empty_cache()
    points = load_training_points(data_path, scales, device)
    internal_xyz = points.internal[:, :3]
    internal_target = points.internal[:, 3:5]
    internal_total = len(internal_xyz)
    used_count = max(1, int(internal_total * args.internal_fraction))
    selected = torch.randperm(internal_total, device=device)[:used_count]
    internal_xyz = internal_xyz[selected]
    internal_target = internal_target[selected]
    boundary_xyz = torch.cat(
        (points.inlet[:, :3], points.outlet[:, :3], points.wall[:, :3]), dim=0
    )
    boundary_target = torch.cat(
        (points.inlet[:, 3:5], points.outlet[:, 3:5], points.wall[:, 3:5]), dim=0
    )
    inlet_slice = slice(0, len(points.inlet))
    outlet_slice = slice(len(points.inlet), len(points.inlet) + len(points.outlet))
    wall_slice = slice(len(points.inlet) + len(points.outlet), len(boundary_xyz))

    model = JacobiNet().to(device=device, dtype=TORCH_TRAINING_DTYPE)
    assert_model_training_dtype(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_START)
    core = JacobiTrainingCore(
        model=model,
        optimizer=optimizer,
        internal_xyz=internal_xyz,
        internal_target=internal_target,
        boundary_xyz=boundary_xyz,
        boundary_target=boundary_target,
        inlet_slice=inlet_slice,
        outlet_slice=outlet_slice,
        wall_slice=wall_slice,
        runtime=args.runtime,
    )
    best_model = deepcopy(model)
    for parameter in best_model.parameters():
        parameter.grad = None
    torch.cuda.reset_peak_memory_stats()

    def one_step(step: int) -> dict[str, torch.Tensor]:
        outputs = core.forward_backward()
        optimizer.step()
        optimizer.param_groups[0]["lr"] = cosine_lr(
            step, args.max_steps, LR_START, LR_END
        )
        return outputs

    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    stop_reason = "maximum_steps"
    completed = 0
    best_boundary = float("inf")
    final_values: dict[str, float] = {}
    best_training_state: dict[str, Any] | None = None

    # Step 1 initializes Adam and writes the initial training record.
    outputs = one_step(1)
    torch.cuda.synchronize(device)
    completed = 1
    final_values = scalar_metrics(outputs)
    lr = cosine_lr(1, args.max_steps, LR_START, LR_END)
    rows.append(
        {
            "step": 1,
            "learning_rate": lr,
            "elapsed_seconds": time.perf_counter() - start,
            **final_values,
        }
    )
    print(json.dumps(rows[-1]), flush=True)
    best_boundary = float(outputs["boundary_mse"].detach())
    copy_model(best_model, model)
    best_training_state = {
        "completed_steps": 1,
        "checkpoint_reason": "historical_best_boundary_at_record",
        "hard_tolerance": args.hard_tol,
        "metrics": dict(final_values),
    }
    if final_values["hard_rmse"] < args.hard_tol:
        stop_reason = "hard_constraint_early_stop"
    rollback = AdamRollback(model, optimizer)

    while stop_reason == "maximum_steps" and completed < args.max_steps:
        block_start = completed + 1
        block_end = min(
            args.max_steps,
            ((completed // args.block_size) + 1) * args.block_size,
        )
        if block_end < block_start:
            block_end = min(args.max_steps, completed + args.block_size)
        rollback.save()
        lr_before_block = optimizer.param_groups[0]["lr"]
        hard_values = torch.empty(
            block_end - block_start + 1, device=device, dtype=TORCH_TRAINING_DTYPE
        )
        last_outputs: dict[str, torch.Tensor] | None = None
        for step in range(block_start, block_end + 1):
            last_outputs = one_step(step)
            hard_values[step - block_start].copy_(
                torch.sqrt(last_outputs["hard_mse"].detach())
            )
        hard_cpu = hard_values.cpu()
        hit_indices = torch.nonzero(hard_cpu < args.hard_tol, as_tuple=False)
        if len(hit_indices):
            hit_step = block_start + int(hit_indices[0, 0])
            rollback.restore()
            optimizer.param_groups[0]["lr"] = lr_before_block
            for step in range(block_start, hit_step + 1):
                last_outputs = one_step(step)
            torch.cuda.synchronize(device)
            completed = hit_step
            assert last_outputs is not None
            final_values = scalar_metrics(last_outputs)
            stop_reason = "hard_constraint_early_stop"
        else:
            completed = block_end
            assert last_outputs is not None
            final_values = scalar_metrics(last_outputs)

        should_record = (
            completed % args.history_frequency == 0
            or stop_reason == "hard_constraint_early_stop"
            or completed == args.max_steps
        )
        if should_record:
            lr = cosine_lr(completed, args.max_steps, LR_START, LR_END)
            row = {
                "step": completed,
                "learning_rate": lr,
                "elapsed_seconds": time.perf_counter() - start,
                **final_values,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            boundary_value = float(last_outputs["boundary_mse"].detach())
            if boundary_value < best_boundary:
                best_boundary = boundary_value
                copy_model(best_model, model)
                best_training_state = {
                    "completed_steps": completed,
                    "checkpoint_reason": "historical_best_boundary_at_record",
                    "hard_tolerance": args.hard_tol,
                    "metrics": dict(final_values),
                }

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if stop_reason == "hard_constraint_early_stop":
        save_model = model
        save_state = {
            "completed_steps": completed,
            "checkpoint_reason": "historical_hard_constraint_early_stop",
            "hard_tolerance": args.hard_tol,
            "metrics": dict(final_values),
        }
    else:
        save_model = best_model
        if best_training_state is None:
            raise RuntimeError("Missing best recorded state")
        save_state = best_training_state
    save_jacobinet_checkpoint(checkpoint, save_model, scales, save_state)

    history_path = checkpoint.parent / "jacobinet_history.csv"
    pd.DataFrame(rows).to_csv(history_path, index=False)
    report = {
        "status": "completed",
        "method_revision": f"jacobinet_{TRAINING_DTYPE_NAME}_v2",
        "acceleration_revision": METHOD_REVISION,
        "case_id": case_id,
        "dtype": TRAINING_DTYPE_NAME,
        "autodiff_dtype": TRAINING_DTYPE_NAME,
        "precision_policy": "fp32_train_fp64_physics_eval_v1",
        "evaluation_dtype": EVALUATION_DTYPE_NAME,
        "device": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "seed": seed,
        "max_steps": args.max_steps,
        "completed_steps": completed,
        "stop_reason": stop_reason,
        "training_seconds": elapsed,
        "seconds_per_step": elapsed / max(completed, 1),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "runtime": {
            "mode": args.runtime,
            "block_size": args.block_size,
            "cuda_graph_setup_seconds": core.setup_seconds,
            "optimizer_captured": False,
            "scheduler_captured": False,
            "exact_first_threshold_rollback": True,
            "checkpoint_writes_during_loop": 0,
        },
        "points": {
            **points.counts(),
            "internal_used": used_count,
            "internal_fraction": args.internal_fraction,
        },
        "physics_scales_for_geometry": scales.to_dict(),
        "final_metrics": final_values,
        "best_recorded_boundary_mse": best_boundary,
        "checkpoint": str(checkpoint),
        "history": str(history_path),
        "objective": "historical MSE_internal + 10*MSE_boundary; float32 training and autodiff",
    }
    report_path = checkpoint.parent / "jacobinet_report.json"
    temporary = report_path.with_suffix(report_path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, report_path)
    print("report=" + str(report_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
