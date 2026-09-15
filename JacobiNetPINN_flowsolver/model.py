"""JacobiNet PINN with one configurable RFF projection per coordinate domain."""

from __future__ import annotations

import torch
from torch import autograd, nn

if __package__ and __package__.startswith("JacobiNetPINN_flowsolver"):
    from .physics import PhysicsScales
    from .runtime import TORCH_TRAINING_DTYPE
    from .rff import BASE_SIGMA, FLOW_RFF_SPEC, DIRECTIONS_PER_DOMAIN, RFFSpec
else:
    from physics import PhysicsScales
    from runtime import TORCH_TRAINING_DTYPE
    from rff import BASE_SIGMA, FLOW_RFF_SPEC, DIRECTIONS_PER_DOMAIN, RFFSpec


class JacobiNet(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        expected = next(self.parameters()).dtype
        if values.dtype != expected:
            raise TypeError(
                f"JacobiNet input dtype {values.dtype} does not match {expected}"
            )
        return self.net(values)


def base_rff_matrices_from_state(
    *,
    legacy_cpu_rng_state: torch.Tensor,
    train_seed: int,
    rff_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create B independently while retaining the exact seed-99 reference directions.

    When the two seeds match, the generator starts from the captured legacy
    state (after JacobiNet construction), so q=1 is bitwise compatible with
    the reference model. Other RFF seeds use their own generator and do not affect MLP weight
    initialisation.
    """
    generator = torch.Generator(device="cpu")
    if int(rff_seed) == int(train_seed):
        generator.set_state(legacy_cpu_rng_state.detach().cpu())
    else:
        generator.manual_seed(int(rff_seed))
    xyz = (
        torch.randn(
            DIRECTIONS_PER_DOMAIN,
            3,
            generator=generator,
            dtype=TORCH_TRAINING_DTYPE,
        )
        / BASE_SIGMA
    )
    rs = (
        torch.randn(
            DIRECTIONS_PER_DOMAIN,
            2,
            generator=generator,
            dtype=TORCH_TRAINING_DTYPE,
        )
        / BASE_SIGMA
    )
    return xyz, rs


def consume_reference_rff_rng_draws() -> None:
    """Advance the global RNG for two random Fourier feature constructors."""
    torch.randn(DIRECTIONS_PER_DOMAIN, 3, dtype=TORCH_TRAINING_DTYPE)
    torch.randn(DIRECTIONS_PER_DOMAIN, 2, dtype=TORCH_TRAINING_DTYPE)


class RandomFourierFeature(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        sigma: float = BASE_SIGMA,
        *,
        base_matrix: torch.Tensor | None = None,
    ):
        super().__init__()
        if out_dim % 2:
            raise ValueError("RFF output dimension must be even")
        if sigma <= 0.0:
            raise ValueError("RFF sigma must be positive")
        expected_shape = (out_dim // 2, in_dim)
        if base_matrix is None:
            matrix = torch.randn(*expected_shape, dtype=TORCH_TRAINING_DTYPE) / float(
                sigma
            )
        else:
            if tuple(base_matrix.shape) != expected_shape:
                raise ValueError(
                    f"RFF base matrix must have shape {expected_shape}; "
                    f"got {tuple(base_matrix.shape)}"
                )
            if base_matrix.dtype != TORCH_TRAINING_DTYPE:
                raise TypeError("RFF base matrix must use the training dtype")
            if not torch.isfinite(base_matrix).all():
                raise ValueError("RFF base matrix must be finite")
            matrix = base_matrix.detach().clone()
        self.register_buffer("B", matrix)

    def _check_values(self, values: torch.Tensor) -> None:
        if values.dtype != self.B.dtype:
            raise TypeError(
                f"RFF input/buffer dtype mismatch: {values.dtype} vs {self.B.dtype}"
            )

    @staticmethod
    def _features(projection: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.cos(projection), torch.sin(projection)), dim=-1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self._check_values(values)
        projection = 2.0 * torch.pi * (values @ self.B.T)
        return self._features(projection)

    def forward_with_coordinate_multipliers(
        self,
        values: torch.Tensor,
        coordinate_multipliers: torch.Tensor,
    ) -> torch.Tensor:
        """Apply all row/coordinate frequencies in a single matrix multiply."""
        self._check_values(values)
        if (
            coordinate_multipliers.dtype != self.B.dtype
            or coordinate_multipliers.device != self.B.device
        ):
            raise TypeError("RFF multipliers must match the B buffer")
        if coordinate_multipliers.shape != self.B.shape:
            raise ValueError("RFF multipliers must have the same shape as B")
        if not torch.isfinite(coordinate_multipliers).all():
            raise ValueError("RFF multipliers must be finite")
        if not torch.all(coordinate_multipliers > 0.0):
            raise ValueError("RFF multipliers must be positive")
        projection = 2.0 * torch.pi * (values @ (self.B * coordinate_multipliers).T)
        return self._features(projection)


class NetPINN(nn.Module):

    def __init__(
        self,
        jacobinet: JacobiNet,
        scales: PhysicsScales,
        *,
        hidden_dim_velocity: int = 128,
        hidden_dim_pressure: int = 128,
        rff_dim_xyz: int = 64,
        rff_dim_rs: int = 64,
        rff_spec: RFFSpec = FLOW_RFF_SPEC,
        base_B_xyz: torch.Tensor | None = None,
        base_B_rs: torch.Tensor | None = None,
        train_seed: int = 99,
        rff_seed: int = 99,
    ):
        super().__init__()
        if rff_dim_xyz != 64 or rff_dim_rs != 64:
            raise ValueError("The model requires 32 xyz and 32 rs directions")
        self.jacobinet = jacobinet
        self.reynolds_number = float(scales.reynolds_number)
        self.inlet_scale = float(scales.inlet_peak_velocity_nondim)
        self.rff_spec = rff_spec
        self.rff_xyz = RandomFourierFeature(
            3, rff_dim_xyz, BASE_SIGMA, base_matrix=base_B_xyz
        )
        self.rff_rs = RandomFourierFeature(
            2, rff_dim_rs, BASE_SIGMA, base_matrix=base_B_rs
        )
        xyz_rows = (
            torch.tensor(rff_spec.xyz_q_rows(), dtype=TORCH_TRAINING_DTYPE)
            .unsqueeze(1)
            .repeat(1, 3)
        )
        rs_rows = torch.tensor(rff_spec.rs_q_rows(), dtype=TORCH_TRAINING_DTYPE)
        self.register_buffer("xyz_frequency_multipliers", xyz_rows, persistent=False)
        self.register_buffer("rs_frequency_multipliers", rs_rows, persistent=False)
        self._xyz_uses_base_frequency = bool(torch.all(xyz_rows == 1.0))
        self._rs_uses_base_frequency = bool(torch.all(rs_rows == 1.0))
        self.register_buffer(
            "xyz_effective_B", self.rff_xyz.B * xyz_rows, persistent=False
        )
        self.register_buffer(
            "rs_effective_B", self.rff_rs.B * rs_rows, persistent=False
        )
        self.register_load_state_dict_post_hook(self._refresh_fourier_cache)
        fused = rff_dim_xyz + rff_dim_rs
        self.net_velocity = nn.Sequential(
            nn.Linear(fused, hidden_dim_velocity),
            nn.SiLU(),
            nn.Linear(hidden_dim_velocity, hidden_dim_velocity),
            nn.SiLU(),
            nn.Linear(hidden_dim_velocity, hidden_dim_velocity),
            nn.SiLU(),
            nn.Linear(hidden_dim_velocity, hidden_dim_velocity),
            nn.SiLU(),
            nn.Linear(hidden_dim_velocity, 3),
        )
        self.net_pressure = nn.Sequential(
            nn.Linear(fused, hidden_dim_pressure),
            nn.SiLU(),
            nn.Linear(hidden_dim_pressure, hidden_dim_pressure),
            nn.SiLU(),
            nn.Linear(hidden_dim_pressure, hidden_dim_pressure),
            nn.SiLU(),
            nn.Linear(hidden_dim_pressure, 1),
        )
        self.model_config = {
            "architecture": "shared_configurable_rff_v2",
            "hidden_dim_velocity": hidden_dim_velocity,
            "hidden_dim_pressure": hidden_dim_pressure,
            "rff_dim_xyz": rff_dim_xyz,
            "rff_dim_rs": rff_dim_rs,
            "base_sigma_xyz": BASE_SIGMA,
            "base_sigma_rs": BASE_SIGMA,
            "shared_projection_count": {"xyz": 1, "rs": 1},
            "rff_spec_id": rff_spec.spec_id,
            "rff_spec_sha256": rff_spec.sha256(),
            "rff_spec": rff_spec.to_dict(),
            "train_seed": int(train_seed),
            "rff_seed": int(rff_seed),
        }

    def _refresh_fourier_cache(self, module, incompatible_keys) -> None:
        """Rebuild non-persistent projections whenever a state dict is loaded."""
        with torch.no_grad():
            self.xyz_effective_B.copy_(self.rff_xyz.B * self.xyz_frequency_multipliers)
            self.rs_effective_B.copy_(self.rff_rs.B * self.rs_frequency_multipliers)

    def _encode_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        if self._xyz_uses_base_frequency:
            return self.rff_xyz(xyz)
        return self.rff_xyz._features(2.0 * torch.pi * (xyz @ self.xyz_effective_B.T))

    def _encode_rs(self, rs: torch.Tensor) -> torch.Tensor:
        if self._rs_uses_base_frequency:
            return self.rff_rs(rs)
        return self.rff_rs._features(2.0 * torch.pi * (rs @ self.rs_effective_B.T))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.dtype != self.rff_xyz.B.dtype:
            raise TypeError(
                f"PINN input dtype {values.dtype} does not match model dtype {self.rff_xyz.B.dtype}"
            )
        xyz = values[:, :3]
        rs = self.jacobinet(xyz)
        radial = rs[:, 0:1]
        axial = rs[:, 1:2]
        shared_features = torch.cat((self._encode_xyz(xyz), self._encode_rs(rs)), dim=1)
        velocity_raw = self.net_velocity(shared_features)
        pressure_raw = self.net_pressure(shared_features)
        velocity_correction = radial * axial * velocity_raw
        u = velocity_correction[:, 0:1]
        v = velocity_correction[:, 1:2]
        w = velocity_correction[:, 2:3] + radial * (1.0 - axial) * self.inlet_scale
        velocity = torch.cat((u, v, w), dim=1)
        pressure = (1.0 - axial) * pressure_raw
        return torch.cat((velocity, pressure), dim=1)


def pde_losses(
    model: NetPINN,
    interior_points: torch.Tensor,
    *,
    build_parameter_graph: bool = True,
    spatial_input_only: bool = False,
    derivative_mode: str = "scalar_vjp",
) -> dict[str, torch.Tensor]:
    """Evaluate the four PDE components.

    Training needs a differentiable graph through the second spatial
    derivatives so that ``total_loss.backward()`` can reach model parameters.
    Validation only needs numerical residual values; disabling that final graph
    avoids the expensive parameter-differentiable Laplacian graph.
    """
    if interior_points.dtype != model.rff_xyz.B.dtype:
        raise TypeError(
            f"PDE/autodiff input dtype {interior_points.dtype} does not match model dtype {model.rff_xyz.B.dtype}"
        )
    derivative_source = (
        interior_points[:, :3] if spatial_input_only else interior_points
    )
    values = derivative_source.detach().clone().requires_grad_(True)
    output = model(values)
    u, v, w, pressure = (output[:, index : index + 1] for index in range(4))
    supported_derivative_modes = {"scalar_vjp", "batched_second_vjp"}
    if derivative_mode not in supported_derivative_modes:
        raise ValueError(f"Unsupported derivative_mode: {derivative_mode}")
    batched_mode = derivative_mode != "scalar_vjp"
    batched_second = derivative_mode == "batched_second_vjp"
    if batched_mode:
        if values.shape[1] != 3:
            raise ValueError(f"{derivative_mode} requires spatial_input_only=True")
        component_seeds = torch.eye(3, dtype=values.dtype, device=values.device)[
            :, None, :
        ].expand(3, len(values), 3)
    grad_u = autograd.grad(u, values, torch.ones_like(u), create_graph=True)[0]
    grad_v = autograd.grad(v, values, torch.ones_like(v), create_graph=True)[0]
    grad_w = autograd.grad(w, values, torch.ones_like(w), create_graph=True)[0]
    grad_p = autograd.grad(
        pressure, values, torch.ones_like(pressure), create_graph=True
    )[0]
    u_x, u_y, u_z = (grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3])
    v_x, v_y, v_z = (grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3])
    w_x, w_y, w_z = (grad_w[:, 0:1], grad_w[:, 1:2], grad_w[:, 2:3])
    p_x, p_y, p_z = (grad_p[:, 0:1], grad_p[:, 1:2], grad_p[:, 2:3])
    if batched_second:
        laplacians = []
        for velocity_gradient in (grad_u, grad_v, grad_w):
            hessian_rows = autograd.grad(
                velocity_gradient,
                values,
                grad_outputs=component_seeds,
                is_grads_batched=True,
                create_graph=build_parameter_graph,
                retain_graph=True,
            )[0]
            laplacians.append(
                (
                    hessian_rows[0, :, 0]
                    + hessian_rows[1, :, 1]
                    + hessian_rows[2, :, 2]
                ).unsqueeze(1)
            )
        lap_u, lap_v, lap_w = laplacians
    else:

        def second(first: torch.Tensor, coordinate: int) -> torch.Tensor:
            return autograd.grad(
                first,
                values,
                grad_outputs=torch.ones_like(first),
                create_graph=build_parameter_graph,
                retain_graph=True,
            )[0][:, coordinate : coordinate + 1]

        lap_u = second(u_x, 0) + second(u_y, 1) + second(u_z, 2)
        lap_v = second(v_x, 0) + second(v_y, 1) + second(v_z, 2)
        lap_w = second(w_x, 0) + second(w_y, 1) + second(w_z, 2)

    def residual_values() -> dict[str, torch.Tensor]:
        inverse_re = 1.0 / model.reynolds_number
        residual_u = u * u_x + v * u_y + w * u_z + p_x - inverse_re * lap_u
        residual_v = u * v_x + v * v_y + w * v_z + p_y - inverse_re * lap_v
        residual_w = u * w_x + v * w_y + w * w_z + p_z - inverse_re * lap_w
        continuity = u_x + v_y + w_z
        return {
            "momentum_u": torch.mean(residual_u.square()),
            "momentum_v": torch.mean(residual_v.square()),
            "momentum_w": torch.mean(residual_w.square()),
            "continuity": torch.mean(continuity.square()),
        }

    if build_parameter_graph:
        return residual_values()
    with torch.no_grad():
        return residual_values()


def boundary_losses(
    model: NetPINN,
    inlet: torch.Tensor,
    outlet: torch.Tensor,
    wall: torch.Tensor,
) -> dict[str, torch.Tensor]:
    inlet_output = model(inlet)
    target = torch.zeros_like(inlet_output[:, :3])
    target[:, 2] = inlet[:, 3] * model.inlet_scale
    outlet_pressure = model(outlet)[:, 3]
    wall_velocity = model(wall)[:, :3]
    return {
        "inlet": torch.mean((inlet_output[:, :3] - target).square()),
        "outlet": torch.mean(outlet_pressure.square()),
        "wall": torch.mean(wall_velocity.square()),
    }
