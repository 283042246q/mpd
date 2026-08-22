"""Inference-only monotone timing spline for Phase 5 planning.

The pretrained diffusion model continues to produce only spatial control
points.  This module owns the independent timing variables and uses
``u(s) = dt/ds`` throughout so the space-time chain rule is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from mpd.parametric_trajectory.trajectory_bspline import BSpline
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


@dataclass(frozen=True)
class TimingSplineEvaluation:
    phase: torch.Tensor
    log_density: torch.Tensor
    u: torch.Tensor
    u_s: torch.Tensor
    time_from_start: torch.Tensor
    duration: torch.Tensor
    q: Optional[torch.Tensor] = None
    dq: Optional[torch.Tensor] = None
    ddq: Optional[torch.Tensor] = None


class TimingSpline:
    """Fixed-basis positive timing density evaluated in batch.

    Timing control points have shape ``[..., K]``.  The first two and last
    two control points are equal for the initial Phase-5 representation,
    which fixes ``u_s`` to zero at both endpoints.  Callers optimize only the
    entries selected by :attr:`optimizable_mask` when endpoint timing must
    remain fixed.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        num_control_points: int = 8,
        degree: int = 3,
        num_phase_points: int = 128,
        u_min: float = 0.05,
        duration_min: float = 2.0,
        duration_max: float = 15.0,
        tensor_args=DEFAULT_TENSOR_ARGS,
    ) -> None:
        if num_control_points < 6:
            raise ValueError("timing spline requires at least 6 control points")
        if degree < 2 or degree >= num_control_points:
            raise ValueError("timing spline degree must be in [2, num_control_points)")
        if num_phase_points < 3:
            raise ValueError("timing spline requires at least 3 phase points")
        if not 0.0 < u_min < duration_min < duration_max:
            raise ValueError("expected 0 < u_min < duration_min < duration_max")

        self.num_control_points = int(num_control_points)
        self.degree = int(degree)
        self.num_phase_points = int(num_phase_points)
        self.u_min = float(u_min)
        self.duration_min = float(duration_min)
        self.duration_max = float(duration_max)
        self.tensor_args = dict(tensor_args)

        basis = BSpline(
            num_pts=self.num_control_points,
            degree=self.degree,
            num_T_pts=self.num_phase_points,
            **self.tensor_args,
        )
        self.basis = basis.N.squeeze(0)
        self.basis_derivative = basis.dN.squeeze(0)
        self.phase = torch.linspace(0.0, 1.0, self.num_phase_points, **self.tensor_args)
        self.quadrature = self._cumulative_trapezoid_matrix()
        self.optimizable_mask = torch.ones(
            self.num_control_points, dtype=torch.bool, device=self.phase.device
        )
        self.optimizable_mask[:2] = False
        self.optimizable_mask[-2:] = False

    def _cumulative_trapezoid_matrix(self) -> torch.Tensor:
        matrix = torch.zeros(
            self.num_phase_points,
            self.num_phase_points,
            **self.tensor_args,
        )
        ds = 1.0 / float(self.num_phase_points - 1)
        for row in range(1, self.num_phase_points):
            matrix[row, 0] = 0.5 * ds
            matrix[row, row] = 0.5 * ds
            if row > 1:
                matrix[row, 1:row] = ds
        return matrix

    @staticmethod
    def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
        return value + torch.log(-torch.expm1(-value))

    def linear_control_points(self, duration, *, batch_shape=()) -> torch.Tensor:
        """Return control points representing exactly ``t(s)=duration*s``."""

        duration = torch.as_tensor(duration, **self.tensor_args)
        if duration.ndim == 0 and batch_shape:
            duration = duration.expand(*batch_shape)
        if not torch.isfinite(duration).all().item():
            raise ValueError("duration contains NaN or Inf")
        if (duration <= self.u_min).any().item():
            raise ValueError("duration must be greater than u_min")
        value = self._inverse_softplus(duration - self.u_min)
        return value[..., None].expand(*value.shape, self.num_control_points).clone()

    def enforce_endpoint_derivatives(self, control_points: torch.Tensor) -> torch.Tensor:
        """Project endpoint pairs so the clamped spline has ``u_s(0)=u_s(1)=0``."""

        self._validate_control_points(control_points)
        projected = control_points.clone()
        projected[..., 1] = projected[..., 0]
        projected[..., -2] = projected[..., -1]
        return projected

    def _validate_control_points(self, control_points: torch.Tensor) -> None:
        if control_points.shape[-1:] != (self.num_control_points,):
            raise ValueError(
                f"timing control points must end in [{self.num_control_points}], "
                f"got {tuple(control_points.shape)}"
            )
        if control_points.device != self.phase.device:
            raise ValueError("timing control points use a different device than the fixed basis")
        if control_points.dtype != self.phase.dtype:
            raise ValueError("timing control points use a different dtype than the fixed basis")
        if not torch.isfinite(control_points).all().item():
            raise ValueError("timing control points contain NaN or Inf")

    def assert_endpoint_derivatives_fixed(self, control_points: torch.Tensor, *, atol=1e-8) -> None:
        self._validate_control_points(control_points)
        start_ok = torch.allclose(control_points[..., 0], control_points[..., 1], atol=atol, rtol=0.0)
        goal_ok = torch.allclose(control_points[..., -2], control_points[..., -1], atol=atol, rtol=0.0)
        if not start_ok or not goal_ok:
            raise ValueError("timing endpoint control-point pairs must be equal")

    def evaluate(
        self,
        control_points: torch.Tensor,
        *,
        q: Optional[torch.Tensor] = None,
        q_s: Optional[torch.Tensor] = None,
        q_ss: Optional[torch.Tensor] = None,
        require_duration_bounds: bool = False,
        require_fixed_endpoint_derivatives: bool = True,
    ) -> TimingSplineEvaluation:
        """Evaluate timing and optional space-time joint derivatives."""

        self._validate_control_points(control_points)
        if require_fixed_endpoint_derivatives:
            self.assert_endpoint_derivatives_fixed(control_points)

        log_density = torch.einsum("hk,...k->...h", self.basis, control_points)
        log_density_s = torch.einsum(
            "hk,...k->...h", self.basis_derivative, control_points
        )
        u = self.u_min + F.softplus(log_density)
        u_s = torch.sigmoid(log_density) * log_density_s
        time_from_start = torch.einsum("jh,...h->...j", self.quadrature, u)
        duration = time_from_start[..., -1]

        if require_duration_bounds and (
            (duration < self.duration_min).any().item()
            or (duration > self.duration_max).any().item()
        ):
            raise ValueError(
                f"timing duration must remain in [{self.duration_min}, {self.duration_max}] seconds"
            )
        if not (torch.diff(time_from_start, dim=-1) > 0.0).all().item():
            raise ValueError("timing spline did not produce strictly increasing time")

        supplied = (q is not None, q_s is not None, q_ss is not None)
        if any(supplied) and not all(supplied):
            raise ValueError("q, q_s and q_ss must be supplied together")
        dq = ddq = None
        if all(supplied):
            expected = control_points.shape[:-1] + (self.num_phase_points,)
            for name, value in (("q", q), ("q_s", q_s), ("q_ss", q_ss)):
                if value.shape[:-1] != expected:
                    raise ValueError(
                        f"{name} must have leading shape {expected}, got {tuple(value.shape)}"
                    )
                if value.device != self.phase.device or value.dtype != self.phase.dtype:
                    raise ValueError(f"{name} must match the timing basis dtype and device")
                if not torch.isfinite(value).all().item():
                    raise ValueError(f"{name} contains NaN or Inf")
            dq = q_s / u[..., None]
            ddq = q_ss / u[..., None].square() - q_s * u_s[..., None] / u[..., None].pow(3)

        return TimingSplineEvaluation(
            phase=self.phase,
            log_density=log_density,
            u=u,
            u_s=u_s,
            time_from_start=time_from_start,
            duration=duration,
            q=q,
            dq=dq,
            ddq=ddq,
        )

    def evaluate_spatial_control_points(
        self,
        timing_control_points: torch.Tensor,
        spatial_control_points: torch.Tensor,
        *,
        position_basis: torch.Tensor,
        velocity_basis: torch.Tensor,
        acceleration_basis: torch.Tensor,
        **kwargs,
    ) -> TimingSplineEvaluation:
        """Evaluate ``q/dq/ddq`` from fixed spatial spline basis tensors."""

        if spatial_control_points.shape[:-2] != timing_control_points.shape[:-1]:
            raise ValueError("spatial and timing candidate dimensions must match")
        basis_shape = (self.num_phase_points, spatial_control_points.shape[-2])
        for name, basis in (
            ("position_basis", position_basis),
            ("velocity_basis", velocity_basis),
            ("acceleration_basis", acceleration_basis),
        ):
            if basis.shape != basis_shape:
                raise ValueError(f"{name} must have shape {basis_shape}, got {tuple(basis.shape)}")
        q = torch.einsum("hk,...kd->...hd", position_basis, spatial_control_points)
        q_s = torch.einsum("hk,...kd->...hd", velocity_basis, spatial_control_points)
        q_ss = torch.einsum("hk,...kd->...hd", acceleration_basis, spatial_control_points)
        return self.evaluate(timing_control_points, q=q, q_s=q_s, q_ss=q_ss, **kwargs)
