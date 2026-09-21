"""Independent-diffusion prior adapter for rigid Marvin transport tasks."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.interpolate import BSpline

from pb_ompl.pb_ompl import fit_bspline_to_path
from scripts.generate_data.generate_marvin_warehouse_cooperative import (
    PHASE_IDS,
    phase_schedule,
)


PRIOR_MODES = ("direct_project", "reference_residual")


@dataclass
class CooperativeAudit:
    valid: bool
    failure_code: str | None
    max_closure_translation_m: float
    max_closure_rotation_rad: float
    checked_points: int


class CooperativePriorAdapter:
    """Build warm starts, project sampled splines, and run the rigid oracle."""

    def __init__(self, generator, planning_task, dataset, config):
        self.generator = generator
        self.planning_task = planning_task
        self.dataset = dataset
        self.config = dict(config)
        self.object_start = None
        self.object_goal = None
        self.q_start = None
        self.q_goal = None
        self.reference_proposal_kind = None
        self.reference_build_seconds = 0.0
        self.projection_seconds = 0.0
        self.projected_candidates = 0
        self.projection_failures = 0

    def set_task(self, object_start, object_goal, q_start, q_goal):
        self.object_start = np.asarray(object_start, dtype=float)
        self.object_goal = np.asarray(object_goal, dtype=float)
        self.q_start = np.asarray(q_start, dtype=float)
        self.q_goal = np.asarray(q_goal, dtype=float)

    def _learnable_from_spline(self, spline):
        import torch

        _, coefficients, _ = spline
        full = torch.as_tensor(
            np.asarray(coefficients).T,
            dtype=self.planning_task.q_pos_start.dtype,
            device=self.planning_task.q_pos_start.device,
        ).unsqueeze(0)
        learnable = self.planning_task.parametric_trajectory.remove_control_points_fn(full)
        return self.dataset.normalize_control_points(learnable)

    def build_reference_normalized(self):
        """Return one closure-valid learnable CP tensor for SDEdit initialization."""

        if self.object_start is None:
            raise RuntimeError("set_task must be called before building a reference")
        started = time.perf_counter()
        for proposal in self.generator.object_proposals(
            self.object_start, self.object_goal
        ):
            coarse = self.generator.continuous_paired_ik(
                proposal, self.q_start, self.q_goal
            )
            if coarse is None:
                continue
            spline = self.generator.optimize_bspline(
                coarse, proposal, self.q_start, self.q_goal
            )
            if spline is None:
                continue
            self.reference_proposal_kind = proposal.kind
            self.reference_build_seconds = time.perf_counter() - started
            return self._learnable_from_spline(spline)
        self.reference_build_seconds = time.perf_counter() - started
        raise RuntimeError("could not build a continuous paired-IK reference")

    def _project_waypoints(self, q_path):
        """Project the right arm to the rigid object inferred from the left arm."""

        projected = np.asarray(q_path, dtype=float).copy()
        projected[0], projected[-1] = self.q_start, self.q_goal
        for index in range(1, len(projected) - 1):
            left_pose = self.generator._pose(projected[index], "left").homogeneous
            world_object = left_pose @ np.linalg.inv(self.generator.object_to_left)
            right_target = world_object @ self.generator.object_to_right
            right = self.generator._solve_arm(
                "right", right_target, projected[index]
            )
            if right is None:
                return None
            projected[index, 7:] = right
        return projected

    def _fit(self, q_path):
        optimization = self.generator.config["joint_optimization"]
        return fit_bspline_to_path(
            q_path,
            bspline_degree=int(optimization["degree"]),
            bspline_num_control_points=int(optimization["control_points"]),
            bspline_zero_vel_at_start_and_goal=True,
            bspline_zero_acc_at_start_and_goal=True,
        )

    def project_normalized(self, normalized):
        """Alternately project dense waypoints and refit the 14-D B-spline."""

        import torch

        started = time.perf_counter()
        control_points = self.dataset.unnormalize_control_points(normalized)
        trajectory = self.planning_task.parametric_trajectory
        projection_points = int(self.config.get("points", 32))
        iterations = int(self.config.get("iterations", 1))
        output = normalized.detach().clone()
        for candidate in range(normalized.shape[0]):
            current = control_points[candidate : candidate + 1]
            fitted = None
            try:
                for _ in range(iterations):
                    q = trajectory.get_q_trajectory(
                        current,
                        None,
                        None,
                        get_type=("pos",),
                        get_time_representation=False,
                    )["pos"][0]
                    q_np = q.detach().cpu().numpy()
                    if len(q_np) != projection_points:
                        phase = np.linspace(0.0, 1.0, len(q_np))
                        target_phase = np.linspace(0.0, 1.0, projection_points)
                        q_np = np.stack(
                            [np.interp(target_phase, phase, q_np[:, j]) for j in range(14)],
                            axis=1,
                        )
                    projected = self._project_waypoints(q_np)
                    if projected is None:
                        raise RuntimeError("paired projection IK failed")
                    fitted = self._fit(projected)
                    current = self.dataset.unnormalize_control_points(
                        self._learnable_from_spline(fitted)
                    )
                output[candidate] = self._learnable_from_spline(fitted)[0]
                self.projected_candidates += 1
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                self.projection_failures += 1
        self.projection_seconds += time.perf_counter() - started
        return output.to(dtype=normalized.dtype, device=normalized.device)

    def audit_path(self, q_path) -> CooperativeAudit:
        """Adaptive closure, robot and phase-aware payload audit."""

        q = np.asarray(q_path, dtype=float)
        max_step = float(
            self.generator.config["validation"]["adaptive_max_joint_step"]
        )
        dense = [q[0]]
        for left, right in zip(q[:-1], q[1:]):
            subdivisions = max(
                1, int(np.ceil(np.max(np.abs(right - left)) / max_step))
            )
            for fraction in np.linspace(0.0, 1.0, subdivisions + 1)[1:]:
                dense.append(left + fraction * (right - left))
        dense = np.asarray(dense)
        translation, rotation = self.generator.closure_errors(dense)
        validation = self.generator.config["validation"]
        max_translation = float(translation.max())
        max_rotation = float(rotation.max())
        if max_translation > float(validation["closure_translation_tolerance"]):
            return CooperativeAudit(
                False, "CLOSED_CHAIN_TRANSLATION", max_translation, max_rotation, len(dense)
            )
        if max_rotation > np.deg2rad(
            float(validation["closure_rotation_tolerance_deg"])
        ):
            return CooperativeAudit(
                False, "CLOSED_CHAIN_ROTATION", max_translation, max_rotation, len(dense)
            )
        for actual_q, expected, label in (
            (dense[0], self.object_start, "START"),
            (dense[-1], self.object_goal, "GOAL"),
        ):
            actual = self.generator.object_state_from_q(actual_q)
            if np.linalg.norm(actual[:3] - expected[:3]) > float(
                validation["object_endpoint_translation_tolerance"]
            ):
                return CooperativeAudit(
                    False,
                    f"OBJECT_{label}_TRANSLATION",
                    max_translation,
                    max_rotation,
                    len(dense),
                )
            yaw_error = abs(
                np.arctan2(
                    np.sin(actual[3] - expected[3]),
                    np.cos(actual[3] - expected[3]),
                )
            )
            if yaw_error > np.deg2rad(
                float(validation["object_endpoint_rotation_tolerance_deg"])
            ):
                return CooperativeAudit(
                    False,
                    f"OBJECT_{label}_ROTATION",
                    max_translation,
                    max_rotation,
                    len(dense),
                )
        phases = phase_schedule(len(dense))
        for index, (state, phase) in enumerate(zip(dense, phases)):
            object_state = self.generator.object_state_from_q(state)
            if not self.generator.cooperative_valid(
                state,
                object_state,
                phase,
                torch_check=bool(validation["torch_collision"]),
                bullet_check=bool(validation["pybullet_collision"]),
            ):
                return CooperativeAudit(
                    False,
                    f"COOPERATIVE_COLLISION_AT_{index}",
                    max_translation,
                    max_rotation,
                    len(dense),
                )
        return CooperativeAudit(
            True, None, max_translation, max_rotation, len(dense)
        )

    def statistics(self):
        return {
            "reference_build_s": self.reference_build_seconds,
            "reference_proposal_kind": self.reference_proposal_kind,
            "projection_s": self.projection_seconds,
            "projected_candidates": self.projected_candidates,
            "projection_failures": self.projection_failures,
        }
