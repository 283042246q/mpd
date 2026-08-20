"""Phase-4 MPD engine with a resident static SDF and dynamic local SDFs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mpd.inference.dynamic_collision import (
    DynamicWorldError,
    FixedCapacityDynamicWorld,
    StaticDynamicCollisionField,
)
from scripts.runtime.runtime_engine import MpdRuntimeEngine, PlanArtifacts


class DynamicMpdRuntimeEngine(MpdRuntimeEngine):
    """A separate Phase-4 entry; the Phase-3 engine remains unchanged."""

    def __init__(
        self,
        config_path: Path,
        runtime_output_root: Path,
        device_text: str = "cuda:0",
        state_callback: Callable[[str], None] | None = None,
        *,
        max_dynamic_objects: int = 16,
        covariance_sigma: float = 3.0,
        process_acceleration_std_m_s2: float = 0.01,
    ) -> None:
        import torch

        super().__init__(
            config_path=config_path,
            runtime_output_root=runtime_output_root,
            device_text=device_text,
            state_callback=state_callback,
        )
        static_field = self.planning_task.get_collision_objects_field()
        if static_field is None:
            raise DynamicWorldError("dynamic runtime requires the resident static collision field")
        self.dynamic_world = FixedCapacityDynamicWorld(
            max_dynamic_objects,
            trajectory_duration_s=float(self.args_inference.trajectory_duration),
            tensor_args=self.tensor_args,
            covariance_sigma=covariance_sigma,
            process_acceleration_std_m_s2=process_acceleration_std_m_s2,
        )
        self.dynamic_field = StaticDynamicCollisionField(static_field, self.dynamic_world)
        self.planning_task.df_collision_objects = self.dynamic_field
        self.planning_task._collision_fields = [
            self.planning_task.df_collision_self,
            self.dynamic_field,
            self.planning_task.df_collision_ws_boundaries,
        ]

        if self.planner.cost_guide is not None:
            collision_cost = self.planner.cost_guide.costs.get("CostTaskSpaceCollisionObjects")
            if collision_cost is not None:
                collision_cost.cost.collision_objects_field = self.dynamic_field
            # Dynamic timing must retain the complete phase axis.  Phase 4 has
            # its own entry so Phase 3's production pruning settings are untouched.
            self.planner.cost_guide.gradient_pruning_enabled = False

        ranked = self.planner.dense_validation_config.get("ranked_early_exit", {})
        ranked["enabled"] = False
        self.planner.dense_validation_config["ranked_early_exit"] = ranked

        # Warm the new full-batch dynamic query/FK path with an empty world.
        # Version zero is internal only; the first external snapshot must be >0.
        self.dynamic_world.frame_id = "fr3_link0"
        self.dynamic_world.stamp_unix_ns = 1
        self.dynamic_world.valid_until_unix_ns = 2**63 - 1
        self.dynamic_world.plan_start_unix_ns = 1
        candidates = int(self.args_inference.n_trajectory_samples)
        template = torch.zeros(
            (candidates, *self.planner.dataset.control_points_dim),
            **self.tensor_args,
        )
        if self.planner.cost_guide is not None:
            self.planner.cost_guide.warmup(tuple(template.shape))
        dense_cfg = self.planner.dense_validation_config
        with torch.no_grad():
            self.planner.dense_validator.validate(
                control_points=template,
                num_points=dense_cfg["runtime_points"],
                check_environment=dense_cfg["check_environment"],
                check_self_collision=dense_cfg["check_self_collision"],
                check_joint_position=dense_cfg["check_joint_position"],
                check_joint_velocity=dense_cfg["check_joint_velocity"],
                check_joint_acceleration=dense_cfg["check_joint_acceleration"],
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def update_world(self, snapshot: dict[str, Any]) -> int:
        return self.dynamic_world.update(snapshot)

    def health(self) -> dict[str, Any]:
        response = super().health()
        response["dynamic_world"] = {
            "max_objects": self.dynamic_world.max_objects,
            "active_objects": int(self.dynamic_world.active.sum().item()),
            "world_version": self.dynamic_world.world_version,
            "frame_id": self.dynamic_world.frame_id,
            "stamp_unix_ns": self.dynamic_world.stamp_unix_ns,
            "valid_until_unix_ns": self.dynamic_world.valid_until_unix_ns,
            "orientation_model": "constant",
            "motion_model": "constant_velocity",
        }
        response["dense_validation"].update(
            fully_warmed=True,
            full_batch=True,
            pruning_used=False,
        )
        return response

    def plan(self, raw_request: dict[str, Any]) -> PlanArtifacts:
        import numpy as np
        import torch

        from torch_robotics.torch_kinematics_tree.geometrics.utils import (
            link_pos_from_link_tensor,
        )
        from torch_robotics.torch_utils.torch_utils import to_numpy

        world_version = raw_request.get("_dynamic_world_version")
        plan_start_unix_ns = raw_request.get("_trajectory_start_unix_ns")
        if isinstance(world_version, bool) or not isinstance(world_version, int):
            raise DynamicWorldError("dynamic plan requires an integer world version")
        if isinstance(plan_start_unix_ns, bool) or not isinstance(plan_start_unix_ns, int):
            raise DynamicWorldError("dynamic plan requires trajectory_start_unix_ns")
        self.dynamic_world.set_plan_start(
            plan_start_unix_ns,
            world_version=world_version,
        )
        artifacts = super().plan(raw_request)
        generated = int(artifacts.result_payload["candidates"]["generated"])
        checked = int(artifacts.result_payload["candidates"]["dense_checked"])
        complete = bool(artifacts.result_payload["candidates"]["dense_complete"])
        if checked != generated or not complete:
            raise DynamicWorldError("Phase-4 final DenseCheck did not evaluate every candidate")

        q_position = torch.as_tensor(artifacts.trajectory_arrays["positions"], **self.tensor_args)
        poses = torch.stack(
            self.planning_task.robot.fk_collision_spheres(q_position), dim=-3
        )
        sphere_positions = link_pos_from_link_tensor(poses)[..., :3]
        topk_q_position = torch.as_tensor(
            artifacts.trajectory_arrays["topk_positions"], **self.tensor_args
        )
        topk_poses = torch.stack(
            self.planning_task.robot.fk_collision_spheres(topk_q_position), dim=-3
        )
        topk_sphere_positions = link_pos_from_link_tensor(topk_poses)[..., :3]
        artifacts.trajectory_arrays.update(
            collision_sphere_positions=to_numpy(sphere_positions, dtype=np.float64),
            topk_collision_sphere_positions=to_numpy(
                topk_sphere_positions, dtype=np.float64
            ),
            collision_sphere_radii=to_numpy(
                self.planning_task.robot.link_collision_spheres_radii,
                dtype=np.float64,
            ),
        )
        artifacts.result_payload["dynamic_world"] = {
            "world_version": world_version,
            "frame_id": self.dynamic_world.frame_id,
            "snapshot_stamp_unix_ns": self.dynamic_world.stamp_unix_ns,
            "valid_until_unix_ns": self.dynamic_world.valid_until_unix_ns,
            "trajectory_start_unix_ns": plan_start_unix_ns,
            "active_object_ids": [object_id for object_id in self.dynamic_world.object_ids if object_id],
            "fixed_timing": True,
            "orientation_model": "constant",
            "motion_model": "constant_velocity",
            "dense_check_pruning_used": False,
        }
        return artifacts
