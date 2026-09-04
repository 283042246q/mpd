#!/usr/bin/env python3
"""Generate collision-free Marvin bimanual Warehouse trajectories.

This is a Marvin-only companion to ``generate_trajectories.py``.  It keeps the
legacy Panda generator untouched and plans the complete 14-DoF state with
PyBullet/OMPL.  The default schedule is 50% random-to-placement and 50%
placement-to-different-placement, with simultaneous:left-only:right-only =
3:1:1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from pb_ompl.pb_ompl import PbOMPLRobot
from scripts.generate_data.generate_trajectories import GenerateDataOMPL
from torch_robotics.torch_kinematics_tree.utils.files import get_configs_path
from torch_robotics.torch_planning_objectives.fields.distance_fields import CollisionObjectDistanceField


JOINT_NAMES = tuple([f"Joint{i}_L" for i in range(1, 8)] + [f"Joint{i}_R" for i in range(1, 8)])
ARM_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}
MODE_SCHEDULE = ("dual_independent", "dual_independent", "dual_independent", "left_only", "right_only")
DEFAULT_REGIONS = {
    "table": {"x": [0.25, 0.85], "y": [-0.22, 0.22], "z": [0.08, 0.18]},
    "left_cabinet": {"x": [0.22, 0.82], "y": [0.50, 0.72], "z": [0.10, 0.72]},
    "right_cabinet": {"x": [0.22, 0.82], "y": [-0.72, -0.50], "z": [0.10, 0.72]},
}
ARM_REGIONS = {
    "left": ("table", "left_cabinet"),
    "right": ("table", "right_cabinet"),
}


def _as_region(value):
    if isinstance(value, dict) and "translation" in value:
        value = value["translation"]
    if not isinstance(value, dict):
        raise ValueError("region must be a mapping")
    result = {}
    for axis in ("x", "y", "z"):
        bounds = value.get(axis)
        if isinstance(bounds, list) and len(bounds) == 1 and isinstance(bounds[0], list):
            bounds = bounds[0]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"region {axis} must contain [low, high]")
        result[axis] = [float(bounds[0]), float(bounds[1])]
    return result


def _load_regions(config):
    regions = {name: dict(bounds) for name, bounds in DEFAULT_REGIONS.items()}
    for name, value in (config.get("placement_regions") or {}).items():
        regions[name] = _as_region(value)
    required = set(DEFAULT_REGIONS)
    missing = required.difference(regions)
    if missing:
        raise ValueError(f"missing placement regions: {sorted(missing)}")
    return regions


class MarvinWarehouseGenerator:
    def __init__(self, config, seed):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.regions = _load_regions(config)
        self.worker = GenerateDataOMPL(
            env_id="EnvWarehouseMarvinBimanual",
            robot_id="RobotMarvinBimanual",
            planner=config.get("planner", "RRTConnect"),
            min_distance_robot_env=float(config.get("min_distance_robot_env", 0.02)),
            pybullet_mode="DIRECT",
            debug=False,
            gripper=False,
        )
        self.interface = self.worker.pbompl_interface
        self.robot = self.worker.robot_pbompl
        self.torch_robot = self.worker.robot_tr
        self.torch_object_field = CollisionObjectDistanceField(
            self.torch_robot,
            df_obj_list_fn=self.worker.env_tr.get_df_obj_list,
            link_margins_for_object_collision_checking_tensor=self.torch_robot.link_collision_spheres_radii,
            cutoff_margin=float(config.get("min_distance_robot_env", 0.02)),
            tensor_args=self.torch_robot.tensor_args,
        )
        self._configure_marvin_self_collision_pairs()
        self.right_robot = PbOMPLRobot(
            self.worker.pybullet_client,
            self.robot.id,
            urdf_path=self.worker.robot_tr.robot_urdf_file,
            link_name_ee="flange_R",
        )

    def _configure_marvin_self_collision_pairs(self):
        """Use the migrated CuRobo link-pair contract, not generic all-pairs."""
        pair_file = Path(get_configs_path()) / "marvin" / "self_collision_pairs.yaml"
        if not pair_file.is_file():
            return
        config = yaml.safe_load(pair_file.read_text()) or {}
        link_indices = self.robot._link_name_to_index
        pairs = []
        for first, second in config.get("pairs", []):
            disabled = {
                tuple(pair)
                for pair in self.config.get("disabled_self_collision_pairs", [])
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            }
            if (first, second) in disabled or (second, first) in disabled:
                continue
            if first not in link_indices or second not in link_indices:
                continue
            pairs.append((link_indices[first], link_indices[second]))
        if pairs:
            self.interface.check_link_pairs = pairs

    def close(self):
        self.worker.terminate()

    def _random_valid_state(self):
        # Marvin's zero configuration is a valid collision-free seed.  Try a
        # small deterministic cloud around it before uniform sampling; the
        # intersection of the mesh and sphere models is intentionally narrow.
        home = np.zeros(14, dtype=np.float64)
        for scale in (0.0, 0.05, 0.10, 0.20, 0.35):
            for _ in range(20):
                candidate = home + self.rng.normal(0.0, scale, size=14)
                candidate = np.clip(candidate, self.robot.joint_bounds_low_np, self.robot.joint_bounds_high_np)
                if self.interface.is_state_valid(candidate, check_bounds=True) and self._torch_state_valid(candidate):
                    return candidate
        for _ in range(int(self.config.get("state_sample_tries", 500))):
            candidate = np.asarray(self.robot.get_random_joint_position(), dtype=np.float64)
            if self.interface.is_state_valid(candidate, check_bounds=True) and self._torch_state_valid(candidate):
                return candidate
        raise RuntimeError(
            "failed to find a state valid in both PyBullet and torch collision models; "
            "increase state_sample_tries or inspect the Marvin collision assets"
        )

    def _torch_state_valid(self, q):
        q_tensor = torch.as_tensor(q, dtype=self.torch_robot.tensor_args["dtype"], device=self.torch_robot.tensor_args["device"])
        if q_tensor.ndim == 1:
            q_tensor = q_tensor.unsqueeze(0)
        collision_positions = self.torch_robot.fk_map_collision(q_tensor)
        self_collision = self.torch_robot.df_collision_self.compute_cost(
            q_tensor, collision_positions, field_type="occupancy"
        )
        object_collision = self.torch_object_field.compute_cost(
            q_tensor, collision_positions, field_type="occupancy"
        )
        return not bool((self_collision | object_collision).any().detach().cpu().item())

    def _sample_position(self, region_name):
        region = self.regions[region_name]
        return np.asarray([self.rng.uniform(*region[axis]) for axis in ("x", "y", "z")], dtype=np.float64)

    def _target_state(self, q_reference, arm, region_name):
        """Sample a collision-free joint state whose EE lies in a placement region."""
        q_reference = np.asarray(q_reference, dtype=np.float64).copy()
        solver = self.robot if arm == "left" else self.right_robot
        arm_slice = ARM_SLICES[arm]
        other_slice = ARM_SLICES["right" if arm == "left" else "left"]
        region = self.regions[region_name]
        for _ in range(int(self.config.get("ik_tries", 30)) * 100):
            q_candidate = q_reference.copy()
            q_candidate[arm_slice] = self.rng.uniform(
                self.robot.joint_bounds_low_np[arm_slice], self.robot.joint_bounds_high_np[arm_slice]
            )
            solver.set_state(q_candidate)
            position = np.asarray(solver.get_ee_pose(q_candidate)[0], dtype=np.float64)
            inside = all(region[axis][0] <= position[index] <= region[axis][1] for index, axis in enumerate(("x", "y", "z")))
            if not inside or not self.interface.is_state_valid(q_candidate, check_bounds=True) or not self._torch_state_valid(q_candidate):
                continue
            if np.linalg.norm(q_candidate[arm_slice] - q_reference[arm_slice]) < float(
                self.config.get("min_active_joint_delta", 0.08)
            ):
                continue
            q_candidate[other_slice] = q_reference[other_slice]
            if self.interface.is_state_valid(q_candidate, check_bounds=True) and self._torch_state_valid(q_candidate):
                return q_candidate
        return None

    def _arm_region(self, arm, region_name=None):
        choices = ARM_REGIONS[arm]
        if region_name is None:
            return choices[int(self.rng.integers(len(choices)))]
        if region_name not in choices:
            raise ValueError(f"{region_name} is not a safe default region for {arm} arm")
        return region_name

    def _sample_endpoint(self, q_reference, mode, region_names):
        q = np.asarray(q_reference, dtype=np.float64).copy()
        active_arms = ("left", "right") if mode == "dual_independent" else (mode.split("_")[0],)
        for arm in active_arms:
            target = self._target_state(q, arm, self._arm_region(arm, region_names.get(arm)))
            if target is None:
                return None
            q = target
        return q if self.interface.is_state_valid(q, check_bounds=True) else None

    def _sample_task(self, mode, direction):
        q_start = self._random_valid_state()
        if direction == "random_to_placement":
            source_regions = {"left": "random", "right": "random"}
            goal_regions = {}
            for arm in ("left", "right"):
                if mode == "left_only" and arm == "right":
                    continue
                if mode == "right_only" and arm == "left":
                    continue
                goal_regions[arm] = self._arm_region(arm)
            q_goal = self._sample_endpoint(q_start, mode, goal_regions)
        else:
            source_regions = {arm: self._arm_region(arm) for arm in ("left", "right")}
            q_start = self._sample_endpoint(q_start, mode, source_regions)
            if q_start is None:
                return None
            goal_regions = {}
            for arm in ("left", "right"):
                if mode == "left_only" and arm == "right":
                    continue
                if mode == "right_only" and arm == "left":
                    continue
                choices = [name for name in ARM_REGIONS[arm] if name != source_regions[arm]]
                goal_regions[arm] = choices[int(self.rng.integers(len(choices)))]
            q_goal = self._sample_endpoint(q_start, mode, goal_regions)
        if q_goal is None:
            return None
        active_arms = ("left", "right") if mode == "dual_independent" else (mode.split("_")[0],)
        if any(
            np.linalg.norm(q_goal[ARM_SLICES[arm]] - q_start[ARM_SLICES[arm]])
            < float(self.config.get("min_active_joint_delta", 0.08))
            for arm in active_arms
        ):
            return None
        return q_start, q_goal, source_regions, goal_regions

    def generate(self, num_trajectories):
        paths = []
        metadata = []
        max_attempts = int(self.config.get("max_attempts_per_trajectory", 30))
        interpolate_num = int(self.config.get("interpolate_num", 128))
        planner_time = float(self.config.get("planner_allowed_time", 10.0))
        direction_cutoff = int(np.ceil(num_trajectories * float(self.config.get("random_to_placement_fraction", 0.5))))
        max_skipped_tasks = int(self.config.get("max_skipped_tasks", max(num_trajectories, 100)))
        skipped_tasks = 0
        while len(paths) < num_trajectories:
            task_id = len(paths)
            mode = MODE_SCHEDULE[task_id % len(MODE_SCHEDULE)]
            direction = "random_to_placement" if task_id < direction_cutoff else "placement_to_placement"
            solved = None
            for _ in range(max_attempts):
                sampled = self._sample_task(mode, direction)
                if sampled is None:
                    continue
                q_start, q_goal, source_regions, goal_regions = sampled
                started = time.perf_counter()
                result = self.interface.plan_start_goal(
                    q_start,
                    q_goal,
                    allowed_time=planner_time,
                    interpolate_num=interpolate_num,
                    simplify_path=bool(self.config.get("simplify_path", True)),
                    fit_bspline=False,
                    debug=False,
                )
                elapsed = time.perf_counter() - started
                if (
                    result.get("success")
                    and result.get("sol_path") is not None
                    and len(result["sol_path"]) > 1
                    and self._torch_state_valid(np.asarray(result["sol_path"]))
                ):
                    solved = (result["sol_path"], q_start, q_goal, source_regions, goal_regions, elapsed)
                    break
            if solved is None:
                skipped_tasks += 1
                print(
                    f"skipping task ({mode}, {direction}): no exact path after "
                    f"{max_attempts} retries ({skipped_tasks}/{max_skipped_tasks} skipped)"
                )
                if skipped_tasks > max_skipped_tasks:
                    raise RuntimeError(
                        f"unable to generate {num_trajectories} trajectories: "
                        f"skipped more than {max_skipped_tasks} tasks"
                    )
                continue
            path, q_start, q_goal, source_regions, goal_regions, elapsed = solved
            paths.append(np.asarray(path, dtype=np.float32))
            metadata.append(
                {
                    "task_id": task_id,
                    "task_mode": mode,
                    "direction": direction,
                    "source_region_left": source_regions.get("left", "random"),
                    "source_region_right": source_regions.get("right", "random"),
                    "goal_region_left": goal_regions.get("left", "inactive"),
                    "goal_region_right": goal_regions.get("right", "inactive"),
                    "planning_time": elapsed,
                    "q_start": q_start,
                    "q_goal": q_goal,
                }
            )
            print(f"generated {len(paths)}/{num_trajectories}: {mode}, {direction}")
        return paths, metadata


def _write_dataset(output, config, paths, metadata, seed):
    output.mkdir(parents=True, exist_ok=True)
    path_array = np.stack(paths).astype(np.float32)
    n = len(metadata)
    active_masks = np.ones((n, 14), dtype=np.bool_)
    for i, item in enumerate(metadata):
        if item["task_mode"] == "left_only":
            active_masks[i, 7:] = False
        elif item["task_mode"] == "right_only":
            active_masks[i, :7] = False
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output / "dataset_merged.hdf5", "w") as handle:
        handle.create_dataset("sol_path", data=path_array, compression="gzip")
        handle.create_dataset("q_start", data=np.stack([item["q_start"] for item in metadata]))
        handle.create_dataset("q_goal", data=np.stack([item["q_goal"] for item in metadata]))
        handle.create_dataset("task_id", data=np.arange(n, dtype=np.int64))
        handle.create_dataset("task_mode", data=[item["task_mode"] for item in metadata], dtype=string_dtype)
        handle.create_dataset("direction", data=[item["direction"] for item in metadata], dtype=string_dtype)
        handle.create_dataset("active_joint_mask", data=active_masks)
        for key in ("source_region_left", "source_region_right", "goal_region_left", "goal_region_right"):
            handle.create_dataset(key, data=[item[key] for item in metadata], dtype=string_dtype)
        handle.create_dataset("planning_time", data=np.asarray([item["planning_time"] for item in metadata]))
        handle.attrs["joint_names"] = np.asarray(JOINT_NAMES, dtype=object)
        handle.attrs["task_ratio"] = "dual_independent:left_only:right_only=3:1:1"
        handle.attrs["direction_ratio"] = "random_to_placement:placement_to_placement=1:1"

    args = {
        "env_id": "EnvWarehouseMarvinBimanual",
        "robot_id": "RobotMarvinBimanual",
        "min_distance_robot_env": float(config.get("min_distance_robot_env", 0.02)),
        "planner": config.get("planner", "RRTConnect"),
        "joint_names": list(JOINT_NAMES),
        "task_family": "independent",
        "task_mode": "dual_independent",
        "num_trajectories": n,
        "seed": int(seed),
    }
    (output / "args.yaml").write_text(yaml.safe_dump(args, sort_keys=False))
    manifest = {
        "schema": "marvin_bimanual_warehouse_dataset/v1",
        "robot_model": "marvin_bimanual",
        "environment": "EnvWarehouseMarvinBimanual",
        "joint_names": list(JOINT_NAMES),
        "num_trajectories": n,
        "task_counts": {mode: sum(item["task_mode"] == mode for item in metadata) for mode in MODE_SCHEDULE},
        "direction_counts": {
            direction: sum(item["direction"] == direction for item in metadata)
            for direction in ("random_to_placement", "placement_to_placement")
        },
        "dataset_sha256": hashlib.sha256((output / "dataset_merged.hdf5").read_bytes()).hexdigest(),
    }
    (output / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (output / "generation_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "generation_summary.json").write_text(json.dumps(manifest, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--num-trajectories", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text()) or {}
    if config.get("env_id", "EnvWarehouseMarvinBimanual") != "EnvWarehouseMarvinBimanual":
        raise ValueError("this generator only supports EnvWarehouseMarvinBimanual")
    if config.get("robot_id", "RobotMarvinBimanual") != "RobotMarvinBimanual":
        raise ValueError("this generator only supports RobotMarvinBimanual")
    num_trajectories = int(args.num_trajectories or config.get("num_trajectories", 1000))
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    output = args.output_dir or Path(config.get("output_dir", "data_trajectories/EnvWarehouse-RobotMarvinBimanual-independent-v1"))
    if args.dry_run:
        print(yaml.safe_dump({"output_dir": str(output), "num_trajectories": num_trajectories, "seed": seed, "regions": _load_regions(config)}))
        return 0
    generator = MarvinWarehouseGenerator(config, seed)
    try:
        paths, metadata = generator.generate(num_trajectories)
    finally:
        generator.close()
    _write_dataset(output, config, paths, metadata, seed)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
