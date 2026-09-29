"""Resident OMPL ST-RRT* baseline for the ToDrawer dynamic-world contract."""

from __future__ import annotations

import math
from pathlib import Path
import time
import uuid

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from scripts.runtime.infer_once import (
    NoValidTrajectoryError,
    RequestValidationError,
    validate_request,
)
from scripts.runtime.runtime_engine import PlanArtifacts
from scripts.runtime.strrt_collision import StrrtCollisionWorld


class StrrtEngine:
    """One robot model and one immutable world snapshot per planning request."""

    def __init__(
        self,
        static_scene: str | Path,
        *,
        target_pose_xyzw: tuple[float, ...] | None = None,
        max_duration_s: float = 14.0,
        solve_budget_s: float = 2.0,
        edge_dt_s: float = 0.02,
        max_joint_step_rad: float = 0.04,
        planner_range: float = 1.5,
        worker_seed: int = 1,
        state_callback=None,
    ) -> None:
        import torch
        from torch_robotics.robots import RobotPanda

        self.instance_id = str(uuid.uuid4())
        self.max_duration_s = float(max_duration_s)
        self.solve_budget_s = float(solve_budget_s)
        self.edge_dt_s = float(edge_dt_s)
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.planner_range = float(planner_range)
        if min(self.max_duration_s, self.solve_budget_s, self.edge_dt_s,
               self.max_joint_step_rad, self.planner_range) <= 0.0:
            raise ValueError("ST-RRT* settings must be positive")
        if not 0 <= int(worker_seed) < 2**32 - 1:
            raise ValueError("worker_seed must be in [0, 2**32 - 1)")
        self.worker_seed = int(worker_seed)
        from ompl import util as ou
        ou.RNG.setSeed(self.worker_seed + 1)
        notify = state_callback or (lambda _state: None)
        notify("LOADING")
        self.robot = RobotPanda(tensor_args={"device": torch.device("cpu"), "dtype": torch.float32})
        self.dynamic_world = StrrtCollisionWorld(self.robot, static_scene)
        self._goal_cache: dict[tuple[float, ...], list[np.ndarray]] = {}
        if target_pose_xyzw is not None:
            self._ik_goals(tuple(float(x) for x in target_pose_xyzw))

    def health(self) -> dict:
        return {
            "planner": "strrtstar",
            "instance_id": self.instance_id,
            "worker_seed": self.worker_seed,
            "ompl_seed": self.worker_seed + 1,
            "world_version": self.dynamic_world.world_version,
            "collision_sphere_count": len(self.dynamic_world.radii),
            "static_obstacle_count": len(self.dynamic_world.static),
        }

    def update_world(self, snapshot: dict) -> int:
        return self.dynamic_world.update(snapshot)

    def _ik_goals(self, pose: tuple[float, ...]) -> list[np.ndarray]:
        import torch

        key = tuple(round(value, 5) for value in pose)
        if key in self._goal_cache:
            return self._goal_cache[key]
        if len(pose) != 7 or not np.isfinite(pose).all():
            raise RequestValidationError("Cartesian target must be a finite xyzw pose")
        target_rotation = Rotation.from_quat(pose[3:])
        lower, upper = self.dynamic_world.q_min, self.dynamic_world.q_max
        rng = np.random.default_rng(1)
        seeds = [np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])]
        seeds.extend(rng.uniform(lower + 0.1, upper - 0.1) for _ in range(15))

        def residual(q: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                transform = self.robot.get_EE_pose(torch.as_tensor(q[None], dtype=torch.float32))[0]
            transform = transform.detach().cpu().numpy()
            rotation_error = (Rotation.from_matrix(transform[:3, :3]).inv() * target_rotation).as_rotvec()
            return np.concatenate((transform[:3, 3] - pose[:3], 0.25 * rotation_error))

        goals = []
        for seed in seeds:
            solution = least_squares(
                residual,
                np.clip(seed, lower + 1e-5, upper - 1e-5),
                bounds=(lower, upper),
                diff_step=1e-3,
                max_nfev=100,
                ftol=1e-6,
                xtol=1e-6,
            )
            error = residual(solution.x)
            if (np.linalg.norm(error[:3]) <= 0.015
                    and np.linalg.norm(error[3:]) / 0.25 <= math.radians(3.0)
                    and self.dynamic_world.static_valid(solution.x)
                    and all(np.linalg.norm(solution.x - old) > 0.1 for old in goals)):
                goals.append(solution.x.copy())
            if len(goals) >= 4:
                break
        if not goals:
            raise NoValidTrajectoryError("Cartesian goal has no collision-free IK solution")
        self._goal_cache[key] = goals
        return goals

    @staticmethod
    def _path_points(path, space) -> tuple[np.ndarray, np.ndarray]:
        states = path.getStates()
        positions = np.asarray([[state[0][index] for index in range(7)] for state in states], dtype=np.float64)
        times = np.asarray([space.getStateTime(state) for state in states], dtype=np.float64)
        if len(times) < 2 or abs(times[0]) > 1e-7 or np.any(np.diff(times) <= 1e-7):
            raise NoValidTrajectoryError("ST-RRT* returned a degenerate time path")
        return positions, times

    def _solve(self, q_start: np.ndarray, goals: list[np.ndarray], start_unix_s: float,
               duration_s: float, budget_s: float) -> tuple[np.ndarray, np.ndarray]:
        from ompl import base as ob, geometric as og
        q_space = ob.RealVectorStateSpace(7)
        bounds = ob.RealVectorBounds(7)
        for index in range(7):
            bounds.setLow(index, float(self.dynamic_world.q_min[index]))
            bounds.setHigh(index, float(self.dynamic_world.q_max[index]))
        q_space.setBounds(bounds)
        # A conservative Euclidean speed bound; the motion validator also checks each joint.
        v_max = float(np.min(self.dynamic_world.dq_max))
        space = ob.SpaceTimeStateSpace(q_space, v_max)
        space.setTimeBounds(0.0, float(duration_s))
        si = ob.SpaceInformation(space)

        def valid(state) -> bool:
            q = np.asarray([state[0][index] for index in range(7)], dtype=np.float64)
            return self.dynamic_world.is_valid(q, start_unix_s + space.getStateTime(state))

        class TimedMotionValidator(ob.MotionValidator):
            def checkMotion(self, first, second):
                t0, t1 = space.getStateTime(first), space.getStateTime(second)
                delta_t = t1 - t0
                if delta_t <= 0.0:
                    return False
                a = np.asarray([first[0][i] for i in range(7)], dtype=np.float64)
                b = np.asarray([second[0][i] for i in range(7)], dtype=np.float64)
                if np.any(np.abs(b - a) > self_limits * delta_t + 1e-9):
                    return False
                count = max(2, math.ceil(delta_t / edge_dt),
                            math.ceil(float(np.max(np.abs(b - a))) / max_step))
                fractions = np.linspace(1.0 / count, 1.0, count)
                samples = a[None] + fractions[:, None] * (b - a)[None]
                times = start_unix_s + t0 + fractions * delta_t
                return self_world.samples_valid(samples, times)

        self_limits = self.dynamic_world.dq_max
        self_world = self.dynamic_world
        edge_dt = self.edge_dt_s
        max_step = self.max_joint_step_rad
        checker = ob.StateValidityCheckerFn(valid)
        validator = TimedMotionValidator(si)
        si.setStateValidityChecker(checker)
        si.setMotionValidator(validator)
        problem = ob.ProblemDefinition(si)
        start = ob.State(space)
        for index, value in enumerate(q_start):
            start()[0][index] = float(value)
        start()[1].position = 0.0
        problem.addStartState(start)
        goal_states = ob.GoalStates(si)
        goal_states.setThreshold(0.05)
        for goal_q in goals:
            goal = ob.State(space)
            for index, value in enumerate(goal_q):
                goal()[0][index] = float(value)
            goal()[1].position = 0.0
            goal_states.addState(goal)
        problem.setGoal(goal_states)
        planner = og.STRRTstar(si)
        planner.setRange(self.planner_range)
        planner.setProblemDefinition(problem)
        planner.setup()
        status = planner.solve(float(budget_s))
        if not status or not problem.hasExactSolution():
            raise NoValidTrajectoryError("ST-RRT* found no exact solution within the planning budget")
        return self._path_points(problem.getSolutionPath(), space)

    @staticmethod
    def _quintic_segment(q0, q1, v0, v1, a0, a1, duration: float, fractions: np.ndarray):
        h = duration
        c0, c1, c2 = q0, h * v0, 0.5 * h * h * a0
        rhs = np.stack((q1 - c0 - c1 - c2, h * v1 - c1 - 2 * c2, h * h * a1 - 2 * c2))
        c3, c4, c5 = np.linalg.solve(np.array([[1, 1, 1], [3, 4, 5], [6, 12, 20]], dtype=float), rhs)
        s = fractions[:, None]
        q = c0 + c1 * s + c2 * s**2 + c3 * s**3 + c4 * s**4 + c5 * s**5
        dq = (c1 + 2 * c2 * s + 3 * c3 * s**2 + 4 * c4 * s**3 + 5 * c5 * s**4) / h
        ddq = (2 * c2 + 6 * c3 * s + 12 * c4 * s**2 + 20 * c5 * s**3) / h**2
        jerk = (6 * c3 + 24 * c4 * s + 60 * c5 * s**2) / h**3
        return q, dq, ddq, jerk

    def _trajectory(self, path_q: np.ndarray, path_t: np.ndarray, dq_start: np.ndarray,
                    ddq_start: np.ndarray, start_unix_s: float, duration_limit_s: float):
        # Remove only edges whose complete space-time interpolation remains valid.
        # This also avoids artificial high jerk at OMPL's densely spaced vertices.
        kept = [0]
        while kept[-1] < len(path_t) - 1:
            source = kept[-1]
            destination = source + 1
            for candidate in range(len(path_t) - 1, source + 1, -1):
                elapsed = path_t[candidate] - path_t[source]
                delta = path_q[candidate] - path_q[source]
                if np.any(np.abs(delta) > self.dynamic_world.dq_max * elapsed):
                    continue
                count = max(2, math.ceil(elapsed / self.edge_dt_s),
                            math.ceil(float(np.max(np.abs(delta))) / self.max_joint_step_rad))
                fractions = np.linspace(1.0 / count, 1.0, count)
                samples = path_q[source][None] + fractions[:, None] * delta[None]
                times = start_unix_s + path_t[source] + elapsed * fractions
                if self.dynamic_world.samples_valid(samples, times):
                    destination = candidate
                    break
            kept.append(destination)
        path_q, path_t = path_q[kept], path_t[kept]
        secants = np.diff(path_q, axis=0) / np.diff(path_t)[:, None]
        tangents = np.zeros_like(path_q)
        tangents[0] = dq_start
        for index in range(1, len(path_q) - 1):
            left, right = secants[index - 1], secants[index]
            tangents[index] = np.where(left * right > 0, np.sign(left) * np.minimum(np.abs(left), np.abs(right)), 0.0)
        tangents[1:-1] = np.clip(tangents[1:-1], -0.5 * self.dynamic_world.dq_max, 0.5 * self.dynamic_world.dq_max)
        accelerations = np.zeros_like(path_q)
        accelerations[0] = ddq_start
        for scale in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
            scaled_t = path_t * scale
            if scaled_t[-1] > duration_limit_s + 1e-9:
                continue
            scaled_v = tangents.copy()
            scaled_v[1:] /= scale
            chunks = []
            for index in range(len(path_t) - 1):
                h = float(scaled_t[index + 1] - scaled_t[index])
                n = max(2, math.ceil(h / self.edge_dt_s) + 1)
                fractions = np.linspace(0.0, 1.0, n)
                if index:
                    fractions = fractions[1:]
                q, dq, ddq, jerk = self._quintic_segment(
                    path_q[index], path_q[index + 1], scaled_v[index], scaled_v[index + 1],
                    accelerations[index], accelerations[index + 1], h, fractions)
                chunks.append((scaled_t[index] + h * fractions, q, dq, ddq, jerk))
            times, q, dq, ddq, jerk = (np.concatenate([chunk[i] for chunk in chunks]) for i in range(5))
            if (np.any(np.abs(dq) > self.dynamic_world.dq_max + 1e-5)
                    or np.any(np.abs(ddq) > self.dynamic_world.ddq_max + 1e-5)
                    or np.any(np.abs(jerk) > 15.0 + 1e-5)
                    or np.any(q < self.dynamic_world.q_min - 1e-6)
                    or np.any(q > self.dynamic_world.q_max + 1e-6)):
                continue
            spheres = self.dynamic_world.spheres_many(q)
            if self.dynamic_world.centers_many_valid(spheres, start_unix_s + times):
                return times, q, dq, ddq, spheres, scale
        raise NoValidTrajectoryError("postprocess_invalid: no safe smooth time scaling")

    def plan(self, raw_request: dict) -> PlanArtifacts:
        request = validate_request(raw_request)
        if request["scene_id"] != "EnvOpenDrawerShelf":
            raise RequestValidationError("ST-RRT* worker supports EnvOpenDrawerShelf only")
        world_version = int(raw_request["_dynamic_world_version"])
        if world_version != self.dynamic_world.world_version:
            raise RequestValidationError("world version changed before planning")
        start_unix_s = int(raw_request["_trajectory_start_unix_ns"]) * 1e-9
        deadline_ns = int(raw_request["_deadline_unix_ns"])
        snapshot = self.dynamic_world.snapshot
        if snapshot is None:
            raise RequestValidationError("dynamic world has not been uploaded")
        duration_limit = min(self.max_duration_s, snapshot["valid_until_s"] - start_unix_s)
        if duration_limit <= 0.1:
            raise NoValidTrajectoryError("prediction horizon does not cover the handoff")
        q_start = request["q_pos_start"]
        dq_start = request["q_vel_start"]
        ddq_start = request["q_acc_start"]
        if not self.dynamic_world.is_valid(q_start, start_unix_s):
            raise NoValidTrajectoryError("start is invalid in the handoff world")
        started = time.perf_counter()
        if request["goal_type"] == "joint":
            goals = [request["q_pos_goal"]]
        else:
            goals = self._ik_goals(tuple(request["ee_pose_goal"].tolist()))
        goals = sorted(goals, key=lambda q: float(np.linalg.norm(q - q_start)))
        remaining = (deadline_ns - time.time_ns()) * 1e-9 - 0.12
        budget = min(self.solve_budget_s, remaining)
        if budget <= 0.05:
            raise NoValidTrajectoryError("deadline leaves no ST-RRT* solve budget")
        path_q, path_t = self._solve(q_start, goals, start_unix_s, duration_limit, budget)
        solve_s = time.perf_counter() - started
        times, q, dq, ddq, spheres, scale = self._trajectory(
            path_q, path_t, dq_start, ddq_start, start_unix_s, duration_limit)
        if request["goal_type"] == "joint":
            if float(np.max(np.abs(q[-1] - request["q_pos_goal"]))) > 0.05:
                raise NoValidTrajectoryError("final joint goal is outside tolerance")
        else:
            import torch
            with torch.no_grad():
                transform = self.robot.get_EE_pose(torch.as_tensor(q[-1][None], dtype=torch.float32))[0]
            transform = transform.detach().cpu().numpy()
            pose = request["ee_pose_goal"]
            position_error = np.linalg.norm(transform[:3, 3] - pose[:3])
            rotation_error = (Rotation.from_matrix(transform[:3, :3]).inv()
                              * Rotation.from_quat(pose[3:])).magnitude()
            if position_error > 0.015 or rotation_error > math.radians(3.0):
                raise NoValidTrajectoryError("final Cartesian goal is outside tolerance")
        hold_times = start_unix_s + np.arange(times[-1], duration_limit + 1e-8, self.edge_dt_s)
        if len(hold_times) and not self.dynamic_world.centers_many_valid(
            np.broadcast_to(spheres[-1], (len(hold_times), *spheres[-1].shape)), hold_times):
            raise NoValidTrajectoryError("terminal hold collides within the prediction horizon")
        total_s = time.perf_counter() - started
        arrays = {
            "joint_names": np.asarray(request["joint_names"], dtype=np.str_),
            "positions": q,
            "velocities": dq,
            "accelerations": ddq,
            "time_from_start": times,
            "collision_sphere_positions": spheres.astype(np.float32),
            "collision_sphere_radii": self.dynamic_world.radii,
        }
        result = {
            "schema_version": 1,
            "status": "success",
            "planner_name": "strrtstar",
            "request_id": request["request_id"],
            "created_unix_time": time.time(),
            "trajectory_file": "trajectory.npz",
            "trajectory_artifact": {"schema_version": 1, "planner": "strrtstar"},
            "timing": {"solve_s": solve_s, "postprocess_s": total_s - solve_s,
                       "total_s": total_s, "inference_total_sec": total_s},
            "trajectory": {"duration_s": float(times[-1]), "time_scale": scale,
                           "waypoint_count": len(path_t), "sample_count": len(times),
                           "terminal_hold_checked_until_s": duration_limit},
        }
        return PlanArtifacts(result_payload=result, trajectory_arrays=arrays)
