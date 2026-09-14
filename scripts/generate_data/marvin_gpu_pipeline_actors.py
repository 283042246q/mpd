"""Spawn-safe actor entrypoint for the Marvin GPU generation pipeline.

Imports for CUDA, Pinocchio and PyBullet are deliberately role-local.  A
spawned actor therefore owns only the native runtime required for its job.
"""
from __future__ import annotations

import traceback

import numpy as np


def _endpoint_command(worker, message):
    result = []
    for job in message["jobs"]:
        task = job["task"]
        for candidate_index, seed in enumerate(job["candidate_seeds"]):
            proposal = worker.propose(
                task["mode"],
                task["direction"],
                task["source"],
                task["goal"],
                seed,
            )
            item = {
                "task_id": task["task_id"],
                "attempt": job["attempt"],
                "candidate_index": candidate_index,
            }
            if proposal is None:
                item["proposal"] = None
            else:
                q_start, q_goal = proposal
                item.update(
                    proposal=True,
                    q_start=q_start,
                    q_goal=q_goal,
                    ee_goal_pose=worker.ee_goal_pose(q_goal),
                )
            result.append(item)
    return result


def _gpu_command(worker, message):
    if message["op"] == "endpoints":
        items = message["items"]
        if not items:
            return []
        states = np.concatenate(
            [np.stack((item["q_start"], item["q_goal"])) for item in items]
        )
        collision = worker.collision_mask(states).detach().cpu().numpy().reshape(-1, 2)
        return [not bool(value.any()) for value in collision]
    if message["op"] == "plan":
        items = message["items"]
        plans = worker.plan_mode_batch(
            message["mode"],
            np.stack([item["q_start"] for item in items]),
            np.stack([item["q_goal"] for item in items]),
            message["query_seeds"],
        )
        result = []
        for item, plan in zip(items, plans):
            result.append(
                {
                    "task_id": item["task_id"],
                    "attempt": item["attempt"],
                    "path": plan.path,
                    "spline": plan.spline,
                    "rejection_reason": plan.rejection_reason,
                    "batch_seconds": plan.batch_seconds,
                    "rrt_iterations": plan.rrt.iterations,
                    "rrt_sampled_edges": plan.rrt.sampled_edges,
                    "rrt_checked_states": plan.rrt.checked_states,
                }
            )
        return result
    raise ValueError(f"unsupported GPU operation: {message['op']}")


def _pybullet_command(worker, message):
    if message["op"] == "endpoints":
        return [
            worker.endpoints_valid(item["q_start"], item["q_goal"])
            for item in message["items"]
        ]
    if message["op"] == "trajectories":
        return [
            worker.trajectory_valid(item["path"], item["spline"])
            for item in message["items"]
        ]
    raise ValueError(f"unsupported PyBullet operation: {message['op']}")


def actor_main(role, config, requests, responses):
    worker = None
    try:
        if role == "endpoint":
            validity_fn = None
            if config.get("gpu_pipeline_endpoint_cpu_sphere_filter", True):
                from scripts.generate_data.marvin_cpu_sphere_checker import (
                    MarvinCpuSphereChecker,
                )

                sphere_checker = MarvinCpuSphereChecker(config)
                validity_fn = sphere_checker.valid
            from scripts.generate_data.marvin_endpoint_proposer import (
                MarvinEndpointProposer,
            )

            worker = MarvinEndpointProposer(config, validity_fn=validity_fn)
            handler = _endpoint_command
        elif role == "gpu":
            from scripts.generate_data.marvin_gpu_planning_backend import (
                MarvinGpuPlanningBackend,
            )

            worker = MarvinGpuPlanningBackend(config)
            handler = _gpu_command
        elif role == "pybullet":
            from scripts.generate_data.marvin_pybullet_auditor import (
                MarvinPyBulletAuditor,
            )

            worker = MarvinPyBulletAuditor(config)
            handler = _pybullet_command
        else:
            raise ValueError(f"unsupported actor role: {role}")
        responses.put({"ready": True, "role": role})
        while True:
            message = requests.get()
            if message is None:
                break
            try:
                responses.put({"ok": True, "result": handler(worker, message)})
            except BaseException as error:
                responses.put(
                    {
                        "ok": False,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
    except BaseException as error:
        responses.put(
            {
                "ready": False,
                "role": role,
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if worker is not None and hasattr(worker, "close"):
            worker.close()
