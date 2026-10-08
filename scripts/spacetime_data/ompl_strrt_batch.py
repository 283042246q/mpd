"""Exact OMPL STRRTstar algorithm, batched with isolated spawned CPU workers."""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from scripts.spacetime_data.batch_spacetime_rrt import SpaceTimeResult
from scripts.spacetime_data.warehouse_world import WarehouseCollision


_WORKER_WORLD = None
_WORKER_SEED = None
_WORKER_SETTINGS = None
_WORKER_ERROR = None


def _initialize_impl(repository, seed, settings):
    global _WORKER_WORLD, _WORKER_SEED, _WORKER_SETTINGS
    bindings = Path(repository) / "deps/pybullet_ompl/ompl/py-bindings"
    sys.path.insert(0, str(bindings))
    from ompl import base, geometric, util
    if not hasattr(geometric, "STRRTstar") or not hasattr(base, "SpaceTimeStateSpace"):
        raise RuntimeError("compiled OMPL SpaceTimeStateSpace/STRRTstar bindings required")
    torch.set_num_threads(1)
    _WORKER_SEED = int(seed + mp.current_process()._identity[-1]) % (2**32 - 1)
    util.RNG.setSeed(_WORKER_SEED)
    util.setLogLevel(util.LOG_WARN)
    _WORKER_SETTINGS = settings
    _WORKER_WORLD = WarehouseCollision("cpu", margin=settings["margin"], chunk_size=settings["collision_chunk"])


def _initialize(repository, seed, settings):
    # A failed Pool initializer otherwise causes an unbounded respawn loop.
    global _WORKER_ERROR
    try:
        _initialize_impl(repository, seed, settings)
    except Exception as exc:
        _WORKER_ERROR = "%s: %s" % (type(exc).__name__, exc)


def _solve(job):
    if _WORKER_ERROR:
        raise RuntimeError("OMPL worker initialization failed: " + _WORKER_ERROR)
    from ompl import base as ob, geometric as og
    world, settings = _WORKER_WORLD, _WORKER_SETTINGS
    world.set_worlds([job["forecast"]])
    robot = world.robot
    started = time.perf_counter()
    statistics = {"backend": "ompl-strrtstar", "worker_seed": _WORKER_SEED,
                  "job_seed": job["seed"], "checked_states": 0}
    q_start, q_goal = np.asarray(job["q_start"]), np.asarray(job["q_goal"])
    velocity = robot.dq_max.cpu().numpy() * settings["velocity_scale"]
    q_space = ob.RealVectorStateSpace(7)
    bounds = ob.RealVectorBounds(7)
    for axis in range(7):
        bounds.setLow(axis, float(robot.q_pos_min[axis]))
        bounds.setHigh(axis, float(robot.q_pos_max[axis]))
    q_space.setBounds(bounds)
    space = ob.SpaceTimeStateSpace(q_space, float(velocity.min()))
    space.setTimeBounds(0., float(job["forecast"].horizon))
    si = ob.SpaceInformation(space)

    def q_of(state):
        return np.array([state[0][axis] for axis in range(7)], dtype=float)

    def valid(state):
        q = q_of(state)
        t = space.getStateTime(state)
        statistics["checked_states"] += 1
        return not bool(world(q[None], [t], [0])[0].item())

    class MotionValidator(ob.MotionValidator):
        def checkMotion(self, first, second):
            t0, t1 = space.getStateTime(first), space.getStateTime(second)
            delta = t1 - t0
            a, b = q_of(first), q_of(second)
            if delta <= 1e-8 or np.any(np.abs(b - a) > velocity * delta + 1e-7):
                return False
            count = max(1, int(np.ceil(delta / settings["time_step"])),
                        int(np.ceil(np.max(np.abs(b - a)) / settings["joint_step"])))
            fractions = np.linspace(0, 1, count + 1)
            samples = a[None] + fractions[:, None] * (b - a)[None]
            times = t0 + fractions * delta
            statistics["checked_states"] += len(samples)
            return not bool(world(samples, times, np.zeros(len(times), dtype=int)).any().item())

    checker = ob.StateValidityCheckerFn(valid)
    validator = MotionValidator(si)
    si.setStateValidityChecker(checker)
    si.setMotionValidator(validator)
    problem = ob.ProblemDefinition(si)
    start, goal = ob.State(space), ob.State(space)
    for axis in range(7):
        start()[0][axis], goal()[0][axis] = float(q_start[axis]), float(q_goal[axis])
    start()[1].position, goal()[1].position = 0., 0.
    problem.addStartState(start)
    goals = ob.GoalStates(si)
    goals.setThreshold(1e-6)
    goals.addState(goal)
    problem.setGoal(goals)
    planner = og.STRRTstar(si)
    planner.setRange(settings["planner_range"])
    planner.setProblemDefinition(problem)
    planner.setup()
    status = planner.solve(settings["solve_seconds"])
    paths = []
    if status and problem.hasExactSolution():
        states = problem.getSolutionPath().getStates()
        path = np.array([list(q_of(state)) + [space.getStateTime(state)] for state in states])
        if len(path) >= 2 and np.all(np.diff(path[:, -1]) > 0):
            paths.append(path)
    statistics["wall_s"] = time.perf_counter() - started
    return SpaceTimeResult(paths, statistics)


class OmplSTRRTBatch:
    """Persistent processes, one independent robot/world/RNG per process."""

    def __init__(self, repository, *, workers=4, seed=1, **settings):
        if workers < 1:
            raise ValueError("workers must be positive")
        # Spawned Python starts with the conda library path. No runtime/ROS
        # launch files or global shell environment need to be modified.
        library = str(Path(sys.executable).resolve().parent.parent / "lib")
        os.environ["LD_LIBRARY_PATH"] = library + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
        self.solve_seconds = settings["solve_seconds"]
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_initialize,
                                                 initargs=(str(repository), seed, settings))

    def plan_batch(self, jobs):
        # A crashed native worker must not leave Pool.map waiting forever.
        return self.pool.map_async(_solve, jobs, chunksize=1).get(timeout=60 + len(jobs) * self.solve_seconds)

    def close(self):
        self.pool.close()
        self.pool.join()

    def terminate(self):
        self.pool.terminate()
        self.pool.join()
