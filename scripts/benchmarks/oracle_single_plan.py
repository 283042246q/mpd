#!/usr/bin/env python3
"""Generate and evaluate frozen Warehouse/ToDrawer single-plan oracle suites.

Usage: python -m scripts.benchmarks.oracle_single_plan {generate,run,report} ...
No ROS, state estimator, obstacle inflation, replanning or handoff is involved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
from queue import Queue
import re
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).parent / "cfgs/oracle_single_plan.yaml"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deps/pybullet_ompl/ompl/py-bindings"))

from mpd.utils.patches import numpy_monkey_patch
numpy_monkey_patch()
from scripts.benchmarks.oracle_sampling import balanced_choices, sample_world, base_sweep_clear
from scripts.runtime.infer_once import (
    _atomic_write_json, _atomic_write_npz, _canonical_json_sha256, _sha256_file,
    _jsonable, EXPECTED_JOINT_NAMES,
)


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve(path):
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    return p if p.is_absolute() else ROOT / p


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if config.get("schema") != "oracle_single_plan_config_v1":
        raise ValueError("unsupported benchmark config")
    if (config["cases_per_scene"] < 1 or config["horizon_s"] != 14.
            or config["nominal_duration_s"] != 10.
            or config["forecast_dt_s"] <= 0 or config["margin_m"] < 0):
        raise ValueError("invalid scene count or timing contract")
    settings = config["inference"]
    if min(settings["candidates"], settings["steps"], settings["cost_points"], settings["output_points"]) < 1:
        raise ValueError("invalid inference budget")
    if settings["validation_dt_s"] <= 0 or settings["terminal_hold_s"] < 0:
        raise ValueError("invalid validation interval")
    if (settings["duration_min_s"], settings["duration_max_s"]) != (2., 14.):
        raise ValueError("the current shared JointDual codec requires duration bounds [2,14]")
    if settings["output_points"] != settings["cost_points"]:
        raise ValueError("legacy and JointDual export grids must have the same point count")
    if settings["duration_max_s"] + settings["terminal_hold_s"] > config["horizon_s"]:
        raise ValueError("execution and hold must fit the authoritative world horizon")
    return config


def model_registry(config):
    """Freeze actual checkpoint identities; never silently replace missing models."""
    from dotmap import DotMap
    from mpd.inference.inference import resolve_model_checkpoint_path
    from scripts.runtime.infer_once import _resolve_model_dir
    models = []
    native = yaml.safe_load(resolve(config["scenes"]["warehouse"]["runtime_config"]).read_text())
    model_dir = _resolve_model_dir(DotMap(native))
    args = yaml.safe_load((model_dir / "args.yaml").read_text())
    spatial = resolve_model_checkpoint_path(model_dir, args.get("use_ema"),
                                            checkpoint=native.get("runtime", {}).get("checkpoint"))
    for name in config["models"]["legacy"]:
        row = dict(id=name, kind="legacy", spatial_checkpoint=str(spatial),
                   weights="ema" if args.get("use_ema") else "model")
        if name.startswith("f"):
            row.update(kind="factorized", method=name[:2], representation="c" if name.endswith("c") else "tau_r",
                timing_checkpoint=str(resolve(config["models"]["timing_c_checkpoint" if name.endswith("c") else "timing_tau_r_checkpoint"])))
        elif name not in ("phase4", "phase4_aligned", "phase5joint"):
            raise ValueError(f"unknown benchmark model {name}")
        models.append(row)
    manifest_path = resolve(config["models"]["jointdual_manifest"])
    manifest = read_json(manifest_path)
    for source in manifest["candidates"]:
        row = deepcopy(source)
        row.update(kind="jointdual", checkpoint=str(resolve(source["checkpoint"])))
        models.append(row)
    if len({m["id"] for m in models}) != len(models):
        raise ValueError("duplicate model IDs")
    hashes = {}
    for model in models:
        for key in ("checkpoint", "spatial_checkpoint", "timing_checkpoint"):
            if key in model:
                path = Path(model[key])
                if not path.is_file():
                    raise FileNotFoundError(f"{model['id']}: missing {key}: {path}")
                if str(path) not in hashes:
                    hashes[str(path)] = _sha256_file(path)
                model[key + "_sha256"] = hashes[str(path)]
    return models, dict(path=str(manifest_path), sha256=_sha256_file(manifest_path))


def scene_collision(scene_id, device, margin):
    import torch
    import torch_robotics.environments as environments
    from scripts.spacetime_data.warehouse_world import WarehouseCollision
    env = getattr(environments, scene_id)(precompute_sdf_obj_fixed=False,
        precompute_sdf_obj_extra=False, tensor_args=dict(device=torch.device(device), dtype=torch.float32))
    return WarehouseCollision(device, margin=margin, environment=env)


def generate(args):
    import torch
    from ompl import util as ou
    from mpd.inference.inference import EvaluationSamplesGenerator
    from dotmap import DotMap
    from mpd.utils.loaders import get_planning_task_and_dataset
    from scripts.runtime.infer_once import _resolve_model_dir
    from scripts.isaaclab.scene_payload import export_isaaclab_scene_payload
    from scripts.spacetime_data.warehouse_world import Forecast, ForecastBatch
    from scripts.spacetime_data.warehouse_world import WarehouseCollision
    from torch_robotics.torch_utils.seed import fix_random_seed
    from scipy.spatial.transform import Rotation

    root = args.output.resolve()
    if (root / "suite.json").exists():
        raise FileExistsError("suite already exists; choose a new output directory (frozen worlds are not overwritten)")
    config = load_config(args.config)
    for name in ("cases_per_scene", "seed"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    for name in ("candidates", "steps", "refinement_steps"):
        if getattr(args, name) is not None:
            config["inference"][name] = getattr(args, name)
    if (config["cases_per_scene"] < 1 or config["seed"] < 0 or config["seed"] >= 2**32
            or config["inference"]["candidates"] < 1 or config["inference"]["steps"] < 1
            or config["inference"]["refinement_steps"] < 0):
        raise ValueError("cases must be positive and seed nonnegative")
    models, manifest = model_registry(config)
    torch.set_num_threads(args.threads)
    fix_random_seed(config["seed"])
    ou.RNG.setSeed(config["seed"])
    ou.setLogLevel(ou.LOG_WARN)
    root.mkdir(parents=True, exist_ok=True)
    suite = dict(schema="oracle_single_plan_suite_v1", status="generating", config=config,
        config_hash=_canonical_json_sha256(config), models=models, jointdual_manifest=manifest,
        runtime_configs={scene: yaml.safe_load(resolve(spec["runtime_config"]).read_text())
                         for scene, spec in config["scenes"].items()},
        jointdual_runtime_config=yaml.safe_load((ROOT / "scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-jointdual.yaml").read_text()),
        protocol=dict(world="prescribed_piecewise_linear", clock="execution_start_zero",
            compute_time_advances_world=False, replanning=False, handoff=False, inflation=False,
            goal="common_cartesian_pose", checkpoint_selection="frozen_before_inference",
            budgets="matched_candidates_and_diffusion_steps; actual branch and guidance costs reported"),
        scenes={}, cases=[])
    _atomic_write_json(root / "suite.json", suite)
    ledger = root / "generation.jsonl"
    for scene_index, (scene, spec) in enumerate(config["scenes"].items()):
        source_path = resolve(spec["single_plan_config"])
        source = yaml.safe_load(source_path.read_text())
        # Reuse the complete single-plan loading path, not just its region YAML.
        # In particular, min_distance_robot_env comes from the dataset args and
        # gripper=True is supplied by the original inference entry point.
        model_dir = _resolve_model_dir(DotMap(source))
        task_args = yaml.safe_load((model_dir / "args.yaml").read_text())
        task_args.update(source)
        task_args.update(model_dir=str(model_dir), gripper=True, reload_data=False, load_indices=True,
            results_dir=str(root / scene), precompute_sdf_obj_fixed=False, precompute_sdf_obj_extra=False,
            tensor_args=dict(device=torch.device("cpu"), dtype=torch.float32))
        task, train_subset, _, val_subset, _ = get_planning_task_and_dataset(**task_args)
        collision = WarehouseCollision("cpu", margin=config["margin_m"], environment=task.env, robot=task.robot)
        region_path = (source_path.parent / source["start_goal_regions_path"]).resolve()
        generator = EvaluationSamplesGenerator(task, train_subset, val_subset, start_goal_source="regions",
            start_goal_regions_path=str(region_path),
            min_distance_q_pos_start_goal=source.get("min_distance_q_pos_start_goal", 0.),
            start_goal_max_sampling_attempts=source.get("start_goal_max_sampling_attempts", 100),
            rotation_z_axis_deg=source.get("rotation_z_axis_deg", 0.), tensor_args=collision.tensor_args)
        worker = generator.generate_data_ompl_worker
        base_low, base_high = worker.pybullet_client.getAABB(worker.robot_pbompl.id, -1)
        payload = export_isaaclab_scene_payload(collision.environment, include_boxes=True)
        scene_meta = dict(scene_id=spec["scene_id"], single_plan_config=str(source_path),
            single_plan_config_sha256=_sha256_file(source_path), region_path=str(region_path),
            region_sha256=_sha256_file(region_path), region_config=yaml.safe_load(region_path.read_text()),
            min_distance_q_pos_start_goal=generator.min_distance_q_pos_start_goal,
            min_distance_robot_env=task.min_distance_robot_env,
            sampling_robot_urdf=str(task.robot.robot_urdf_file),
            sampling_robot_urdf_sha256=_sha256_file(Path(task.robot.robot_urdf_file)),
            static_scene_sha256=_canonical_json_sha256(payload),
            base_aabb_low=list(base_low), base_aabb_high=list(base_high))
        suite["scenes"][scene] = scene_meta
        _atomic_write_json(root / scene / "static_scene.json", payload)
        rng = np.random.default_rng(config["seed"] + 1000003 * scene_index)
        counts = balanced_choices(config["cases_per_scene"], list(range(6)), config["obstacle_count_weights"], rng)
        arrivals = balanced_choices(config["cases_per_scene"], config["arrival_modes"], config["arrival_weights"], rng)
        moving = [i for i, p in enumerate(collision.robot.collision_sphere_parent_links)
                  if p not in ("panda_link0", "panda_link1", "panda_link2")]
        try:
            for index, (count, mode) in enumerate(zip(counts, arrivals)):
                case_id = f"{scene}-{index:04d}"
                case_seed = int(config["seed"] + scene_index * 1000003 + index * 997)
                fix_random_seed(case_seed)
                reference = None
                for attempt in range(config["max_task_attempts"]):
                    try:
                        qs, qg, goal = generator.get_data_sample(index)
                        answer = worker.run(1, qs.tolist(), qg.tolist(),
                            planner_allowed_time=config["rrt_solve_seconds"], interpolate_num=config["reference_points"],
                            simplify_path=True, fit_bspline=False, max_tries=1)
                        if not answer:
                            raise RuntimeError("static RRTConnect did not solve within its budget")
                        reference = np.asarray(answer[0]["sol_path"], dtype=np.float64)
                        if reference.ndim != 2 or reference.shape[1] != 7:
                            raise ValueError("unexpected RRT reference shape")
                        if collision.static_collision(reference).any():
                            raise RuntimeError("RRT reference fails common static/self sphere validation")
                        break
                    except (RuntimeError, AssertionError) as error:
                        reference = None
                        with ledger.open("a") as stream:
                            stream.write(json.dumps(dict(case_id=case_id, stage="static_task", attempt=attempt,
                                                        error=str(error))) + "\n")
                if reference is None:
                    raise RuntimeError(f"cannot obtain static reference for {case_id}; see generation ledger")
                centers = collision.centers(reference).numpy()
                case_rng = np.random.default_rng(case_seed)
                rejections = Counter()
                for world_attempt in range(config["max_world_attempts"]):
                    world, info = sample_world(centers, moving, count, mode, case_rng, config)
                    if not base_sweep_clear(world, base_low, base_high, config["margin_m"]):
                        rejections["base_swept_aabb"] += 1; continue
                    forecast = Forecast(world["times"], world["positions"], world["shapes"], world["sizes"],
                        np.asarray(info["event_times"]), info["motion_modes"], np.asarray(info["anchors"]).reshape(count, 3))
                    batch = ForecastBatch([forecast])
                    initial = batch.clearance(torch.as_tensor(centers[:1]), collision.robot.link_collision_spheres_radii,
                        torch.zeros(1), torch.zeros(1, dtype=torch.long))
                    if bool((initial <= config["margin_m"]).any()):
                        rejections["initial_arm_collision"] += 1; continue
                    break
                else:
                    raise RuntimeError(f"cannot sample base-safe world for {case_id}: {dict(rejections)}")
                transform = goal.numpy()
                pose = np.r_[transform[:, 3], Rotation.from_matrix(transform[:, :3]).as_quat()].tolist()
                request = dict(schema_version=1, request_id=case_id, scene_id=spec["scene_id"],
                    joint_names=list(EXPECTED_JOINT_NAMES), q_pos_start=qs.tolist(),
                    goal_type="cartesian", ee_pose_goal=pose, ik_candidates=0, seed=case_seed)
                case_dir = root / scene / case_id
                _atomic_write_json(case_dir / "request.json", request)
                _atomic_write_npz(case_dir / "world.npz", compressed=True, **world)
                _atomic_write_npz(case_dir / "reference.npz", compressed=True, q=reference,
                    phase=np.linspace(0, 1, len(reference)),
                    time_from_start=np.linspace(0, config["nominal_duration_s"], len(reference)), q_goal_reference=qg.numpy())
                info.update(id=case_id, scene=scene, seed=case_seed, world_attempts=world_attempt + 1,
                    generation_rejections=dict(rejections), start_goal_metadata=generator.last_sample_metadata,
                    static_reference_planner="OMPL RRTConnect", reference_used_for="anchors_and_event_times_only",
                    reference_nominal_duration_s=config["nominal_duration_s"],
                    request_sha256=_sha256_file(case_dir / "request.json"), world_sha256=_sha256_file(case_dir / "world.npz"),
                    reference_sha256=_sha256_file(case_dir / "reference.npz"), directory=str(case_dir.relative_to(root)))
                _atomic_write_json(case_dir / "case.json", info)
                suite["cases"].append(info)
                _atomic_write_json(root / "suite.json", suite)
                print(f"generated {case_id} count={count} arrival={info['arrival_mode']}", flush=True)
        finally:
            worker.terminate()
    suite["status"] = "ready"
    suite["case_count"] = len(suite["cases"])
    suite["suite_hash"] = _canonical_json_sha256({k: v for k, v in suite.items() if k != "suite_hash"})
    _atomic_write_json(root / "suite.json", suite)
    print(f"frozen {len(suite['cases'])} scenarios, {len(models)} models: {root}", flush=True)
    return 0


def runtime_config(suite, scene, model, output):
    config, settings = suite["config"], suite["config"]["inference"]
    spec = config["scenes"][scene]
    if model["kind"] == "jointdual":
        data = deepcopy(suite.get("jointdual_runtime_config") or yaml.safe_load(
            (ROOT / "scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-jointdual.yaml").read_text()))
        data.update(scene_id=spec["scene_id"], candidates=settings["candidates"], top_k=settings["top_k"],
            cost_points=settings["cost_points"], output_points=settings["output_points"], collision_margin=config["margin_m"],
            ee_position_tolerance=settings["ee_position_tolerance_m"],
            ee_orientation_tolerance_deg=settings["ee_orientation_tolerance_deg"])
        data["sampler"].update(steps=settings["steps"], guide_fraction=settings["guide_fraction"],
                               refinement_steps=settings["refinement_steps"], eta=0.)
    else:
        data = deepcopy(suite.get("runtime_configs", {}).get(scene) or yaml.safe_load(resolve(spec["runtime_config"]).read_text()))
        data.update(n_trajectory_samples=settings["candidates"], num_T_pts=settings["cost_points"],
                    trajectory_duration=config["nominal_duration_s"], margin_for_dense_collision_checking=config["margin_m"])
        data["runtime"]["runtime_top_k_valid_trajectories"] = settings["top_k"]
        data["dense_validation"].update(runtime_points=settings["cost_points"], benchmark_points=settings["cost_points"])
        data["dense_validation"]["ranked_early_exit"]["enabled"] = False
        data["ddim"].update(ddim_sampling_timesteps=settings["steps"], ddim_eta=0.,
            t_start_guide_steps_fraction=settings["guide_fraction"], n_guide_steps=1)
        # One refinement per late diffusion step, plus the shared final budget.
        # F samplers already own their final refinement, so do not apply it twice.
        data["extra_mp_steps"] = settings["refinement_steps"] if model["kind"] == "legacy" else 0
    path = output / "runtime.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def build_engine(suite, scene, model, output, device):
    from mpd.inference.prescribed_world import PrescribedDynamicWorld
    from scripts.runtime.dynamic_runtime_engine import DynamicMpdRuntimeEngine
    from scripts.runtime.space_time_runtime_engine import SpaceTimeMpdRuntimeEngine
    from scripts.runtime.factorized_runtime_engine import FactorizedMpdRuntimeEngine
    from scripts.runtime.jointdual_runtime_engine import JointDualMpdRuntimeEngine
    config, settings = suite["config"], suite["config"]["inference"]
    kwargs = dict(config_path=runtime_config(suite, scene, model, output),
                  runtime_output_root=output, device_text=device)
    if model["kind"] == "jointdual":
        collision = scene_collision(config["scenes"][scene]["scene_id"], device, config["margin_m"])
        return JointDualMpdRuntimeEngine(joint_checkpoint=Path(model["checkpoint"]), collision_backend=collision,
                                        allow_static_scene_transfer=True, **kwargs)
    kwargs["dynamic_world_factory"] = PrescribedDynamicWorld
    kwargs["allow_benchmark_budgets"] = True
    if model["id"] in ("phase4", "phase4_aligned"):
        return DynamicMpdRuntimeEngine(aligned=model["id"] == "phase4_aligned", **kwargs)
    kwargs["space_time_settings"] = dict(duration_min=settings["duration_min_s"],
        duration_max=settings["duration_max_s"], nominal_duration=config["nominal_duration_s"], corridor_a_enabled=False)
    if model["kind"] == "factorized":
        return FactorizedMpdRuntimeEngine(timing_checkpoint=Path(model["timing_checkpoint"]), adapt_spatial_basis=True,
            factorized_settings=dict(method=model["method"], space_steps=settings["steps"], timing_steps=settings["steps"],
                refinement_steps=settings["refinement_steps"], space_guide_fraction=settings["guide_fraction"],
                timing_guide_fraction=settings["guide_fraction"], eta=0.), **kwargs)
    return SpaceTimeMpdRuntimeEngine(timing_mode="phase5_joint", **kwargs)


def validate_artifact(artifacts, collision, world, request, settings):
    """Evaluate the selected trajectory using the same executable interpolant."""
    import torch
    from scipy.interpolate import BPoly
    from scipy.spatial.transform import Rotation
    arrays = artifacts.trajectory_arrays
    q, dq, ddq = [np.asarray(arrays["topk_" + k])[0] for k in ("positions", "velocities", "accelerations")]
    native_t = np.asarray(arrays["topk_time_from_start"][0] if "topk_time_from_start" in arrays else arrays["time_from_start"])
    if (native_t.ndim != 1 or len(native_t) != len(q) or native_t[0] != 0 or not (np.diff(native_t) > 0).all()
            or not all(np.isfinite(v).all() for v in (q, dq, ddq, native_t))):
        return dict(valid=False, reasons=["invalid_exported_trajectory"])
    duration = float(native_t[-1])
    hold = settings["terminal_hold_s"]
    if not settings["duration_min_s"] <= duration <= settings["duration_max_s"] or duration + hold > float(world.times[-1]) + 1e-6:
        return dict(valid=False, reasons=["duration_or_world_coverage"], duration_s=duration)
    times = np.unique(np.r_[native_t, np.linspace(0., duration, int(np.ceil(duration / settings["validation_dt_s"])) + 1),
                            world.times.cpu().numpy()[world.times.cpu().numpy() <= duration]])
    polynomial = BPoly.from_derivatives(native_t, np.stack([q, dq, ddq], axis=1))
    dense_q, dense_dq, dense_ddq = [polynomial(times, nu=i) for i in range(3)]
    if hold:
        h = duration + np.linspace(0, hold, int(np.ceil(hold / settings["validation_dt_s"])) + 1)[1:]
        times = np.r_[times, h]
        dense_q = np.r_[dense_q, np.repeat(q[-1:], len(h), axis=0)]
        dense_dq = np.r_[dense_dq, np.zeros((len(h), 7))]
        dense_ddq = np.r_[dense_ddq, np.zeros((len(h), 7))]
    values = []
    with torch.no_grad():
        for begin in range(0, len(times), 256):
            part = torch.as_tensor(dense_q[begin:begin + 256], **collision.tensor_args)
            positions = collision.centers(part)
            radius = collision.robot.link_collision_spheres_radii
            static = (collision.field.object_signed_distances(positions) - radius).amin((-1, -2))
            own = collision.robot.df_collision_self.compute_embodiment_signed_distances(part, positions).amin(-1)
            t = torch.as_tensor(times[begin:begin + 256], **collision.tensor_args)[None]
            dynamic = (world.minimum_signed_distance(positions[None], t)[0] - radius).amin(-1)
            values.append(torch.stack((static, own, dynamic), -1).cpu().numpy())
        clearance = np.concatenate(values).min(0)
        achieved = collision.robot.get_EE_pose(torch.as_tensor(q[-1:], **collision.tensor_args))[0].cpu().numpy()
    target = np.asarray(request["ee_pose_goal"])
    position_error = float(np.linalg.norm(achieved[:3, 3] - target[:3]))
    angle = float((Rotation.from_quat(target[3:]).inv() * Rotation.from_matrix(achieved[:3, :3])).magnitude() * 180 / np.pi)
    robot = collision.robot
    q_min, q_max = robot.q_pos_min.cpu().numpy(), robot.q_pos_max.cpu().numpy()
    velocity = float(np.max(np.abs(dense_dq) / robot.dq_max.cpu().numpy()))
    acceleration = float(np.max(np.abs(dense_ddq) / robot.ddq_max.cpu().numpy()))
    boundary = max(float(np.max(np.abs(q[0] - request["q_pos_start"]))),
                   *(float(np.max(np.abs(v))) for v in (dq[0], ddq[0], dq[-1], ddq[-1])))
    checks = dict(static_collision=clearance[0] <= collision.margin,
        self_collision=clearance[1] <= collision.margin, dynamic_collision=clearance[2] <= collision.margin,
        joint_position=bool(((dense_q < q_min - 1e-5) | (dense_q > q_max + 1e-5)).any()),
        joint_velocity=velocity > 1 + 1e-4, joint_acceleration=acceleration > 1 + 1e-4,
        boundary=boundary > settings["boundary_tolerance"],
        ee_position=position_error > settings["ee_position_tolerance_m"],
        ee_orientation=angle > settings["ee_orientation_tolerance_deg"])
    return dict(valid=not any(checks.values()), reasons=[k for k, v in checks.items() if v],
        duration_s=duration, path_length_rad=float(np.linalg.norm(np.diff(dense_q, axis=0), axis=1).sum()),
        ee_position_error_m=position_error, ee_orientation_error_deg=angle, boundary_error=boundary,
        velocity_utilization=velocity, acceleration_utilization=acceleration,
        minimum_static_clearance_m=float(clearance[0]), minimum_self_clearance_m=float(clearance[1]),
        minimum_dynamic_clearance_m=float(clearance[2]), validation_points=len(times),
        interpolant="quintic_hermite_from_exported_q_dq_ddq")


def verify_suite(root):
    suite = read_json(root / "suite.json")
    if suite.get("status") != "ready" or suite.get("schema") != "oracle_single_plan_suite_v1":
        raise ValueError("suite is not a completed frozen oracle suite")
    expected = _canonical_json_sha256({k: v for k, v in suite.items() if k != "suite_hash"})
    if expected != suite["suite_hash"]:
        raise ValueError("suite manifest changed after freezing")
    return suite


def audit(root):
    """Audit serialized worlds independently of planner outcomes."""
    suite = verify_suite(root)
    issues, scenes = [], {}
    config = suite["config"]
    for name, scene in suite["scenes"].items():
        cases = [c for c in suite["cases"] if c["scene"] == name]
        expected = Counter(balanced_choices(config["cases_per_scene"], list(range(6)),
                                           config["obstacle_count_weights"], np.random.default_rng(0)))
        actual = Counter(c["object_count"] for c in cases)
        if expected != actual:
            issues.append(dict(scene=name, error="obstacle count quota mismatch"))
        if "min_distance_robot_env" not in scene or "fixed_gripper" not in scene.get("sampling_robot_urdf", ""):
            issues.append(dict(scene=name, error="missing original single-plan sampling contract"))
        scenes[name] = dict(cases=len(cases), counts=dict(sorted(actual.items())),
            arrivals=dict(Counter(c["arrival_mode"] for c in cases)),
            motion_modes=dict(Counter(m for c in cases for m in c["motion_modes"])),
            min_distance_robot_env=scene.get("min_distance_robot_env"),
            sampling_robot_urdf=scene.get("sampling_robot_urdf"))
    for case in suite["cases"]:
        try:
            folder = root / case["directory"]
            for name in ("request", "world", "reference"):
                file = folder / (name + (".json" if name == "request" else ".npz"))
                if _sha256_file(file) != case[name + "_sha256"]:
                    raise ValueError(f"{name} content hash mismatch")
            with np.load(folder / "world.npz", allow_pickle=False) as file:
                arrays = {key: file[key] for key in ("times", "positions", "shapes", "sizes")}
            count = case["object_count"]
            t = arrays["times"]
            if (t[0] != 0. or t[-1] != config["horizon_s"] or not (np.diff(t) > 0).all()
                    or arrays["positions"].shape != (len(t), count, 3)
                    or arrays["sizes"].shape != (count, 3)
                    or not all(np.isfinite(a).all() for a in arrays.values())):
                raise ValueError("invalid world shape, values or time coverage")
            scene = suite["scenes"][case["scene"]]
            if not base_sweep_clear(arrays, scene["base_aabb_low"], scene["base_aabb_high"], config["margin_m"]):
                raise ValueError("saved obstacle sweep intersects protected base region")
            if not np.allclose(np.asarray(case["phases"]) * config["nominal_duration_s"], case["event_times"], atol=1e-12):
                raise ValueError("event time does not use reference phase")
            for i, (event, anchor) in enumerate(zip(case["event_times"], case["anchors"])):
                knots = np.flatnonzero(t == event)
                if len(knots) != 1 or not np.allclose(arrays["positions"][knots[0], i], anchor, atol=1e-10):
                    raise ValueError("saved event does not pass through its reference anchor")
            events = np.unique(case["event_times"])
            if count > 1:
                expected_groups = {"simultaneous": 1, "sequential": count, "grouped": 2}[case["arrival_mode"]]
                if len(events) != expected_groups:
                    raise ValueError("arrival grouping disagrees with metadata")
            with np.load(folder / "reference.npz", allow_pickle=False) as reference:
                request = read_json(folder / "request.json")
                if not np.allclose(reference["q"][0], request["q_pos_start"], atol=1e-7):
                    raise ValueError("reference and request have different starts")
            if case["inflation"] != "none" or case["obstacle_static_collision"] != "allowed":
                raise ValueError("unexpected obstacle collision/prediction policy")
        except Exception as error:
            issues.append(dict(case_id=case["id"], error=str(error)))
    result = dict(schema="oracle_single_plan_audit_v1", valid=not issues, issues=issues,
                  suite_hash=suite["suite_hash"], cases=len(suite["cases"]), models=len(suite["models"]), scenes=scenes)
    _atomic_write_json(root / "audit.json", result)
    return result


def implementation_hashes():
    paths = ["scripts/benchmarks/oracle_single_plan.py", "scripts/benchmarks/oracle_sampling.py",
        "mpd/inference/prescribed_world.py", "scripts/runtime/infer_once.py", "scripts/runtime/runtime_engine.py",
        "scripts/runtime/dynamic_runtime_engine.py", "scripts/runtime/space_time_runtime_engine.py",
        "scripts/runtime/factorized_runtime_engine.py", "scripts/runtime/jointdual_runtime_engine.py",
        "scripts/spacetime_data/warehouse_world.py"]
    return {path: _sha256_file(ROOT / path) for path in paths}


def execution_contract(suite, device):
    return dict(schema="oracle_single_plan_execution_v1", suite_hash=suite["suite_hash"], device=device,
        implementation_hashes=implementation_hashes(),
        runtime_configs=suite.get("runtime_configs") or {scene: yaml.safe_load(resolve(spec["runtime_config"]).read_text())
            for scene, spec in suite["config"]["scenes"].items()},
        jointdual_runtime_config=suite.get("jointdual_runtime_config") or yaml.safe_load(
            (ROOT / "scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-jointdual.yaml").read_text()))


def experiment_root(root, name):
    if name in ("", ".", "..") or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("run-name must be a simple directory name")
    return root if name == "default" else root / "experiments" / name


def cpu_partitions(jobs, threads):
    """Keep sibling threads together so workers do not share physical cores."""
    groups = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((topology / "physical_package_id").read_text().strip(),
               (topology / "core_id").read_text().strip())
        groups.setdefault(key, []).append(cpu)
    # On hybrid Intel machines do not mix SMT P cores with single-thread E
    # cores across models. Prefer the class with the largest sibling count.
    siblings = max(map(len, groups.values()))
    partitions, current = [], []
    for cpus in (g for g in groups.values() if len(g) == siblings):
        current.extend(cpus)
        if len(current) >= threads:
            partitions.append(current); current = []
    if len(partitions) < jobs:
        raise ValueError("not enough distinct CPU core groups for the requested jobs/threads")
    return partitions[:jobs]


def worker(args):
    if args.cpu_affinity:
        os.sched_setaffinity(0, {int(x) for x in args.cpu_affinity.split(",")})
    import torch
    from mpd.inference.prescribed_world import PrescribedDynamicWorld
    from scripts.runtime.infer_once import NoValidTrajectoryError, ResultValidationError
    root = args.suite.resolve(); suite = verify_suite(root)
    experiment = experiment_root(root, args.run_name)
    contract_file = experiment / "execution_contract.json"
    contract = read_json(contract_file) if contract_file.exists() else execution_contract(suite, args.device)
    if (contract["suite_hash"] != suite["suite_hash"] or contract["device"] != args.device
            or contract["implementation_hashes"] != implementation_hashes()):
        raise ValueError("execution code, device or suite changed; use a new experiment directory")
    suite.update(runtime_configs=contract["runtime_configs"], jointdual_runtime_config=contract["jointdual_runtime_config"])
    torch.set_num_threads(args.threads)
    model = next(m for m in suite["models"] if m["id"] == args.model)
    cases = [c for c in suite["cases"] if c["scene"] == args.scene]
    if args.case_limit is not None:
        cases = cases[:args.case_limit]
    for key in ("checkpoint", "spatial_checkpoint", "timing_checkpoint"):
        if key in model and _sha256_file(Path(model[key])) != model[key + "_sha256"]:
            raise ValueError(f"checkpoint changed since suite generation: {model[key]}")
    output = experiment / "results" / args.scene / args.model
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    engine = build_engine(suite, args.scene, model, output, args.device)
    health = engine.health()
    if health["scene_sha256"] != suite["scenes"][args.scene]["static_scene_sha256"]:
        raise ValueError("planner static geometry differs from the frozen benchmark scene")
    if model["kind"] != "jointdual" and health["checkpoint_sha256"] != model["spatial_checkpoint_sha256"]:
        raise ValueError("loaded spatial checkpoint differs from frozen manifest")
    _atomic_write_json(output / "engine.json", dict(health=health, load_and_warmup_s=time.perf_counter() - started,
        model=model, device=args.device, suite_hash=suite["suite_hash"], oracle_world=True, inflation=False,
        execution_contract_sha256=_canonical_json_sha256(contract), torch_version=torch.__version__,
        torch_threads=torch.get_num_threads(), cpu_affinity=sorted(os.sched_getaffinity(0)),
        gpu_name=torch.cuda.get_device_name() if args.device.startswith("cuda") else None))
    collision = scene_collision(suite["config"]["scenes"][args.scene]["scene_id"], args.device, suite["config"]["margin_m"])
    for case in cases:
        for repeat in range(args.repeats):
            run_dir = output / case["id"] / f"repeat-{repeat:02d}"
            result_path = run_dir / "benchmark.json"
            if args.resume and result_path.exists():
                old = read_json(result_path)
                if old.get("suite_hash") != suite["suite_hash"] or old.get("device") != args.device:
                    raise ValueError("cannot resume results from a different suite/device")
                continue
            case_dir = root / case["directory"]
            for name in ("request", "world", "reference"):
                file = case_dir / (name + (".json" if name == "request" else ".npz"))
                if _sha256_file(file) != case[name + "_sha256"]:
                    raise ValueError(f"frozen case changed: {file}")
            request = read_json(case_dir / "request.json")
            request["seed"] = int((case["seed"] + repeat * 10000019) % 2**32)
            world = PrescribedDynamicWorld(16, trajectory_duration_s=14., tensor_args=collision.tensor_args,
                                          fused_reduction_enabled=True, shape_grouping_enabled=True)
            with np.load(case_dir / "world.npz", allow_pickle=False) as arrays:
                world.load_forecast(**{k: arrays[k] for k in ("times", "positions", "shapes", "sizes")})
            request.update(_dynamic_world_version=world.world_version, _trajectory_start_unix_ns=1)
            if model["kind"] != "jointdual":
                with np.load(case_dir / "world.npz", allow_pickle=False) as arrays:
                    version = engine.dynamic_world.load_forecast(**{k: arrays[k] for k in ("times", "positions", "shapes", "sizes")})
                request["_dynamic_world_version"] = version
            row = dict(schema="oracle_single_plan_result_v1", suite_hash=suite["suite_hash"], scene=args.scene,
                model_id=model["id"], case_id=case["id"], repeat=repeat, seed=request["seed"], device=args.device,
                object_count=case["object_count"], arrival_mode=case["arrival_mode"], motion_modes=case["motion_modes"],
                request_sha256=case["request_sha256"], world_sha256=case["world_sha256"], success=False)
            row["execution_contract_sha256"] = _canonical_json_sha256(contract)
            run_dir.mkdir(parents=True, exist_ok=True)
            for name in ("trajectory.npz", "result.json", "benchmark.json"):
                (run_dir / name).unlink(missing_ok=True)
            _atomic_write_json(run_dir / "request.json", request)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                artifacts = engine.plan(request, oracle_world=world) if model["kind"] == "jointdual" else engine.plan(request)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                row["plan_wall_s"] = time.perf_counter() - started
                validation_started = time.perf_counter()
                check = validate_artifact(artifacts, collision, world, request, suite["config"]["inference"])
                row.update(validation=check, success=check["valid"],
                    status="success" if check["valid"] else "common_validation_failed",
                    common_validation_wall_s=time.perf_counter() - validation_started,
                    candidates=artifacts.result_payload.get("candidates"), timing=artifacts.result_payload.get("timing"))
                _atomic_write_npz(run_dir / "trajectory.npz", compressed=True, **artifacts.trajectory_arrays)
                _atomic_write_json(run_dir / "result.json", artifacts.result_payload)
            except Exception as error:
                row.update(plan_wall_s=time.perf_counter() - started,
                    status=("no_valid_trajectory" if isinstance(error, NoValidTrajectoryError) else
                            "native_validation_failed" if isinstance(error, ResultValidationError) else "inference_error"),
                    error_type=type(error).__name__, error_message=str(error))
            row["total_wall_s"] = time.perf_counter() - started
            _atomic_write_json(result_path, row)
            print(f"{args.model} {case['id']} repeat={repeat} {row['status']} {row['total_wall_s']:.3f}s", flush=True)
    return 0


def report(root, run_name="default"):
    suite = verify_suite(root)
    experiment = experiment_root(root, run_name)
    rows = [read_json(p) for p in sorted((experiment / "results").glob("*/*/*/repeat-*/benchmark.json"))]
    if any(r.get("suite_hash") != suite["suite_hash"] for r in rows):
        raise ValueError("refusing to aggregate results from different frozen suites")
    groups = {}
    for scene in suite["scenes"]:
        for model in suite["models"]:
            selected = [r for r in rows if r["scene"] == scene and r["model_id"] == model["id"]]
            valid = [r for r in selected if r["success"]]
            groups[scene + "/" + model["id"]] = dict(scene=scene, model_id=model["id"],
                completed=len(selected), successes=len(valid), success_rate=len(valid) / len(selected) if selected else None,
                status_counts=dict(Counter(r["status"] for r in selected)),
                mean_plan_wall_s=float(np.mean([r["plan_wall_s"] for r in selected])) if selected else None,
                mean_success_duration_s=float(np.mean([r["validation"]["duration_s"] for r in valid])) if valid else None,
                by_obstacle_count={str(n): dict(total=sum(r["object_count"] == n for r in selected),
                    success=sum(r["object_count"] == n and r["success"] for r in selected)) for n in range(6)},
                by_arrival_mode={mode: dict(total=sum(r["arrival_mode"] == mode for r in selected),
                    success=sum(r["arrival_mode"] == mode and r["success"] for r in selected))
                    for mode in ("empty", "single", "simultaneous", "sequential", "grouped")})
            groups[scene + "/" + model["id"]]["missing_case_ids"] = sorted(
                {c["id"] for c in suite["cases"] if c["scene"] == scene} - {r["case_id"] for r in selected})
    payload = dict(schema="oracle_single_plan_report_v1", suite_hash=suite["suite_hash"],
        expected_cases=len(suite["cases"]), expected_models=len(suite["models"]), result_count=len(rows), groups=groups)
    payload["devices"] = sorted({r["device"] for r in rows})
    payload["complete_for_one_repeat"] = all(not g["missing_case_ids"] for g in groups.values())
    payload["execution_contracts"] = sorted({r.get("execution_contract_sha256", "legacy_debug") for r in rows})
    if len(payload["devices"]) > 1:
        raise ValueError("refusing to aggregate mixed-device runs")
    contract_file = experiment / "execution_contract.json"
    payload["resources"] = read_json(contract_file).get("resources") if contract_file.exists() else None
    _atomic_write_json(experiment / "report.json", payload)
    text = ["# 已知障碍运动的单次规划 benchmark", "", f"Suite: `{suite['suite_hash']}`", "",
        "世界使用相同的冻结分段线性轨迹，无预测膨胀、重规划或 handoff。耗时包含本轮规划与验证；加载预热另存 engine.json。",
        "", "仅运行部分 case 的结果属于子集实验；未运行项显示 —，不算作失败。不同设备结果应分目录保存。", "",
        "| 场景 | 模型 | 已完成 | 公共验证成功 | 成功率 | 平均规划秒 | 成功轨迹平均时长秒 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    if payload["resources"]:
        resource = payload["resources"]
        text[4:4] = [f"设备：{', '.join(payload['devices']) or '尚未产生结果'}；资源协议：`{resource['latency_protocol']}`，"
                     f"jobs={resource['jobs']}，每 worker Torch threads={resource['threads']}。并行 CPU 实验共享内存和 I/O，耗时用于描述该配置，延迟排名需顺序复测。", ""]
    def fmt(v):
        return "—" if v is None else f"{v:.4f}"
    for g in groups.values():
        text.append(f"| {g['scene']} | {g['model_id']} | {g['completed']} | {g['successes']} | {fmt(g['success_rate'])} | {fmt(g['mean_plan_wall_s'])} | {fmt(g['mean_success_duration_s'])} |")
    text.extend(["", "各自成功样本的平均时长仅作描述，不能单独作为质量排名；完整失败分类及障碍数/到达模式分组见 report.json。", ""])
    (experiment / "report.md").write_text("\n".join(text))
    return payload


def run(args):
    root = args.suite.resolve(); suite = verify_suite(root)
    experiment = experiment_root(root, args.run_name)
    selected = args.models or [m["id"] for m in suite["models"]]
    if set(selected) - {m["id"] for m in suite["models"]}:
        raise ValueError("unknown selected model")
    scenes = args.scenes or list(suite["scenes"])
    if set(scenes) - set(suite["scenes"]):
        raise ValueError("unknown selected scene")
    failures = []
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; choose --device cpu for a CPU experiment")
        if args.jobs != 1:
            raise ValueError("GPU benchmark runs must use --jobs 1")
    partitions = cpu_partitions(args.jobs, args.threads) if args.device == "cpu" and args.jobs > 1 else [[]]
    contract_file = experiment / "execution_contract.json"
    contract = execution_contract(suite, args.device)
    contract["resources"] = dict(jobs=args.jobs, threads=args.threads, cpu_partitions=partitions,
        latency_protocol="disjoint_cpu_core_groups_shared_memory" if args.jobs > 1 else "sequential")
    if contract_file.exists() and read_json(contract_file) != contract:
        raise ValueError("execution code/config/device changed; use a new experiment directory")
    _atomic_write_json(contract_file, contract)
    leases = Queue()
    for partition in partitions:
        leases.put(partition)

    def evaluate(scene, model):
        partition = leases.get()
        try:
            log = experiment / "results" / scene / model / "worker.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "scripts.benchmarks.oracle_single_plan", "_worker", "--suite", str(root),
                   "--scene", scene, "--model", model, "--device", args.device, "--threads", str(args.threads),
                   "--repeats", str(args.repeats), "--run-name", args.run_name]
            if partition:
                cmd += ["--cpu-affinity", ",".join(map(str, partition))]
            if args.case_limit is not None:
                cmd += ["--case-limit", str(args.case_limit)]
            if args.resume:
                cmd += ["--resume"]
            print(f"running {scene}/{model}; log: {log}", flush=True)
            environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                               OPENBLAS_NUM_THREADS="1")
            with log.open("a" if args.resume else "w") as stream:
                answer = subprocess.run(cmd, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            return dict(scene=scene, model=model, exit_code=answer.returncode, log=str(log))
        finally:
            leases.put(partition)

    completed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(evaluate, scene, model) for scene in scenes for model in selected]
        for future in as_completed(futures):
            result = future.result(); completed.append(result)
            if result["exit_code"]:
                failures.append(result)
            report(root, args.run_name)
            _atomic_write_json(experiment / "run_status.json", dict(worker_failures=failures, completed_workers=completed,
                requested_models=selected, requested_scenes=scenes, repeats=args.repeats, case_limit=args.case_limit,
                jobs=args.jobs, status="running" if len(completed) < len(futures) else "complete"))
    return 1 if failures else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    gen.add_argument("--output", type=Path, required=True)
    for name in ("cases-per-scene", "seed", "candidates", "steps", "refinement-steps"):
        gen.add_argument("--" + name, type=int)
    gen.add_argument("--threads", type=int, default=4)
    runner = sub.add_parser("run")
    runner.add_argument("--jobs", type=int, default=1)
    runner.add_argument("--models", nargs="+")
    runner.add_argument("--scenes", nargs="+")
    worker_p = sub.add_parser("_worker")
    worker_p.add_argument("--scene", required=True)
    worker_p.add_argument("--model", required=True)
    worker_p.add_argument("--cpu-affinity")
    for child in (runner, worker_p):
        child.add_argument("--suite", type=Path, required=True)
        child.add_argument("--run-name", default="default")
        child.add_argument("--device", default="cuda:0")
        child.add_argument("--threads", type=int, default=4)
        child.add_argument("--repeats", type=int, default=1)
        child.add_argument("--case-limit", type=int)
        child.add_argument("--resume", action="store_true")
    rep = sub.add_parser("report")
    rep.add_argument("--suite", type=Path, required=True)
    rep.add_argument("--run-name", default="default")
    audit_p = sub.add_parser("audit")
    audit_p.add_argument("--suite", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if getattr(args, "jobs", 1) < 1 or getattr(args, "repeats", 1) < 1 or getattr(args, "threads", 1) < 1 or (getattr(args, "case_limit", None) is not None and args.case_limit < 1):
        raise ValueError("repeats, threads and case-limit must be positive")
    if args.command == "generate":
        return generate(args)
    if args.command == "run":
        return run(args)
    if args.command == "_worker":
        return worker(args)
    if args.command == "audit":
        result = audit(args.suite.resolve())
        print(json.dumps(result, indent=2))
        return 0 if result["valid"] else 1
    report(args.suite.resolve(), args.run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
