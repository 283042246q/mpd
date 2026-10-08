"""Reload and independently audit the saved Warehouse spacetime dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import BSpline
import torch

from mpd.datasets.spacetime_schema import load_panda_robot_bundle, sha256_file
from mpd.parametric_trajectory.timing_fitting import TimingSplineNumpy
from mpd.parametric_trajectory.normalized_timing import NormalizedTimingSplineNumpy, minimum_duration_for_path
from scripts.spacetime_data.batch_spacetime_rrt import edge_collision_mask
from scripts.spacetime_data.generate_warehouse_spacetime import REPOSITORY, SCHEMA, read_progress
from scripts.spacetime_data.warehouse_trajectory_fit import audit_decoded
from scripts.spacetime_data.warehouse_world import Forecast, WarehouseCollision


def read_forecast(sample):
    metadata = json.loads(sample.attrs["world_metadata_json"])
    return Forecast(np.asarray(sample["world/times"]), np.asarray(sample["world/positions"]),
                    np.asarray(sample["world/shape"]), np.asarray(sample["world/size"]),
                    np.asarray(sample["world/event_times"]), metadata["modes"], np.asarray(sample["world/anchors"]))


def validate_dataset(root, *, device="cuda:0", limit=None):
    root = Path(root)
    manifest = json.loads((root / "config.json").read_text())
    index = json.loads((root / "index.json").read_text())
    if hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() != index["config_hash"]:
        raise ValueError("configuration hash differs from index")
    robot = load_panda_robot_bundle(REPOSITORY)
    if manifest["schema"] != SCHEMA or manifest["robot_hashes"] != robot.hashes:
        raise ValueError("schema/robot mismatch")
    for filename, key in (("robot.urdf", "urdf_sha256"), ("joint_limits.yaml", "joint_limits_sha256"),
                          ("collision_spheres.yaml", "collision_spheres_sha256"),
                          ("collision_parent_bounds.yaml", "collision_parent_bounds_sha256")):
        if sha256_file(root / "robot" / filename) != robot.hashes[key]:
            raise ValueError("exported robot asset changed: " + filename)
    for name, expected in manifest["static_geometry_hashes"].items():
        if sha256_file(REPOSITORY / "mpd/torch_robotics/torch_robotics/environments" / name) != expected:
            raise ValueError("static warehouse geometry changed: " + name)
    accepted, attempts, failures, committed = read_progress(root, index["config_hash"])
    entries = [entry for split in index["splits"].values() for entry in split]
    if sorted((e["file"], e["group"]) for e in entries) != sorted((e["file"], e["group"]) for e in committed):
        raise ValueError("index and committed shard contents differ; regenerate index with --resume")
    seen = {}
    for split, samples in index["splits"].items():
        for sample in samples:
            family = sample["family_id"]
            if family in seen and seen[family] != split:
                raise ValueError("source family leaked across splits")
            seen[family] = split
    collision = WarehouseCollision(device, margin=manifest["margin"], chunk_size=manifest["collision_chunk"])
    codec = TimingSplineNumpy(num_phase_points=2049)
    normalized = NormalizedTimingSplineNumpy(num_phase_points=2049)
    errors, summaries = [], []
    for entry in entries[:limit]:
        try:
            with h5py.File(root / entry["file"], "r") as shard:
                sample = shard[entry["group"]]
                forecast = read_forecast(sample)
                collision.set_worlds([forecast])
                spline = BSpline(np.asarray(sample["spatial/knots"]), np.asarray(sample["spatial/control_points"]),
                                 int(sample.attrs["degree"]))
                evaluation = codec.evaluate(np.asarray(sample["timing/control_points"]))
                quality, reason = audit_decoded(spline, codec.phase, evaluation.time_from_start,
                                                 evaluation.u, evaluation.u_s, collision, 0, forecast,
                                                 validation_dt=0.005)
                if quality is None:
                    raise ValueError("decoded c failed: " + str(reason))
                np.testing.assert_allclose(spline([0, 1]), np.stack([sample["condition/q_start"], sample["condition/q_goal"]]), atol=1e-6)
                for derivative in (1, 2):
                    np.testing.assert_allclose(spline.derivative(derivative)([0, 1]), 0, atol=1e-6)
                np.testing.assert_allclose(sample["timing/reference_time"], evaluation.time_from_start, atol=1e-7)
                execution_time = np.asarray(sample["execution/time"])
                if not np.all(np.diff(execution_time) > 0) or abs(execution_time[-1] - evaluation.duration) > 1e-6:
                    raise ValueError("saved execution time mismatch")
                phase = np.interp(execution_time, evaluation.time_from_start, evaluation.phase)
                np.testing.assert_allclose(sample["execution/q"], spline(phase), atol=1e-6)
                if bool(collision(np.asarray(sample["execution/q"]), execution_time, np.zeros(len(execution_time), dtype=int)).any().item()):
                    raise ValueError("saved execution collision")
                for name in sample["raw"]:
                    if not name.startswith("solution_"):
                        continue
                    path = np.asarray(sample["raw/" + name])
                    delta_t = np.diff(path[:, -1])
                    if path.ndim != 2 or path.shape[1] != 8 or not np.isfinite(path).all() or np.any(delta_t <= 0):
                        raise ValueError("invalid raw time path")
                    velocity = collision.robot.dq_max.cpu().numpy() * manifest["velocity_scale"]
                    if np.any(np.abs(np.diff(path[:, :7], axis=0)) > delta_t[:, None] * velocity + 1e-5):
                        raise ValueError("raw edge velocity violation")
                    tensor = torch.as_tensor(path, **collision.tensor_args)
                    blocked, _ = edge_collision_mask(tensor[:-1][None], tensor[1:][None],
                                                     torch.zeros((1, len(path) - 1), device=tensor.device, dtype=torch.long),
                                                     collision, joint_step=manifest["joint_step"] / 2,
                                                     time_step=manifest["edge_dt"] / 2)
                    if blocked.any().item():
                        raise ValueError("raw edge collision")
                tau_valid = bool(sample.attrs["tau_r_valid"])
                if tau_valid:
                    latent = np.asarray(sample["timing/tau_r"])
                    shape = normalized.evaluate(latent[1:])
                    t_min = minimum_duration_for_path(spline, shape, dq_max=collision.robot.dq_max.cpu().numpy(),
                                                      ddq_max=collision.robot.ddq_max.cpu().numpy(), duration_floor=2.).value
                    np.testing.assert_allclose(t_min, sample["timing/T_min"][()], atol=1e-6)
                    duration = t_min + (forecast.horizon - t_min) / (1 + np.exp(-latent[0]))
                    tau_quality, reason = audit_decoded(spline, normalized.phase, shape.normalized_time * duration,
                                                         shape.density * duration, shape.density_s * duration,
                                                         collision, 0, forecast, validation_dt=0.005)
                    if tau_quality is None:
                        raise ValueError("decoded tau_r failed: " + str(reason))
                summaries.append({"attempt_id": entry["attempt_id"], "duration": quality["duration"],
                                  "clearance_min": quality["clearance_min"].tolist(), "tau_r_valid": tau_valid,
                                  "modes": forecast.modes})
        except (ValueError, AssertionError, KeyError) as exc:
            errors.append({"entry": entry, "error": str(exc)})
    return {"accepted": accepted, "attempts": attempts, "checked": len(summaries) + len(errors),
            "passed": len(summaries), "errors": errors, "samples": summaries}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    torch.set_num_threads(1)
    result = validate_dataset(args.root, device=args.device, limit=args.limit)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
