#!/usr/bin/env python3
"""Validate and launch Marvin bimanual Warehouse training."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from scripts.train.train_marvin_bimanual import results_dir_for_variant, validate_config, run_training


def validate_dataset(config):
    """Fail before training on an old scene, different TCP or unverified spline."""
    import h5py
    import numpy as np
    from mpd.paths import DATASET_BASE_DIR
    from scripts.generate_data.generate_marvin_warehouse_bimanual import file_sha256, JOINT_NAMES
    from torch_robotics.environments.env_warehouse_marvin_bimanual import EnvWarehouseMarvinBimanual
    from mpd.datasets.trajectories_dataset_bspline import adjust_bspline_number_control_points

    root = Path(DATASET_BASE_DIR) / config["dataset_subdir"]
    manifest = yaml.safe_load((root / "manifest.yaml").read_text())
    generation = yaml.safe_load((root / "generation_config.yaml").read_text())
    args = yaml.safe_load((root / "args.yaml").read_text())
    if args.get("env_id") != "EnvWarehouseMarvinBimanual" or args.get("robot_id") != "RobotMarvinBimanual":
        raise ValueError("dataset environment/robot mismatch")
    if manifest.get("scene_version") != EnvWarehouseMarvinBimanual.scene_version:
        raise ValueError("old Warehouse scene: regenerate or migrate the v3 dataset")
    if manifest.get("schema") != "marvin_bimanual_warehouse_dataset/v3" or manifest.get(
        "ee_goal_schema"
    ) != "marvin_dual_pika_tcp/v1":
        raise ValueError("dataset manifest does not declare the Marvin dual-slot EE schema")
    robot_dir = Path(__file__).resolve().parents[2] / "mpd/torch_robotics/torch_robotics/data/urdf/robots/marvin"
    if manifest.get("model_sha256") != file_sha256(robot_dir / "marvin_pika_bimanual_mpd.urdf"):
        raise ValueError("dataset robot URDF differs from current Pika model")
    if (
        manifest.get("asset_sha256")
        != yaml.safe_load((robot_dir / "pika_assets.lock.yaml").read_text())["asset_sha256"]
    ):
        raise ValueError("dataset collision/TCP assets differ from current model")
    dataset_file = root / config.get("dataset_file_merged", "dataset_merged.hdf5")
    if file_sha256(dataset_file) != manifest["dataset_sha256"]:
        raise ValueError("dataset hash does not match completed generation manifest")
    desired_count = int(config.get("bspline_num_control_points_desired", 22))
    count, _ = adjust_bspline_number_control_points(desired_count, True, True, True, True)
    if config.get("bspline_num_control_points_exact", False):
        count = desired_count
    if generation.get("bspline_num_control_points") != count or generation.get("bspline_degree") != config.get(
        "bspline_degree", 5
    ):
        raise ValueError("training spline differs from the collision-validated generation spline")
    with h5py.File(dataset_file, "r") as data:
        if not data.attrs.get("validated_splines") or tuple(data.attrs["joint_names"]) != JOINT_NAMES:
            raise ValueError("dataset needs validated splines in canonical joint order")
        n = len(data["sol_path"])
        if n < 2 or data["sol_path"].shape[-1] != 14 or data["bspline_params_cc"].shape != (n, 14, count):
            raise ValueError("invalid path/control point shape")
        if data["ee_goal_pose"].shape != (n, 2, 3, 4) or data["active_ee_mask"].shape != (n, 2):
            raise ValueError("dataset lacks dual-slot EE goals/masks")
        if data.attrs.get("ee_goal_schema") != "marvin_dual_pika_tcp/v1":
            raise ValueError("unknown or missing dual Pika TCP schema")
        for start in range(0, n, 256):
            cc = data["bspline_params_cc"][start : start + 256]
            pose = data["ee_goal_pose"][start : start + 256]
            mask = data["active_ee_mask"][start : start + 256]
            if not np.isfinite(cc).all() or not np.isfinite(pose).all():
                raise ValueError("nonfinite training control points/EE goals")
            rotations = pose[..., :3]
            identity = np.einsum("...ji,...jk->...ik", rotations, rotations)
            if not np.allclose(identity, np.eye(3), atol=2e-4) or not np.isin(mask, [0, 1]).all():
                raise ValueError("invalid EE rotation matrix or activity mask")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-dataset", action="store_true", help="also validate data without training")
    parser.add_argument("--dataset-subdir")
    parser.add_argument("--device")
    parser.add_argument("--num-train-steps", type=int)
    parser.add_argument("--results-dir")
    parser.add_argument(
        "--warm-start-checkpoint",
        type=Path,
        help="load a *_state_dict.pth model/EMA checkpoint and restart optimization at step 0",
    )
    parser.add_argument("--network-variant", type=str.upper, choices=("A", "B", "C", "D"))
    parser.add_argument(
        "--no-summary", action="store_true", help="skip trajectory sampling/plots for a short training smoke test"
    )
    args = parser.parse_args(argv)
    if args.warm_start_checkpoint is not None and args.results_dir is None:
        parser.error("--warm-start-checkpoint requires a new --results-dir to avoid overwriting the source run")
    config = yaml.safe_load(args.config.read_text()) or {}
    if args.network_variant is not None:
        config["bimanual_network_variant"] = args.network_variant
    if args.results_dir is None and "bimanual_network_variant" in config and config.get("results_dir"):
        config["results_dir"] = results_dir_for_variant(
            config["results_dir"], str(config["bimanual_network_variant"]).upper()
        )
    config = validate_config(config)
    if not config.get("context_ee_goal_pose") or not config.get("context_ee_goal_pose_bimanual"):
        raise ValueError("Warehouse Marvin training requires scheme-3 dual-slot EE context")
    for key in ("dataset_subdir", "device", "num_train_steps", "results_dir"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.warm_start_checkpoint is not None:
        checkpoint = args.warm_start_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Warm-start state_dict not found: {checkpoint}")
        if checkpoint.stat().st_size == 0:
            raise ValueError(f"Warm-start state_dict is empty: {checkpoint}")
        config["warm_start_checkpoint"] = str(checkpoint)
    if args.no_summary:
        config["summary_class"] = None
    dataset_subdir = str(config.get("dataset_subdir", ""))
    if not dataset_subdir.startswith("EnvWarehouse-RobotMarvinBimanual"):
        raise ValueError("Warehouse training requires an EnvWarehouse-RobotMarvinBimanual dataset")
    if config["task_family"] != "independent":
        raise ValueError("Warehouse cooperative object constraints are not implemented")
    print(
        f"validated Marvin Warehouse {config['task_family']} config: state_dim=14 "
        f"raw_context_dim=40 network_variant={config['bimanual_network_variant']}"
    )
    if args.check_dataset or not args.dry_run:
        validate_dataset(config)
    if args.dry_run or args.check_dataset:
        return 0
    return run_training(config)


if __name__ == "__main__":
    raise SystemExit(main())
