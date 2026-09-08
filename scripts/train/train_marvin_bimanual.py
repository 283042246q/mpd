#!/usr/bin/env python3
"""Validated thin entrypoint around the existing dimension-agnostic trainer."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import yaml


def validate_config(config: dict) -> dict:
    if config.get("robot_model") != "marvin_bimanual":
        raise ValueError("robot_model must be marvin_bimanual")
    if config.get("task_family") not in {"independent", "cooperative"}:
        raise ValueError("task_family must be independent or cooperative")
    if int(config.get("state_dim", 14)) != 14 or not config.get("context_qs", True):
        raise ValueError("Marvin checkpoints require state_dim=14 and context_qs=true")
    use_ee = bool(config.get("context_ee_goal_pose", False))
    use_dual_ee = bool(config.get("context_ee_goal_pose_bimanual", False))
    if use_ee != use_dual_ee:
        raise ValueError("Marvin EE conditioning must use both context_ee_goal_pose and dual-slot bimanual mode")
    expected_q_dim = 14 if use_dual_ee else 28
    if int(config.get("context_q_dim", expected_q_dim)) != expected_q_dim:
        raise ValueError(f"Marvin context_q_dim must be {expected_q_dim} for this context mode")
    if use_dual_ee and int(config.get("raw_context_dim", 40)) != 40:
        raise ValueError("dual-slot context is q_start(14) + left EE(12) + right EE(12) + mask(2) = 40")
    return config


def run_training(config):
    from scripts.train.train import experiment

    # Direct calls bypass experiment_launcher's CLI default resolution. Save
    # the complete model/diffusion configuration for checkpoint reconstruction.
    resolved = {
        name: parameter.default
        for name, parameter in inspect.signature(experiment).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    resolved.update(config)
    return experiment(**resolved)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate without starting a GPU training job")
    args = parser.parse_args(argv)
    config = validate_config(yaml.safe_load(args.config.read_text()) or {})
    raw_dim = 40 if config.get("context_ee_goal_pose_bimanual") else 28
    print(f"validated Marvin {config['task_family']} config: state_dim=14 raw_context_dim={raw_dim}")
    if args.dry_run:
        return 0
    return run_training(config)


if __name__ == "__main__":
    raise SystemExit(main())
