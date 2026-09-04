#!/usr/bin/env python3
"""Validate and launch Marvin bimanual Warehouse training."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from scripts.train.train_marvin_bimanual import validate_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args, unknown = parser.parse_known_args(argv)
    config = validate_config(yaml.safe_load(args.config.read_text()) or {})
    dataset_subdir = str(config.get("dataset_subdir", ""))
    if not dataset_subdir.startswith("EnvWarehouse-RobotMarvinBimanual"):
        raise ValueError("Warehouse training requires an EnvWarehouse-RobotMarvinBimanual dataset")
    print(f"validated Marvin Warehouse {config['task_family']} config: state_dim=14 context_q_dim=28")
    if args.dry_run:
        return 0
    from scripts.train.train import experiment

    return experiment(**config)


if __name__ == "__main__":
    raise SystemExit(main())
