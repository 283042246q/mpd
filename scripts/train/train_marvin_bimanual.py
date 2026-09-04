#!/usr/bin/env python3
"""Validated thin entrypoint around the existing dimension-agnostic trainer."""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml


def validate_config(config: dict) -> dict:
    if config.get("robot_model") != "marvin_bimanual":
        raise ValueError("robot_model must be marvin_bimanual")
    if config.get("task_family") not in {"independent", "cooperative"}:
        raise ValueError("task_family must be independent or cooperative")
    if int(config.get("state_dim", 14)) != 14 or int(config.get("context_q_dim", 28)) != 28:
        raise ValueError("Marvin checkpoints require state_dim=14 and context_q_dim=28")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate without starting a GPU training job")
    args, _ = parser.parse_known_args(argv)
    config = validate_config(yaml.safe_load(args.config.read_text()) or {})
    print(f"validated Marvin {config['task_family']} config: state_dim=14 context_q_dim=28")
    if args.dry_run:
        return 0
    from scripts.train.train import experiment
    # The existing experiment launcher owns CLI/config expansion.  Passing the
    # validated file keeps all optimizer/network behavior identical to Franka.
    return experiment(config_file=str(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
