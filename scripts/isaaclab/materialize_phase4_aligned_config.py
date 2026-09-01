#!/usr/bin/env python3
"""Materialize one independently switched Phase-4 aligned ROS configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def materialize_config(
    base: Path,
    output: Path,
    *,
    deviation: bool,
    relative_hysteresis: bool,
    clearance_split: bool,
    tail_kinematic: bool,
) -> dict:
    payload = yaml.safe_load(base.read_text(encoding="utf-8"))
    try:
        parameters = payload["mpd_dynamic_replanner"]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise ValueError("base config has no replanner ROS parameters") from error
    parameters["cost_deviation_weight"] = 0.15 if deviation else 0.0
    parameters["relative_switching_hysteresis"] = (
        0.10 if relative_hysteresis else 0.0
    )
    parameters["split_terminal_hold_clearance"] = bool(clearance_split)
    parameters["cost_tail_kinematic_weight"] = 4.0 if tail_kinematic else 1.0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload


def _enabled(value: str) -> bool:
    return value == "on"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    for name in (
        "deviation",
        "relative-hysteresis",
        "clearance-split",
        "tail-kinematic",
    ):
        parser.add_argument(f"--{name}", choices=("on", "off"), default="on")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    materialize_config(
        args.base,
        args.output,
        deviation=_enabled(args.deviation),
        relative_hysteresis=_enabled(args.relative_hysteresis),
        clearance_split=_enabled(args.clearance_split),
        tail_kinematic=_enabled(args.tail_kinematic),
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
