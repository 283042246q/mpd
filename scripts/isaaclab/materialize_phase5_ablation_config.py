#!/usr/bin/env python3
"""Materialize one independently varied Phase-5 ROS scoring configuration."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml


def materialize_config(
    base: Path,
    output: Path,
    *,
    deviation: bool,
    adaptive_deviation: bool,
    relative_hysteresis: bool,
    clearance_split: bool,
    tail_kinematic_weight: float,
    terminal_hold_clearance_weight: float,
) -> dict:
    weights = (tail_kinematic_weight, terminal_hold_clearance_weight)
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("Phase-5 ablation weights must be finite and non-negative")
    payload = yaml.safe_load(base.read_text(encoding="utf-8"))
    try:
        parameters = payload["mpd_space_time_replanner"]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise ValueError("base config has no replanner ROS parameters") from error
    parameters["cost_deviation_weight"] = 0.15 if deviation else 0.0
    parameters["adaptive_deviation_weight_enabled"] = bool(adaptive_deviation)
    parameters["relative_switching_hysteresis"] = 0.10 if relative_hysteresis else 0.0
    parameters["split_terminal_hold_clearance"] = bool(clearance_split)
    parameters["cost_tail_kinematic_weight"] = float(tail_kinematic_weight)
    parameters["cost_terminal_hold_clearance_weight"] = float(terminal_hold_clearance_weight)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload


def _enabled(value: str) -> bool:
    return value == "on"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--deviation", choices=("on", "off"), default="on")
    parser.add_argument("--adaptive-deviation", choices=("on", "off"), default="on")
    parser.add_argument("--relative-hysteresis", choices=("on", "off"), default="on")
    parser.add_argument("--clearance-split", choices=("on", "off"), default="on")
    parser.add_argument("--tail-kinematic-weight", type=float, default=3.0)
    parser.add_argument("--terminal-hold-clearance-weight", type=float, default=0.5)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    materialize_config(
        args.base,
        args.output,
        deviation=_enabled(args.deviation),
        adaptive_deviation=_enabled(args.adaptive_deviation),
        relative_hysteresis=_enabled(args.relative_hysteresis),
        clearance_split=_enabled(args.clearance_split),
        tail_kinematic_weight=args.tail_kinematic_weight,
        terminal_hold_clearance_weight=args.terminal_hold_clearance_weight,
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
