#!/usr/bin/env python3
"""Serve Phase-5 inference-only space-time MPD on a separate socket."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.runtime.infer_dynamic_server import DynamicResidentPlannerService
from scripts.runtime.infer_once import DEFAULT_CONFIG_PATH
from scripts.runtime.space_time_runtime_engine import SpaceTimeMpdRuntimeEngine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase-5 space-time MPD service.")
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-dynamic-objects", type=int, default=16)
    parser.add_argument("--covariance-sigma", type=float, default=3.0)
    parser.add_argument("--process-acceleration-std", type=float, default=0.01)
    parser.add_argument(
        "--timing-mode",
        choices=(
            "phase5_scalar_duration",
            "phase5_timing_only",
            "phase5_joint",
        ),
        default="phase5_joint",
    )
    parser.add_argument("--timing-control-points", type=int, default=8)
    parser.add_argument("--timing-degree", type=int, default=3)
    parser.add_argument("--u-min", type=float, default=0.05)
    parser.add_argument("--duration-min", type=float, default=6.0)
    parser.add_argument("--duration-max", type=float, default=14.0)
    parser.add_argument("--nominal-duration", type=float, default=10.0)
    parser.add_argument("--timing-learning-rate", type=float, default=0.08)
    parser.add_argument(
        "--static-spatial-pruning",
        dest="static_spatial_pruning",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-static-spatial-pruning",
        dest="static_spatial_pruning",
        action="store_false",
    )
    parser.add_argument(
        "--dynamic-space-time-pruning",
        action="store_true",
        default=False,
        help="Reserved fail-closed switch; candidate-specific pruning is not implemented.",
    )
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    settings = {
        "num_timing_control_points": args.timing_control_points,
        "timing_degree": args.timing_degree,
        "u_min": args.u_min,
        "duration_min": args.duration_min,
        "duration_max": args.duration_max,
        "nominal_duration": args.nominal_duration,
        "timing_learning_rate": args.timing_learning_rate,
    }

    def engine_factory(state_callback):
        return SpaceTimeMpdRuntimeEngine(
            config_path=args.config,
            runtime_output_root=args.output_root,
            device_text=args.device,
            state_callback=state_callback,
            timing_mode=args.timing_mode,
            space_time_settings=settings,
            max_dynamic_objects=args.max_dynamic_objects,
            covariance_sigma=args.covariance_sigma,
            process_acceleration_std_m_s2=args.process_acceleration_std,
            static_spatial_pruning_enabled=args.static_spatial_pruning,
            dynamic_space_time_pruning_enabled=args.dynamic_space_time_pruning,
        )

    service = DynamicResidentPlannerService(
        args.socket,
        args.output_root,
        engine_factory,
        trajectory_compression=False,
    )
    service.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
