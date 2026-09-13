#!/usr/bin/env python3
"""Serve learned factorized F1/F2/F3 MPD on a separate socket."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.runtime.factorized_runtime_engine import FactorizedMpdRuntimeEngine
from scripts.runtime.infer_dynamic_server import DynamicResidentPlannerService
from scripts.runtime.infer_once import DEFAULT_CONFIG_PATH


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the learned factorized F1/F2/F3 space-time MPD service."
    )
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timing-checkpoint", required=True, type=Path)
    parser.add_argument("--method", choices=("f1", "f2", "f3"), default="f1")
    parser.add_argument(
        "--adapt-spatial-basis",
        action="store_true",
        help=(
            "Evaluate the timing encoder with the runtime spatial spline basis. "
            "Required when its control-point count differs from training."
        ),
    )
    parser.add_argument("--max-dynamic-objects", type=int, default=16)
    parser.add_argument("--covariance-sigma", type=float, default=3.0)
    parser.add_argument("--process-acceleration-std", type=float, default=0.01)
    parser.add_argument("--duration-min", type=float, default=2.0)
    parser.add_argument("--duration-max", type=float, default=14.0)
    parser.add_argument("--nominal-duration", type=float, default=10.0)
    parser.add_argument("--u-min", type=float, default=0.05)
    parser.add_argument("--spatial-dynamic-max-grad-norm", type=float, default=2.0)
    parser.add_argument("--space-steps", type=int, default=32)
    parser.add_argument("--timing-steps", type=int, default=100)
    parser.add_argument("--space-guide-fraction", type=float, default=0.3)
    parser.add_argument("--timing-guide-fraction", type=float, default=0.3)
    parser.add_argument("--space-lr", type=float, default=0.01)
    parser.add_argument("--timing-lr", type=float, default=0.02)
    parser.add_argument("--refinement-steps", type=int, default=5)
    parser.add_argument("--weak-dynamic-scale", type=float, default=0.2)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--alternating-rounds", type=int, default=1)
    parser.add_argument("--timing-steps-per-space-step", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--dynamic-guidance", dest="dynamic_guidance", action="store_true", default=True
    )
    parser.add_argument(
        "--no-dynamic-guidance", dest="dynamic_guidance", action="store_false"
    )
    parser.add_argument(
        "--dynamic-selection", dest="dynamic_selection", action="store_true", default=True
    )
    parser.add_argument(
        "--no-dynamic-selection", dest="dynamic_selection", action="store_false"
    )
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
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    space_time_settings = {
        "num_timing_control_points": 8,
        "timing_degree": 3,
        "u_min": args.u_min,
        "duration_min": args.duration_min,
        "duration_max": args.duration_max,
        "nominal_duration": args.nominal_duration,
        "spatial_dynamic_max_grad_norm": args.spatial_dynamic_max_grad_norm,
        "dynamic_guidance_enabled": args.dynamic_guidance,
    }
    factorized_settings = {
        key: getattr(args, key)
        for key in (
            "method",
            "space_steps",
            "timing_steps",
            "space_guide_fraction",
            "timing_guide_fraction",
            "space_lr",
            "timing_lr",
            "refinement_steps",
            "weak_dynamic_scale",
            "eta",
            "alternating_rounds",
            "timing_steps_per_space_step",
        )
    }

    def engine_factory(state_callback):
        return FactorizedMpdRuntimeEngine(
            config_path=args.config,
            runtime_output_root=args.output_root,
            device_text=args.device,
            state_callback=state_callback,
            timing_checkpoint=args.timing_checkpoint,
            factorized_settings=factorized_settings,
            adapt_spatial_basis=args.adapt_spatial_basis,
            space_time_settings=space_time_settings,
            max_dynamic_objects=args.max_dynamic_objects,
            covariance_sigma=args.covariance_sigma,
            process_acceleration_std_m_s2=args.process_acceleration_std,
            static_spatial_pruning_enabled=args.static_spatial_pruning,
            dynamic_space_time_pruning_enabled=False,
            dynamic_selection_enabled=args.dynamic_selection,
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
