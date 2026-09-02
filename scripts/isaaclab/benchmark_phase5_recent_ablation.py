#!/usr/bin/env python3
"""Run paired Phase-5 joint one-factor ablations against Phase-4 controls."""

from __future__ import annotations

from datetime import datetime

from scripts.isaaclab import benchmark_todrawer_random as benchmark


ABLATION_MODE_SPECS = {
    "phase4": ("phase4", None),
    "phase4_aligned": ("phase4_aligned", None),
    "phase5_all": ("phase5", "phase5_joint"),
    "phase5_no_deviation": ("phase5", "phase5_joint"),
    "phase5_no_adaptive_deviation": ("phase5", "phase5_joint"),
    "phase5_no_relative_hysteresis": ("phase5", "phase5_joint"),
    "phase5_no_clearance_split": ("phase5", "phase5_joint"),
    "phase5_no_tail_kinematic": ("phase5", "phase5_joint"),
    "phase5_no_dynamic_guidance": ("phase5", "phase5_joint"),
    "phase5_no_dynamic_selection": ("phase5", "phase5_joint"),
    "phase5_old_hold_0_2": ("phase5", "phase5_joint"),
    "phase5_old_tail_4": ("phase5", "phase5_joint"),
    "phase5_old_dynamic_grad_cap_1": ("phase5", "phase5_joint"),
}

ABLATION_PIPELINE_ARGS = {
    "phase5_no_deviation": ("--phase5-deviation", "off"),
    "phase5_no_adaptive_deviation": (
        "--phase5-adaptive-deviation",
        "off",
    ),
    "phase5_no_relative_hysteresis": (
        "--phase5-relative-hysteresis",
        "off",
    ),
    "phase5_no_clearance_split": ("--phase5-clearance-split", "off"),
    "phase5_no_tail_kinematic": ("--phase5-tail-weight", "1.0"),
    "phase5_no_dynamic_guidance": ("--phase5-mpd-guidance", "off"),
    "phase5_no_dynamic_selection": ("--phase5-mpd-selection", "off"),
    "phase5_old_hold_0_2": ("--phase5-hold-weight", "0.2"),
    "phase5_old_tail_4": ("--phase5-tail-weight", "4.0"),
    "phase5_old_dynamic_grad_cap_1": (
        "--phase5-dynamic-grad-cap",
        "1.0",
    ),
}


def planned_run_count(scenario_count: int, repeats: int, modes) -> int:
    return int(scenario_count) * int(repeats) * len(modes)


def _parser():
    original_specs = benchmark.MODE_SPECS
    original_pipeline_args = benchmark.MODE_PIPELINE_ARGS
    try:
        benchmark.MODE_SPECS = dict(ABLATION_MODE_SPECS)
        benchmark.MODE_PIPELINE_ARGS = dict(ABLATION_PIPELINE_ARGS)
        parser = benchmark._parser()
    finally:
        benchmark.MODE_SPECS = original_specs
        benchmark.MODE_PIPELINE_ARGS = original_pipeline_args
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.description = __doc__
    parser.set_defaults(
        output_dir=benchmark.REPO_ROOT / "scripts" / "isaaclab" / "logs" / "phase5-recent-ablation" / timestamp,
        scenario_count=40,
        repeats=2,
        modes=list(ABLATION_MODE_SPECS),
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.repeats < 1 or args.scenario_count < 1:
        raise SystemExit("scenario-count and repeats must be positive")
    if args.dry_run:
        args.skip_build = True
    print(
        "planned paired runs: "
        f"{planned_run_count(args.scenario_count, args.repeats, args.modes)} "
        f"({args.scenario_count} scenarios x {args.repeats} repeats x "
        f"{len(args.modes)} modes)"
    )
    benchmark.MODE_SPECS = dict(ABLATION_MODE_SPECS)
    benchmark.MODE_PIPELINE_ARGS = dict(ABLATION_PIPELINE_ARGS)
    return benchmark.run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
