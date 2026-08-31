#!/usr/bin/env python3
"""Run a paired one-factor-off ablation of all Phase-4 aligned changes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from scripts.isaaclab import benchmark_todrawer_random as benchmark


ABLATION_MODE_SPECS = {
    "phase4": ("phase4", None),
    "aligned_all": ("phase4_aligned", None),
    "aligned_no_deviation": ("phase4_aligned", None),
    "aligned_no_relative_hysteresis": ("phase4_aligned", None),
    "aligned_no_clearance_split": ("phase4_aligned", None),
    "aligned_no_tail_kinematic": ("phase4_aligned", None),
    "aligned_no_dynamic_guidance": ("phase4_aligned", None),
    "aligned_no_dynamic_selection": ("phase4_aligned", None),
}

ABLATION_PIPELINE_ARGS = {
    "aligned_no_deviation": ("--aligned-deviation", "off"),
    "aligned_no_relative_hysteresis": (
        "--aligned-relative-hysteresis",
        "off",
    ),
    "aligned_no_clearance_split": ("--aligned-clearance-split", "off"),
    "aligned_no_tail_kinematic": ("--aligned-tail-kinematic", "off"),
    "aligned_no_dynamic_guidance": ("--aligned-mpd-guidance", "off"),
    "aligned_no_dynamic_selection": ("--aligned-mpd-selection", "off"),
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
        output_dir=benchmark.REPO_ROOT
        / "scripts"
        / "isaaclab"
        / "logs"
        / "phase4-aligned-ablation"
        / timestamp,
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
