from pathlib import Path

import yaml

from scripts.isaaclab.benchmark_phase4_aligned_ablation import (
    ABLATION_MODE_SPECS,
    ABLATION_PIPELINE_ARGS,
    _parser,
    planned_run_count,
)
from scripts.isaaclab.materialize_phase4_aligned_config import materialize_config


ALIGNED_CONFIG = Path(
    "/home/eric/Projects/physical_ai_runtime/src/motion_planning/motion_planners/"
    "mpd_dynamic_planner_adapter/config/replan_dynamic_aligned.yaml"
)


def test_ablation_defaults_to_640_paired_runs():
    args = _parser().parse_args([])

    assert args.scenario_count == 40
    assert args.repeats == 2
    assert args.modes == list(ABLATION_MODE_SPECS)
    assert planned_run_count(args.scenario_count, args.repeats, args.modes) == 640
    assert len(ABLATION_PIPELINE_ARGS) == 6


def test_materialized_ros_config_switches_each_aligned_feature(tmp_path):
    output = tmp_path / "ablated.yaml"

    payload = materialize_config(
        ALIGNED_CONFIG,
        output,
        deviation=False,
        relative_hysteresis=True,
        clearance_split=False,
        tail_kinematic=True,
    )

    parameters = payload["mpd_dynamic_replanner"]["ros__parameters"]
    assert parameters["cost_deviation_weight"] == 0.0
    assert parameters["relative_switching_hysteresis"] == 0.10
    assert parameters["split_terminal_hold_clearance"] is False
    assert parameters["cost_tail_kinematic_weight"] == 4.0
    assert yaml.safe_load(output.read_text(encoding="utf-8")) == payload


def test_materialized_ros_config_uses_full_adaptive_deviation_weight(tmp_path):
    payload = materialize_config(
        ALIGNED_CONFIG,
        tmp_path / "aligned.yaml",
        deviation=True,
        relative_hysteresis=True,
        clearance_split=True,
        tail_kinematic=True,
    )

    parameters = payload["mpd_dynamic_replanner"]["ros__parameters"]
    assert parameters["cost_deviation_weight"] == 0.15


def test_every_one_factor_mode_changes_exactly_one_pipeline_switch():
    assert ABLATION_MODE_SPECS["phase4"] == ("phase4", None)
    assert ABLATION_MODE_SPECS["aligned_all"] == ("phase4_aligned", None)
    assert all(
        arguments[-1] == "off" and len(arguments) == 2
        for arguments in ABLATION_PIPELINE_ARGS.values()
    )
