from pathlib import Path

import yaml

from scripts.isaaclab.benchmark_phase5_recent_ablation import (
    ABLATION_MODE_SPECS,
    ABLATION_PIPELINE_ARGS,
    _parser,
    planned_run_count,
)
from scripts.isaaclab.materialize_phase5_ablation_config import materialize_config


PHASE5_CONFIG = Path(
    "/home/eric/Projects/physical_ai_runtime/src/motion_planning/motion_planners/"
    "mpd_dynamic_planner_adapter/config/replan_space_time.yaml"
)


def test_ablation_defaults_to_1040_paired_runs():
    args = _parser().parse_args([])

    assert args.scenario_count == 40
    assert args.repeats == 2
    assert args.modes == list(ABLATION_MODE_SPECS)
    assert planned_run_count(args.scenario_count, args.repeats, args.modes) == 1040
    assert len(ABLATION_PIPELINE_ARGS) == 10


def test_materialized_phase5_config_uses_latest_full_values(tmp_path):
    output = tmp_path / "phase5.yaml"
    payload = materialize_config(
        PHASE5_CONFIG,
        output,
        deviation=True,
        adaptive_deviation=True,
        relative_hysteresis=True,
        clearance_split=True,
        tail_kinematic_weight=3.0,
        terminal_hold_clearance_weight=0.5,
    )

    parameters = payload["mpd_space_time_replanner"]["ros__parameters"]
    assert parameters["cost_deviation_weight"] == 0.15
    assert parameters["adaptive_deviation_weight_enabled"] is True
    assert parameters["relative_switching_hysteresis"] == 0.10
    assert parameters["split_terminal_hold_clearance"] is True
    assert parameters["cost_tail_kinematic_weight"] == 3.0
    assert parameters["cost_terminal_hold_clearance_weight"] == 0.5
    assert yaml.safe_load(output.read_text(encoding="utf-8")) == payload


def test_recent_rollbacks_are_distinct_from_feature_removals():
    assert ABLATION_MODE_SPECS["phase4"] == ("phase4", None)
    assert ABLATION_MODE_SPECS["phase4_aligned"] == ("phase4_aligned", None)
    assert ABLATION_PIPELINE_ARGS["phase5_no_tail_kinematic"][-1] == "1.0"
    assert ABLATION_PIPELINE_ARGS["phase5_old_tail_4"][-1] == "4.0"
    assert ABLATION_PIPELINE_ARGS["phase5_old_hold_0_2"][-1] == "0.2"
    assert ABLATION_PIPELINE_ARGS["phase5_old_dynamic_grad_cap_1"][-1] == "1.0"
