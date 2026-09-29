import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts.isaaclab.benchmark_todrawer_random import (
    BASE_CROSSINGS,
    BENCHMARK_GENERATION_REVISION,
    CATEGORIES,
    DEFAULT_ENVIRONMENT_COUNT_PER_CATEGORY,
    DEFAULT_FACTORIZED_C_CHECKPOINT,
    DEFAULT_FACTORIZED_TAU_R_CHECKPOINT,
    DEFAULT_MODES,
    DEFAULT_PLANNER_REPEATS,
    DEFAULT_TIMING_PROTOCOL,
    DIFFICULTIES,
    GENERATION_REVISION,
    MAX_GENERATION_RESAMPLE_ATTEMPTS,
    MODE_SPECS,
    PREDICTION_HORIZON_S,
    _existing_rows,
    _calibrate_crossing_from_report,
    _normalize_metrics,
    _paired_summary,
    _parse_ros_log,
    _parser,
    _timing_profile_mode,
    _trajectory_segment_metrics,
    extract_run_metrics,
    generate_benchmark_suite,
    generate_suite,
    main,
    materialize_scenario_for_mode,
    materialize_suite,
    object_position_at,
    write_reports,
)
from scripts.isaaclab.todrawer_scenario_validation import (
    STARTUP_SAFE_ANCHOR_BOUNDS,
    static_interaction_clearance,
    trajectory_clearances,
)
from scripts.isaaclab.validate_todrawer_random_suite import validate_suite
from scripts.isaaclab.test_todrawer_first_plan_calibration import _summarize as summarize_first_plan_test


def test_hard_scene_median_first_plan_calibrates_next_crossing_with_parent_offsets(tmp_path, monkeypatch):
    from scripts.isaaclab.run_todrawer_f3c_until_success import MODE_TIMING_PROFILES

    original = MODE_TIMING_PROFILES["joint_corridor_a"]
    monkeypatch.setitem(MODE_TIMING_PROFILES, "joint_corridor_a", original)
    report = tmp_path / "runs.csv"
    report.write_text(
        "mode,difficulty,first_plan_completed_from_world_s\n"
        "joint_corridor_a,easy,2.1\n"
        "joint_corridor_a,hard,4.7\n"
        "joint_corridor_a,hard,4.9\n",
        encoding="utf-8",
    )

    calibration = _calibrate_crossing_from_report(report, ["joint_corridor_a"])

    parent = MODE_TIMING_PROFILES["joint"]
    profile = MODE_TIMING_PROFILES["joint_corridor_a"]
    offset = 4.8 - parent.first_plan_completed_s
    assert calibration["modes"]["joint_corridor_a"]["first_plan_completed_s"] == pytest.approx(4.8)
    assert profile.first_plan_completed_s == pytest.approx(4.8)
    assert profile.significant_motion_start_s == pytest.approx(parent.significant_motion_start_s + offset)
    assert profile.crossing_shift_min_s == pytest.approx(parent.crossing_shift_min_s + offset)
    assert profile.expected_goal_s == pytest.approx(parent.expected_goal_s + offset)


def test_non_corridor_calibration_uses_hard_plan_median(tmp_path, monkeypatch):
    from scripts.isaaclab.run_todrawer_f3c_until_success import MODE_TIMING_PROFILES

    original = MODE_TIMING_PROFILES["f2_c"]
    monkeypatch.setitem(MODE_TIMING_PROFILES, "f2_c", original)
    report = tmp_path / "runs.csv"
    report.write_text(
        "mode,difficulty,category,first_plan_completed_from_world_s\n"
        "f2_c,hard,fast_crossing,2.0\n"
        "f2_c,hard,fast_crossing,2.2\n"
        "f2_c,hard,inflated_dense,2.8\n",
        encoding="utf-8",
    )
    calibration = _calibrate_crossing_from_report(report, ["f2_c"])
    profile = MODE_TIMING_PROFILES["f2_c"]
    assert calibration["modes"]["f2_c"]["first_plan_completed_s"] == pytest.approx(2.2)
    assert profile.first_plan_completed_s == pytest.approx(2.2)
    assert profile.expected_goal_s == pytest.approx(original.expected_goal_s + 2.2 - original.first_plan_completed_s)


def test_first_plan_test_counts_failed_results_and_keeps_missing_separate():
    rows = [
        {"mode": "f1_c", "difficulty": "hard", "category": "fast_crossing",
         "first_plan_completed_from_world_s": "2.0", "first_plan_status": "success"},
        {"mode": "f1_c", "difficulty": "hard", "category": "inflated_dense",
         "first_plan_completed_from_world_s": "3.0", "first_plan_status": "no_valid_trajectory"},
        {"mode": "f1_c", "difficulty": "hard", "category": "mixed_motion_multi",
         "first_plan_completed_from_world_s": "", "first_plan_status": ""},
    ]
    result = summarize_first_plan_test(rows, ["f1_c"], 2.5)["f1_c"]
    assert result["median_s"] == pytest.approx(2.5)
    assert result["over_threshold_count"] == 1
    assert result["planning_failed_results"] == 1
    assert result["returned_results"] == 2
    assert result["missing_results"] == 1


def test_compatibility_modes_calibrate_independently(tmp_path, monkeypatch):
    from scripts.isaaclab.run_todrawer_f3c_until_success import MODE_TIMING_PROFILES

    monkeypatch.setitem(MODE_TIMING_PROFILES, "joint", MODE_TIMING_PROFILES["joint"])
    monkeypatch.setitem(MODE_TIMING_PROFILES, "scalar_duration", MODE_TIMING_PROFILES["scalar_duration"])
    report = tmp_path / "runs.csv"
    report.write_text(
        "mode,difficulty,first_plan_completed_from_world_s\n"
        "scalar_duration,hard,9.0\n"
        "joint,hard,2.0\n",
        encoding="utf-8",
    )
    calibration = _calibrate_crossing_from_report(report, ["scalar_duration", "joint"])
    assert set(calibration["modes"]) == {"joint", "scalar_duration"}
    assert MODE_TIMING_PROFILES["joint"].first_plan_completed_s == pytest.approx(2.0)
    assert MODE_TIMING_PROFILES["scalar_duration"].first_plan_completed_s == pytest.approx(9.0)


def test_stale_first_plans_calibrate_crossing_after_observed_robot_motion(tmp_path, monkeypatch):
    from scripts.isaaclab.run_todrawer_f3c_until_success import MODE_TIMING_PROFILES

    original = MODE_TIMING_PROFILES["f1_c_corridor_a"]
    monkeypatch.setitem(MODE_TIMING_PROFILES, "f1_c_corridor_a", original)
    report = tmp_path / "runs.csv"
    report.write_text(
        "mode,difficulty,first_plan_completed_from_world_s,first_significant_motion_from_world_s\n"
        "f1_c_corridor_a,hard,3.6,14.0\n",
        encoding="utf-8",
    )
    calibration = _calibrate_crossing_from_report(
        report, ["f1_c_corridor_a"], motion_floor=True,
    )
    parent = MODE_TIMING_PROFILES["f1_c"]
    profile = MODE_TIMING_PROFILES["f1_c_corridor_a"]
    assert calibration["modes"]["f1_c_corridor_a"]["offset_source"] == "first_motion"
    assert profile.first_plan_completed_s == pytest.approx(3.6)
    assert profile.significant_motion_start_s == pytest.approx(14.0)
    assert profile.crossing_shift_min_s == pytest.approx(
        parent.crossing_shift_min_s + 14.0 - parent.significant_motion_start_s
    )


def test_random_suite_is_deterministic_and_covers_categories():
    first = generate_suite(len(CATEGORIES), 42)
    second = generate_suite(len(CATEGORIES), 42)

    assert first == second
    assert [item["category"] for item in first["scenarios"]] == list(CATEGORIES)
    assert first["schema_version"] == 4
    assert first["generation_policy"]["revision"] == GENERATION_REVISION
    assert all(1 <= len(item["objects"]) <= 3 for item in first["scenarios"])
    assert all(item["schema"] == "mpd_todrawer_dynamic_scenario" for item in first["scenarios"])
    assert all(item["schema_version"] == 3 for item in first["scenarios"])
    assert "first_execution_accepted" not in json.dumps(first)


def test_random_suite_uses_stratified_motion_and_structural_feasibility():
    suite = generate_suite(5 * len(CATEGORIES), 1234)

    assert suite["generation_policy"]["maximum_simultaneous_crossings"] == 2
    motion_types = set()
    for scenario in suite["scenarios"]:
        assert scenario["difficulty"] in DIFFICULTIES
        objects = scenario["objects"]
        corridor_ids = [item["corridor_id"] for item in objects]
        assert len(corridor_ids) == len(set(corridor_ids))
        for item in objects:
            motion_types.add(item["motion_model"])
            assert 0.08 <= item["speed_m_s"] <= 0.32
            assert item["anchor_id"] == item["corridor_id"]
            assert item["schedule_role"]
            if item["motion_model"] == "constant_acceleration":
                acceleration = item["motion"]["longitudinal_acceleration_m_s2"]
                for elapsed in (0.0, 35.0):
                    velocity = item["speed_m_s"] + acceleration * (elapsed - item["crossing_time_s"])
                    assert 0.04 - 1.0e-12 <= velocity <= 0.38 + 1.0e-12
            inflation = item["inflation"]
            horizon_inflation = inflation["base_m"] + PREDICTION_HORIZON_S * inflation["horizon_rate_m_s"]
            assert horizon_inflation <= 0.23 + 1.0e-12

    assert motion_types == {
        "constant_velocity",
        "constant_acceleration",
        "sinusoidal_curve",
        "smooth_speed_variation",
        "curved_speed_variation",
    }


def test_every_object_reaches_its_workspace_anchor_during_arm_motion():
    suite = generate_suite(10 * len(CATEGORIES), 9271)

    for scenario in suite["scenarios"]:
        assert scenario["primary_crossing_window_s"] == [4.0, 6.5]
        assert scenario["motion_clock"] == {
            "mode": "first_robot_state_after_scenario_load",
            "independent_of_execution": True,
        }
        for item in scenario["objects"]:
            assert object_position_at(item, item["crossing_time_s"]) == pytest.approx(
                item["anchor_position"], abs=1.0e-12
            )


def test_arrival_patterns_preserve_simultaneous_and_staggered_semantics():
    suite = generate_suite(20 * len(CATEGORIES), 7319)

    for scenario in suite["scenarios"]:
        times = sorted(item["crossing_time_s"] for item in scenario["objects"])
        category = scenario["category"]
        if category in {"simultaneous_multi", "curved_crossing"}:
            assert times[-1] - times[0] <= 0.20 + 1.0e-12
        elif category in {"staggered_multi", "accelerating_crossing"}:
            assert all(0.70 <= second - first <= 1.10 for first, second in zip(times, times[1:]))
        elif category == "inflated_dense":
            assert all(0.74 <= second - first <= 1.06 for first, second in zip(times, times[1:]))
        elif category == "fast_crossing" and len(times) == 2:
            assert 0.60 <= times[1] - times[0] <= 0.90
        elif category == "uncertain_motion":
            assert 0.65 <= times[1] - times[0] <= 1.00
        elif category == "mixed_motion_multi":
            by_role = {role: [] for role in ("simultaneous", "delayed")}
            for item in scenario["objects"]:
                by_role[item["schedule_role"]].append(item["crossing_time_s"])
            assert max(by_role["simultaneous"]) - min(by_role["simultaneous"]) <= 0.20
            center = sum(by_role["simultaneous"]) / 2.0
            assert 0.90 <= by_role["delayed"][0] - center <= 1.20


def test_generated_objects_never_enter_robot_base_exclusion_volume():
    suite = generate_suite(20 * len(CATEGORIES), 8113)

    for scenario in suite["scenarios"]:
        for item in scenario["objects"]:
            clearance = trajectory_clearances(item)
            assert clearance.robot_base_m > 0.0
            assert item["minimum_robot_base_clearance_m"] == pytest.approx(clearance.robot_base_m)
            assert item["minimum_static_environment_clearance_m"] == pytest.approx(clearance.static_environment_m)


def test_suite_contract_validator_and_anchor_jitter():
    suite = generate_suite(10 * len(CATEGORIES), 2081)
    validate_suite(suite)
    anchors = {item["id"]: item["anchor"] for item in BASE_CROSSINGS}
    for scenario in suite["scenarios"]:
        if scenario["category"] == "safe_control":
            assert {item["anchor_id"] for item in scenario["objects"]} <= {"S0", "S1"}
            continue
        limit = 0.020 if scenario["category"] == "inflated_dense" else 0.015
        for item in scenario["objects"]:
            assert all(
                abs(actual - nominal) <= limit + 1e-12
                for actual, nominal in zip(item["anchor_position"], anchors[item["anchor_id"]])
            )


def test_dense_and_mixed_motion_models_are_not_bound_to_temporal_role():
    suite = generate_suite(100 * len(CATEGORIES), 3187)
    roles = {category: {} for category in ("inflated_dense", "mixed_motion_multi")}
    for scenario in suite["scenarios"]:
        if scenario["category"] not in roles:
            continue
        for item in scenario["objects"]:
            roles[scenario["category"]].setdefault(item["motion_model"], set()).add(item["schedule_role"])
    assert all(len(values) >= 2 for category in roles.values() for values in category.values())


def test_generation_has_bounded_resampling(monkeypatch):
    import scripts.isaaclab.benchmark_todrawer_random as benchmark

    calls = 0

    def reject(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise ValueError("forced invalid geometry")

    monkeypatch.setattr(benchmark, "validate_trajectory_clearance", reject)
    with pytest.raises(RuntimeError, match="after 50 attempts"):
        benchmark.generate_suite(1, 9)
    assert calls == MAX_GENERATION_RESAMPLE_ATTEMPTS


def test_same_category_arrivals_have_bounded_random_variation():
    suite = generate_suite(20 * len(CATEGORIES), 6197)
    arrivals_by_category = {category: [] for category in CATEGORIES}
    for scenario in suite["scenarios"]:
        arrivals_by_category[scenario["category"]].append(
            tuple(round(item["crossing_time_s"], 6) for item in scenario["objects"])
        )

    assert all(len(set(arrivals)) > 1 for arrivals in arrivals_by_category.values())


def test_benchmark_defaults_to_paired_corridor_comparison():
    args = _parser().parse_args([])

    assert args.environment_count_per_category == DEFAULT_ENVIRONMENT_COUNT_PER_CATEGORY
    assert args.planner_repeats == DEFAULT_PLANNER_REPEATS
    assert args.timing_protocol == DEFAULT_TIMING_PROTOCOL == "motion_aligned"
    assert args.modes == list(DEFAULT_MODES)
    assert DEFAULT_MODES == (
        "phase4_aligned",
        "joint",
        "joint_corridor_a",
        "f1_tau_r",
        "f1_tau_r_corridor_a",
        "f1_c",
        "f1_c_corridor_a",
        "f3_tau_r",
    )
    assert args.categories is None
    assert MODE_SPECS["phase4"] == ("phase4", None)
    assert MODE_SPECS["phase4_aligned"] == ("phase4_aligned", None)
    assert MODE_SPECS["f1"] == ("factorized", None)
    assert MODE_SPECS["f2"] == ("factorized", None)
    assert MODE_SPECS["f3"] == ("factorized", None)
    assert MODE_SPECS["f1_c"] == ("factorized", None)
    assert MODE_SPECS["f3_tau_r"] == ("factorized", None)
    assert MODE_SPECS["joint_corridor_a"] == ("phase5", "phase5_joint")
    assert MODE_SPECS["f1_tau_r_corridor_a"] == ("factorized", None)
    assert MODE_SPECS["f1_c_corridor_a"] == ("factorized", None)
    assert args.factorized_c_checkpoint == DEFAULT_FACTORIZED_C_CHECKPOINT
    assert args.factorized_tau_r_checkpoint == DEFAULT_FACTORIZED_TAU_R_CHECKPOINT


def _shared_object_geometry(scenario):
    return [
        {
            "anchor_position": item["anchor_position"],
            "direction": item["direction"],
            "speed_m_s": item["speed_m_s"],
            "size_xyz": item["local_sdf"]["size_xyz"],
            "motion": item["motion"],
        }
        for item in scenario["objects"]
    ]


def test_paired_benchmark_suite_rejects_invalid_environments_and_shares_geometry():
    modes = ["phase4", "joint", "f3_c", "f3_tau_r"]
    suite = generate_benchmark_suite(
        1,
        1234,
        timing_protocol="motion_aligned",
        modes=modes,
    )

    assert suite["schema_version"] == 6
    assert suite["generation_policy"]["revision"] == BENCHMARK_GENERATION_REVISION
    assert suite["generation_policy"]["static_environment_intersection_policy"] == "allowed_and_reported"
    assert suite["scenario_count"] == len(CATEGORIES)
    assert suite["environment_count_per_category"] == 1
    assert suite["generation_policy"]["environment_randomized_fields"] == [
        "anchor_position",
        "direction",
        "speed_m_s",
        "local_sdf.size_xyz",
        "motion_model",
        "motion_parameters",
        "crossing_time_s",
    ]
    assert [item["category"] for item in suite["scenarios"]] == list(CATEGORIES)
    validate_suite(suite)

    saw_mode_specific_crossing = False
    saw_static_furniture_penetration = False
    for environment in suite["scenarios"]:
        variants = [materialize_scenario_for_mode(environment, mode) for mode in modes]
        assert all(_shared_object_geometry(variant) == _shared_object_geometry(variants[0]) for variant in variants[1:])
        crossing_sets = {tuple(item["crossing_time_s"] for item in variant["objects"]) for variant in variants}
        saw_mode_specific_crossing |= len(crossing_sets) > 1
        for variant in variants:
            for item in variant["objects"]:
                lower, upper = STARTUP_SAFE_ANCHOR_BOUNDS[item["anchor_id"]]
                assert all(low <= actual <= high for actual, low, high in zip(item["anchor_position"], lower, upper))
                interaction_clearance = static_interaction_clearance(item)
                assert item["minimum_static_interaction_clearance_m"] == pytest.approx(
                    interaction_clearance
                )
                saw_static_furniture_penetration |= interaction_clearance <= 0.0
                assert item["minimum_initial_franka_clearance_m"] > 0.005
            if len(variant["objects"]) >= 2:
                assert variant["attempt_sampling"]["direction_line_contract"]["satisfied"]
    assert saw_mode_specific_crossing
    assert saw_static_furniture_penetration


def test_absolute_world_time_shares_crossing_times_as_well_as_geometry():
    modes = ["phase4", "joint", "f3_tau_r"]
    suite = generate_benchmark_suite(
        1,
        4321,
        timing_protocol="absolute_world_time",
        modes=modes,
    )

    for environment in suite["scenarios"]:
        variants = [materialize_scenario_for_mode(environment, mode) for mode in modes]
        expected = [item["crossing_time_s"] for item in variants[0]["objects"]]
        assert all([item["crossing_time_s"] for item in variant["objects"]] == expected for variant in variants[1:])


def test_both_protocols_share_geometry_but_keep_separate_timing_variants(tmp_path):
    modes = ["joint", "joint_corridor_a", "f1_c", "f1_c_corridor_a"]
    suite = materialize_suite(
        tmp_path,
        1,
        9876,
        timing_protocol="both",
        modes=modes,
        planner_repeats=1,
    )

    assert suite["timing_protocols"] == ["absolute_world_time", "motion_aligned"]
    validate_suite(suite)
    environment = suite["scenarios"][0]
    variants = {
        protocol: {
            mode: materialize_scenario_for_mode(
                environment, mode, timing_protocol=protocol
            )
            for mode in modes
        }
        for protocol in suite["timing_protocols"]
    }
    reference = _shared_object_geometry(variants["absolute_world_time"][modes[0]])
    assert all(
        _shared_object_geometry(variant) == reference
        for protocol_variants in variants.values()
        for variant in protocol_variants.values()
    )
    absolute_times = {
        tuple(item["crossing_time_s"] for item in variant["objects"])
        for variant in variants["absolute_world_time"].values()
    }
    assert len(absolute_times) == 1
    assert variants["motion_aligned"]["joint_corridor_a"]["objects"][0][
        "crossing_time_s"
    ] > variants["motion_aligned"]["joint"]["objects"][0]["crossing_time_s"]
    for protocol in suite["timing_protocols"]:
        for mode in modes:
            assert (tmp_path / "scenarios" / protocol / environment["id"] / f"{mode}.json").is_file()

    report_rows = [
        {
            "scenario_id": environment["id"],
            "repeat": 0,
            "mode": "joint",
            "timing_protocol": protocol,
            "pipeline_completed": True,
            "manifest_available": True,
            "goal_reached": True,
            "first_plan_completed_from_world_s": first_plan_s,
            "goal_time_s": first_plan_s + 5.0,
            "inference_total_mean_s": 0.4,
        }
        for protocol, first_plan_s in (
            ("absolute_world_time", 2.0),
            ("motion_aligned", 4.0),
        )
    ]
    write_reports(tmp_path, report_rows, suite, ["joint"])
    summary = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert summary["by_timing_protocol"]["absolute_world_time"]["run_count"] == 1
    assert summary["by_timing_protocol"]["motion_aligned"]["run_count"] == 1
    assert summary["by_timing_protocol"]["absolute_world_time"]["by_mode"]["joint"][
        "first_plan_completed_from_world_s"
    ]["mean"] == pytest.approx(2.0)
    report = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "时间协议分开汇总" in report
    assert "`absolute_world_time`" in report
    assert "`motion_aligned`" in report


def test_furniture_rejecting_suite_cannot_be_reused(tmp_path):
    (tmp_path / "suite.json").write_text(
        json.dumps(
            {
                "suite_seed": 42,
                "environment_count_per_category": 1,
                "planner_repeats": 1,
                "timing_protocol": "motion_aligned",
                "modes": ["phase4"],
                "generation_policy": {"revision": "paired-startup-safe-environments-v2"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="use a new output directory"):
        materialize_suite(
            tmp_path,
            1,
            42,
            timing_protocol="motion_aligned",
            modes=["phase4"],
            planner_repeats=1,
        )


@pytest.mark.parametrize(
    ("mode", "profile"),
    [
        ("aligned_all", "phase4_aligned"),
        ("aligned_no_dynamic_guidance", "phase4_aligned"),
        ("phase5_all", "joint"),
        ("phase5_no_deviation", "joint"),
    ],
)
def test_ablation_modes_reuse_parent_timing_profiles(mode, profile):
    assert _timing_profile_mode(mode) == profile


@pytest.mark.parametrize(
    "mode",
    ["joint_corridor_a", "f1_tau_r_corridor_a", "f1_c_corridor_a"],
)
def test_corridor_modes_use_independent_timing_profiles(mode):
    from scripts.isaaclab.run_todrawer_f3c_until_success import MODE_TIMING_PROFILES

    assert _timing_profile_mode(mode) == mode
    assert mode in MODE_TIMING_PROFILES


def test_ros_log_extracts_world_clock_and_initial_warmup(tmp_path):
    path = tmp_path / "ros.log"
    path.write_text(
        "scenario world clock started unix_ns=12300000000 mode=first_joint_state\n"
        "initial dynamic-world warm-up complete and first planning submitted "
        "unix_ns=12800000000 from_world_s=0.500000 observations=5 "
        "track_age_s=0.400000\n"
        "[node] [13.0] dynamic plan rejected: RequestValidationError: "
        "q_pos_start is in collision in the configured MPD scene.\n"
        "[node] [14.0] goal reached; holding position\n",
        encoding="utf-8",
    )

    parsed = _parse_ros_log(path)

    assert parsed["world_start_unix_s"] == pytest.approx(12.3)
    assert parsed["first_planning_submit_from_world_s"] == pytest.approx(0.5)
    assert parsed["initial_world_warmup_observations"] == 5
    assert parsed["initial_world_warmup_age_s"] == pytest.approx(0.4)
    assert parsed["goal_reached"] is True


def test_failed_run_keeps_planner_and_online_outcomes_without_manifest(tmp_path):
    attempt = tmp_path / "attempt-001"
    request_dir = attempt / "planner-results" / "request-0001"
    request_dir.mkdir(parents=True)
    (request_dir / "response.json").write_text(json.dumps({
        "planner_result_status": "success",
        "online_acceptance_status": "STALE",
        "online_rejection_reason": "deadline_expired_after_planning",
        "server_round_trip_sec": 4.2,
    }), encoding="utf-8")
    (request_dir / "result.json").write_text(json.dumps({
        "status": "success", "created_unix_time": 12.4,
    }), encoding="utf-8")
    (attempt / "ros-replan.log").write_text(
        "scenario world clock started unix_ns=10000000000\n", encoding="utf-8",
    )
    metrics = extract_run_metrics(attempt, {"mode": "joint_corridor_a"}, 1)
    assert not metrics["manifest_available"]
    assert metrics["planner_result_status_counts"] == {"success": 1}
    assert metrics["online_acceptance_status_counts"] == {"STALE": 1}
    assert metrics["deadline_expired_after_planning_count"] == 1
    assert metrics["server_round_trip_mean_s"] == pytest.approx(4.2)
    assert metrics["first_plan_completed_from_world_s"] == pytest.approx(2.4)
    assert metrics["first_plan_status"] == "success"


def test_realized_joint_path_uses_only_active_interval(tmp_path):
    episode = tmp_path / "episode"
    archive = episode / "plans" / "plan-0000" / "trajectory.npz"
    archive.parent.mkdir(parents=True)
    np.savez_compressed(
        archive,
        positions=np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        velocities=np.zeros((3, 2)),
        accelerations=np.zeros((3, 2)),
        time_from_start=np.asarray([0.0, 1.0, 2.0]),
    )
    plans = [
        {
            "trajectory": "plans/plan-0000/trajectory.npz",
            "start_s": 5.0,
            "active_from_s": 5.5,
            "active_until_s": 6.5,
        }
    ]

    l2, l1 = _trajectory_segment_metrics(episode / "replay-manifest.json", plans)

    assert l2 == pytest.approx(1.0)
    assert l1 == pytest.approx(1.0)


def test_extract_metrics_and_report_from_synthetic_completed_run(tmp_path):
    attempt = tmp_path / "runs" / "scenario-000" / "repeat-00" / "joint" / "attempt-001"
    episode = attempt / "episode"
    archive = episode / "plans" / "plan-0000" / "trajectory.npz"
    archive.parent.mkdir(parents=True)
    np.savez_compressed(
        archive,
        positions=np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]),
        velocities=np.zeros((3, 2)),
        accelerations=np.zeros((3, 2)),
        time_from_start=np.asarray([0.0, 1.0, 2.0]),
    )
    manifest = {
        "duration_s": 4.0,
        "plans": [
            {
                "id": "plan-0000",
                "status": "accepted",
                "trajectory": "plans/plan-0000/trajectory.npz",
                "start_s": 1.0,
                "active_from_s": 1.0,
                "active_until_s": 3.0,
                "phase_timing": {"mpd_suffix_s": 8.5},
                "candidate_clearance_diagnostics": [
                    {
                        "candidate_index": 0,
                        "composite_cost": 0.2,
                        "hard_minimum_clearance_m": 0.04,
                        "common_window_minimum_clearance_m": 0.03,
                        "clearance_mean_cost": 0.1,
                        "clearance_cvar_cost": 0.3,
                    }
                ],
            }
        ],
        "events": [{"type": "handoff", "time_s": 1.2}],
    }
    (episode / "replay-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (attempt / "to_drawer-replan-timing.json").write_text(
        json.dumps(
            {
                "maximum_uncovered_command_gap_s": 0.0,
                "guarded_terminal_hold_s": 0.4,
                "maximum_controller_reference_jump_rad": 0.01,
            }
        ),
        encoding="utf-8",
    )
    (attempt / "ros-replan.log").write_text(
        "scenario world clock started unix_ns=1000000000 mode=first_joint_state\n"
        "[1.000] dynamic MPD replanner started\n"
        '[2.000] top-K candidate rejections: [{"reason": "dynamic_collision"}]\n'
        "[3.000] goal reached; holding position\n",
        encoding="utf-8",
    )
    result_dir = attempt / "planner-results" / "request-1"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "status": "success",
                "created_unix_time": 2.4,
                "timing": {"inference_total_sec": 0.4},
                "trajectory": {
                    "minimum_environment_clearance_m": 0.02,
                    "minimum_self_clearance_m": 0.08,
                },
                "space_time_guidance": {
                    "settings": {"spatial_dynamic_max_grad_norm": 2.0},
                    "corridor_a": {
                        "enabled": True,
                        "elapsed_s": 0.12,
                        "grid_build_s": 0.02,
                        "branch_selection_s": 0.01,
                        "refinement_forward_s": 0.04,
                        "refinement_backward_s": 0.03,
                        "final_validation_s": 0.05,
                        "profiled_total_s": 0.15,
                        "changed": 3,
                        "dense_fallbacks": 1,
                        "runtime_identity": {
                            "guidance_mode": "phase5_joint",
                            "factorized_method": None,
                            "factorized_representation": None,
                            "corridor_variant": "corridor_a",
                        },
                    },
                    "steps": [
                        {
                            "spatial_clip_ratio": 0.25,
                            "timing_clip_ratio": 0.5,
                            "spatial_gradient_norm_mean": 1.2,
                            "static_dynamic_gradient_cosine_mean": -0.2,
                            "static_dynamic_gradient_conflict_ratio": 0.75,
                            "static_dynamic_gradient_cosine_valid_ratio": 0.8,
                        }
                    ],
                },
                "factorized": {
                    "representation": "tau_r",
                    "timing_checkpoint_step": 60000,
                    "timing_checkpoint_sha256": "a" * 64,
                    "spatial_basis_adapted": True,
                    "denoiser_evaluations": {"space": 32, "timing": 100},
                },
            }
        ),
        encoding="utf-8",
    )
    failed_result_dir = attempt / "planner-results" / "request-0"
    failed_result_dir.mkdir(parents=True)
    (failed_result_dir / "result.json").write_text(
        json.dumps(
            {
                "status": "no_valid_trajectory",
                "created_unix_time": 2.2,
                "timing": {"inference_total_sec": 0.3},
            }
        ),
        encoding="utf-8",
    )
    run_spec = {
        "scenario_id": "scenario-000",
        "category": "single_crossing",
        "repeat": 0,
        "mode": "joint_corridor_a",
        "corridor_a_enabled": True,
        "planner_seed": 42,
    }

    metrics = extract_run_metrics(attempt, run_spec, 0)

    assert metrics["pipeline_completed"]
    assert metrics["manifest_available"]
    assert metrics["failure_class"] is None
    assert metrics["goal_reached"]
    assert metrics["goal_time_s"] == pytest.approx(2.0)
    assert metrics["first_plan_completed_from_world_s"] == pytest.approx(1.2)
    assert metrics["first_plan_status"] == "no_valid_trajectory"
    assert metrics["joint_l2_path_rad"] == pytest.approx(1.0)
    assert metrics["hard_minimum_clearance_m"] == pytest.approx(0.04)
    assert metrics["guard_dynamic_collision_rejections"] == 1
    assert metrics["spatial_dynamic_grad_cap"] == pytest.approx(2.0)
    assert metrics["spatial_clip_ratio"] == pytest.approx(0.25)
    assert metrics["static_dynamic_gradient_cosine_mean"] == pytest.approx(-0.2)
    assert metrics["factorized_representation"] == "tau_r"
    assert metrics["factorized_timing_checkpoint_step"] == 60000
    assert metrics["factorized_spatial_basis_adapted"] is True
    assert metrics["factorized_space_nfe_mean"] == pytest.approx(32.0)
    assert metrics["factorized_timing_nfe_mean"] == pytest.approx(100.0)
    assert metrics["corridor_a_refinement_mean_s"] == pytest.approx(0.12)
    assert metrics["corridor_a_grid_build_mean_s"] == pytest.approx(0.02)
    assert metrics["corridor_a_branch_selection_mean_s"] == pytest.approx(0.01)
    assert metrics["corridor_a_refinement_forward_mean_s"] == pytest.approx(0.04)
    assert metrics["corridor_a_refinement_backward_mean_s"] == pytest.approx(0.03)
    assert metrics["corridor_a_final_validation_mean_s"] == pytest.approx(0.05)
    assert metrics["corridor_a_profiled_total_mean_s"] == pytest.approx(0.15)
    assert metrics["corridor_a_changed_candidates_mean"] == pytest.approx(3.0)
    assert metrics["corridor_a_dense_fallbacks_mean"] == pytest.approx(1.0)
    assert metrics["corridor_a_payload_match_count"] == 1
    assert metrics["corridor_a_payload_mismatch_count"] == 0

    suite = generate_suite(1, 42)
    write_reports(tmp_path, [metrics], suite)
    report = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "ToDrawer 随机动态重规划基准报告" in report
    assert "场景类型与难度" in report
    assert "joint" in report
    assert "时长、路径与推理耗时" in report
    assert "仅到达目标运行的路径与执行时长" in report
    assert "完整配对结果" in report
    assert "Clearance 汇总" in report
    assert "难度分层结果" in report
    assert "规划轨迹时长 mean s" in report
    assert "Phase 5 / Factorized 梯度裁剪诊断" in report
    assert "Corridor A 诊断" in report
    assert (tmp_path / "report" / "runs.csv").is_file()
    summary = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert summary["schema_version"] == 6
    assert summary["timing_protocols"] == ["motion_aligned"]
    assert "motion_aligned" in summary["by_timing_protocol"]


def test_missing_manifest_dds_startup_is_infrastructure_failure(tmp_path):
    attempt = tmp_path / "attempt-001"
    attempt.mkdir()
    (attempt / "ros-replan.log").write_text(
        "rmw_create_node: failed to create domain, error Error\n",
        encoding="utf-8",
    )

    metrics = extract_run_metrics(
        attempt,
        {
            "scenario_id": "scenario-001",
            "category": "staggered_multi",
            "repeat": 0,
            "mode": "joint",
        },
        1,
    )

    assert not metrics["manifest_available"]
    assert not metrics["pipeline_completed"]
    assert metrics["failure_class"] == "dds_startup"


def test_old_terminal_clip_failure_is_revalidated(tmp_path):
    attempt = tmp_path / "attempt-001"
    episode = attempt / "episode"
    episode.mkdir(parents=True)
    (episode / "replay-manifest.json").write_text(
        json.dumps(
            {
                "duration_s": 10.000001,
                "plans": [
                    {
                        "id": "terminal",
                        "status": "accepted",
                        "active_from_s": 10.0,
                        "active_until_s": 10.000001,
                        "phase_timing": {
                            "planning_submitted_s": 8.0,
                            "bridge_start_s": 10.0,
                            "handoff_s": 10.2,
                            "mpd_suffix_s": 8.75,
                        },
                    }
                ],
                "events": [],
            }
        ),
        encoding="utf-8",
    )
    (attempt / "pipeline.log").write_text(
        "ValueError: executed plan 0 has inconsistent phase timing\n",
        encoding="utf-8",
    )

    normalized = _normalize_metrics(
        {
            "attempt_dir": attempt.as_posix(),
            "pipeline_completed": False,
            "pipeline_returncode": 1,
            "error": None,
        }
    )

    assert normalized["pipeline_completed"]
    assert normalized["pipeline_revalidated"]
    assert normalized["terminal_clipped_plan_count"] == 1
    assert normalized["maximum_command_gap_s"] == pytest.approx(0.0)
    assert normalized["maximum_uncovered_command_gap_s"] == pytest.approx(0.0)
    assert normalized["failure_class"] is None


def test_reversed_active_interval_is_a_real_continuity_failure(tmp_path):
    attempt = tmp_path / "attempt-001"
    episode = attempt / "episode"
    episode.mkdir(parents=True)
    (episode / "replay-manifest.json").write_text(
        json.dumps({"duration_s": 10.0, "plans": [], "events": []}),
        encoding="utf-8",
    )
    (attempt / "pipeline.log").write_text(
        "plans[4]: active interval is outside the trajectory duration\n",
        encoding="utf-8",
    )

    normalized = _normalize_metrics(
        {
            "attempt_dir": attempt.as_posix(),
            "pipeline_completed": False,
            "pipeline_returncode": 1,
        }
    )

    assert normalized["manifest_available"]
    assert normalized["failure_class"] == "command_continuity"


def test_paired_summary_requires_manifest_from_all_report_modes():
    rows = [
        {
            "scenario_id": "scenario-000",
            "repeat": 0,
            "mode": mode,
            "manifest_available": True,
            "pipeline_completed": True,
            "goal_reached": mode != "joint",
            "brake_count": int(mode == "phase4"),
            "execution_duration_s": 2.0,
            "joint_l2_path_rad": 1.0,
            "joint_l1_travel_rad": 2.0,
        }
        for mode in MODE_SPECS
    ]

    paired = _paired_summary(rows)

    assert paired["cell_count"] == 1
    assert paired["by_mode"]["phase4"]["goal_reached"] == 1
    assert paired["by_mode"]["phase4"]["goal_and_brake_runs"] == 1
    assert paired["by_mode"]["joint"]["goal_reached"] == 0


def test_paired_summary_can_compare_only_requested_six_modes():
    selected = ("phase4", "phase4_aligned", "joint", "f1", "f2", "f3")
    rows = [
        {
            "scenario_id": "scenario-000",
            "repeat": 0,
            "mode": mode,
            "manifest_available": True,
            "pipeline_completed": True,
            "goal_reached": True,
        }
        for mode in selected
    ]

    paired = _paired_summary(rows, selected)

    assert paired["cell_count"] == 1
    assert tuple(paired["by_mode"]) == selected


def test_factorized_dry_run_writes_paired_commands_with_explicit_basis(tmp_path):
    checkpoint = tmp_path / "timing.pt"
    checkpoint.write_bytes(b"checkpoint-placeholder")
    output = tmp_path / "benchmark"
    selected = ("phase4", "phase4_aligned", "joint", "f1", "f2", "f3")

    assert (
        main(
            [
                "--output-dir",
                str(output),
                "--environment-count-per-category",
                "1",
                "--planner-repeats",
                "1",
                "--modes",
                *selected,
                "--factorized-timing-checkpoint",
                str(checkpoint),
                "--dry-run",
            ]
        )
        == 0
    )

    specs = {}
    for mode in selected:
        path = next((output / "runs" / "scenario-000" / "repeat-00" / mode).glob("*/run-spec.json"))
        specs[mode] = json.loads(path.read_text(encoding="utf-8"))
    assert {spec["planner_seed"] for spec in specs.values()} == {20260829}
    for method in ("f1", "f2", "f3"):
        command = specs[method]["command"]
        assert command[command.index("--phase") + 1] == "factorized"
        assert command[command.index("--factorized-method") + 1] == method
        assert "--factorized-adapt-spatial-basis" in command
        assert specs[method]["factorized_spatial_basis_adapted"] is True


def test_space_separated_modes_run_only_requested_subset(tmp_path):
    output = tmp_path / "selected-modes"
    selected = ("phase4_aligned", "f1_c_corridor_a", "f3_tau_r")
    assert main([
        "--output-dir", str(output),
        "--environment-count-per-category", "1",
        "--planner-repeats", "1",
        "--categories", "single_crossing",
        "--modes", *selected, "f1_c_corridor_a",
        "--dry-run",
    ]) == 0
    suite = json.loads((output / "suite.json").read_text(encoding="utf-8"))
    assert suite["modes"] == list(selected)
    mode_dirs = {path.name for path in (output / "runs" / "scenario-000" / "repeat-00").iterdir()}
    assert mode_dirs == set(selected)


def test_c_and_tau_r_factorized_modes_run_together_with_best_defaults(tmp_path):
    output = tmp_path / "benchmark"
    selected = (
        "phase4",
        "phase4_aligned",
        "joint",
        "f1_c",
        "f2_c",
        "f3_c",
        "f1_tau_r",
        "f2_tau_r",
        "f3_tau_r",
    )

    assert (
        main(
            [
                "--output-dir",
                str(output),
                "--environment-count-per-category",
                "1",
                "--planner-repeats",
                "1",
                "--modes",
                *selected,
                "--dry-run",
            ]
        )
        == 0
    )

    seeds = set()
    for mode in selected:
        path = next((output / "runs" / "scenario-000" / "repeat-00" / mode).glob("*/run-spec.json"))
        spec = json.loads(path.read_text(encoding="utf-8"))
        seeds.add(spec["planner_seed"])
        if mode.endswith("_c"):
            assert spec["factorized_representation"] == "c"
            assert spec["factorized_timing_checkpoint"] == (DEFAULT_FACTORIZED_C_CHECKPOINT.resolve().as_posix())
        elif mode.endswith("_tau_r"):
            assert spec["factorized_representation"] == "tau_r"
            assert spec["factorized_timing_checkpoint"] == (DEFAULT_FACTORIZED_TAU_R_CHECKPOINT.resolve().as_posix())
        if mode.startswith("f"):
            assert spec["factorized_method"] == mode.split("_")[0]
    assert seeds == {20260829}


def test_corridor_comparison_dry_run_pairs_world_seed_and_checkpoints(tmp_path):
    output = tmp_path / "corridor-benchmark"
    assert main([
        "--output-dir", str(output),
        "--environment-count-per-category", "1",
        "--planner-repeats", "1",
        "--dry-run",
    ]) == 0
    validate_suite(json.loads((output / "suite.json").read_text(encoding="utf-8")))

    specs = {}
    scenarios = {}
    for mode in DEFAULT_MODES:
        spec_path = next((output / "runs" / "scenario-000" / "repeat-00" / mode).glob("*/run-spec.json"))
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        specs[mode] = spec
        scenarios[mode] = json.loads(Path(spec["scenario_file"]).read_text(encoding="utf-8"))
        assert spec["corridor_a_enabled"] == mode.endswith("_corridor_a")
        assert ("--corridor-a" in spec["command"]) == mode.endswith("_corridor_a")
    assert {spec["planner_seed"] for spec in specs.values()} == {20260829}
    for baseline, enabled in (
        ("joint", "joint_corridor_a"),
        ("f1_tau_r", "f1_tau_r_corridor_a"),
        ("f1_c", "f1_c_corridor_a"),
    ):
        assert _shared_object_geometry(scenarios[baseline]) == _shared_object_geometry(
            scenarios[enabled]
        )
        assert min(
            item["crossing_time_s"] for item in scenarios[enabled]["objects"]
        ) > min(item["crossing_time_s"] for item in scenarios[baseline]["objects"])
        assert [
            (item["anchor_id"], item["schedule_role"])
            for item in scenarios[baseline]["anchor_schedule"]
        ] == [
            (item["anchor_id"], item["schedule_role"])
            for item in scenarios[enabled]["anchor_schedule"]
        ]
        assert specs[baseline]["phase"] == specs[enabled]["phase"]
        if baseline.startswith("f1"):
            assert specs[baseline]["factorized_timing_checkpoint"] == specs[enabled]["factorized_timing_checkpoint"]
            assert specs[baseline]["factorized_method"] == specs[enabled]["factorized_method"] == "f1"
    assert "--corridor-a" not in specs["phase4_aligned"]["command"]
    assert "--corridor-a" not in specs["f3_tau_r"]["command"]


def test_existing_rows_keep_historical_infrastructure_attempt_count(tmp_path):
    root = tmp_path / "runs" / "scenario-000" / "repeat-00" / "joint"
    first = root / "attempt-001"
    second = root / "attempt-002"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "ros-replan.log").write_text("rmw_create_node: failed to create domain\n", encoding="utf-8")
    common = {"scenario_id": "scenario-000", "repeat": 0, "mode": "joint"}
    (first / "run-metrics.json").write_text(
        json.dumps(
            {
                **common,
                "attempt_dir": first.as_posix(),
                "pipeline_completed": False,
            }
        ),
        encoding="utf-8",
    )
    (second / "run-metrics.json").write_text(
        json.dumps(
            {
                **common,
                "attempt_dir": second.as_posix(),
                "pipeline_completed": True,
            }
        ),
        encoding="utf-8",
    )

    rows = _existing_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["attempt_count"] == 2
    assert rows[0]["infrastructure_failure_attempts"] == 1
    assert rows[0]["pipeline_completed"]


def test_benchmark_rejects_invalid_explicit_ros_domain():
    script = Path("scripts/isaaclab/benchmark_todrawer_random.py")

    result = subprocess.run(
        [
            "/home/eric/anaconda3/envs/mpd-splines-public/bin/python",
            script.as_posix(),
            "--ros-domain-id",
            "233",
            "--report-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "ros-domain-id must lie in [0, 232]" in result.stderr
    GENERATION_REVISION,
    MAX_GENERATION_RESAMPLE_ATTEMPTS,
