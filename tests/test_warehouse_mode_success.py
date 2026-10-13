from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.spacetime_data.benchmark_warehouse_mode_success import command_for_case, summarize_case, verify_case_config


def report():
    return dict(attempts=100, accepted=40, tau_r_valid_samples=40, session_wall_s=50., config_hash="fixture",
                failure_counts={"source_static_self_or_limits": 20, "world_anchor_or_initial_collision": 10,
                                "teacher_no_solution": 15, "dynamic_collision": 15},
                session_pipeline_statistics=dict(submitted=70, completed=70, raw_solved_queries=55))


def test_all_attempt_denominator_and_solver_conditioned_rate_are_distinct():
    row = summarize_case(report(), mode="linear", obstacles=3, attempts=100)
    assert row["attempt_success_rate"] == 0.4
    assert row["acceptance_given_submitted"] == 40 / 70
    assert row["postprocess_rejected"] == 15
    assert row["source_failed"] == 20 and row["world_failed"] == 10


def test_resumed_session_does_not_replace_cumulative_attempt_counts():
    resumed = report()
    resumed["session_pipeline_statistics"] = dict(prepared=25, submitted=14, completed=14,
                                                 raw_solved_queries=10, accepted=6)
    row = summarize_case(resumed, mode="linear", obstacles=5, attempts=100)
    assert row["attempt_success_rate"] == 0.4
    assert row["submitted"] == 70 and row["raw_solved"] == 55


def test_partial_or_inconsistent_reports_cannot_be_published_as_complete():
    with pytest.raises(ValueError, match="incomplete"):
        summarize_case(report(), mode="linear", obstacles=5, attempts=101)
    broken = report()
    broken["accepted"] = 41
    with pytest.raises(ValueError, match="inconsistent"):
        summarize_case(broken, mode="linear", obstacles=3, attempts=100)


def test_target_exceeds_attempt_budget_to_prevent_success_dependent_stopping():
    args = SimpleNamespace(device="cuda:0", source="source.hdf5", attempts=100, query_batch_size=8,
                           solve_seconds=5., seed=7, start_index=0)
    command = command_for_case(args, "helix", 5, "output")
    def value(key):
        return command[command.index(key) + 1]
    assert value("--num-trajectories") == "101"
    assert value("--max-attempts") == "100"
    assert value("--motion-modes") == "helix" and value("--obstacles") == "5"
    assert "--require-tau-r" in command


def test_summarization_rejects_wrong_seed_or_different_source_and_software():
    args = SimpleNamespace(device="cuda:0", source=Path("source.hdf5"), query_batch_size=8,
                           solve_seconds=5., seed=7, start_index=0)
    config = dict(backend="torch-forest", device=args.device, source=str(args.source.resolve()),
                  seed=7, start_index=0, source_group_size=2, query_batch_size=8, solve_seconds=5.,
                  torch_batch_policy="fill", require_tau_r=True, edges_per_query=16, goal_roots=4,
                  max_iterations=512, collision_chunk=512, motion_modes=["linear"], obstacles=3,
                  software={"generator": "fixture"})
    common = verify_case_config(config, args, "linear", 3)
    other = dict(config, motion_modes=["orbit"], obstacles=5)
    assert verify_case_config(other, args, "orbit", 5, common) == common
    with pytest.raises(ValueError, match="benchmark parameters"):
        verify_case_config(dict(config, seed=8), args, "linear", 3)
    with pytest.raises(ValueError, match="changed between cases"):
        verify_case_config(dict(other, software={"generator": "changed"}), args, "orbit", 5, common)
