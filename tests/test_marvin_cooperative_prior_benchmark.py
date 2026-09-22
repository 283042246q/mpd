import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from mpd.bimanual.cooperative_sources import request_from_config_source
from scripts.inference import benchmark_marvin_cooperative_priors as benchmark


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/inference/cfgs/" "config_EnvWarehouse-RobotMarvinBimanual-cooperative-independent-prior.yaml"
REGIONS = (
    ROOT / "scripts/inference/cfgs/start_goal_regions/" "EnvWarehouse-RobotMarvinBimanual-cooperative-regions.yaml"
)


def test_scenario_catalog_covers_production_matrix_and_three_difficulties():
    _, scenarios = benchmark.load_scenarios(benchmark.DEFAULT_SCENARIOS, benchmark.DEFAULT_REGIONS)
    production = yaml.safe_load(REGIONS.read_text(encoding="utf-8"))
    matrix_pairs = {(row["start"], row["goal"]) for row in production["sampling_matrix"]}
    benchmark_pairs = {(spec["start_region"], spec["goal_region"]) for spec in scenarios.values()}

    assert benchmark_pairs == matrix_pairs
    assert {
        difficulty: sum(spec["difficulty"] == difficulty for spec in scenarios.values())
        for difficulty in benchmark.DIFFICULTIES
    } == {"easy": 2, "medium": 4, "hard": 2}


def test_four_cases_are_exact_prior_projection_cartesian_product():
    base = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    original = deepcopy(base)
    cases = benchmark.build_cases(base, n_trajectory_samples=3)

    assert base == original
    assert len(cases) == 4
    assert {
        (
            case["cooperative_inference"]["prior_mode"],
            case["cooperative_inference"]["closed_chain_projection"],
        )
        for case in cases.values()
    } == {
        ("direct_project", True),
        ("direct_project", False),
        ("reference_residual", True),
        ("reference_residual", False),
    }
    assert all(case["n_trajectory_samples"] == 3 for case in cases.values())
    assert all(case["runtime_top_k_valid_trajectories"] <= 3 for case in cases.values())


def test_model_resolution_pins_latest_numbered_checkpoint_not_mutable_current(tmp_path):
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (run / "args.yaml").write_text(
        "use_ema: true\nbimanual_network_variant: D\ndataset_subdir: new-data\n",
        encoding="utf-8",
    )
    (checkpoints / "ema_model_current.pth").write_bytes(b"mutable")
    (checkpoints / "ema_model__iter_000010.pth").write_bytes(b"ten")
    (checkpoints / "ema_model__iter_000020.pth").write_bytes(b"twenty")
    (checkpoints / "ema_model__iter_000020_state_dict.pth").write_bytes(b"state")

    selected_run, checkpoint, train, selection = benchmark._resolve_model_run({"model_dir_ddpm_bspline": str(run)})

    assert selected_run == run
    assert checkpoint.name == "ema_model__iter_000020.pth"
    assert train["dataset_subdir"] == "new-data"
    assert selection == "latest_numbered"
    manifest = {
        "model": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": benchmark._sha256(checkpoint),
        }
    }
    benchmark._verify_model_fingerprint(manifest)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        benchmark._verify_model_fingerprint(manifest)


def test_prepare_freezes_one_request_and_reuses_it_on_resume(tmp_path, monkeypatch):
    template = request_from_config_source(
        CONFIG,
        source="seed_file",
        source_path=None,
        sample_index=0,
        seed=7,
        request_id="template",
    )
    calls = []

    def fake_request(path, *, generator_config_path, request_id, seed):
        calls.append((path, generator_config_path, request_id, seed))
        request = deepcopy(template)
        request["request_id"] = request_id
        request["seed"] = seed
        request["scene"]["start_goal_source"] = {
            "type": "regions",
            "start_region": "start",
            "goal_region": "goal",
        }
        request["scene"]["object_start_state_xyz_yaw"] = [0.4, 0.0, 0.2, 0.0]
        request["scene"]["object_goal_state_xyz_yaw"] = [0.7, 0.1, 0.2, 0.2]
        return request

    monkeypatch.setattr(benchmark, "request_from_regions", fake_request)
    regions = tmp_path / "regions.yaml"
    regions.write_text(
        yaml.safe_dump(
            {
                "schema": "marvin_bimanual_cooperative_regions/v1",
                "object_regions": {"start": {}, "goal": {}},
                "sampling_matrix": [{"start": "start", "goal": "goal", "weight": 1.0}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    generation = tmp_path / "generation.yaml"
    generation.write_text("task_mode: cooperative_rigid\n", encoding="utf-8")
    scenarios = {
        "sample": {
            "difficulty": "easy",
            "transition": "adjacent_longitudinal",
            "description": "test pair",
            "start_region": "start",
            "goal_region": "goal",
        }
    }
    output = tmp_path / "benchmark"

    report = benchmark.prepare_tasks(
        output,
        scenarios,
        regions,
        generation,
        tasks_per_scenario=1,
        max_endpoint_attempts=2,
        endpoint_seed=100,
    )
    request_path = Path(report["tasks"][0]["request_path"])
    frozen = json.loads(request_path.read_text(encoding="utf-8"))
    assert report["complete"]
    assert report["tasks"][0]["object_translation_m"] > 0.3
    assert frozen["scene"]["start_goal_source"]["benchmark_scenario"] == "sample"
    assert len(calls) == 1

    resumed = benchmark.prepare_tasks(
        output,
        scenarios,
        regions,
        generation,
        tasks_per_scenario=1,
        max_endpoint_attempts=2,
        endpoint_seed=100,
    )
    assert resumed == report
    assert len(calls) == 1


def test_summary_groups_case_difficulty_and_cooperative_timings(tmp_path):
    rows = [
        {
            "case": "direct_project__projection_on",
            "difficulty": "easy",
            "scenario": "easy_near_to_middle",
            "status": "success",
            "wall_seconds": 2.0,
            "timing": {
                "inference_total_s": 1.5,
                "generator_s": 0.4,
                "guide_s": 0.7,
                "dense_validation_s": 0.2,
            },
            "cooperative_inference": {
                "reference_build_s": 0.0,
                "projection_s": 0.3,
            },
            "candidates": {"valid": 2},
            "validation": {
                "max_closure_translation_error_m": 0.003,
                "max_closure_rotation_error_rad": 0.01,
            },
        },
        {
            "case": "reference_residual__projection_off",
            "difficulty": "hard",
            "scenario": "hard_left_to_right_offset",
            "status": "no_valid_trajectory",
            "failure_reason": "NoValidTrajectoryError",
            "wall_seconds": 3.0,
            "timing": None,
            "cooperative_inference": None,
            "candidates": None,
            "validation": None,
        },
    ]
    for index, row in enumerate(rows):
        path = tmp_path / "runs" / str(index) / "benchmark-result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row), encoding="utf-8")
    manifest = {"task_count": 2, "seeds": [1], "cases": list(range(4))}

    summary = benchmark.summarize_experiment(tmp_path, manifest)

    assert summary["expected_runs"] == 8
    assert summary["completed_runs"] == 2
    assert len(summary["by_difficulty_case"]) == 2
    easy = next(row for row in summary["by_difficulty_case"] if row["difficulty"] == "easy")
    assert easy["success_rate"] == 1.0
    assert easy["mean_projection_s"] == 0.3
    hard = next(row for row in summary["by_difficulty_case"] if row["difficulty"] == "hard")
    assert hard["failure_reason_counts"] == {"NoValidTrajectoryError": 1}
    assert "n/a" in (tmp_path / "summary.md").read_text(encoding="utf-8")
