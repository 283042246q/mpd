import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from mpd.bimanual.start_goal_sources import _base_request, _pose_from_matrix
from scripts.inference.benchmark_marvin_region_models import (
    DEFAULT_MODELS,
    DEFAULT_SCENARIOS,
    _collect_rows,
    _group_trials,
    _aggregate,
    load_model_cases,
    load_scenarios,
    prepare_tasks,
)


def _valid_request(request_id="test"):
    request = _base_request(
        request_id=request_id,
        seed=0,
        q_start=[0.0] * 14,
        q_goal=[0.1] * 14,
        source={"type": "dataset", "row": 0},
    )
    pose = _pose_from_matrix([[1, 0, 0, 0.5], [0, 1, 0, 0.2], [0, 0, 1, 0.1]])
    request["left_goal_pose"] = pose
    request["right_goal_pose"] = pose
    return request


def test_catalog_covers_all_difficulties_ood_and_shared_workspace():
    scenarios = load_scenarios(DEFAULT_SCENARIOS)
    assert {spec["difficulty"] for spec in scenarios.values()} == {
        "easy", "medium", "hard", "extreme"
    }
    assert {spec["support"] for spec in scenarios.values()} >= {
        "dataset_exact", "in_distribution", "structured_ood",
        "strict_random_ood", "shared_workspace_ood",
    }
    assert {spec["transition"] for spec in scenarios.values()} >= {
        "id_to_id", "id_to_structured_ood", "structured_ood_to_id",
        "structured_ood_to_structured_ood", "id_to_random_ood",
        "random_ood_to_id", "random_ood_to_random_ood", "id_to_shared_ood",
    }
    shared = [spec for spec in scenarios.values() if spec.get("workspace_overlap")]
    assert len(shared) >= 3
    assert any(
        spec.get("goal") == {
            "left": "left_random_ood_centerline",
            "right": "right_random_ood_centerline",
        }
        for spec in shared
    )


def test_checked_in_registry_pins_two_b_c_d_and_leaves_a_interface_disabled():
    registry = yaml.safe_load(DEFAULT_MODELS.read_text())
    assert registry["models"]["A"]["enabled"] is False
    assert registry["models"]["A"]["selection"] == {
        "type": "top_validation", "top_k": 2
    }
    expected = {"B": [955000, 715000], "C": [715000, 835000], "D": [305000, 665000]}
    for variant, steps in expected.items():
        selection = registry["models"][variant]["selection"]
        assert selection["type"] == "explicit"
        assert [item["step"] for item in selection["checkpoints"]] == steps


def test_model_loader_skips_future_a_and_builds_two_explicit_cases(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "args.yaml").write_text(
        "bimanual_network_variant: B\ndataset_subdir: reversed\n", encoding="utf-8"
    )
    for step in (10, 20):
        (run / "checkpoints" / f"ema_model__iter_{step:06d}.pth").write_bytes(b"model")
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump({
        "schema": "marvin_bimanual_model_registry/v1",
        "dataset_subdir": "reversed",
        "models": {
            "A": {
                "enabled": False, "run_dir": str(tmp_path / "future-a"),
                "network_variant": "A",
                "selection": {"type": "top_validation", "top_k": 2},
            },
            "B": {
                "enabled": True, "run_dir": str(run), "network_variant": "B",
                "selection": {"type": "explicit", "checkpoints": [
                    {"step": 10, "file": "ema_model__iter_000010.pth", "val_loss": 0.2},
                    {"step": 20, "file": "ema_model__iter_000020.pth", "val_loss": 0.3},
                ]},
            },
        },
    }), encoding="utf-8")
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"runtime": {}, "gradient_pruning": {}}), encoding="utf-8")
    _, cases, skipped = load_model_cases(registry, base, ["A", "B"], strict=True)
    assert [case["id"] for case in cases] == ["B-10", "B-20"]
    assert [case["step"] for case in cases] == [10, 20]
    assert skipped == [{"variant": "A", "reason": "disabled", "note": None}]
    assert all(case["config"]["collision_optimization"]["pair_streaming"]["enabled"] for case in cases)


def test_prepare_records_sampling_shortfall_separately(tmp_path, monkeypatch):
    import scripts.inference.benchmark_marvin_region_models as benchmark

    calls = []
    def fake_region(*args, **kwargs):
        calls.append(kwargs["request_id"])
        if len(calls) == 1:
            raise benchmark.StartGoalSamplingError("no endpoint")
        request = _valid_request(kwargs["request_id"])
        request["scene"]["start_goal_source"] = {"type": "regions"}
        return request

    monkeypatch.setattr(benchmark, "request_from_regions", fake_region)
    regions = tmp_path / "regions.yaml"
    regions.write_text("schema: marvin_bimanual_regions/v1\n", encoding="utf-8")
    scenarios = {
        "test": {
            "difficulty": "hard", "support": "structured_ood",
            "transition": "id_to_structured_ood",
            "start": {"left": "left_a", "right": "right_a"},
            "goal": {"left": "left_b", "right": "right_b"},
        }
    }
    report = prepare_tasks(tmp_path / "out", scenarios, regions, {}, 2, 3, 1, 1.0)
    assert report["complete"]
    assert report["scenarios"]["test"]["attempts"] == 3
    assert report["scenarios"]["test"]["accepted"] == 2
    assert report["scenarios"]["test"]["errors"] == {"StartGoalSamplingError": 1}
    assert all(Path(task["request_path"]).is_file() for task in report["tasks"])
    prepare_tasks(tmp_path / "out", scenarios, regions, {}, 2, 3, 1, 1.0)
    assert len(calls) == 3


def test_early_success_is_a_complete_trial_without_later_batch_files(tmp_path):
    task = {
        "scenario": "shared", "task_index": 0, "difficulty": "extreme",
        "support": "shared_workspace_ood", "workspace_overlap": True,
        "transition": "id_to_shared_ood",
    }
    manifest = {
        "models": [{"id": "D-305k"}], "seeds": [7], "candidate_batches": 4,
    }
    folder = tmp_path / "runs/D-305k/shared/task-000/seed-7/batch-00"
    folder.mkdir(parents=True)
    (folder / "benchmark-result.json").write_text(json.dumps({
        "model": "D-305k", "variant": "D", "checkpoint_step": 305000,
        "scenario": "shared", "task_index": 0, "seed": 7, "batch": 0,
        "difficulty": "extreme", "support": "shared_workspace_ood",
        "transition": "id_to_shared_ood",
        "workspace_overlap": True, "status": "success", "wall_seconds": 1.5,
        "gpu_memory": {"peak_device_used_mib": 100}, "candidates": {"valid": 1},
    }), encoding="utf-8")
    rows = _collect_rows(tmp_path, manifest, {"tasks": [task]})
    trials = _group_trials(rows, 4)
    assert len(rows) == 1
    assert trials[0]["complete"] and trials[0]["success"]
    assert trials[0]["batches_attempted"] == 1


def test_infrastructure_error_is_not_a_planning_failure():
    base = {
        "model": "D-305k", "variant": "D", "checkpoint_step": 305000,
        "scenario": "test", "task_index": 0, "seed": 1, "difficulty": "hard",
        "support": "structured_ood", "transition": "id_to_structured_ood",
        "workspace_overlap": False, "complete": True, "batches_attempted": 1,
        "wall_seconds": 1.0, "peak_device_used_mib": 100, "valid_candidates": 0,
    }
    trials = [
        {**base, "success": False, "statuses": {"no_valid_trajectory": 1}},
        {**base, "seed": 2, "success": False, "statuses": {"cuda_oom": 1}},
    ]
    row = _aggregate(trials, ("model",))[0]
    assert row["planning_evaluable_trials"] == 1
    assert row["infrastructure_or_contract_trials"] == 1
    assert row["planning_success_rate"] == 0
    assert row["end_to_end_success_rate"] == 0
