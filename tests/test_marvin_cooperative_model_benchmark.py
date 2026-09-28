import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import yaml

from scripts.inference import benchmark_marvin_cooperative_models as benchmark


def test_effective_object_bounds_cover_current_cooperative_regions(tmp_path):
    source = benchmark.ROOT / "data_generation_cfgs/EnvWarehouse-RobotMarvinBimanual-cooperative.yaml"
    effective = benchmark._effective_generation_config(tmp_path, source, benchmark.DEFAULT_REGIONS)
    config = yaml.safe_load(effective.read_text(encoding="utf-8"))
    regions = yaml.safe_load(benchmark.DEFAULT_REGIONS.read_text(encoding="utf-8"))["object_regions"]
    low_y, high_y = config["object_planning"]["bounds"][1]
    assert low_y <= min(region["y"][0] for region in regions.values())
    assert high_y >= max(region["y"][1] for region in regions.values())
    assert config["object_planning"]["bounds"][0] == [0.38, 0.72]


def test_requested_checkpoints_resolve_on_compatible_training_runs():
    models = benchmark.resolve_models(benchmark.DEFAULT_REGISTRY, list(benchmark.DEFAULT_MODELS))
    assert [model["id"] for model in models] == list(benchmark.DEFAULT_MODELS)
    assert len({model["dataset_subdir"] for model in models}) == 1
    assert all(Path(model["checkpoint"]).is_file() for model in models)


def test_frozen_requests_and_seed_are_shared_across_models(tmp_path, monkeypatch):
    output = tmp_path / "comparison"
    source_files = [tmp_path / f"source-{name}.yaml" for name in ("config", "scenario", "regions", "registry")]
    for path in source_files:
        path.write_text("schema: test\n", encoding="utf-8")
    config_path, scenario_path, regions_path, registry_path = source_files
    request = {"request_id": "fixed-task", "seed": 47, "q_start": [0.0] * 14}
    request_path = output / "tasks/requests/task-000.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    tasks = {
        "tasks": [
            {
                "scenario": "easy_near_to_middle",
                "difficulty": "easy",
                "task_index": 0,
                "endpoint_seed": 47,
                "request_path": str(request_path),
                "object_translation_m": 0.15,
            }
        ],
        "sampling_shortfalls": {},
    }
    checkpoints = []
    models = []
    for model_id in ("D-600k", "A-835k"):
        variant, step, filename = benchmark._model_spec(model_id)
        run = tmp_path / model_id
        checkpoint = run / "checkpoints" / filename
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(model_id.encode())
        checkpoints.append(checkpoint)
        models.append(
            {
                "id": model_id,
                "variant": variant,
                "step": step,
                "run_dir": str(run),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": benchmark._sha256(checkpoint),
                "checkpoint_size": checkpoint.stat().st_size,
                "dataset_subdir": "same-data",
            }
        )
    monkeypatch.setattr(benchmark, "resolve_models", lambda *_: deepcopy(models))
    args = SimpleNamespace(
        models=[model["id"] for model in models],
        n_trajectory_samples=4,
        seeds=[11, 12],
        device="cpu",
        backend="contract_stub",
        timeout_s=20.0,
        gpu_poll_interval=0.2,
    )
    base = {
        "n_trajectory_samples": 16,
        "runtime_top_k_valid_trajectories": 8,
        "runtime": {"network_variant": "D"},
        "cooperative_inference": {
            "prior_mode": "direct_project",
            "closed_chain_projection": False,
        },
    }
    original = deepcopy(base)
    manifest = benchmark._materialize(
        output, config_path, base, scenario_path, regions_path, registry_path, config_path, tasks, args
    )
    assert base == original
    for model in manifest["models"]:
        config = yaml.safe_load(Path(model["config_path"]).read_text(encoding="utf-8"))
        assert config["cooperative_inference"]["prior_mode"] == "reference_residual"
        assert config["cooperative_inference"]["closed_chain_projection"] is True
        assert config["runtime"]["network_variant"] == model["variant"]
        assert config["checkpoint"] == Path(model["checkpoint"]).name
        assert config["n_trajectory_samples"] == 4

    seen = []

    def fake_execute(config_path, artifact, run_args):
        assert not (artifact / "result.json").exists()
        frozen = json.loads(run_args.request.read_text(encoding="utf-8"))
        seen.append((Path(config_path).stem, frozen, run_args.device))
        return {
            "status": "success",
            "wall_seconds": 1.0,
            "timing": {"inference_total_s": 0.5},
            "candidates": {"valid": 2},
        }

    monkeypatch.setattr(benchmark, "execute_case", fake_execute)
    benchmark._run(output, manifest, tasks)
    benchmark._run(output, manifest, tasks)
    assert len(seen) == 4  # resume skips all completed reports
    for seed in (11, 12):
        paired = [frozen for _, frozen, _ in seen if frozen["seed"] == seed]
        assert len(paired) == 2
        assert paired[0] == paired[1]
    assert all(device == "cpu" for _, _, device in seen)
    summary = benchmark._summarize(output, manifest, tasks)
    assert summary["expected_runs"] == summary["completed_runs"] == 4
    assert all(row["success_rate"] == 1.0 for row in summary["by_model"])

    interrupted = output / "runs/D-600k/easy/easy_near_to_middle/task-000/seed-11"
    (interrupted / "input-request.json").write_bytes(b"")
    (interrupted / "benchmark-result.json").write_bytes(b"")
    (interrupted / "result.json").write_text('{"status":"success"}', encoding="utf-8")
    assert benchmark._summarize(output, manifest, tasks)["completed_runs"] == 3
    benchmark._run(output, manifest, tasks)
    assert len(seen) == 5
    assert benchmark._summarize(output, manifest, tasks)["completed_runs"] == 4
    assert list(interrupted.glob("input-request.json.incomplete-*"))
    assert list(interrupted.glob("benchmark-result.json.incomplete-*"))
    assert list(interrupted.glob("result.json.incomplete-*"))

    checkpoints[0].write_bytes(b"changed")
    import pytest

    with pytest.raises(RuntimeError, match="pinned checkpoint changed"):
        benchmark._verify_fingerprints(manifest, tasks)
