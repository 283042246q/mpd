import json
from types import SimpleNamespace

import pytest

from scripts.inference import run_marvin_cooperative_d600k_videos as runner


def _fixture(tmp_path, count=8):
    tasks = {"tasks": []}
    for index in range(count):
        name = f"scenario_{index}"
        request_path = tmp_path / "tasks" / f"{name}.json"
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(json.dumps({"request_id": name, "seed": 47000 + index}), encoding="utf-8")
        tasks["tasks"].append({"scenario": name, "difficulty": "easy", "task_index": 0,
                               "endpoint_seed": 47000 + index, "request_path": str(request_path),
                               "transition": "test", "start_region": "start", "goal_region": "goal"})
    manifest = {"models": [{"id": "D-600k", "config_path": str(tmp_path / "D-600k.yaml")}],
                "seeds": [1101, 1102, 1103], "device": "cpu", "timeout_s": 900,
                "gpu_poll_interval": 0.2, "generation_config": str(tmp_path / "generation.yaml")}
    args = SimpleNamespace(max_endpoint_tasks_per_scenario=0, max_endpoint_attempts=20,
                           isaaclab_conda_env="env_isaaclab",
                           isaaclab_device="cuda:0", isaaclab_timeout_s=900)
    return tasks, manifest, args


def test_retries_current_task_then_records_before_next_and_resumes(tmp_path, monkeypatch):
    tasks, manifest, args = _fixture(tmp_path)
    monkeypatch.setattr(runner.benchmark, "_verify_fingerprints", lambda *_: None)
    events = []

    def fake_execute(config, artifact, run_args):
        request = json.loads(run_args.request.read_text(encoding="utf-8"))
        scenario = request["request_id"].split("-seed-")[0]
        events.append(("plan", scenario, request["seed"]))
        status = "fault" if scenario == "scenario_0" and request["seed"] == 1101 else "success"
        return {"status": status, "wall_seconds": 1.0,
                "result_error": {"message": "could not build a continuous paired-IK reference"} if status == "fault" else None}

    def fake_replay(artifact, replay_args):
        events.append(("record", artifact.parents[1].name))
        (artifact / "isaaclab-run.json").write_text('{"status":"completed"}', encoding="utf-8")
        (artifact / "isaaclab-replay.mp4").write_bytes(b"video")
        return "completed"

    monkeypatch.setattr(runner, "execute_case", fake_execute)
    monkeypatch.setattr(runner, "_replay", fake_replay)
    first = runner._run(tmp_path, manifest, tasks, args)
    assert first["completed_tasks"] == 8
    assert events[:4] == [("plan", "scenario_0", 1101), ("plan", "scenario_0", 1102),
                          ("record", "scenario_0"), ("plan", "scenario_1", 1101)]
    assert len([item for item in events if item[0] == "record"]) == 8
    before = list(events)
    assert runner._run(tmp_path, manifest, tasks, args)["completed_tasks"] == 8
    assert events == before


def test_recording_error_stops_without_advancing_and_resumes_same_attempt(tmp_path, monkeypatch):
    tasks, manifest, args = _fixture(tmp_path, count=2)
    monkeypatch.setattr(runner.benchmark, "_verify_fingerprints", lambda *_: None)
    calls = []

    def fake_execute(config, artifact, run_args):
        calls.append(artifact)
        return {"status": "success", "wall_seconds": 1.0}

    monkeypatch.setattr(runner, "execute_case", fake_execute)
    monkeypatch.setattr(runner, "_replay", lambda *_: (_ for _ in ()).throw(RuntimeError("Isaac failed")))
    with pytest.raises(RuntimeError, match="Isaac failed"):
        runner._run(tmp_path, manifest, tasks, args)
    assert len(calls) == 1
    assert tasks["tasks"][0]["scenario"] in str(calls[0])

    def fake_replay(artifact, replay_args):
        (artifact / "isaaclab-run.json").write_text('{"status":"completed"}', encoding="utf-8")
        (artifact / "isaaclab-replay.mp4").write_bytes(b"video")
        return "completed"

    monkeypatch.setattr(runner, "_replay", fake_replay)
    assert runner._run(tmp_path, manifest, tasks, args)["completed_tasks"] == 2
    assert len(calls) == 2  # original MPD success was reused; only task 1 was planned


def test_three_failures_resample_same_scenario_before_next(tmp_path, monkeypatch):
    tasks, manifest, args = _fixture(tmp_path, count=2)
    monkeypatch.setattr(runner.benchmark, "_verify_fingerprints", lambda *_: None)
    events = []

    def fake_resample(output, run_manifest, original, index, endpoint_attempts):
        events.append(("resample", original["scenario"], index))
        request_path = output / "tasks" / f"{original['scenario']}-task-{index:03d}.json"
        request_path.write_text(json.dumps({"request_id": f"{original['scenario']}-task-{index:03d}"}), encoding="utf-8")
        return {**original, "task_index": index, "request_path": str(request_path),
                "endpoint_seed": original["endpoint_seed"] + index}

    def fake_execute(config, artifact, run_args):
        events.append(("plan", artifact.parents[1].name, artifact.parent.name, run_args.seed))
        return {"status": "no_valid_trajectory" if artifact.parent.name == "task-000"
                and "scenario_0" in str(artifact) else "success", "wall_seconds": 1.0}

    def fake_replay(artifact, replay_args):
        events.append(("record", artifact.parents[1].name, artifact.parent.name))
        (artifact / "isaaclab-run.json").write_text('{"status":"completed"}', encoding="utf-8")
        (artifact / "isaaclab-replay.mp4").write_bytes(b"video")
        return "completed"

    monkeypatch.setattr(runner, "_resampled_task", fake_resample)
    monkeypatch.setattr(runner, "execute_case", fake_execute)
    monkeypatch.setattr(runner, "_replay", fake_replay)
    summary = runner._run(tmp_path, manifest, tasks, args)
    assert summary["completed_tasks"] == 2
    assert events[:6] == [
        ("plan", "scenario_0", "task-000", 1101),
        ("plan", "scenario_0", "task-000", 1102),
        ("plan", "scenario_0", "task-000", 1103),
        ("resample", "scenario_0", 1),
        ("plan", "scenario_0", "task-001", 1101),
        ("record", "scenario_0", "task-001"),
    ]
    assert summary["tasks"][0]["completed_endpoint_task_index"] == 1


def test_three_seed_limit_stops_without_advancing(tmp_path, monkeypatch):
    tasks, manifest, args = _fixture(tmp_path, count=2)
    args.max_endpoint_tasks_per_scenario = 1
    monkeypatch.setattr(runner.benchmark, "_verify_fingerprints", lambda *_: None)
    calls = []

    def fake_execute(config, artifact, run_args):
        calls.append(artifact)
        return {"status": "no_valid_trajectory", "wall_seconds": 1.0}

    monkeypatch.setattr(runner, "execute_case", fake_execute)
    with pytest.raises(RuntimeError, match="not advancing"):
        runner._run(tmp_path, manifest, tasks, args)
    assert len(calls) == 3
    assert all("scenario_0" in str(path) for path in calls)


def test_resampled_endpoint_is_frozen_and_uses_new_seed_block(tmp_path, monkeypatch):
    tasks, manifest, _ = _fixture(tmp_path, count=1)
    original = tasks["tasks"][0]
    region_path = tmp_path / "tasks/effective-regions/scenario_0.yaml"
    region_path.parent.mkdir(parents=True)
    region_path.write_text("schema: test\n", encoding="utf-8")
    seeds = []

    def fake_request(path, *, generator_config_path, request_id, seed):
        seeds.append(seed)
        return {"request_id": request_id, "seed": seed,
                "scene": {"start_goal_source": {},
                          "object_start_state_xyz_yaw": [0, 0, 0, 0],
                          "object_goal_state_xyz_yaw": [1, 0, 0, 0]}}

    monkeypatch.setattr(runner, "request_from_regions", fake_request)
    monkeypatch.setattr(runner.BimanualRequest, "from_dict", lambda _: None)
    monkeypatch.setattr(runner, "_task_metrics", lambda _: {"object_translation_m": 1.0})
    first = runner._resampled_task(tmp_path, manifest, original, 1, 20)
    assert first["endpoint_seed"] == 47001
    assert runner._resampled_task(tmp_path, manifest, original, 1, 20) == first
    second = runner._resampled_task(tmp_path, manifest, original, 2, 20)
    assert second["endpoint_seed"] == 47021
    assert seeds == [47001, 47021]


def test_legacy_extra_attempts_are_preserved_but_not_repeated(tmp_path, monkeypatch):
    tasks, manifest, args = _fixture(tmp_path, count=1)
    monkeypatch.setattr(runner.benchmark, "_verify_fingerprints", lambda *_: None)
    original = tasks["tasks"][0]
    old_artifacts = []
    for attempt, seed in enumerate([1101, 1102, 1103, 11101, 11102, 11103]):
        artifact = runner._attempt_dir(tmp_path, original, attempt, seed)
        artifact.mkdir(parents=True)
        runner._atomic_json(artifact / "benchmark-result.json",
                            {"model": "D-600k", "scenario": original["scenario"],
                             "task_index": 0, "inference_seed": seed,
                             "status": "no_valid_trajectory", "wall_seconds": 1.0})
        old_artifacts.append(artifact)
    calls = []

    def fake_resample(output, run_manifest, base, index, endpoint_attempts):
        path = output / "tasks" / "task-001.json"
        path.write_text('{"request_id":"scenario_0-task-001"}', encoding="utf-8")
        return {**base, "task_index": index, "request_path": str(path), "endpoint_seed": 47001}

    def fake_execute(config, artifact, run_args):
        calls.append(artifact)
        return {"status": "success", "wall_seconds": 1.0}

    def fake_replay(artifact, replay_args):
        (artifact / "isaaclab-run.json").write_text('{"status":"completed"}', encoding="utf-8")
        (artifact / "isaaclab-replay.mp4").write_bytes(b"video")
        return "completed"

    monkeypatch.setattr(runner, "_resampled_task", fake_resample)
    monkeypatch.setattr(runner, "execute_case", fake_execute)
    monkeypatch.setattr(runner, "_replay", fake_replay)
    summary = runner._run(tmp_path, manifest, tasks, args)
    assert len(calls) == 1 and "task-001" in str(calls[0])
    assert all((path / "benchmark-result.json").exists() for path in old_artifacts)
    assert len(summary["tasks"][0]["attempts"]) == 7


def test_cuda_oom_stops_instead_of_retrying(tmp_path, monkeypatch):
    tasks, manifest, args = _fixture(tmp_path, count=2)
    monkeypatch.setattr(runner.benchmark, "_verify_fingerprints", lambda *_: None)
    calls = []

    def fake_execute(config, artifact, run_args):
        calls.append(artifact)
        return {"status": "cuda_oom", "wall_seconds": 1.0}

    monkeypatch.setattr(runner, "execute_case", fake_execute)
    with pytest.raises(RuntimeError, match="non-retryable MPD failure"):
        runner._run(tmp_path, manifest, tasks, args)
    assert len(calls) == 1


def test_replay_uses_isaaclab_and_requires_video(tmp_path, monkeypatch):
    _, _, args = _fixture(tmp_path, count=1)
    artifact = tmp_path / "attempt"
    artifact.mkdir()
    commands = []

    def fake_subprocess(command, **kwargs):
        commands.append(command)
        (artifact / "isaaclab-run.json").write_text('{"status":"completed"}', encoding="utf-8")
        (artifact / "isaaclab-replay.mp4").write_bytes(b"video" if len(commands) == 1 else b"")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess)
    assert runner._replay(artifact, args) == "completed"
    assert "--isaaclab-capture" in commands[0]
    assert "--isaaclab-replay" in commands[0]
    assert commands[0][commands[0].index("--isaaclab-conda-env") + 1] == "env_isaaclab"

    with pytest.raises(RuntimeError, match="recording failed"):
        runner._replay(artifact, args)
