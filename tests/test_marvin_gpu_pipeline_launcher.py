import json
import subprocess
import sys

import numpy as np
import yaml

from scripts.generate_data.launch_generate_marvin_warehouse_gpu_pipeline import (
    DEFAULT_CONFIG,
    RestartableActor,
    SEGMENT_INCOMPLETE_EXIT_CODE,
    _append_actor_process_event,
    _candidate_waves,
    _factor_dual_candidates,
    _dynamic_proposals,
    _write_pipeline_telemetry,
    _supervise_short_processes,
    generate_checkpoint,
    main,
)
from scripts.generate_data.marvin_gpu_task_contract import TaskContract


class FakeEndpointActor:
    def __init__(self):
        self.message = None

    def send(self, message):
        self.message = message

    def receive(self):
        result = []
        for job in self.message["jobs"]:
            task = job["task"]
            q_start = np.zeros(14)
            q_goal = np.zeros(14)
            if task["mode"] in ("dual_independent", "left_only"):
                q_goal[:7] = 0.5
            if task["mode"] in ("dual_independent", "right_only"):
                q_goal[7:] = -0.5
            result.append(
                {
                    "task_id": task["task_id"],
                    "attempt": job["attempt"],
                    "candidate_index": 0,
                    "proposal": True,
                    "q_start": q_start,
                    "q_goal": q_goal,
                    "ee_goal_pose": np.zeros((2, 3, 4), dtype=np.float32),
                }
            )
        return result

    def poll(self):
        return True, self.receive()


class FakeGpuActor:
    def __init__(self):
        self.plan_batches = []

    def request(self, message):
        if message["op"] == "endpoints":
            return [True] * len(message["items"])
        self.plan_batches.append((message["mode"], len(message["items"])))
        result = []
        for item in message["items"]:
            path = np.linspace(item["q_start"], item["q_goal"], 128)
            result.append(
                {
                    "task_id": item["task_id"],
                    "attempt": item["attempt"],
                    "path": path,
                    "spline": (np.zeros(28), np.zeros((14, 22)), 5),
                    "rejection_reason": None,
                    "batch_seconds": 0.1,
                    "rrt_iterations": 1,
                    "rrt_sampled_edges": 16,
                    "rrt_checked_states": 32,
                }
            )
        return result


class FakeBulletActor:
    def request(self, message):
        return [True] * len(message["items"])


class ImmediateResponseQueue:
    def __init__(self, response):
        self.response = response

    def get_nowait(self):
        return self.response


def test_gpu_launcher_import_does_not_eagerly_load_native_workers():
    command = (
        "import sys; "
        "import scripts.generate_data.launch_generate_marvin_warehouse_gpu_pipeline; "
        "assert 'ompl' not in sys.modules; "
        "assert 'pybullet' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_actor_restart_limits_are_consecutive_and_lifetime():
    actor = object.__new__(RestartableActor)
    actor.name = "test-actor"
    actor.max_consecutive_restarts = 3
    actor.max_total_restarts = 2
    actor.restart_count = 0
    actor.consecutive_restart_count = 0
    actor.pending = {"op": "work"}
    actor.pending_started = 0.0
    actor._terminate = lambda *args, **kwargs: None
    actor._start = lambda: None

    actor._restart_after_failure()
    assert actor.restart_count == actor.consecutive_restart_count == 1

    actor.pending = {"op": "work"}
    actor.pending_started = 0.0
    actor.responses = ImmediateResponseQueue({"ok": True, "result": 7})
    assert actor.poll() == (True, 7)
    assert actor.consecutive_restart_count == 0

    actor._restart_after_failure()
    assert actor.restart_count == 2
    assert actor.consecutive_restart_count == 1
    with np.testing.assert_raises_regex(RuntimeError, "exceeded 2 total"):
        actor._restart_after_failure()

    actor.restart_count = 0
    actor.consecutive_restart_count = 0
    actor.max_consecutive_restarts = 1
    actor.max_total_restarts = 10
    actor._restart_after_failure()
    with np.testing.assert_raises_regex(RuntimeError, "exceeded 1 consecutive"):
        actor._restart_after_failure()


def test_actor_process_events_are_durably_journaled(tmp_path):
    event = {"name": "gpu", "role": "gpu", "pid": 1234, "start_index": 1}
    _append_actor_process_event(tmp_path, event)
    _append_actor_process_event(tmp_path, {**event, "pid": 5678, "start_index": 2})

    lines = (tmp_path / "actor_processes.jsonl").read_text().splitlines()
    assert [json.loads(line)["pid"] for line in lines] == [1234, 5678]


def test_checkpoint_keeps_mixed_contract_but_dispatches_homogeneous_batches():
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    definitions = [TaskContract(config, config["seed"]).build(i) for i in range(10)]
    gpu = FakeGpuActor()
    paths, metadata, stats = generate_checkpoint(
        config,
        definitions,
        [FakeEndpointActor(), FakeEndpointActor()],
        gpu,
        FakeBulletActor(),
    )

    assert len(paths) == len(metadata) == 10
    assert [item["task_id"] for item in metadata] == list(range(10))
    assert gpu.plan_batches == [
        ("dual_independent", 6),
        ("left_only", 2),
        ("right_only", 2),
    ]
    assert stats["accepted"] == 10
    assert stats["gpu_plan_batches"] == 3
    assert stats["gpu_plan_queries"] == 10
    assert stats["candidate_waves"] == 1
    assert stats["stale_candidates_discarded"] == 0
    assert stats["checkpoint_wall_milliseconds"] >= 0
    assert stats["endpoint_proposal_wall_milliseconds"] >= 0
    assert stats["gpu_endpoint_audit_wall_milliseconds"] >= 0
    assert stats["pybullet_endpoint_audit_wall_milliseconds"] >= 0
    assert stats["gpu_plan_audit_wall_milliseconds"] >= 0
    assert stats["pybullet_trajectory_audit_wall_milliseconds"] >= 0


def test_gpu_launcher_dry_run_does_not_start_actors(tmp_path, capsys):
    assert main(
        [
            "--num-trajectories",
            "10",
            "--output-dir",
            str(tmp_path / "dry"),
            "--dry-run",
        ]
    ) == 0
    assert not (tmp_path / "dry").exists()
    output = capsys.readouterr().out
    assert "active_unfinished=80" in output
    assert "max_open_shards=8" in output


def test_supervised_dry_run_reports_segment_limits_without_spawning(tmp_path, capsys):
    assert main(
        [
            "--num-trajectories",
            "10",
            "--output-dir",
            str(tmp_path / "dry"),
            "--supervised-short-processes",
            "--segment-seconds",
            "720",
            "--segment-max-accepted-tasks",
            "50",
            "--dry-run",
        ]
    ) == 0
    assert not (tmp_path / "dry").exists()
    output = capsys.readouterr().out
    assert "supervised_short_processes=True" in output
    assert "segment_seconds=720" in output
    assert "segment_tasks=50" in output


def test_outer_supervisor_relaunches_only_clean_incomplete_segments(
    tmp_path, monkeypatch
):
    returncodes = iter((SEGMENT_INCOMPLETE_EXIT_CODE, 0))
    commands = []

    class FakeProcess:
        def __init__(self, command):
            commands.append(command)
            self.pid = 1000 + len(commands)
            self.returncode = next(returncodes)

        def wait(self):
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", FakeProcess)

    assert _supervise_short_processes(
        ["--supervised-short-processes", "--num-trajectories", "10"],
        tmp_path,
        segment_seconds=600,
        segment_tasks=0,
        delay=0,
    ) == 0
    assert len(commands) == 2
    assert all(command[-1] == "--segment-child" for command in commands)
    events = [
        json.loads(line)
        for line in (tmp_path / "supervisor_segments.jsonl").read_text().splitlines()
    ]
    assert [event["returncode"] for event in events if event["event"] == "segment_exited"] == [
        SEGMENT_INCOMPLETE_EXIT_CODE,
        0,
    ]


def test_gpu_launcher_uses_start_to_exclusive_end_task_ids(tmp_path, capsys):
    assert main(
        [
            "--start-shard",
            "20",
            "--num-trajectories",
            "50",
            "--output-dir",
            str(tmp_path / "range"),
            "--dry-run",
        ]
    ) == 0
    assert not (tmp_path / "range").exists()
    output = capsys.readouterr().out
    assert "30 trajectories for task IDs [20, 50)" in output


def test_endpoint_jobs_are_dynamically_refilled_one_at_a_time():
    actors = [FakeEndpointActor(), FakeEndpointActor()]
    jobs = [
        {
            "task": {
                "task_id": task_id,
                "mode": "dual_independent",
                "direction": "random_to_random",
            },
            "attempt": 1,
            "candidate_indices": [task_id],
            "candidate_seeds": [task_id],
        }
        for task_id in range(5)
    ]
    result = _dynamic_proposals(actors, jobs, idle_sleep=0)
    assert sorted(item["task_id"] for item in result) == list(range(5))
    assert all(len(actor.message["jobs"]) == 1 for actor in actors)


def test_candidate_waves_prioritize_distinct_unfinished_tasks():
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    contract = TaskContract(config, config["seed"])
    definitions = {task_id: contract.build(task_id) for task_id in (0, 1, 3)}
    endpoints = {
        task_id: [
            {"task_id": task_id, "candidate_index": candidate}
            for candidate in range(3)
        ]
        for task_id in definitions
    }
    waves = list(_candidate_waves(endpoints, definitions, accepted={1: object()}))
    assert len(waves) == 3
    for wave, by_mode in waves:
        items = [item for values in by_mode.values() for item in values]
        assert {(item["task_id"], item["candidate_index"]) for item in items} == {
            (0, wave),
            (3, wave),
        }


def test_dual_candidates_cross_arm_pairs_but_single_arm_candidates_do_not():
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    contract = TaskContract(config, config["seed"])
    definitions = {index: contract.build(index) for index in (0, 3)}
    proposals = []
    for task_id in (0, 3):
        for candidate in range(2):
            proposals.append(
                {
                    "task_id": task_id,
                    "attempt": 1,
                    "candidate_index": candidate,
                    "q_start": np.r_[
                        np.full(7, candidate + 1), np.full(7, candidate + 11)
                    ],
                    "q_goal": np.r_[
                        np.full(7, candidate + 21), np.full(7, candidate + 31)
                    ],
                    "ee_goal_pose": np.stack(
                        (np.full((3, 4), candidate + 41), np.full((3, 4), candidate + 51))
                    ),
                }
            )
    result = _factor_dual_candidates(proposals, definitions, maximum=64)
    dual = [item for item in result if item["task_id"] == 0]
    single = [item for item in result if item["task_id"] == 3]
    assert len(dual) == 4 and len(single) == 2
    crossed = dual[1]
    np.testing.assert_array_equal(crossed["q_start"][:7], np.ones(7))
    np.testing.assert_array_equal(crossed["q_start"][7:], np.full(7, 12))
    np.testing.assert_array_equal(crossed["ee_goal_pose"][0], np.full((3, 4), 41))
    np.testing.assert_array_equal(crossed["ee_goal_pose"][1], np.full((3, 4), 52))


def test_pipeline_telemetry_is_written_beside_and_into_manifest(tmp_path):
    (tmp_path / "manifest.yaml").write_text("schema: test\n")
    telemetry = {
        "schema": "telemetry/v1",
        "counters": {
            "accepted": 10,
            "endpoint_actor_restarts": 3,
            "gpu_actor_restarts": 1,
            "pybullet_actor_restarts": 0,
        },
    }
    _write_pipeline_telemetry(tmp_path, telemetry)
    assert yaml.safe_load((tmp_path / "pipeline_telemetry.yaml").read_text()) == telemetry
    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert manifest["pipeline_telemetry"] == telemetry
    assert manifest["stats"]["endpoint_actor_restarts"] == 3
    assert manifest["stats"]["gpu_actor_restarts"] == 1
    assert manifest["stats"]["pybullet_actor_restarts"] == 0
