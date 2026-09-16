from pathlib import Path

import numpy as np
import yaml

from scripts.generate_data.launch_generate_marvin_warehouse_gpu_pipeline import (
    DEFAULT_CONFIG,
)
from scripts.generate_data.marvin_gpu_streaming import StreamingCoordinator
from scripts.generate_data.marvin_gpu_task_contract import TaskContract


class ImmediateActor:
    def __init__(self, role, fail_first_attempt_task=None):
        self.role = role
        self.fail_first_attempt_task = fail_first_attempt_task
        self.response = None
        self.restart_count = 0
        self.plan_batches = []
        self.recycle_count = 0

    def send(self, message):
        if self.response is not None:
            raise RuntimeError("actor is busy")
        if message["op"] == "telemetry":
            self.response = {"host_peak_rss_kib": 1234}
        elif self.role == "endpoint":
            values = []
            for job in message["jobs"]:
                task = job["task"]
                for candidate in job["candidate_indices"]:
                    q_start = np.zeros(14)
                    q_goal = np.zeros(14)
                    if task["mode"] in ("dual_independent", "left_only"):
                        q_goal[:7] = 0.5 + 0.001 * candidate
                    if task["mode"] in ("dual_independent", "right_only"):
                        q_goal[7:] = -0.5 - 0.001 * candidate
                    values.append(
                        {
                            "task_id": task["task_id"],
                            "attempt": job["attempt"],
                            "candidate_index": candidate,
                            "proposal": True,
                            "q_start": q_start,
                            "q_goal": q_goal,
                            "ee_goal_pose": np.zeros((2, 3, 4), dtype=np.float32),
                        }
                    )
            self.response = values
        elif self.role == "gpu" and message["op"] == "endpoints":
            self.response = [True] * len(message["items"])
        elif self.role == "gpu":
            self.plan_batches.append(
                (message["mode"], [item["task_id"] for item in message["items"]])
            )
            values = []
            for item in message["items"]:
                reject = (
                    item["task_id"] == self.fail_first_attempt_task
                    and item["attempt"] == 1
                )
                path = None if reject else np.linspace(item["q_start"], item["q_goal"], 128)
                values.append(
                    {
                        "task_id": item["task_id"],
                        "attempt": item["attempt"],
                        "path": path,
                        "spline": None
                        if reject
                        else (np.zeros(28), np.zeros((14, 22)), 5),
                        "rejection_reason": "rrt" if reject else None,
                        "batch_seconds": 0.01,
                        "rrt_iterations": 1,
                        "rrt_sampled_edges": 16,
                        "rrt_checked_states": 32,
                    }
                )
            self.response = values
        else:
            self.response = [True] * len(message["items"])

    def poll(self):
        if self.response is None:
            return False, None
        result, self.response = self.response, None
        return True, result

    def request(self, message):
        self.send(message)
        ready, result = self.poll()
        assert ready
        return result

    def recycle(self):
        self.recycle_count += 1


class RecordingSpool:
    def __init__(self):
        self.progress = []
        self.stats = []

    def restore(self, shard):
        return []

    def save_progress(self, task, shard_size):
        self.progress.append((task.task_id, task.attempt, shard_size))

    def save_stats(self, shard):
        self.stats.append(dict(shard.stats))


def test_streaming_window_batches_across_shards_and_publishes_independently(tmp_path):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    assert config["gpu_pipeline_active_unfinished_tasks"] == 80
    assert config["gpu_pipeline_max_open_shards"] == 8
    assert config["gpu_query_batch_size_dual"] == 12
    assert config["gpu_query_batch_size_left"] == 16
    assert config["gpu_query_batch_size_right"] == 16
    assert config["gpu_pipeline_endpoint_workers"] == 2
    assert config["gpu_pipeline_endpoint_recycle_jobs"] == 100
    config.update(
        gpu_pipeline_active_unfinished_tasks=20,
        gpu_pipeline_max_open_shards=2,
        gpu_pipeline_plan_batch_max_wait_seconds=0,
        gpu_pipeline_pybullet_recycle_trajectories=0,
        gpu_pipeline_endpoint_recycle_jobs=2,
    )
    specs = [
        (start, 10, tmp_path / f"{start:09d}") for start in (0, 10)
    ]
    endpoints = [ImmediateActor("endpoint") for _ in range(4)]
    gpu = ImmediateActor("gpu", fail_first_attempt_task=0)
    bullet = ImmediateActor("pybullet")
    published = []

    def publish(path, saved_config, start, paths, metadata, stats):
        published.append((start, paths, metadata, stats))

    coordinator = StreamingCoordinator(
        config,
        specs,
        TaskContract(config, config["seed"]),
        endpoints,
        gpu,
        bullet,
        publish,
    )
    completed, stats = coordinator.run()

    assert sorted(path.name for path in completed) == ["000000000", "000000010"]
    assert [item[0] for item in published] == [10, 0]
    dual_batches = [task_ids for mode, task_ids in gpu.plan_batches if mode == "dual_independent"]
    assert any(min(task_ids) < 10 <= max(task_ids) for task_ids in dual_batches)
    assert all(len(paths) == len(metadata) == 10 for _, paths, metadata, _ in published)
    expected_dual_queries = sum(
        len(task_ids)
        for mode, task_ids in gpu.plan_batches
        if mode == "dual_independent"
    )
    assert stats["gpu_plan_queries/dual_independent"] == expected_dual_queries
    endpoints[0].restart_count = 2
    gpu.restart_count = 1
    bullet.restart_count = 0
    telemetry = coordinator.telemetry()
    assert telemetry["batch_occupancy"]["dual_independent"] > 0
    assert telemetry["counters"]["max_open_shards"] == 2
    assert telemetry["counters"]["max_active_window_tasks"] == 20
    assert telemetry["counters"]["max_active_unfinished_tasks"] == 20
    assert telemetry["counters"]["endpoint_actor_recycles"] > 0
    assert telemetry["counters"]["endpoint_actor_restarts"] == 2
    assert telemetry["counters"]["gpu_actor_restarts"] == 1
    assert telemetry["counters"]["pybullet_actor_restarts"] == 0
    assert sum(actor.recycle_count for actor in endpoints) == telemetry["counters"][
        "endpoint_actor_recycles"
    ]
    assert telemetry["actors"]["gpu"]["host_peak_rss_kib"] == 1234
    assert telemetry["configuration"]["rrt_edges_per_query"] == config[
        "gpu_rrt_edges_per_query"
    ]
    assert telemetry["configuration"]["rrt_nearest_chunk_size"] == config[
        "gpu_rrt_nearest_chunk_size"
    ]
    assert telemetry["configuration"]["collision_batch_size"] == config[
        "gpu_collision_batch_size"
    ]
    assert telemetry["configuration"]["endpoint_recycle_jobs"] == 2
    assert all(
        "endpoint_actor_restarts" not in shard_stats
        for _, _, _, shard_stats in published
    )


def test_streaming_rejects_more_than_one_planned_candidate_per_task(tmp_path):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    config["gpu_pipeline_max_inflight_candidates_per_task"] = 2
    actors = [ImmediateActor("endpoint")]

    with np.testing.assert_raises_regex(ValueError, "exactly one planned candidate"):
        StreamingCoordinator(
            config,
            [(0, 10, tmp_path / "000000000")],
            TaskContract(config, config["seed"]),
            actors,
            ImmediateActor("gpu"),
            ImmediateActor("pybullet"),
            lambda *args: None,
        )


def test_attempt_progress_is_saved_once_and_stats_are_debounced(tmp_path):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    config.update(
        gpu_pipeline_active_unfinished_tasks=1,
        gpu_pipeline_max_open_shards=1,
        gpu_pipeline_stats_flush_accepted=2,
        gpu_pipeline_stats_flush_seconds=3600,
    )
    endpoint = ImmediateActor("endpoint")
    spool = RecordingSpool()
    coordinator = StreamingCoordinator(
        config,
        [(0, 1, tmp_path / "000000000")],
        TaskContract(config, config["seed"]),
        [endpoint],
        ImmediateActor("gpu"),
        ImmediateActor("pybullet"),
        lambda *args: None,
        spool,
    )
    coordinator.window.fill(coordinator._open_shard)

    assert coordinator._dispatch_endpoints()
    assert coordinator._poll_endpoints()
    task = coordinator._task(0)
    for candidate in task.candidates.values():
        candidate.reject("test")
    assert coordinator._dispatch_endpoints()

    assert spool.progress == [(0, 1, 1)]
    assert not coordinator._flush_due_stats()
    shard = coordinator._shard(0)
    shard.stats["accepted"] = 2
    coordinator.stats_dirty.add(shard.start)
    assert coordinator._flush_due_stats()
    assert len(spool.stats) == 1
