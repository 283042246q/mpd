from pathlib import Path

import numpy as np

from scripts.generate_data.marvin_gpu_scheduler import (
    ActiveShardWindow,
    CandidateStage,
    ShardState,
    TaskState,
    TaskStatus,
)
from scripts.generate_data.marvin_gpu_task_contract import TaskDefinition


def definition(task_id):
    return TaskDefinition(
        task_id,
        "dual_independent",
        "random_to_random",
        {"left": "random", "right": "random"},
        {"left": "random", "right": "random"},
    )


def shard(start, size, path):
    tasks = {
        task_id: TaskState(definition(task_id), start)
        for task_id in range(start, start + size)
    }
    return ShardState(start, size, Path(path), tasks)


def test_task_candidate_lifecycle_and_acceptance_stales_other_candidates():
    task = TaskState(definition(3), 0)
    assert task.begin_attempt(3) == 1
    first = task.add_candidate({"value": 1})
    second = task.add_candidate({"value": 2})
    task.finish_endpoint_work()
    first.advance(CandidateStage.PYBULLET_ENDPOINT)
    second.reject("gpu_endpoint")
    assert task.accept(np.zeros((2, 14)), {"task_id": 3})
    assert task.status == TaskStatus.ACCEPTED
    assert first.stage == CandidateStage.STALE
    assert second.stage == CandidateStage.REJECTED
    assert not task.accept(None, None)


def test_hard_attempt_limit_marks_task_failed():
    task = TaskState(definition(2), 0)
    assert task.begin_attempt(1) == 1
    task.finish_endpoint_work()
    assert task.begin_attempt(1) is None
    assert task.status == TaskStatus.HARD_FAILED


def test_active_window_opens_and_removes_independent_shards():
    specs = [(start, 10, Path(f"shard-{start}")) for start in range(0, 100, 10)]
    window = ActiveShardWindow(specs, max_tasks=40, max_shards=4)
    opened = window.fill(shard)
    assert [item.start for item in opened] == [0, 10, 20, 30]
    assert window.task_count == 40

    slow = window.open[0].tasks[7]
    for start in (10, 20, 30):
        for task in window.open[start].tasks.values():
            task.accept(np.zeros((2, 14)), {"task_id": task.task_id})
        assert window.open[start].complete
        window.remove(start)
    assert not window.open[0].complete and slow.status == TaskStatus.ACTIVE

    opened = window.fill(shard)
    assert [item.start for item in opened] == [40, 50, 60]
    assert window.task_count == 40
