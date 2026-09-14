from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts.generate_data.marvin_gpu_partial_spool import PartialTaskSpool
from scripts.generate_data.marvin_gpu_scheduler import ShardState, TaskState
from scripts.generate_data.marvin_gpu_task_contract import TaskDefinition


def definition(task_id):
    return TaskDefinition(
        task_id,
        "left_only",
        "random_to_random",
        {"left": "random", "right": "inactive"},
        {"left": "random", "right": "inactive"},
    )


def make_shard(tmp_path):
    tasks = {task_id: TaskState(definition(task_id), 0) for task_id in range(10)}
    return ShardState(0, 10, Path(tmp_path) / "shards/000000000", tasks)


def accept(task):
    path = np.linspace(np.zeros(14), np.ones(14), 128)
    metadata = {
        "task_id": task.task_id,
        "task_mode": task.definition.mode,
        "direction": task.definition.direction,
        "q_start": path[0],
        "q_goal": path[-1],
        "planning_time": 1.25,
        "bspline": (np.zeros(28), np.zeros((14, 22)), 5),
        "ee_goal_pose": np.zeros((2, 3, 4), dtype=np.float32),
        "joint_path_length": 2.0,
        "source_region_left": task.definition.source["left"],
        "source_region_right": task.definition.source["right"],
        "goal_region_left": task.definition.goal["left"],
        "goal_region_right": task.definition.goal["right"],
    }
    task.accept(path, metadata)


def test_partial_spool_round_trip_and_cleanup(tmp_path):
    config = {"seed": 7, "planner": "GpuMultiQueryRRTConnect"}
    spool = PartialTaskSpool(tmp_path, config)
    shard = make_shard(tmp_path)
    accept(shard.tasks[3])
    shard.stats = Counter(task_attempts=4, accepted=1)
    spool.save_task(shard.tasks[3])
    spool.save_stats(shard)

    restored = make_shard(tmp_path)
    assert spool.restore(restored) == [3]
    assert restored.tasks[3].accepted
    np.testing.assert_array_equal(
        restored.tasks[3].accepted_path, shard.tasks[3].accepted_path
    )
    assert restored.tasks[3].accepted_metadata["planning_time"] == 1.25
    assert restored.stats["task_attempts"] == 4

    spool.clear_shard(0)
    assert not (tmp_path / ".inflight/000000000").exists()


def test_partial_spool_rejects_different_configuration(tmp_path):
    first = PartialTaskSpool(tmp_path, {"seed": 1})
    first.prepare_shard(0, 10)
    second = PartialTaskSpool(tmp_path, {"seed": 2})
    with pytest.raises(ValueError, match="contract differs"):
        second.prepare_shard(0, 10)
