from collections import Counter

import yaml

from scripts.generate_data.generate_marvin_warehouse_bimanual import (
    DEFAULT_CONFIG,
    _direction_schedule,
    _placement_distributions,
)
from scripts.generate_data.marvin_gpu_task_contract import (
    TaskContract,
    direction_schedule,
    placement_distributions,
    stable_seed,
)


def test_gpu_contract_matches_cpu_schedule_and_marginals():
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    assert direction_schedule(config["trajectory_direction_weights"]) == _direction_schedule(
        config["trajectory_direction_weights"]
    )
    gpu = placement_distributions(config)
    cpu = _placement_distributions(config)
    for arm in ("left", "right"):
        assert gpu[arm][0] == cpu[arm][0]
        assert (gpu[arm][1] == cpu[arm][1]).all()


def test_gpu_contract_preserves_mixed_shard_quotas_and_fixed_categories():
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    contract = TaskContract(config, config["seed"])
    tasks = [contract.build(task_id) for task_id in range(100)]
    assert Counter(task.mode for task in tasks) == {
        "dual_independent": 60,
        "left_only": 20,
        "right_only": 20,
    }
    assert Counter(task.direction for task in tasks) == {
        "placement_to_placement": 45,
        "random_to_placement": 35,
        "placement_to_random": 10,
        "random_to_random": 10,
    }
    assert contract.build(7) == contract.build(7)


def test_stable_seed_depends_on_task_attempt_and_stage():
    values = {
        stable_seed(1, task, attempt, stage)
        for task in range(2)
        for attempt in range(2)
        for stage in ("endpoint", "rrt")
    }
    assert len(values) == 8
