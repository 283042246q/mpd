import subprocess
import sys

import numpy as np
import yaml

from scripts.generate_data.launch_generate_marvin_warehouse_gpu_pipeline import (
    DEFAULT_CONFIG,
    _factor_dual_candidates,
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


def test_gpu_launcher_import_does_not_eagerly_load_native_workers():
    command = (
        "import sys; "
        "import scripts.generate_data.launch_generate_marvin_warehouse_gpu_pipeline; "
        "assert 'ompl' not in sys.modules; "
        "assert 'pybullet' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


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
    assert stats["checkpoint_wall_milliseconds"] >= 0
    assert stats["endpoint_proposal_wall_milliseconds"] >= 0
    assert stats["gpu_endpoint_audit_wall_milliseconds"] >= 0
    assert stats["pybullet_endpoint_audit_wall_milliseconds"] >= 0
    assert stats["gpu_plan_audit_wall_milliseconds"] >= 0
    assert stats["pybullet_trajectory_audit_wall_milliseconds"] >= 0


def test_gpu_launcher_dry_run_does_not_start_actors(tmp_path):
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
