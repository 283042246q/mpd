import json

import h5py
import numpy as np
import pytest
import torch

from scripts.spacetime_data import generate_warehouse_spacetime as generation
from scripts.spacetime_data.batch_spacetime_rrt import SpaceTimeResult


class FakeWarehouse:
    margin = 0.005
    tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}

    def __init__(self, *args, **kwargs):
        from types import SimpleNamespace
        self.robot = SimpleNamespace(q_pos_min=torch.full((7,), -2.), q_pos_max=torch.full((7,), 2.),
                                     dq_max=torch.ones(7), ddq_max=torch.full((7,), 3.))

    def static_collision(self, q):
        return torch.zeros(len(q), dtype=torch.bool)

    def centers(self, q):
        return torch.tensor([0.4, 0., 0.5]).expand(len(q), 12, 3)

    def set_worlds(self, worlds):
        pass

    def __call__(self, q, times, ids):
        return torch.zeros(len(q), dtype=torch.bool)

    def clearances(self, q, times, ids):
        return torch.ones((len(q), 3))


class FakePlanner:
    def __init__(self, *args, **kwargs):
        pass

    def plan_batch(self, starts, goals, *args, **kwargs):
        paths = [np.c_[np.linspace(a, b, 8), np.linspace(0, 5, 8)] for a, b in zip(starts, goals)]
        return [SpaceTimeResult([path], {"backend": "fixture"}) for path in paths]


def test_atomic_dataset_resume_and_source_family_split(tmp_path, monkeypatch):
    source = tmp_path / "source.hdf5"
    path = np.zeros((32, 7))
    path[:, 0] = np.linspace(-0.5, 0.5, 32)
    with h5py.File(source, "w") as file:
        file["sol_path"] = np.stack([path, path[::-1]])
        file["task_id"] = [10, 10]
    output = tmp_path / "output"
    monkeypatch.setattr(generation, "WarehouseCollision", FakeWarehouse)
    monkeypatch.setattr(generation, "TorchSpaceTimeForest", FakePlanner)
    command = ["--source", str(source), "--output-root", str(output), "--device", "cpu",
               "--num-trajectories", "2", "--query-batch-size", "2", "--obstacles", "1",
               "--nominal-duration", "4"]
    assert generation.main(command) == 0
    shards = list((output / "shards").glob("*.hdf5"))
    assert len(shards) == 1 and not list(output.rglob("*.inprogress"))
    with h5py.File(shards[0]) as file:
        assert len(file["samples"]) == 2
        for sample in file["samples"].values():
            assert sample.attrs["accepted"]
            assert sample["world/positions"].shape[-2:] == (1, 3)
            assert sample["spatial/control_points"].shape == (29, 7)
    index = json.loads((output / "index.json").read_text())
    assert sorted(len(v) for v in index["splits"].values()) == [0, 0, 2]
    hashes_before = [file.read_bytes() for file in shards]
    # Simulate a crash after the atomic shard commit but before index publish.
    (output / "index.json").unlink()
    assert generation.main(command + ["--resume"]) == 0
    assert (output / "index.json").exists()
    assert hashes_before == [file.read_bytes() for file in shards]
    # Extending a completed run adds samples, never rewrites committed shards.
    extended = command.copy()
    extended[extended.index("--num-trajectories") + 1] = "3"
    assert generation.main(extended + ["--resume"]) == 0
    assert len(list((output / "shards").glob("*.hdf5"))) == 2
    assert hashes_before[0] == shards[0].read_bytes()
    assert json.loads((output / "report.json").read_text())["accepted"] == 3
    with pytest.raises(ValueError, match="do not match"):
        generation.main(command + ["--resume", "--margin", "0.01"])


def test_invalid_generation_arguments_are_rejected():
    with pytest.raises(SystemExit):
        generation.parse_args(["--output-root", "/tmp/unused", "--query-batch-size", "0"])
    with pytest.raises(SystemExit):
        generation.parse_args(["--output-root", "/tmp/unused", "--velocity-scale", "2"])


@pytest.mark.parametrize("policy,expected_sizes", [("fill", [3]), ("legacy", [1, 1, 1])])
def test_torch_batch_policy_refills_rejected_candidates_without_losing_attempts(tmp_path, monkeypatch,
                                                                              policy, expected_sizes):
    source = tmp_path / "source.hdf5"
    path = np.zeros((32, 7))
    path[:, 0] = np.linspace(-0.5, 0.5, 32)
    with h5py.File(source, "w") as file:
        file["sol_path"] = np.stack([path, path[::-1]])
        file["task_id"] = [10, 10]
    sizes = []

    class RecordingPlanner(FakePlanner):
        def plan_batch(self, starts, goals, *args, **kwargs):
            sizes.append(len(starts))
            return super().plan_batch(starts, goals, *args, **kwargs)

    original_prepare = generation.prepare_job

    def prepare(args, source, collision, current):
        if current % 2 == 0:
            return None, "source_static_self_or_limits"
        return original_prepare(args, source, collision, current)

    monkeypatch.setattr(generation, "WarehouseCollision", FakeWarehouse)
    monkeypatch.setattr(generation, "TorchSpaceTimeForest", RecordingPlanner)
    monkeypatch.setattr(generation, "prepare_job", prepare)
    output = tmp_path / "output"
    command = ["--source", str(source), "--output-root", str(output), "--device", "cpu",
               "--num-trajectories", "3", "--query-batch-size", "3", "--torch-batch-policy", policy,
               "--obstacles", "1", "--nominal-duration", "4", "--max-attempts", "12"]
    assert generation.main(command) == 0
    assert sizes == expected_sizes
    report = json.loads((output / "report.json").read_text())
    progress = generation.read_progress(output, report["config_hash"])
    assert progress[0:2] == (3, 6)
    assert progress[2] == {"source_static_self_or_limits": 3}
    assert sorted(entry["attempt_id"] for entry in progress[3]) == [1, 3, 5]
    stats = report["session_pipeline_statistics"]
    assert stats["prepared"] == 6 and stats["submitted"] == stats["completed"] == stats["accepted"] == 3
    assert stats["planned_batches"] == len(sizes)
    assert stats["effective_batch_size_mean"] == 3 / len(sizes)
    assert stats["preparation_wall_s"] >= 0 and stats["postprocess_wall_s"] >= 0
    hashes = [file.read_bytes() for file in sorted((output / "shards").glob("*.hdf5"))]
    assert generation.main(command + ["--resume"]) == 0
    assert [file.read_bytes() for file in sorted((output / "shards").glob("*.hdf5"))] == hashes
    assert json.loads((output / "report.json").read_text())["session_pipeline_statistics"] == stats
    extended = command.copy()
    extended[extended.index("--num-trajectories") + 1] = "4"
    assert generation.main(extended + ["--resume"]) == 0
    extended_report = json.loads((output / "report.json").read_text())
    extended_progress = generation.read_progress(output, extended_report["config_hash"])
    assert extended_progress[0:2] == (4, 8)
    assert sorted(entry["attempt_id"] for entry in extended_progress[3]) == [1, 3, 5, 7]
    assert [file.read_bytes() for file in sorted((output / "shards").glob("*.hdf5"))[:len(hashes)]] == hashes


@pytest.mark.parametrize("all_invalid", [False, True])
def test_torch_fill_stops_at_attempt_budget_and_commits_partial_or_empty_batch(tmp_path, monkeypatch, all_invalid):
    source = tmp_path / "source.hdf5"
    path = np.zeros((32, 7))
    path[:, 0] = np.linspace(-0.5, 0.5, 32)
    with h5py.File(source, "w") as file:
        file["sol_path"] = np.stack([path, path[::-1]])
        file["task_id"] = [10, 10]
    original_prepare = generation.prepare_job

    def prepare(args, source, collision, current):
        if all_invalid or current % 2 == 0:
            return None, "source_static_self_or_limits"
        return original_prepare(args, source, collision, current)

    monkeypatch.setattr(generation, "WarehouseCollision", FakeWarehouse)
    monkeypatch.setattr(generation, "TorchSpaceTimeForest", FakePlanner)
    monkeypatch.setattr(generation, "prepare_job", prepare)
    output = tmp_path / "output"
    command = ["--source", str(source), "--output-root", str(output), "--device", "cpu",
               "--num-trajectories", "3", "--query-batch-size", "3", "--obstacles", "1",
               "--nominal-duration", "4", "--max-attempts", "4"]
    assert generation.main(command) == 2
    report = json.loads((output / "report.json").read_text())
    expected = 0 if all_invalid else 2
    assert generation.read_progress(output, report["config_hash"])[0:2] == (expected, 4)
    assert report["session_pipeline_statistics"]["submitted"] == expected
    assert not report["complete"]


def test_second_writer_is_rejected_before_loading_models(tmp_path):
    import fcntl
    with (tmp_path / ".writer.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another writer"):
            generation.main(["--output-root", str(tmp_path), "--device", "cpu"])


def test_streaming_dataset_recovers_committed_prefix_and_restores_each_world(tmp_path, monkeypatch):
    from multiprocessing.pool import ThreadPool
    from scripts.spacetime_data import ompl_strrt_batch

    class RecordingWarehouse(FakeWarehouse):
        def set_worlds(self, worlds):
            self.worlds = worlds

    class AsyncPlanner:
        def __init__(self, *args, **kwargs):
            self.pool = ThreadPool(2)

        def submit(self, job):
            def solve():
                path = np.c_[np.linspace(job["q_start"], job["q_goal"], 8), np.linspace(0, 5, 8)]
                return SpaceTimeResult([path], {"backend": "fixture", "wall_s": 0.01})
            return self.pool.apply_async(solve)

        def close(self):
            self.pool.close()
            self.pool.join()

        terminate = close

    source = tmp_path / "source.hdf5"
    path = np.zeros((32, 7))
    path[:, 0] = np.linspace(-0.5, 0.5, 32)
    with h5py.File(source, "w") as file:
        file["sol_path"] = np.stack([path, path[::-1]])
        file["task_id"] = [10, 10]
    output = tmp_path / "stream"
    monkeypatch.setattr(generation, "WarehouseCollision", RecordingWarehouse)
    monkeypatch.setattr(ompl_strrt_batch, "OmplSTRRTBatch", AsyncPlanner)
    original_validate, original_write = generation.validate_result, generation.write_shard
    validated = []

    def validate(args, collision, query, job, result):
        assert query == 0 and collision.worlds == [job["forecast"]]
        validated.append(job["attempt_id"])
        return original_validate(args, collision, query, job, result)

    def interrupted_write(output, shard_index, *args):
        if shard_index == 1:
            raise RuntimeError("interrupted before atomic commit")
        return original_write(output, shard_index, *args)

    monkeypatch.setattr(generation, "validate_result", validate)
    monkeypatch.setattr(generation, "write_shard", interrupted_write)
    command = ["--source", str(source), "--output-root", str(output), "--device", "cpu",
               "--backend", "ompl-strrtstar", "--num-trajectories", "5", "--cpu-workers", "2",
               "--query-batch-size", "2", "--ompl-queue-size", "4", "--obstacles", "1",
               "--nominal-duration", "4"]
    with pytest.raises(RuntimeError, match="interrupted"):
        generation.main(command)
    shard = next((output / "shards").glob("*.hdf5"))
    committed = shard.read_bytes()
    with h5py.File(shard) as file:
        assert file.attrs["attempt_start"] == 0 and file.attrs["attempt_stop"] == 2
        assert list(file["samples"]) == ["000000000", "000000001"]
    monkeypatch.setattr(generation, "write_shard", original_write)
    validated.clear()
    assert generation.main(command + ["--resume"]) == 0
    assert shard.read_bytes() == committed
    assert sorted(validated) == [2, 3, 4]
    report = json.loads((output / "report.json").read_text())
    assert report["accepted"] == report["attempts"] == 5
    stats = report["session_pipeline_statistics"]
    assert stats["submitted"] == stats["completed"] == stats["accepted"] == 3
    assert stats["queue_capacity"] == 4
    assert generation.read_progress(output, report["config_hash"])[0:2] == (5, 5)
    assert generation.main(command + ["--resume"]) == 0
    assert json.loads((output / "report.json").read_text())["session_pipeline_statistics"] == stats
