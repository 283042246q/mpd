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


def test_second_writer_is_rejected_before_loading_models(tmp_path):
    import fcntl
    with (tmp_path / ".writer.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another writer"):
            generation.main(["--output-root", str(tmp_path), "--device", "cpu"])
