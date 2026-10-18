"""Conversion contracts, independent refit gates, and crash-safe resume."""
import json
import os
import signal
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import yaml

from mpd.datasets.spacetime_schema import load_panda_robot_bundle, sha256_file
from mpd.parametric_trajectory.normalized_timing import NormalizedTimingSplineNumpy, minimum_duration_for_path
from scipy.interpolate import BSpline
from scripts.spacetime_data import generate_warehouse_spacetime as base
from scripts.spacetime_data import refit_warehouse_eegoal as refit
from scripts.spacetime_data.warehouse_world import Forecast
from scripts.spacetime_data.warehouse_trajectory_fit import fit_teacher


class CollisionFixture:
    margin = .005
    tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}

    def __init__(self, blocked=False):
        self.blocked = blocked
        self.robot = SimpleNamespace(q_pos_min=torch.full((7,), -2.), q_pos_max=torch.full((7,), 2.),
            dq_max=torch.ones(7), ddq_max=torch.full((7,), 3.), get_EE_pose=self.ee_pose)

    @staticmethod
    def ee_pose(q):
        pose = torch.cat((torch.eye(3), q[:3, None]), dim=1)
        return pose[None]

    def set_worlds(self, worlds):
        self.worlds = worlds

    def clearances(self, q, times, query_ids):
        result = torch.ones((len(q), 3))
        if self.blocked:
            result[:, 2] = -.1
        return result


def source_fixture(root, *, empty=False):
    root.mkdir()
    bundle = load_panda_robot_bundle(base.REPOSITORY)
    config = {"schema": base.SCHEMA, "robot_hashes": bundle.hashes,
              "static_environment": "EnvWarehouse", "source_size": 100, "source_mtime_ns": 5,
              "source_group_size": 2, "oracle_forecast": True,
              "static_geometry_hashes": {"env_warehouse.py": sha256_file(
                  base.REPOSITORY / "mpd/torch_robotics/torch_robotics/environments/env_warehouse.py")}}
    h = refit.digest(config)
    base.atomic_json(root / "config.json", config)
    (root / "manifest.yaml").write_text(yaml.safe_dump({"schema": base.SCHEMA, "config_hash": h}))
    (root / "shards").mkdir()
    forecast = Forecast(np.array([0., 1.5, 14.]), np.tile([[[.4, 0., .5]]], (3, 1, 1)),
        np.array([0]), np.array([[.05, 0., 0.]]), np.array([1.5]), ["linear"], np.array([[.4, 0., .5]]))
    if empty:
        forecast = Forecast(np.array([0., 14.]), np.empty((2, 0, 3)), np.empty(0, dtype=int),
                            np.empty((0, 3)), np.empty(0), [], np.empty((0, 3)))
    path = np.zeros((8, 8)); path[:, 0] = np.linspace(-.5, .5, 8); path[:, -1] = np.linspace(0, 5., 8)
    fitted, reason = fit_teacher(path, CollisionFixture(), 0, forecast, require_tau_r=True)
    assert fitted is not None, reason
    job = dict(attempt_id=0, source_row=2, family_id=1, source_task_id=7, seed=123,
               forecast=forecast, q_start=path[0, :7], q_goal=path[-1, :7],
               fitted_solution_index=0, fit_wall_s=0.)
    # First raw is deliberately malformed; conversion must try the second.
    broken = path.copy(); broken[-1, 0] += .1
    result = SimpleNamespace(paths=[broken, path], statistics={"backend": "test"})
    shard_path = base.write_shard(root, 0, 0, 1, [(job, result, fitted)], {}, h)
    with h5py.File(shard_path, "a") as f:
        g = f["samples/000000000"]
        g.attrs.update(world_id=3, obstacle_count=len(forecast.shape), direction=0, world_seed=77)
        # Prove Tmin and timing are recomputed, never copied as P29 targets.
        g["timing/T_min"][...] = -123.
    entry = {"file": "shards/shard-000000.hdf5", "group": "samples/000000000", "family_id": 1}
    base.atomic_json(root / "index.json", {"schema": base.SCHEMA, "config_hash": h,
        "splits": {"train": [entry], "validation": [], "test": []}})
    return shard_path


def args_for(root, output, resume=False):
    argv = ["--dataset-root", str(root), "--output-root", str(output), "--shard-size", "1"]
    return refit.parse_args(argv + (["--resume"] if resume else []))


def test_refit_p21_eegoal_timing_and_resume(tmp_path):
    root, output = tmp_path / "source", tmp_path / "converted"
    source = source_fixture(root)
    source_hash = sha256_file(source)
    output.mkdir()
    assert refit.run(args_for(root, output), collision=CollisionFixture()) == 0
    assert sha256_file(source) == source_hash
    manifest = yaml.safe_load((output / "manifest.yaml").read_text())
    assert manifest["spatial"]["num_learnable_control_points"] == 16
    assert manifest["goal_type"] == "ee_pose"
    with h5py.File(output / "shards/shard-000000.hdf5") as f:
        g = f["samples/000000000"]
        p, learned = g["spatial/control_points"][:], g["spatial/learnable_control_points"][:]
        assert p.shape == (21, 7) and learned.shape == (16, 7)
        restored = np.concatenate((np.repeat(p[:1], 3, axis=0), learned[:-1],
                                   np.repeat(learned[-1:], 3, axis=0)))
        np.testing.assert_array_equal(restored, p)
        np.testing.assert_allclose(g["condition/ee_goal_position"][:], p[-1, :3])
        assert g["condition/ee_goal_pose"].shape == (3, 4)
        assert g.attrs["split"] == "train" and g.attrs["world_seed"] == 77
        assert int(g["raw/fitted_solution_index"][()]) == 1
        assert int(g.attrs["original_fitted_solution_index"]) == 0
        tau_r = g["timing/tau_r"][:]
        spline = BSpline(g["spatial/knots"][:], p, 5)
        shape = NormalizedTimingSplineNumpy(num_phase_points=2049).evaluate(tau_r[1:])
        tmin = minimum_duration_for_path(spline, shape, dq_max=np.ones(7),
                                        ddq_max=np.full(7, 3.), duration_floor=2.).value
        np.testing.assert_allclose(g["timing/T_min"][()], tmin)
        duration = tmin + max(0., 14. - tmin) / (1 + np.exp(-tau_r[0]))
        np.testing.assert_allclose(duration, g["timing/reference_time"][-1], atol=1e-6)
        ledger = json.loads(f["conversion_outcomes"].asstr()[0])
        assert ledger["raw_attempts"][0]["reason"] == "raw_boundary_or_shape"
    # Simulate loss of the derived index/report after the atomic shard commit.
    (output / "index.json").unlink(); (output / "report.json").unlink()
    assert refit.run(args_for(root, output, resume=True), collision=CollisionFixture(blocked=True)) == 0
    assert json.loads((output / "report.json").read_text())["accepted"] == 1
    assert len(list((output / "shards").glob("*.hdf5"))) == 1
    # Parent index can grow; resume consumes only the original frozen snapshot.
    (root / "index.json").write_text("{}")
    assert refit.run(args_for(root, output, resume=True), collision=CollisionFixture()) == 0


def test_rejects_new_collision_and_keeps_failure_ledger(tmp_path):
    root, output = tmp_path / "source", tmp_path / "converted"
    source_fixture(root); output.mkdir()
    refit.run(args_for(root, output), collision=CollisionFixture(blocked=True))
    report = json.loads((output / "report.json").read_text())
    assert report["run_finished"] and report["accepted"] == 0
    assert report["failure_counts"] == {"dynamic_collision": 1}
    with h5py.File(output / "shards/shard-000000.hdf5") as f:
        assert len(f["samples"]) == 0
        assert json.loads(f["conversion_outcomes"].asstr()[0])["status"] == "fit_rejected"


def test_resume_rejects_modified_committed_source(tmp_path):
    root, output = tmp_path / "source", tmp_path / "converted"
    source = source_fixture(root); output.mkdir()
    refit.run(args_for(root, output), collision=CollisionFixture())
    with h5py.File(source, "a") as f:
        f.attrs["changed"] = True
    with pytest.raises(ValueError, match="shard changed"):
        refit.run(args_for(root, output, resume=True), collision=CollisionFixture())


def test_freeze_rejects_cross_root_family_split_conflict(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    source_fixture(one); source_fixture(two)
    idx = json.loads((two / "index.json").read_text())
    idx["splits"]["test"] = idx["splits"]["train"]; idx["splits"]["train"] = []
    base.atomic_json(two / "index.json", idx)
    with pytest.raises(ValueError, match="split conflict"):
        refit.freeze_inputs([one, two], 0, load_panda_robot_bundle(base.REPOSITORY))


def test_empty_world_and_partial_crash_resume(tmp_path, monkeypatch):
    root, output = tmp_path / "source", tmp_path / "converted"
    shard = source_fixture(root, empty=True)
    with h5py.File(shard, "a") as f:
        f.copy("samples/000000000", "samples/000000001")
        f["samples/000000001"].attrs["attempt_id"] = 1
        f.attrs["attempt_stop"] = 2
    idx = json.loads((root / "index.json").read_text())
    idx["splits"]["train"].append(dict(idx["splits"]["train"][0], group="samples/000000001"))
    base.atomic_json(root / "index.json", idx)
    output.mkdir()
    original_commit = refit.commit

    def interrupted_commit(*args, **kwargs):
        original_commit(*args, **kwargs)
        raise RuntimeError("simulated crash after first atomic commit")

    monkeypatch.setattr(refit, "commit", interrupted_commit)
    with pytest.raises(RuntimeError, match="simulated crash"):
        refit.run(args_for(root, output), collision=CollisionFixture())
    monkeypatch.setattr(refit, "commit", original_commit)
    refit.run(args_for(root, output, resume=True), collision=CollisionFixture())
    report = json.loads((output / "report.json").read_text())
    assert report["run_finished"] and report["processed"] == report["accepted"] == 2
    assert report["per_obstacle_count"] == {"0": {"accepted": 2}}
    assert len(list((output / "shards").glob("*.hdf5"))) == 2
    with h5py.File(output / "shards/shard-000001.hdf5") as f:
        g = f["samples/000000001"]
        assert g["world/positions"].shape == (2, 0, 3)
        assert g["world/event_times"].shape == (0,)


def two_row_source(root):
    shard = source_fixture(root)
    with h5py.File(shard, "a") as f:
        f.copy("samples/000000000", "samples/000000001")
        f["samples/000000001"].attrs["attempt_id"] = 1
        f.attrs["attempt_stop"] = 2
    index = json.loads((root / "index.json").read_text())
    index["splits"]["train"].append(dict(index["splits"]["train"][0], group="samples/000000001"))
    base.atomic_json(root / "index.json", index)


def fixture_native_worker(args, snapshot, config_hash, number, begin, sender, writer_lock=None):
    """Spawnable fixture: abrupt native-like death, including lost commit notice."""
    marker = args.output_root / "fixture_crashed"
    crash = not marker.exists() or args.crash_phase == "always"
    if crash:
        marker.touch()
        if args.crash_phase != "after_commit":
            staging = args.output_root / ".refit-staging/shards"
            staging.mkdir(parents=True, exist_ok=True)
            (staging / "shard-000000.hdf5.inprogress").write_bytes(b"incomplete HDF5")
            os.kill(os.getpid(), signal.SIGKILL)
    refit.process_shard(args, snapshot, config_hash, number, begin, CollisionFixture())
    if crash:
        os.kill(os.getpid(), signal.SIGKILL)
    sender.send({"shard": number})
    sender.close()


@pytest.mark.parametrize("phase", ["before_commit", "after_commit"])
def test_native_worker_death_replays_only_uncommitted_input(tmp_path, phase):
    root, output = tmp_path / "source", tmp_path / "converted"
    two_row_source(root)
    output.mkdir(); (output / "shards").mkdir()
    args = args_for(root, output)
    args.crash_phase, args.worker_shards, args.worker_retries = phase, 1, 1
    snapshot = refit.freeze_inputs([root], 0, load_panda_robot_bundle(base.REPOSITORY))
    paths = list(refit.isolated_shards(args, snapshot, "test-identity", 0, 0,
                                     worker_target=fixture_native_worker))
    assert [p.name for p in paths] == ["shard-000000.hdf5", "shard-000001.hdf5"]
    outcomes, entries = refit.restore(output, "test-identity", snapshot)
    assert [r["conversion_id"] for r in outcomes] == [0, 1]
    assert len(entries) == 2 and all(r["status"] == "accepted" for r in outcomes)
    assert not list((output / ".refit-staging/shards").glob("*"))


def test_repeated_native_failure_stops_without_rejection_or_skipping(tmp_path):
    root, output = tmp_path / "source", tmp_path / "converted"
    source_fixture(root)
    output.mkdir(); (output / "shards").mkdir()
    args = args_for(root, output)
    args.crash_phase, args.worker_retries = "always", 1
    snapshot = refit.freeze_inputs([root], 0, load_panda_robot_bundle(base.REPOSITORY))
    with pytest.raises(RuntimeError, match="no inputs skipped"):
        list(refit.isolated_shards(args, snapshot, "test-identity", 0, 0,
                                  worker_target=fixture_native_worker))
    assert not list((output / "shards").glob("*.hdf5"))
    assert len((output / "refit_faults.log").read_text().splitlines()) == 2


def fixture_lock_worker(writer_lock, connection):
    with os.fdopen(writer_lock.fd, "a"):
        connection.send("locked")
        connection.recv()
    connection.close()


def test_spawn_writer_keeps_exclusive_lock_after_supervisor_fd_closes(tmp_path):
    path = tmp_path / ".writer.lock"
    lock = path.open("a")
    refit.fcntl.flock(lock, refit.fcntl.LOCK_EX | refit.fcntl.LOCK_NB)
    context = refit.multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(target=fixture_lock_worker,
        args=(refit.SpawnWriterLock(lock.fileno()), child))
    try:
        worker.start(); child.close()
        assert parent.poll(15) and parent.recv() == "locked"
        lock.close()
        with path.open("a") as contender:
            with pytest.raises(BlockingIOError):
                refit.fcntl.flock(contender, refit.fcntl.LOCK_EX | refit.fcntl.LOCK_NB)
        parent.send("stop")
        worker.join(timeout=15)
        assert worker.exitcode == 0
        with path.open("a") as contender:
            refit.fcntl.flock(contender, refit.fcntl.LOCK_EX | refit.fcntl.LOCK_NB)
    finally:
        lock.close(); parent.close(); child.close()
        if worker.pid is not None:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
            worker.close()


def test_legacy_resume_audit_preserves_identity_and_never_initializes_cuda(tmp_path, monkeypatch):
    root, output = tmp_path / "source", tmp_path / "converted"
    two_row_source(root); output.mkdir()
    original_commit = refit.commit

    def interrupted_commit(*args, **kwargs):
        original_commit(*args, **kwargs)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(refit, "commit", interrupted_commit)
    with pytest.raises(RuntimeError, match="interrupted"):
        refit.run(args_for(root, output), collision=CollisionFixture())
    monkeypatch.setattr(refit, "commit", original_commit)
    # Model a pre-upgrade output, including the original immutable shard hash.
    config = json.loads((output / "config.json").read_text())
    config["options"]["software"][refit.CONTROLLER] = next(iter(refit.COMPATIBLE_CONTROLLERS))
    config_hash = refit.digest(config)
    base.atomic_json(output / "config.json", config)
    manifest = yaml.safe_load((output / "manifest.yaml").read_text())
    manifest["config_hash"] = config_hash
    (output / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
    shard = output / "shards/shard-000000.hdf5"
    with h5py.File(shard, "a") as f:
        f.attrs["config_hash"] = config_hash
    protected = {p: sha256_file(p) for p in (shard, output / "config.json", output / "input_snapshot.json")}

    def forbidden(*args, **kwargs):
        raise AssertionError("audit must not initialize fitting/CUDA")

    monkeypatch.setattr(refit, "WarehouseCollision", forbidden)
    monkeypatch.setattr(refit, "isolated_shards", forbidden)
    args = args_for(root, output, resume=True)
    args.audit_only = True
    refit.run(args)
    assert protected == {p: sha256_file(p) for p in protected}
    report = json.loads((output / "report.json").read_text())
    assert report["processed"] == report["accepted"] == 1
    assert not report["run_finished"] and report["config_hash"] == config_hash
    # A known controller upgrade cannot authorize a numerical/config change.
    args.margin += .001
    with pytest.raises(ValueError, match="options/software differ"):
        refit.run(args)
    args.margin -= .001
    config["options"]["software"]["scripts/spacetime_data/warehouse_trajectory_fit.py"] = "changed"
    base.atomic_json(output / "config.json", config)
    with pytest.raises(ValueError, match="options/software differ"):
        refit.run(args)
