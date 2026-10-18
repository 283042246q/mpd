import json
import multiprocessing
import pickle

import h5py
import pytest
import torch
from torch.utils.data import DataLoader

from mpd.datasets.warehouse_joint_cache import build_tensor_cache, CachedWarehouseJointDataset
from mpd.datasets.warehouse_joint_dataset import (
    WarehouseJointDataset, build_snapshot, collate_joint, identity, atomic_json,
)
from scripts.spacetime_data import refit_warehouse_eegoal as refit
from test_warehouse_eegoal_refit import source_fixture, args_for, CollisionFixture


@pytest.fixture
def cached_snapshot(tmp_path):
    roots = []
    # Include EMPTY and KNOWN worlds in one cache, with different token counts.
    for name, empty in (("known", False), ("empty", True)):
        raw, converted = tmp_path / name, tmp_path / (name + "-converted")
        source_fixture(raw, empty=empty); converted.mkdir()
        refit.run(args_for(raw, converted), collision=CollisionFixture())
        roots.append(converted)
    snapshot = tmp_path / "snapshot.json"
    build_snapshot(roots, snapshot)
    cache = build_tensor_cache(snapshot)
    return snapshot, cache


@pytest.mark.parametrize("representation", ["c", "tau_r"])
def test_cache_matches_hdf5_and_reuses_without_hdf5(cached_snapshot, monkeypatch, representation):
    snapshot, cache = cached_snapshot
    raw = WarehouseJointDataset(snapshot, representation=representation)
    cached = CachedWarehouseJointDataset(snapshot, cache, representation=representation)
    expected = [raw[i] for i in range(len(raw))]; raw.close()
    for i, row in enumerate(expected):
        actual = cached[i]
        assert actual.keys() == row.keys()
        for name in row:
            torch.testing.assert_close(actual[name], row[name], atol=0, rtol=0)
    stamps = {p.name: p.stat().st_mtime_ns for p in cache.iterdir()}
    def forbidden(*args, **kwargs):
        raise AssertionError("cached iteration must not open HDF5")
    monkeypatch.setattr(h5py, "File", forbidden)
    assert build_tensor_cache(snapshot, cache) == cache
    assert {p.name: p.stat().st_mtime_ns for p in cache.iterdir()} == stamps
    restored = pickle.loads(pickle.dumps(cached))
    assert not restored._arrays
    one = next(iter(DataLoader(restored, batch_size=2, collate_fn=collate_joint)))
    assert one["world_features"].shape == (2, 32, 13)
    assert one["world_mask"].sum().item() == 32
    # KIN maps no world files, and collation still works.
    kin = CachedWarehouseJointDataset(snapshot, cache, representation=representation, include_world=False)
    next(iter(DataLoader(kin, batch_size=2, collate_fn=collate_joint)))
    assert all(not name.startswith("world_") for name in kin._arrays)


def _cache_worker(dataset, connection):
    try:
        batch = collate_joint([dataset[0], dataset[1]])
        connection.send({key: value.numpy() for key, value in batch.items()})
    finally:
        connection.close(); dataset.close()


def test_cache_worker_reopens_memmaps(cached_snapshot):
    snapshot, cache = cached_snapshot
    cached = CachedWarehouseJointDataset(snapshot, cache)
    expected = collate_joint([cached[0], cached[1]])
    # Test spawn+reopen via a pipe: the sandbox blocks torch DataLoader's
    # shared-tensor socket transport even for a plain TensorDataset.
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    worker = context.Process(target=_cache_worker, args=(cached, writer))
    worker.start(); writer.close()
    try:
        assert reader.poll(15), "spawned cache reader did not respond"
        actual = reader.recv()
        for key in expected:
            torch.testing.assert_close(torch.from_numpy(actual[key]), expected[key], atol=0, rtol=0)
        worker.join(5)
        assert worker.exitcode == 0
    finally:
        reader.close()
        if worker.is_alive():
            worker.terminate(); worker.join(5)


def test_cache_rejects_changes_and_incomplete_publication(cached_snapshot, tmp_path, monkeypatch):
    snapshot, cache = cached_snapshot
    wrong = json.loads(snapshot.read_text())
    wrong["normalization"]["c"]["mean"][0] += 1
    wrong.pop("identity"); wrong["identity"] = identity(wrong)
    other = tmp_path / "other-snapshot.json"; atomic_json(other, wrong)
    with pytest.raises(ValueError, match="identity/reader mismatch"):
        CachedWarehouseJointDataset(other, cache)
    def crash(*args):
        raise RuntimeError("injected build failure")
    monkeypatch.setattr(WarehouseJointDataset, "_read_sample", crash)
    broken = tmp_path / "broken"
    with pytest.raises(RuntimeError, match="injected build failure"):
        build_tensor_cache(snapshot, broken)
    assert not broken.exists()
    assert not list(tmp_path.glob("broken.building-*"))
    # A changed cache file is rejected, never silently rebuilt in place.
    with (cache / "p.npy").open("ab") as handle:
        handle.write(b"bad")
    with pytest.raises(ValueError, match="frozen tensor cache changed"):
        CachedWarehouseJointDataset(snapshot, cache)


def test_cache_still_checks_frozen_source(cached_snapshot):
    snapshot, cache = cached_snapshot
    value = json.loads(snapshot.read_text())
    with h5py.File(value["entries"][0]["file"], "a") as handle:
        handle.attrs["modified"] = True
    with pytest.raises(ValueError, match="frozen shard changed"):
        CachedWarehouseJointDataset(snapshot, cache)
