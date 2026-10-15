import hashlib
import json

import h5py
import numpy as np
import pytest

from scripts.spacetime_data import generate_warehouse_multiworld as multi
from scripts.spacetime_data import generate_warehouse_spacetime as base
from scripts.spacetime_data.multiworld_sampling import make_multiworld_forecast
from scripts.spacetime_data.warehouse_world import make_forecast
from tests.test_warehouse_spacetime_generation import FakePlanner, FakeWarehouse


def fixtures(tmp_path, families=5):
    source = tmp_path / "source.hdf5"
    paths = []
    for family in range(families):
        path = np.zeros((32, 7))
        path[:, 0] = np.linspace(-.4 + .01 * family, .4 + .01 * family, 32)
        paths.extend([path, path[::-1]])
    with h5py.File(source, "w") as file:
        file["sol_path"] = np.stack(paths)
        file["task_id"] = np.arange(2 * families) + 100
    roots = []
    for count, ids in ((3, range(families - 1)), (5, range(1, families))):
        root = tmp_path / f"legacy-{count}"
        root.mkdir()
        config = {"source_size": source.stat().st_size, "source_mtime_ns": source.stat().st_mtime_ns,
                  "schema": base.SCHEMA, "source_group_size": 2, "obstacles": count}
        (root / "config.json").write_text(json.dumps(config))
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        entries = [{"family_id": family} for family in ids]
        (root / "index.json").write_text(json.dumps({"config_hash": config_hash, "splits": {"train": entries}}))
        roots.append(root)
    command = ["--output-root", str(tmp_path / "out"), "--source", str(source), "--device", "cpu",
               "--num-families", str(families), "--query-batch-size", "2", "--checkpoint-batches", "1",
               "--eligible-roots", *map(str, roots)]
    return source, command


@pytest.mark.parametrize("count", [0, 1, 2])
def test_low_counts_are_identical_to_original_sampler(count):
    centers = np.broadcast_to([.4, 0., .5], (64, 12, 3)).copy()
    for seed in range(10):
        original = make_forecast(centers, 8., 14., np.random.default_rng(seed), object_count=count)
        updated = make_multiworld_forecast(centers, 8., 14., np.random.default_rng(seed), object_count=count)
        for name in ("times", "positions", "shape", "size", "event_times", "anchors"):
            np.testing.assert_array_equal(getattr(original, name), getattr(updated, name))
        assert original.modes == updated.modes


@pytest.mark.parametrize("count", [3, 4, 5])
def test_high_counts_use_widened_fraction_for_both_anchor_and_event(count):
    centers = np.zeros((64, 12, 3))
    centers[:] = [.4, 0., .5]
    centers[:, :, 0] = np.linspace(.2, .8, 64)[:, None]
    events = []
    for seed in range(12):
        forecast = make_multiworld_forecast(centers, 8., 14., np.random.default_rng(seed), object_count=count)
        assert np.all((forecast.event_times >= 2.) & (forecast.event_times <= 6.4))
        for obj, event in enumerate(forecast.event_times):
            index = round(event / 8 * 63)
            np.testing.assert_allclose(forecast.anchors[obj], centers[index, 0])
            knot = np.flatnonzero(forecast.times == event)[0]
            np.testing.assert_allclose(forecast.positions[knot, obj], forecast.anchors[obj], atol=1e-12)
        events.extend(forecast.event_times)
    assert max(events) > 5.2


def test_family_union_selection_directions_and_stable_world_keys(tmp_path):
    source, command = fixtures(tmp_path)
    args = multi.parse_args(command)
    plan, selection = multi.select_families(args)
    assert selection["eligible_families"] == 5
    assert sorted(row["family_id"] for row in plan) == list(range(5))
    assert all(row["source_row"] == 2 * row["family_id"] + row["direction"] for row in plan)
    assert all(row["source_task_id"] == 100 + row["source_row"] for row in plan)
    smaller = multi.parse_args(command + ["--num-families", "3"])
    assert multi.select_families(smaller)[0] == plan[:3]
    for direction, code in (("forward", 0), ("reverse", 1)):
        fixed, _ = multi.select_families(multi.parse_args(command + ["--direction", direction]))
        assert all(row["direction"] == code for row in fixed)
    namespace = selection["source_namespace"]
    assert len({multi.keyed_seed(args.seed, namespace, 20, 7, 0, n, 0) for n in multi.COUNTS}) == 6
    with pytest.raises(ValueError, match="eligible union"):
        multi.select_families(multi.parse_args(command + ["--num-families", "6"]))
    # A different source file must not silently reuse a historical family index.
    config_path = args.eligible_roots[0] / "config.json"
    config = json.loads(config_path.read_text())
    config["source_size"] += 1
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="contract mismatch"):
        multi.select_families(args)


def patch_teacher(monkeypatch, *, fail=False, interrupt_at=None):
    batches = []

    class RecordingWarehouse(FakeWarehouse):
        def set_worlds(self, worlds):
            self.worlds = worlds

    class Planner(FakePlanner):
        def __init__(self, collision, *args, **kwargs):
            self.collision = collision

        def plan_batch(self, starts, goals, *args, **kwargs):
            if interrupt_at is not None and len(batches) == interrupt_at:
                raise RuntimeError("simulated interruption")
            counts = [len(world.shape) for world in self.collision.worlds]
            assert len(set(counts)) == 1
            batches.append(counts)
            return super().plan_batch(starts, goals, *args, **kwargs)

    def validate(args, collision, query, job, result):
        if fail and len(job["forecast"].shape) == 4:
            return None, "fixture_fit_failure"
        job["fit_wall_s"], job["fitted_solution_index"] = 0., 0
        # The scheduler tests exercise the real atomic writer and resume logic;
        # fitted-trajectory correctness is covered by the real-robot smoke audit.
        fitted = {"quality": {"duration": 7.}, "tau_r_valid": True, "tau_r_reject_reason": None,
                  "tau_r_quality": {"duration": 7.}, "degree": 5, "phase_eta": .05,
                  "P": np.zeros((29, 7)), "knots": np.arange(35), "c": np.zeros(8),
                  "tau_r": np.zeros(6), "T_min": 2., "reference_phase": np.linspace(0, 1, 8),
                  "reference_time": np.linspace(0, 7, 8), "execution_q": job["reference"],
                  "execution_dq": np.zeros_like(job["reference"]), "execution_ddq": np.zeros_like(job["reference"]),
                  "execution_time": np.linspace(0, 7, len(job["reference"]))}
        return (job, result, fitted), None

    monkeypatch.setattr(multi, "WarehouseCollision", RecordingWarehouse)
    monkeypatch.setattr(multi, "TorchSpaceTimeForest", Planner)
    monkeypatch.setattr(base, "validate_result", validate)
    return batches


def test_all_counts_batch_by_count_persist_family_direction_and_resume(tmp_path, monkeypatch):
    _, command = fixtures(tmp_path)
    batches = patch_teacher(monkeypatch)
    assert multi.main(command) == 0
    assert batches == [[count] * size for size in (2, 2, 1) for count in multi.COUNTS]
    output = tmp_path / "out"
    report = json.loads((output / "report.json").read_text())
    assert report["accepted"] == report["processed_worlds"] == 30
    assert report["run_finished"] and report["complete_families"] == 5
    elapsed = report["session_wall_s"]
    plan = {row["family_id"]: row for row in json.loads((output / "families.json").read_text())}
    ledger = []
    for path in sorted((output / "shards").glob("*.hdf5")):
        with h5py.File(path) as shard:
            ledger.extend(json.loads(value) for value in shard["task_outcomes"].asstr()[:])
            for sample in shard["samples"].values():
                row = plan[int(sample.attrs["family_id"])]
                assert sample.attrs["source_row"] == row["source_row"]
                assert sample.attrs["direction"] == row["direction"]
                assert sample.attrs["obstacle_count"] == len(sample["world/shape"])
    assert len({(row["family_id"], row["world_id"]) for row in ledger}) == 30
    before = {path: base.sha256_file(path) for path in (output / "shards").glob("*.hdf5")}
    (output / "index.json").unlink()
    assert multi.main(command + ["--resume"]) == 0
    assert (output / "index.json").exists()
    assert {path: base.sha256_file(path) for path in before} == before
    assert json.loads((output / "report.json").read_text())["session_wall_s"] == elapsed
    with pytest.raises(ValueError, match="do not match"):
        multi.main(command + ["--resume", "--direction", "forward"])


def test_failed_slots_stay_in_ledger_and_do_not_repeat_on_resume(tmp_path, monkeypatch):
    _, command = fixtures(tmp_path, families=3)
    patch_teacher(monkeypatch, fail=True)
    assert multi.main(command) == 2
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["run_finished"] and not report["complete"]
    assert report["accepted"] == 15 and report["failed_worlds"] == 3
    assert report["family_success_histogram"] == {"5": 3}
    assert report["per_obstacle_count"]["4"]["status_counts"] == {"fit_rejected": 3}
    batches = patch_teacher(monkeypatch)
    assert multi.main(command + ["--resume"]) == 2
    assert not batches


def test_interrupted_generation_rebuilds_cursor_from_committed_shards(tmp_path, monkeypatch):
    _, command = fixtures(tmp_path)
    patch_teacher(monkeypatch, interrupt_at=3)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        multi.main(command)
    output = tmp_path / "out"
    before = {path: base.sha256_file(path) for path in (output / "shards").glob("*.hdf5")}
    assert len(before) == 3
    patch_teacher(monkeypatch)
    assert multi.main(command + ["--resume"]) == 0
    assert all(base.sha256_file(path) == digest for path, digest in before.items())
    report = json.loads((output / "report.json").read_text())
    assert report["accepted"] == report["processed_worlds"] == 30


def test_invalid_worlds_fill_batches_and_leave_explicit_terminal_outcomes(tmp_path, monkeypatch):
    _, command = fixtures(tmp_path, families=5)
    batches = patch_teacher(monkeypatch)
    original = multi.make_multiworld_forecast
    failures = []

    def sampler(*args, **kwargs):
        if kwargs["object_count"] == 3:
            failures.append(1)
            raise ValueError("fixture has no usable anchor")
        return original(*args, **kwargs)

    monkeypatch.setattr(multi, "make_multiworld_forecast", sampler)
    assert multi.main(command + ["--max-world-attempts", "2"]) == 2
    assert len(failures) == 10
    assert all(3 not in batch for batch in batches)
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["processed_worlds"] == 30 and report["accepted"] == 25
    assert report["per_obstacle_count"]["3"]["status_counts"] == {"invalid_scene": 5}


def test_rejected_reference_does_not_leave_a_hole_in_valid_batches(tmp_path, monkeypatch):
    _, command = fixtures(tmp_path)
    batches = patch_teacher(monkeypatch)
    original = multi.WarehouseCollision

    class InvalidReference(original):
        def static_collision(self, q):
            result = super().static_collision(q)
            if np.isclose(np.asarray(q)[:, 0].mean(), .02):
                result[:] = True
            return result

    monkeypatch.setattr(multi, "WarehouseCollision", InvalidReference)
    assert multi.main(command) == 2
    assert len(batches) == 12 and all(len(batch) == 2 for batch in batches)
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["accepted"] == 24 and report["processed_worlds"] == 30
    assert report["failure_counts"] == {"source_static_self_or_limits": 6}


def test_dry_run_freezes_plan_without_constructing_planner(tmp_path, monkeypatch):
    _, command = fixtures(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not initialize collision environment")

    monkeypatch.setattr(multi, "WarehouseCollision", forbidden)
    assert multi.main(command + ["--dry-run"]) == 0
    output = tmp_path / "out"
    assert json.loads((output / "report.json").read_text())["pending_worlds"] == 30
    patch_teacher(monkeypatch)
    assert multi.main(command + ["--resume"]) == 0
