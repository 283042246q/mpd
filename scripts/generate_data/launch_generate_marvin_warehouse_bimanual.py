#!/usr/bin/env python3
"""Launch resumable CPU shards (3 workers / 500 tasks, matching Panda)."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from collections import Counter
from collections import deque
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import h5py
import yaml

from scripts.generate_data.generate_marvin_warehouse_bimanual import (
    DEFAULT_CONFIG,
    EE_GOAL_SCHEMA,
    MarvinWarehouseGenerator,
    PRE_RRT_FILTERS,
    _write_dataset,
    file_sha256,
    validate_config,
)


def shard_complete(path, config, start, count):
    manifest_path = path / "manifest.yaml"
    if not manifest_path.is_file() or not (path / "generation_config.yaml").is_file():
        return False
    try:
        manifest = yaml.safe_load(manifest_path.read_text())
        saved_config = yaml.safe_load((path / "generation_config.yaml").read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return False
    if not isinstance(manifest, dict) or not isinstance(saved_config, dict):
        return False
    if saved_config != config:
        raise ValueError(f"existing shard config differs: {path}")
    try:
        with h5py.File(path / "dataset_merged.hdf5", "r") as data:
            valid = (
                len(data["sol_path"]) == count
                and data["task_id"][0] == start
                and data["task_id"][-1] == start + count - 1
                and data["ee_goal_pose"].shape == (count, 2, 3, 4)
                and data["active_ee_mask"].shape == (count, 2)
                and data.attrs.get("ee_goal_schema") == EE_GOAL_SCHEMA
            )
    except (OSError, KeyError, IndexError):
        return False
    try:
        return valid and file_sha256(path / "dataset_merged.hdf5") == manifest["dataset_sha256"]
    except (OSError, KeyError):
        return False


def fsync_tree(path):
    """Make a completed staging directory durable before publishing it."""
    path = Path(path)
    for child in path.iterdir():
        if child.is_file():
            descriptor = os.open(child, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def quarantine_incomplete_shard(path):
    """Move a corrupt published shard aside without deleting evidence."""
    path = Path(path)
    suffix = f"{time.time_ns()}-{os.getpid()}"
    quarantine = path.parent / f".{path.name}.corrupt-{suffix}"
    os.replace(path, quarantine)
    print(f"[{path.name}] quarantined incomplete shard -> {quarantine.name}", flush=True)
    return quarantine


def run_shard(config, root, start, count):
    path = Path(root) / "shards" / f"{start:09d}"
    if shard_complete(path, config, start, count):
        return str(path)
    if path.exists():
        quarantine_incomplete_shard(path)
    staging = path.parent / f".{path.name}.incomplete-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    label = f"shard {start:09d}"
    generator = MarvinWarehouseGenerator(
        config,
        int(config.get("seed", 0)) + start,
        progress_label=label,
    )
    try:
        paths, metadata = generator.generate(count, start)
        _write_dataset(staging, config, paths, metadata, int(config.get("seed", 0)) + start, generator.stats)
        fsync_tree(staging)
        os.replace(staging, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        print(f"[{label}] complete: {count}/{count} -> {path}", flush=True)
    finally:
        generator.close()
    return str(path)


def run_shards_resilient(
    config,
    root,
    shards,
    workers,
    max_restarts,
    *,
    _executor_factory=ProcessPoolExecutor,
    _wait=wait,
    _shard_complete=shard_complete,
    _run_shard=run_shard,
):
    """Run shards in rolling slots, using a fresh process for every shard.

    A normal ``ProcessPoolExecutor`` reuses worker processes.  That is unsafe
    here because PyBullet/OMPL native heap corruption has previously surfaced
    after a worker handled several trajectories.  Each in-flight shard
    therefore owns a single-worker executor.  As soon as its future finishes,
    that executor is destroyed and a fresh one immediately takes the next
    pending shard; other slots do not have to reach a wave barrier first.

    The underscored callables are dependency-injection hooks for scheduler
    tests.  Production callers use the defaults above.
    """
    pending = deque()
    completed = {}
    for start, count in shards:
        path = Path(root) / "shards" / f"{start:09d}"
        if _shard_complete(path, config, start, count):
            completed[start] = str(path)
        else:
            pending.append((start, count))
    if completed:
        print(f"resume: reusing {len(completed)}/{len(shards)} complete shards", flush=True)
    failures = Counter()
    context = mp.get_context("spawn")
    in_flight = {}

    def launch_next():
        start, count = pending.popleft()
        # One executor owns exactly one submitted task, guaranteeing a fresh
        # spawned process without relying on max_tasks_per_child (Python 3.8).
        pool = _executor_factory(max_workers=1, mp_context=context)
        try:
            future = pool.submit(_run_shard, config, str(root), start, count)
        except BaseException:
            pool.shutdown(wait=True)
            pending.appendleft((start, count))
            raise
        in_flight[future] = (pool, start, count)

    try:
        while pending and len(in_flight) < workers:
            launch_next()

        while in_flight:
            done, _ = _wait(tuple(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                pool, start, count = in_flight.pop(future)
                try:
                    future.result()
                except BrokenProcessPool as error:
                    # This executor belongs only to this shard, so native
                    # failure does not invalidate the other rolling slots.
                    print(f"[shard {start:09d}] native worker crashed: {error!r}", flush=True)
                except Exception as error:
                    print(f"[shard {start:09d}] worker failed: {error!r}", flush=True)
                finally:
                    pool.shutdown(wait=True)

                path = Path(root) / "shards" / f"{start:09d}"
                if _shard_complete(path, config, start, count):
                    completed[start] = str(path)
                else:
                    completed.pop(start, None)
                    failures[start] += 1
                    if failures[start] > max_restarts:
                        raise RuntimeError(
                            f"shard {start:09d} failed after {max_restarts} restarts; "
                            f"staging directories were retained under {Path(root) / 'shards'}"
                        )
                    print(
                        f"[shard {start:09d}] incomplete after worker exit; "
                        f"retry {failures[start]}/{max_restarts}",
                        flush=True,
                    )
                    pending.append((start, count))

                # Refill this slot immediately.  Other in-flight shards keep
                # running and do not form a synchronization barrier.
                if pending and len(in_flight) < workers:
                    launch_next()
    finally:
        # Normally empty.  On Ctrl-C or a fatal retry error, do not leak child
        # processes or their native PyBullet/OMPL resources.
        for pool, _, _ in in_flight.values():
            pool.shutdown(wait=True)
    return [completed[start] for start, _ in shards]


def build_shards(num_trajectories, max_shard_size, workers):
    """Build quota-safe shards while keeping available workers occupied.

    Every shard is a multiple of ten so each independently generated shard
    retains the 5:5 direction and 3:1:1 mode schedule. ``max_shard_size`` is
    an upper bound, not a request to leave workers idle.
    """
    natural_count = (num_trajectories + max_shard_size - 1) // max_shard_size
    shard_count = max(natural_count, min(workers, num_trajectories // 10))
    blocks, remainder = divmod(num_trajectories // 10, shard_count)
    counts = [(blocks + (i < remainder)) * 10 for i in range(shard_count)]
    if any(count <= 0 or count > max_shard_size for count in counts):
        raise ValueError("could not construct positive quota-safe shards within the requested maximum")
    starts = []
    start = 0
    for count in counts:
        starts.append((start, count))
        start += count
    return starts


def merge_shards(root, paths, config):
    root = Path(root)
    paths = sorted(map(Path, paths))
    target = root / "dataset_merged.hdf5"
    if target.exists():
        raise FileExistsError(target)
    temporary = root / "dataset_merged.incomplete.hdf5"
    count = sum(yaml.safe_load((p / "manifest.yaml").read_text())["num_trajectories"] for p in paths)
    task_counts, direction_counts, stats = Counter(), Counter(), Counter()
    region_counts = {}
    with h5py.File(temporary, "x") as dest:
        offset = 0
        for path in paths:
            manifest = yaml.safe_load((path / "manifest.yaml").read_text())
            task_counts.update(manifest["task_counts"])
            direction_counts.update(manifest["direction_counts"])
            stats.update(manifest["stats"])
            for key, counts in manifest.get("region_counts", {}).items():
                region_counts.setdefault(key, Counter()).update(counts)
            with h5py.File(path / "dataset_merged.hdf5", "r") as source:
                n = len(source["sol_path"])
                if offset == 0:
                    dest.attrs.update(source.attrs)
                    for key, value in source.items():
                        dest.create_dataset(
                            key, shape=(count,) + value.shape[1:], dtype=value.dtype, compression="gzip"
                        )
                for key, value in source.items():
                    dest[key][offset : offset + n] = value[:]
                offset += n
    temporary.rename(target)
    shutil.copyfile(paths[0] / "args.yaml", root / "args.yaml")
    (root / "generation_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest.update(
        num_trajectories=count,
        task_counts=dict(task_counts),
        direction_counts=dict(direction_counts),
        region_counts={key: dict(counts) for key, counts in region_counts.items()},
        stats=dict(stats),
        dataset_sha256=file_sha256(target),
        shards=[str(p.relative_to(root)) for p in paths],
    )
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-trajectories", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--tasks-per-shard", type=int)
    parser.add_argument(
        "--worker-lifetime-trajectories",
        type=int,
        help="Maximum successful trajectories handled by one fresh native worker process.",
    )
    parser.add_argument("--max-worker-restarts-per-shard", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sampler", choices=("region_ik", "joint_fk"))
    parser.add_argument("--pre-rrt-filter", choices=PRE_RRT_FILTERS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.sampler:
        config["sampler"] = args.sampler
    if args.pre_rrt_filter:
        config["pre_rrt_filter"] = args.pre_rrt_filter
    validate_config(config)
    n = args.num_trajectories if args.num_trajectories is not None else config["num_trajectories"]
    workers = args.workers if args.workers is not None else config.get("workers", 3)
    shard_size = args.tasks_per_shard if args.tasks_per_shard is not None else config.get("tasks_per_shard", 500)
    worker_lifetime = (
        args.worker_lifetime_trajectories
        if args.worker_lifetime_trajectories is not None
        else config.get("worker_lifetime_trajectories", 10)
    )
    max_restarts = (
        args.max_worker_restarts_per_shard
        if args.max_worker_restarts_per_shard is not None
        else config.get("max_worker_restarts_per_shard", 3)
    )
    if (
        n <= 0
        or n % 10
        or shard_size <= 0
        or shard_size % 10
        or worker_lifetime <= 0
        or worker_lifetime % 10
        or workers < 1
        or max_restarts < 0
    ):
        raise ValueError("counts/shard/worker lifetime must be positive multiples of 10; workers >= 1; restarts >= 0")
    root = args.output_dir or Path(config["output_dir"])
    effective_shard_size = min(shard_size, worker_lifetime)
    shards = build_shards(n, effective_shard_size, workers)
    active_workers = min(workers, len(shards))
    print(
        f"{config['sampler']}, pre_rrt_filter={config.get('pre_rrt_filter', 'none')}: "
        f"{n} trajectories, {active_workers}/{workers} active CPU workers, "
        f"{len(shards)} fresh-worker shards (size <= {effective_shard_size}) -> {root}",
        flush=True,
    )
    if args.dry_run:
        return 0
    if (root / "dataset_merged.hdf5").exists():
        raise FileExistsError(root / "dataset_merged.hdf5")
    root.mkdir(parents=True, exist_ok=True)
    paths = run_shards_resilient(config, root, shards, workers, max_restarts)
    merge_shards(root, paths, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
