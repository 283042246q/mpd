#!/usr/bin/env python3
"""Launch resumable cooperative Marvin context shards and merge them."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import h5py
import yaml

from scripts.generate_data.generate_marvin_warehouse_bimanual import file_sha256
from scripts.generate_data.generate_marvin_warehouse_cooperative import (
    CONTEXT_DATASETS,
    DATASET_SCHEMA,
    DEFAULT_CONFIG,
    MarvinWarehouseCooperativeGenerator,
    SOLUTION_DATASETS,
    TASK_MODE,
    validate_config,
    write_dataset,
)
from scripts.generate_data.launch_generate_marvin_warehouse_bimanual import (
    fsync_tree,
    quarantine_incomplete_shard,
    run_shards_resilient,
)


def build_context_shards(num_contexts, max_contexts_per_shard, workers):
    if num_contexts <= 0 or max_contexts_per_shard <= 0 or workers <= 0:
        raise ValueError("context counts and workers must be positive")
    natural = (num_contexts + max_contexts_per_shard - 1) // max_contexts_per_shard
    shard_count = max(natural, min(workers, num_contexts))
    base, remainder = divmod(num_contexts, shard_count)
    counts = [base + (index < remainder) for index in range(shard_count)]
    shards, start = [], 0
    for count in counts:
        shards.append((start, count))
        start += count
    return shards


def shard_complete(path, config, start, count):
    path = Path(path)
    try:
        manifest = yaml.safe_load((path / "manifest.yaml").read_text())
        saved_config = yaml.safe_load((path / "generation_config.yaml").read_text())
        if saved_config != config:
            raise ValueError(f"existing shard config differs: {path}")
        with h5py.File(path / "dataset_merged.hdf5", "r") as data:
            valid = (
                manifest["schema"] == DATASET_SCHEMA
                and manifest["num_contexts"] == count
                and len(data["context_task_id"]) == count
                and int(data["context_task_id"][0]) == start
                and int(data["context_task_id"][-1]) == start + count - 1
                and all(len(data[key]) == len(data["sol_path"]) for key in SOLUTION_DATASETS)
                and all(len(data[key]) == count for key in CONTEXT_DATASETS)
            )
        return valid and file_sha256(path / "dataset_merged.hdf5") == manifest["dataset_sha256"]
    except (OSError, KeyError, IndexError, TypeError, UnicodeDecodeError, yaml.YAMLError):
        return False


def run_shard(config, root, start, count):
    path = Path(root) / "shards" / f"{start:09d}"
    if shard_complete(path, config, start, count):
        return str(path)
    if path.exists():
        quarantine_incomplete_shard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.incomplete-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    generator = MarvinWarehouseCooperativeGenerator(config, int(config.get("seed", 0)) + start, f"shard {start:09d}")
    try:
        paths, metadata, contexts = generator.generate(count, start)
        write_dataset(staging, config, paths, metadata, contexts, int(config.get("seed", 0)) + start, generator.stats)
        fsync_tree(staging)
        os.replace(staging, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        generator.close()
    return str(path)


def _copy_dataset(dest, source, key, offset):
    value = source[key]
    if key not in dest:
        shape = list(value.shape)
        shape[0] = 0
        maxshape = list(value.shape)
        maxshape[0] = None
        dest.create_dataset(
            key,
            shape=tuple(shape),
            maxshape=tuple(maxshape),
            dtype=value.dtype,
            compression="gzip" if value.ndim > 1 else None,
        )
    target = dest[key]
    target.resize(offset + len(value), axis=0)
    target[offset : offset + len(value)] = value[:]


def merge_shards(root, paths, config):
    root = Path(root)
    paths = sorted(map(Path, paths), key=lambda path: int(path.name))
    target = root / "dataset_merged.hdf5"
    if target.exists():
        raise FileExistsError(target)
    temporary = root / "dataset_merged.incomplete.hdf5"
    solution_offset = context_offset = 0
    stats, proposal_counts, failures, histogram = Counter(), Counter(), Counter(), Counter()
    with h5py.File(temporary, "x") as dest:
        for path in paths:
            manifest = yaml.safe_load((path / "manifest.yaml").read_text())
            stats.update(manifest.get("stats", {}))
            proposal_counts.update(manifest.get("proposal_counts", {}))
            failures.update(manifest.get("failed_context_reasons", {}))
            histogram.update({int(key): value for key, value in manifest.get("context_solution_histogram", {}).items()})
            with h5py.File(path / "dataset_merged.hdf5", "r") as source:
                if not dest.attrs:
                    dest.attrs.update(source.attrs)
                for key in SOLUTION_DATASETS:
                    _copy_dataset(dest, source, key, solution_offset)
                for key in CONTEXT_DATASETS:
                    _copy_dataset(dest, source, key, context_offset)
                solution_offset += len(source["sol_path"])
                context_offset += len(source["context_task_id"])
    os.replace(temporary, target)
    first_manifest = yaml.safe_load((paths[0] / "manifest.yaml").read_text())
    manifest = {
        **first_manifest,
        "num_contexts": context_offset,
        "num_trajectories": solution_offset,
        "context_solution_histogram": dict(histogram),
        "failed_context_reasons": dict(failures),
        "proposal_counts": dict(proposal_counts),
        "stats": dict(stats),
        "dataset_sha256": file_sha256(target),
        "shards": [str(path.relative_to(root)) for path in paths],
    }
    shutil.copyfile(paths[0] / "args.yaml", root / "args.yaml")
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (root / "generation_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (root / "generation_summary.json").write_text(json.dumps(manifest, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-contexts", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--contexts-per-shard", type=int)
    parser.add_argument("--worker-lifetime-trajectories", type=int)
    parser.add_argument("--max-worker-restarts-per-shard", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--disable-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.disable_fallback:
        config["fallback"]["enabled"] = False
    validate_config(config)
    launcher = config["launcher"]
    count = args.num_contexts if args.num_contexts is not None else int(config["dataset"]["num_contexts"])
    workers = args.workers if args.workers is not None else int(launcher["workers"])
    requested_shard = (
        args.contexts_per_shard if args.contexts_per_shard is not None else int(launcher["contexts_per_shard"])
    )
    lifetime = (
        args.worker_lifetime_trajectories
        if args.worker_lifetime_trajectories is not None
        else int(launcher["worker_lifetime_trajectories"])
    )
    restarts = (
        args.max_worker_restarts_per_shard
        if args.max_worker_restarts_per_shard is not None
        else int(launcher["max_worker_restarts_per_shard"])
    )
    if count <= 0 or workers <= 0 or requested_shard <= 0 or lifetime <= 0 or restarts < 0:
        raise ValueError("counts/lifetime/workers must be positive and restarts nonnegative")
    target_per_context = int(config["dataset"]["solutions_per_task_target"])
    lifetime_contexts = max(1, lifetime // target_per_context)
    effective_shard = min(requested_shard, lifetime_contexts)
    shards = build_context_shards(count, effective_shard, workers)
    root = args.output_dir or Path(config["output_dir"])
    print(
        f"{TASK_MODE}: {count} contexts x {target_per_context} target solutions, {min(workers, len(shards))}/{workers} workers, {len(shards)} resumable shards (<= {effective_shard} contexts) -> {root}",
        flush=True,
    )
    if args.dry_run:
        return 0
    if (root / "dataset_merged.hdf5").exists():
        raise FileExistsError(root / "dataset_merged.hdf5")
    root.mkdir(parents=True, exist_ok=True)
    paths = run_shards_resilient(
        config,
        root,
        shards,
        workers,
        restarts,
        _shard_complete=shard_complete,
        _run_shard=run_shard,
    )
    merge_shards(root, paths, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
