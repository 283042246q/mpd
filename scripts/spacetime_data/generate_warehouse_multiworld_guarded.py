"""Run the existing multiworld generator with storage preflight checks.

This operational wrapper leaves the original generator and its software hashes
unchanged, so previously committed runs keep their strict resume contract.
Checks reduce ENOSPC risk; concurrent writers and filesystem quotas can still
consume space between a check and a write.
"""
from __future__ import annotations

import argparse
import errno
import math
import os
from pathlib import Path
import sys

import numpy as np

from scripts.spacetime_data import generate_warehouse_multiworld as generation


GIB = 1024 ** 3
WRITE_FLOOR = 64 * 1024 ** 2


class StorageBudgetExceeded(RuntimeError):
    pass


class StorageGuard:
    def __init__(self, min_free_gib=10., min_free_inodes=1024, safety_factor=2.):
        self.reserve = math.ceil(min_free_gib * GIB)
        self.inode_reserve = min_free_inodes
        self.safety_factor = safety_factor
        self.previous_shard_bytes = 0
        self.requested_worlds = 0

    def check(self, output, estimated_bytes=WRITE_FLOOR):
        usage = os.statvfs(output)
        available = usage.f_bavail * usage.f_frsize
        required = self.reserve + math.ceil(self.safety_factor * max(WRITE_FLOOR, estimated_bytes))
        if available < required:
            raise StorageBudgetExceeded(
                f"output filesystem has {available / GIB:.2f} GiB available; "
                f"need {required / GIB:.2f} GiB including reserve and write headroom")
        if usage.f_files > 0 and usage.f_favail < self.inode_reserve + 16:
            raise StorageBudgetExceeded(
                f"output filesystem has {usage.f_favail} free inodes; "
                f"need at least {self.inode_reserve + 16}")

    def prepare(self, args):
        # Called inside generation.main's exclusive writer lock. These files
        # have never been published to output/shards and are safe to discard.
        stale = args.output_root / ".batch-staging" / "shards"
        reclaimed = 0
        for path in stale.glob("shard-*.hdf5*"):
            reclaimed += path.stat().st_size
            path.unlink()
        paths = sorted((args.output_root / "shards").glob("*.hdf5"))
        self.previous_shard_bytes = max((path.stat().st_size for path in paths[-32:]), default=0)
        self.requested_worlds = args.num_trajectories
        self.check(args.output_root, self.previous_shard_bytes)
        print(f"Storage guard: reserve={self.reserve / GIB:g} GiB, "
              f"min_free_inodes={self.inode_reserve}, safety_factor={self.safety_factor:g}, "
              f"committed_shards={len(paths)}, discarded_staging_bytes={reclaimed}", file=sys.stderr, flush=True)

    def estimate_shard(self, records, outcomes):
        total = WRITE_FLOOR + len(outcomes) * 4096
        for job, result, fitted in records:
            forecast = job["forecast"]
            arrays = [forecast.times, forecast.positions, forecast.shape, forecast.size,
                      forecast.event_times, forecast.anchors, job["q_start"], job["q_goal"], *result.paths]
            arrays += [value for value in fitted.values() if isinstance(value, np.ndarray)]
            total += sum(np.asarray(value).nbytes for value in arrays)
            # HDF5 groups, datasets, chunk headers and attributes, beyond arrays.
            total += 256 * 1024
        return max(total, self.previous_shard_bytes)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--min-free-gib", type=float, default=10.)
    parser.add_argument("--min-free-inodes", type=int, default=1024)
    parser.add_argument("--disk-safety-factor", type=float, default=2.)
    options, remaining = parser.parse_known_args(argv)
    if (not math.isfinite(options.min_free_gib) or options.min_free_gib < 0
        or options.min_free_inodes < 0 or not math.isfinite(options.disk_safety_factor)
        or options.disk_safety_factor < 1):
        parser.error("require finite min-free-gib >= 0, min-free-inodes >= 0, disk-safety-factor >= 1")
    if "--help" in remaining or "-h" in remaining:
        print(parser.format_help())
        return generation.main(remaining)
    guard = StorageGuard(options.min_free_gib, options.min_free_inodes, options.disk_safety_factor)
    original_generate, original_commit, original_publish = generation._generate, generation.commit_batch, generation.publish

    def generate(args):
        guard.prepare(args)
        return original_generate(args)

    def commit(output, shard_index, begin, outcomes, records, failures, config_hash, statistics):
        guard.check(output, guard.estimate_shard(records, outcomes))
        original_commit(output, shard_index, begin, outcomes, records, failures, config_hash, statistics)
        guard.previous_shard_bytes = (output / "shards" / f"shard-{shard_index:06d}.hdf5").stat().st_size
        if begin + len(outcomes) < guard.requested_worlds:
            # Stop immediately after a successful commit rather than spend a
            # GPU batch on work that cannot currently be written. Its ledger
            # remains authoritative even if index/report have not caught up.
            guard.check(output, guard.previous_shard_bytes)

    def publish(output, config_hash, plan, state, elapsed):
        # The index is rewritten atomically, so its temporary copy needs space
        # in addition to the existing index. Include JSON formatting overhead.
        guard.check(output, max(WRITE_FLOOR, len(state[3]) * 2048))
        return original_publish(output, config_hash, plan, state, elapsed)

    generation._generate, generation.commit_batch, generation.publish = generate, commit, publish
    try:
        return generation.main(remaining)
    except StorageBudgetExceeded as error:
        print(f"Storage stop: {error}. Committed shards were preserved. "
              "Free or expand output storage, then rerun with the same parameters and --resume.",
              file=sys.stderr, flush=True)
        return 3
    except OSError as error:
        if error.errno not in (errno.ENOSPC, errno.EDQUOT):
            raise
        print(f"Storage write failed: {error}. Keep committed shards; free space/quota and use --resume.",
              file=sys.stderr, flush=True)
        return 3
    finally:
        generation._generate, generation.commit_batch, generation.publish = original_generate, original_commit, original_publish


if __name__ == "__main__":
    raise SystemExit(main())
