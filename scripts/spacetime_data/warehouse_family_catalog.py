"""Export/load a portable historical solved-family pool, without loading trajectories."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from scripts.spacetime_data import generate_warehouse_spacetime as base


FORMAT = "warehouse_solved_family_catalog_v1"
DEFAULT_CATALOG = Path(__file__).with_name("assets") / "warehouse_solved_families_ge3_v1.npz"
DEFAULT_ELIGIBLE_ROOTS = tuple(
    base.REPOSITORY / f"data_trajectories_spacetime/Warehouse-TorchSpaceTime-v1-{n}obstacle"
    for n in (3, 4, 5)
)


def source_contract(path):
    stat = path.stat()
    identity = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "source_group_size": 2}
    namespace = int(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:8], 16)
    with h5py.File(path, "r") as source:
        shape = source["sol_path"].shape
        if len(shape) != 3 or shape[0] % 2 or shape[1] < 3 or shape[2] != 7:
            raise ValueError("source must be a doubled [2*families,waypoints,7] dataset")
        task_ids = np.asarray(source["task_id"])
        if task_ids.shape != (shape[0],):
            raise ValueError("source task_id shape mismatch")
    return identity, namespace, shape[0], task_ids


def scan_history(roots, identity, source_rows):
    pool, evidence = set(), []
    for root in roots:
        config = json.loads((root / "config.json").read_text())
        index = json.loads((root / "index.json").read_text())
        expected = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if index.get("config_hash") != expected or config.get("schema") != base.SCHEMA:
            raise ValueError("historical dataset contract mismatch: " + str(root))
        if (config.get("obstacles", -1) < 3 or config.get("source_group_size") != 2
            or config.get("source_size") != identity["size"]
            or config.get("source_mtime_ns") != identity["mtime_ns"]):
            raise ValueError("historical dataset must use >=3 obstacles and the same doubled source: " + str(root))
        families = {int(entry["family_id"]) for entries in index["splits"].values() for entry in entries}
        if any(family < 0 or 2 * family + 1 >= source_rows for family in families):
            raise ValueError("historical family ID outside source")
        pool.update(families)
        evidence.append({"root": str(root), "obstacles": config["obstacles"], "families": len(families),
                         "config_sha256": base.sha256_file(root / "config.json"),
                         "index_sha256": base.sha256_file(root / "index.json")})
    return np.array(sorted(pool), dtype="<i8"), evidence


def family_digest(families):
    return hashlib.sha256(np.asarray(families, dtype="<i8").tobytes()).hexdigest()


def export_catalog(source, roots, output):
    identity, namespace, source_rows, _ = source_contract(source)
    families, evidence = scan_history(roots, identity, source_rows)
    if not len(families):
        raise ValueError("historical indexes contain no eligible families")
    splits = Counter()
    for family in families:
        bucket = int(hashlib.sha256(str(int(family)).encode()).hexdigest()[:8], 16) % 100
        splits["train" if bucket < 80 else ("validation" if bucket < 90 else "test")] += 1
    metadata = {"format": FORMAT, "source_identity": identity, "source_namespace": namespace,
                "source_rows": source_rows, "eligibility_min_obstacles": 3,
                "criterion": "family accepted at least once in one historical dataset with >=3 obstacles",
                "family_id_definition": "source_row // 2; both directions grouped",
                "family_count": len(families), "split_counts": dict(splits),
                "family_ids_sha256": family_digest(families), "historical_datasets": evidence}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".inprogress")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, family_id=families, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    base.atomic_json(output.with_suffix(".json"), metadata)
    return metadata


def load_catalog(path, identity, source_rows):
    with np.load(path, allow_pickle=False) as catalog:
        families = np.asarray(catalog["family_id"])
        metadata = json.loads(str(catalog["metadata_json"].item()))
    if metadata.get("format") != FORMAT or metadata.get("eligibility_min_obstacles") != 3:
        raise ValueError("unsupported solved-family catalog format")
    if metadata.get("source_identity") != identity or metadata.get("source_rows") != source_rows:
        raise ValueError("family catalog belongs to a different source")
    if (families.ndim != 1 or families.dtype.kind not in "iu" or not len(families)
        or np.any(families < 0) or np.any(families >= source_rows // 2)
        or np.any(families[1:] <= families[:-1])):
        raise ValueError("family catalog IDs must be unique, sorted and within source")
    if metadata.get("family_count") != len(families) or metadata.get("family_ids_sha256") != family_digest(families):
        raise ValueError("family catalog IDs/count/hash mismatch")
    # Historical paths are evidence only: generation must not reopen them.
    return families.astype(np.int64), metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=base.SOURCE)
    parser.add_argument("--eligible-roots", type=Path, nargs="+", default=list(DEFAULT_ELIGIBLE_ROOTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args(argv)
    metadata = export_catalog(args.source.resolve(), [root.resolve() for root in args.eligible_roots], args.output.resolve())
    print(json.dumps({"catalog": str(args.output.resolve()), "families": metadata["family_count"],
                      "splits": metadata["split_counts"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
