"""Immutable, memory-mapped training tensors shared by all four JointDual runs."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset

from mpd.datasets.spacetime_schema import sha256_file
from mpd.datasets.warehouse_joint_dataset import (
    WarehouseJointDataset, WORLD_CONTRACT, atomic_json, identity, load_snapshot,
)
from mpd.parametric_trajectory.joint_timing_codec import CODEC_CONTRACT

CACHE_VERSION = "jointdual_tensor_cache_v1"
FIXED = {"p": (np.dtype("float32"), (16, 7)), "task": (np.dtype("float32"), (19,)),
         "z_c": (np.dtype("float32"), (6,)), "z_tau_r": (np.dtype("float32"), (6,)),
         "world_state": (np.dtype("int64"), ())}


def cache_contract(snapshot):
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), root / "mpd/datasets/warehouse_joint_dataset.py",
             root / "mpd/datasets/spacetime_legacy.py",
             root / "mpd/parametric_trajectory/timing_fitting.py"]
    return dict(version=CACHE_VERSION, snapshot_identity=snapshot["identity"],
                world_contract=WORLD_CONTRACT, codec_contract=CODEC_CONTRACT,
                reader_hashes={p.name: sha256_file(p) for p in files})


def default_cache_path(snapshot):
    path = Path(snapshot).resolve()
    contract = cache_contract(load_snapshot(path))
    return path.parent / ".jointdual_tensor_cache" / identity(contract)


def _read_manifest(path, snapshot):
    manifest = json.loads((path / "manifest.json").read_text())
    saved = dict(manifest); expected = saved.pop("identity")
    if identity(saved) != expected or manifest["contract"] != cache_contract(snapshot):
        raise ValueError("tensor cache identity/reader mismatch; build a new cache directory")
    if manifest["count"] != len(snapshot["entries"]) or manifest["tokens"] < manifest["count"]:
        raise ValueError("tensor cache row/token count mismatch")
    specifications = {name + ".npy": (dtype, [manifest["count"], *shape])
                      for name, (dtype, shape) in FIXED.items()}
    specifications["world_offsets.npy"] = (np.dtype("int64"), [manifest["count"] + 1])
    specifications["world_features.bin"] = (np.dtype("float32"), [manifest["tokens"], 13])
    specifications["world_mask.bin"] = (np.dtype("bool"), [manifest["tokens"]])
    if set(manifest["files"]) != set(specifications):
        raise ValueError("tensor cache file set mismatch")
    for name, (dtype, shape) in specifications.items():
        meta = manifest["files"][name]
        stat = (path / name).stat()
        if (meta["shape"] != shape or meta["dtype"] != dtype.str
            or stat.st_size != meta["size"] or stat.st_mtime_ns != meta["mtime_ns"]):
            raise ValueError("frozen tensor cache changed: " + name)
    return manifest


def build_tensor_cache(snapshot, output=None, *, progress_every=4096):
    """Publish only complete caches; readers never consume staging files.

    A file lock serializes simultaneous first launches of the four training runs.
    The snapshot and its source shard stats are checked even on cache reuse.
    """
    source = WarehouseJointDataset(snapshot)
    output = Path(output).resolve() if output is not None else default_cache_path(snapshot)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (output.parent / (output.name + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if output.exists():
                _read_manifest(output, source.snapshot)
                return output
            staging = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
            arrays = {}
            try:
                count = len(source.snapshot["entries"])
                for name, (dtype, shape) in FIXED.items():
                    arrays[name] = np.lib.format.open_memmap(staging / (name + ".npy"), mode="w+",
                                                            dtype=dtype, shape=(count, *shape))
                arrays["world_offsets"] = np.lib.format.open_memmap(staging / "world_offsets.npy",
                    mode="w+", dtype=np.int64, shape=(count + 1,))
                offset = 0; arrays["world_offsets"][0] = 0
                print(json.dumps(dict(tensor_cache="building", path=str(output), samples=count)), flush=True)
                with (staging / "world_features.bin").open("wb") as features, (staging / "world_mask.bin").open("wb") as masks:
                    for i, entry in enumerate(source.snapshot["entries"]):
                        row = source._read_sample(entry)
                        for name in FIXED:
                            arrays[name][i] = row[name].numpy()
                        row["world_features"].numpy().tofile(features)
                        row["world_mask"].numpy().tofile(masks)
                        offset += len(row["world_mask"])
                        arrays["world_offsets"][i+1] = offset
                        if progress_every and (i+1) % progress_every == 0:
                            print(json.dumps(dict(tensor_cache="building", processed=i+1, samples=count)), flush=True)
                    features.flush(); os.fsync(features.fileno())
                    masks.flush(); os.fsync(masks.fileno())
                for array in arrays.values():
                    array.flush()
                files = {}
                for name, array in arrays.items():
                    files[name + ".npy"] = dict(shape=list(array.shape), dtype=array.dtype.str)
                files["world_features.bin"] = dict(shape=[offset, 13], dtype=np.dtype("float32").str)
                files["world_mask.bin"] = dict(shape=[offset], dtype=np.dtype("bool").str)
                arrays.clear()
                for name, meta in files.items():
                    path = staging / name
                    with path.open("rb") as handle:
                        os.fsync(handle.fileno())
                    stat = path.stat()
                    meta.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha256_file(path))
                # Refuse publication if a source was modified during construction.
                load_snapshot(snapshot)
                manifest = dict(contract=cache_contract(source.snapshot), count=count, tokens=offset, files=files)
                manifest["identity"] = identity(manifest)
                atomic_json(staging / "manifest.json", manifest)
                os.rename(staging, output)
                print(json.dumps(dict(tensor_cache="ready", path=str(output), samples=count,
                                      bytes=sum(m["size"] for m in files.values()))), flush=True)
            finally:
                arrays.clear()
                if staging.exists():
                    shutil.rmtree(staging)
    finally:
        source.close()
    return output


class CachedWarehouseJointDataset(Dataset):
    def __init__(self, snapshot, cache, split="train", representation="tau_r", include_world=True):
        if representation not in ("c", "tau_r") or split not in ("train", "validation", "test"):
            raise ValueError("invalid representation or split")
        self.snapshot = load_snapshot(snapshot)
        self.cache = Path(cache).resolve()
        self.manifest = _read_manifest(self.cache, self.snapshot)
        self.indices = [i for i, entry in enumerate(self.snapshot["entries"]) if entry["split"] == split]
        self.entries = [self.snapshot["entries"][i] for i in self.indices]
        self.normalization = self.snapshot["normalization"]
        self.representation, self.include_world = representation, include_world
        self._arrays, self._pid = {}, os.getpid()

    def __len__(self):
        return len(self.indices)

    def _array(self, name):
        if self._pid != os.getpid():
            self.close(); self._pid = os.getpid()
        if name not in self._arrays:
            # Copy-on-write permits safe torch views without modifying disk data.
            if name.endswith(".npy"):
                value = np.load(self.cache / name, mmap_mode="c", allow_pickle=False)
                meta = self.manifest["files"][name]
                if list(value.shape) != meta["shape"] or value.dtype.str != meta["dtype"]:
                    raise ValueError("tensor cache array header mismatch")
            else:
                meta = self.manifest["files"][name]
                value = np.memmap(self.cache / name, mode="c", dtype=meta["dtype"], shape=tuple(meta["shape"]))
            self._arrays[name] = value
        return self._arrays[name]

    def __getitem__(self, index):
        i = self.indices[index]
        result = {name: torch.from_numpy(self._array(name + ".npy")[i]) for name in ("p", "task")}
        result["z"] = torch.from_numpy(self._array("z_" + self.representation + ".npy")[i])
        result["sample_index"] = torch.tensor(index)
        if self.include_world:
            lo, hi = self._array("world_offsets.npy")[i:i+2]
            if not 0 <= lo < hi <= self.manifest["tokens"]:
                raise ValueError("invalid tensor cache world offsets")
            result.update(world_features=torch.from_numpy(self._array("world_features.bin")[lo:hi]),
                          world_mask=torch.from_numpy(self._array("world_mask.bin")[lo:hi]),
                          world_state=torch.tensor(int(self._array("world_state.npy")[i])))
        return result

    def close(self):
        # Outstanding tensor views keep their arrays alive; never unmap underneath them.
        self._arrays.clear()

    def __getstate__(self):
        return dict(self.__dict__, _arrays={})
