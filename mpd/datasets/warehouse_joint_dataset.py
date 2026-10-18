"""Frozen P21/EE-goal data views shared by all four JointDual variants."""
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from mpd.datasets.spacetime_schema import load_panda_robot_bundle, sha256_file
from mpd.datasets.spacetime_legacy import open_uniform_knots
from mpd.parametric_trajectory.timing_fitting import encode_timing_control_points

SCHEMA = "warehouse_spacetime_eegoal_v1"
SNAPSHOT_VERSION = "jointdual_snapshot_v1"
WORLD_CONTRACT = dict(version="jointdual_world_v1", time_points=32, horizon=14., feature_dim=13)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def world_features(times, positions, shapes, sizes):
    """Fixed physical time grid; velocity is the right segment slope at knots."""
    t = np.linspace(0., 14., 32)
    times, positions = np.asarray(times), np.asarray(positions)
    m = len(shapes)
    if times[0] > 0 or times[-1] < 14 or np.any(np.diff(times) <= 0):
        raise ValueError("world must cover [0,14] with strictly increasing knots")
    if positions.shape != (len(times), m, 3) or np.asarray(sizes).shape != (m, 3):
        raise ValueError("world dimensions disagree")
    if not m:
        return np.zeros((1, 13), np.float32), np.zeros(1, bool), 1
    upper = np.searchsorted(times, t, side="right").clip(1, len(times) - 1)
    lo = upper - 1
    slope = (positions[upper] - positions[lo]) / (times[upper] - times[lo])[:, None, None]
    center = positions[lo] + slope * (t - times[lo])[:, None, None]
    shape = np.broadcast_to(np.eye(3)[shapes][None], (32, m, 3))
    size = np.broadcast_to(sizes[None], (32, m, 3))
    seconds = np.broadcast_to(t[:, None, None], (32, m, 1))
    features = np.concatenate((center, slope, shape, size, seconds), axis=-1)
    features = features.transpose(1, 0, 2).reshape(-1, 13).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("nonfinite world features")
    return features, np.ones(len(features), bool), 0


def read_label(g):
    if not g.attrs.get("accepted", False) or not g.attrs.get("tau_r_valid", False):
        raise ValueError("not valid in both representations")
    for key in ("quality_json", "tau_r_quality_json"):
        quality = json.loads(g.attrs[key])
        if not quality or not 2. <= quality["duration"] <= 14.:
            raise ValueError("missing quality or invalid duration")
    full = g["spatial/control_points"][:]
    p = g["spatial/learnable_control_points"][:]
    qs, qg = g["condition/q_start"][:], g["condition/q_goal"][:]
    pose = g["condition/ee_goal_pose"][:]
    zc = encode_timing_control_points(g["timing/control_points"][:])
    zr = g["timing/tau_r"][:]
    if (full.shape != (21, 7) or p.shape != (16, 7) or pose.shape != (3, 4)
        or qs.shape != (7,) or qg.shape != (7,) or zr.shape != (6,)
        or int(g.attrs["degree"]) != 5 or float(g.attrs["phase_eta"]) != .05
        or not np.allclose(g["spatial/knots"][:], open_uniform_knots(21, 5))):
        raise ValueError("P21/EE spline contract mismatch")
    expanded = np.concatenate((np.repeat(qs[None], 3, 0), p[:-1], np.repeat(p[-1:], 3, 0)))
    if not np.allclose(full, expanded, atol=1e-7) or not np.allclose(p[-1], qg, atol=1e-7):
        raise ValueError("generated-point layout mismatch")
    task = np.r_[qs, pose[:, 3], pose[:, :3].reshape(9)]
    if not all(np.isfinite(v).all() for v in (p, zc, zr, task)):
        raise ValueError("nonfinite label")
    if not np.allclose(pose[:, :3].T @ pose[:, :3], np.eye(3), atol=1e-5):
        raise ValueError("invalid EE rotation")
    return p, {"c": zc, "tau_r": zr}, task


def build_snapshot(roots, output):
    bundle = load_panda_robot_bundle(Path(__file__).resolve().parents[2])
    geometry_dir = Path(__file__).resolve().parents[1] / "torch_robotics/torch_robotics/environments"
    geometry_hashes = {name: sha256_file(geometry_dir / name)
                       for name in ("env_warehouse.py", "env_table_shelf.py", "primitives.py")}
    entries, families, sources, files, rejects = [], {}, [], {}, []
    sums = {k: [0, np.zeros(d), np.zeros(d)] for k, d in (("c", 6), ("tau_r", 6), ("ee", 12))}
    for root in map(lambda x: Path(x).resolve(), roots):
        manifest = yaml.safe_load((root / "manifest.yaml").read_text())
        index = json.loads((root / "index.json").read_text())
        if manifest["schema"] != SCHEMA or index["schema"] != SCHEMA:
            raise ValueError("convert raw data to P21/EE before building a snapshot")
        if manifest["config_hash"] != index["config_hash"]:
            raise ValueError("index/manifest identity mismatch")
        if identity(json.loads((root / "config.json").read_text())) != index["config_hash"]:
            raise ValueError("config identity mismatch")
        if any(manifest["robot"][key] != value for key, value in bundle.hashes.items()):
            raise ValueError("robot assets differ from canonical Panda")
        if manifest["spatial"]["num_control_points"] != 21 or manifest["timing"]["codec_grid_size"] != 2049:
            raise ValueError("unsupported spatial/timing contract")
        parent_snapshot = json.loads((root / "input_snapshot.json").read_text())
        for source in parent_snapshot["sources"]:
            if any(geometry_hashes.get(k) != v for k, v in source["static_geometry_hashes"].items()):
                raise ValueError("static geometry differs from the refit source")
        sources.append(dict(root=str(root), manifest_sha256=sha256_file(root / "manifest.yaml")))
        for split in ("train", "validation", "test"):
            for e in index["splits"][split]:
                path = (root / e["file"]).resolve()
                if root / "shards" not in path.parents or path.suffix != ".hdf5":
                    raise ValueError("entry outside committed shards")
                with h5py.File(path, "r") as f:
                    if f.attrs["schema"] != SCHEMA or f.attrs["config_hash"] != index["config_hash"]:
                        raise ValueError("shard contract mismatch")
                    g = f[e["group"]]
                    family = f"{g.attrs['source_namespace']}:{int(g.attrs['family_id'])}"
                    if family in families and families[family] != split:
                        raise ValueError("family split conflict")
                    families[family] = split
                    if str(g.attrs["split"]) != split:
                        raise ValueError("index/group split mismatch")
                    try:
                        p, targets, task = read_label(g)
                        world_features(g["world/times"][:], g["world/positions"][:],
                                       g["world/shape"][:], g["world/size"][:])
                    except (ValueError, KeyError) as exc:
                        rejects.append(dict(file=str(path), group=e["group"], reason=str(exc)))
                        continue
                    key = f"{path}:{e['group']}"
                    entries.append(dict(file=str(path), group=e["group"], split=split,
                                        family=family, sample_key=key))
                    if split == "train":
                        for name, value in dict(targets, ee=task[7:]).items():
                            state = sums[name]
                            state[0] += 1; state[1] += value; state[2] += value ** 2
                stat = path.stat()
                files[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if not sums["c"][0]:
        raise ValueError("snapshot needs valid training labels")
    if len({e["sample_key"] for e in entries}) != len(entries):
        raise ValueError("duplicate samples across roots")
    statistics = {}
    for name, (count, total, squares) in sums.items():
        mean = total / count
        std = np.maximum(np.sqrt(np.maximum(0, squares / count - mean ** 2)), 1e-6)
        statistics[name] = dict(mean=mean.tolist(), std=std.tolist())
    norm = dict(q_mean=((bundle.q_min + bundle.q_max) / 2).tolist(),
                q_std=((bundle.q_max - bundle.q_min) / 2).tolist(), **statistics)
    snapshot = dict(version=SNAPSHOT_VERSION, entries=entries, files=files, sources=sources,
        normalization=norm, robot_hashes=bundle.hashes, static_geometry_hashes=geometry_hashes,
        world_contract=WORLD_CONTRACT,
        rejected=rejects, split_counts={s: sum(e["split"] == s for e in entries)
                                      for s in ("train", "validation", "test")})
    snapshot["identity"] = identity(snapshot)
    if Path(output).exists():
        raise FileExistsError(output)
    atomic_json(output, snapshot)
    return snapshot


def load_snapshot(snapshot):
    value = json.loads(Path(snapshot).read_text())
    saved = dict(value); expected = saved.pop("identity")
    if saved["version"] != SNAPSHOT_VERSION or identity(saved) != expected:
        raise ValueError("corrupted snapshot")
    for name, expected in value["files"].items():
        stat = Path(name).stat()
        if expected != dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns):
            raise ValueError("frozen shard changed")
    return value


class WarehouseJointDataset(Dataset):
    def __init__(self, snapshot, split="train", representation="tau_r", max_open_files=8,
                 include_world=True):
        if representation not in ("c", "tau_r") or split not in ("train", "validation", "test"):
            raise ValueError("invalid representation or split")
        self.snapshot = load_snapshot(snapshot)
        self.entries = [e for e in self.snapshot["entries"] if e["split"] == split]
        self.normalization = self.snapshot["normalization"]
        self.representation, self.max_open_files = representation, max_open_files
        self.include_world = include_world
        self._handles, self._pid = OrderedDict(), os.getpid()

    def __len__(self):
        return len(self.entries)

    def _open(self, name):
        if os.getpid() != self._pid:
            self.close(); self._pid = os.getpid()
        if name in self._handles:
            self._handles.move_to_end(name)
            return self._handles[name]
        self._handles[name] = h5py.File(name, "r")
        if len(self._handles) > self.max_open_files:
            self._handles.popitem(last=False)[1].close()
        return self._handles[name]

    def _read_sample(self, entry):
        """One HDF5 read produces both standardized timing targets for the cache."""
        g = self._open(entry["file"])[entry["group"]]
        p, targets, task = read_label(g)
        n = self.normalization
        p = (p - n["q_mean"]) / n["q_std"]
        task = np.r_[(task[:7] - n["q_mean"]) / n["q_std"],
                     (task[7:] - n["ee"]["mean"]) / n["ee"]["std"]]
        result = {"p": torch.tensor(p, dtype=torch.float32), "task": torch.tensor(task, dtype=torch.float32)}
        for representation, z in targets.items():
            stats = n[representation]
            result["z_" + representation] = torch.tensor((z - stats["mean"]) / stats["std"], dtype=torch.float32)
        if self.include_world:
            features, mask, state = world_features(g["world/times"][:], g["world/positions"][:],
                                                   g["world/shape"][:], g["world/size"][:])
            result.update(world_features=torch.from_numpy(features), world_mask=torch.from_numpy(mask),
                          world_state=torch.tensor(state))
        return result

    def __getitem__(self, index):
        result = self._read_sample(self.entries[index])
        result["z"] = result.pop("z_" + self.representation)
        result.pop("z_tau_r" if self.representation == "c" else "z_c")
        return dict(result, sample_index=torch.tensor(index))

    def close(self):
        for f in self._handles.values():
            f.close()
        self._handles.clear()

    def __getstate__(self):
        state = dict(self.__dict__); state["_handles"] = OrderedDict()
        return state


def collate_joint(batch):
    if "world_mask" not in batch[0]:
        return {k: torch.stack([row[k] for row in batch]) for k in batch[0]}
    count = max(len(row["world_mask"]) for row in batch)
    features = torch.zeros(len(batch), count, 13)
    mask = torch.zeros(len(batch), count, dtype=torch.bool)
    for i, row in enumerate(batch):
        n = len(row["world_mask"])
        features[i, :n] = row["world_features"]; mask[i, :n] = row["world_mask"]
    result = {k: torch.stack([row[k] for row in batch]) for k in batch[0]
              if k not in ("world_features", "world_mask")}
    return dict(result, world_features=features, world_mask=mask)
