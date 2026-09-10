#!/usr/bin/env python3
"""Build a local Foam-based, guidance-only Marvin/Pika collision profile."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import yaml


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "mpd/torch_robotics/torch_robotics/data"
URDF = DATA / "urdf/robots/marvin/marvin_pika_bimanual_mpd.urdf"
PRODUCTION = DATA / "configs/marvin/pika"
DEFAULT_OUTPUT = DATA / "configs/marvin/pika_foam_guide"
TOOL_LINK_SPECS = {
    "left_pika_adaptor_link": {"target": 4, "branch": 4, "depth": 1},
    "left_gripper_base_link": {"target": 38, "branch": 7, "depth": 2},
    "left_gripper_left_link": {"target": 3, "branch": 10, "depth": 1},
    "left_gripper_right_link": {"target": 3, "branch": 10, "depth": 1},
    "right_pika_adaptor_link": {"target": 4, "branch": 4, "depth": 1},
    "right_gripper_base_link": {"target": 38, "branch": 7, "depth": 2},
    "right_gripper_left_link": {"target": 3, "branch": 10, "depth": 1},
    "right_gripper_right_link": {"target": 3, "branch": 10, "depth": 1},
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rpy_matrix(value):
    roll, pitch, yaw = value
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _load_link_meshes():
    root = ET.parse(URDF).getroot()
    result = {}
    for link in root.findall("link"):
        name = link.get("name")
        if name not in TOOL_LINK_SPECS:
            continue
        pieces = []
        sources = []
        for collision in link.findall("collision"):
            mesh_element = collision.find("geometry/mesh")
            if mesh_element is None:
                raise ValueError(f"{name} contains a non-mesh collision geometry")
            mesh_path = URDF.parent / mesh_element.get("filename")
            loaded = trimesh.load_mesh(mesh_path, process=False)
            if isinstance(loaded, trimesh.Scene):
                loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
            scale = np.fromstring(mesh_element.get("scale", "1 1 1"), sep=" ")
            loaded.apply_scale(scale)
            origin = collision.find("origin")
            if origin is not None:
                xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                transform = np.eye(4)
                transform[:3, :3] = _rpy_matrix(rpy)
                transform[:3, 3] = xyz
                loaded.apply_transform(transform)
            pieces.append(loaded)
            sources.append(
                {"path": str(mesh_path.relative_to(ROOT)), "sha256": _sha256(mesh_path)}
            )
        result[name] = (trimesh.util.concatenate(pieces), sources)
    missing = set(TOOL_LINK_SPECS) - set(result)
    if missing:
        raise ValueError(f"URDF is missing Pika collision links: {sorted(missing)}")
    return result


def _read_sph(path, offset):
    lines = path.read_text().splitlines()
    counts = [int(line.split(":", 1)[1]) for line in lines if "Num" in line]
    best = [float(line.split(":", 1)[1]) for line in lines if "Best" in line]
    worst = [float(line.split(":", 1)[1]) for line in lines if "Worst" in line]
    mean = [float(line.split(":", 1)[1]) for line in lines if "Mean" in line]
    levels = []
    # makeTreeOctree uses the compact legacy format: the first line is
    # ``number_of_levels branch_factor`` followed by branch**level entries.
    if not counts and lines:
        header = lines[0].split()
        if len(header) == 2 and all(value.isdigit() for value in header):
            n_levels, branch = map(int, header)
            cursor = 1
            for level in range(n_levels):
                count = branch**level
                spheres = []
                for line in lines[cursor : cursor + count]:
                    values = [float(value) for value in line.split()]
                    if len(values) >= 4 and values[3] > 0:
                        spheres.append(
                            [*(np.asarray(values[:3]) + offset).tolist(), values[3]]
                        )
                cursor += count
                levels.append(
                    {
                        "level": level,
                        "spheres": spheres,
                        "mean_error": None,
                        "best_error": None,
                        "worst_error": None,
                    }
                )
            return levels
    if not (len(counts) == len(best) == len(worst) == len(mean)):
        raise RuntimeError(f"Malformed Foam sphere file: {path}")
    for level, (count, best_error, worst_error, mean_error) in enumerate(
        zip(counts, best, worst, mean)
    ):
        start = 1 + sum(counts[:level])
        spheres = []
        for line in lines[start : start + count]:
            values = [float(value) for value in line.split()]
            if len(values) < 4 or values[3] <= 0:
                continue
            spheres.append([*(np.asarray(values[:3]) + offset).tolist(), values[3]])
        levels.append(
            {
                "level": level,
                "spheres": spheres,
                "mean_error": mean_error,
                "best_error": best_error,
                "worst_error": worst_error,
            }
        )
    if not levels:
        raise RuntimeError(
            f"Foam did not write parseable sphere levels to {path}: {lines[:20]}"
        )
    return levels


def _run_foam(
    mesh,
    *,
    foam_root,
    executable,
    method,
    branch,
    depth,
    target,
    inflation,
    num_cover,
    init_spheres,
    min_spheres,
    optimise,
    verify,
    manifold_leaves,
    simplify_ratio,
    expand,
    merge,
):
    lower, upper = mesh.bounds
    offset = 0.5 * (lower + upper)
    centered = mesh.copy()
    centered.apply_translation(-offset)
    # Foam's OBJ loader expects normals; this mirrors foam.compute_spheres_helper.
    _ = centered.vertex_normals
    with tempfile.TemporaryDirectory(prefix="marvin-foam-") as directory:
        directory = Path(directory)
        original_obj = directory / "original.obj"
        manifold_obj = directory / "manifold.obj"
        obj = directory / "input.obj"
        original_obj.write_text(trimesh.exchange.obj.export_obj(centered))
        manifold_executable = foam_root / "foam/external/manifold_old"
        simplify_executable = foam_root / "foam/external/simplify_old"
        subprocess.run(
            [
                str(manifold_executable),
                str(original_obj),
                str(manifold_obj),
                str(manifold_leaves),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                str(simplify_executable),
                "-i",
                str(manifold_obj),
                "-o",
                str(obj),
                "-r",
                str(simplify_ratio),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        processed = trimesh.load_mesh(obj, process=False)
        if isinstance(processed, trimesh.Scene):
            processed = trimesh.util.concatenate(tuple(processed.geometry.values()))
        if not len(processed.faces):
            raise RuntimeError("Foam preprocessing produced an empty mesh")
        _ = processed.vertex_normals
        obj.write_text(trimesh.exchange.obj.export_obj(processed))
        command = [
            str(executable),
            "-nopause",
            "-branch",
            str(branch),
            "-depth",
            str(depth),
        ]
        if verify:
            command.insert(2, "-verify")
        if method in {"medial", "grid", "spawn"}:
            command.extend(
                [
                    "-testerLevels",
                    "2",
                    "-numCover",
                    str(num_cover),
                    "-minCover",
                    "2",
                ]
            )
        if method == "medial":
            command.extend(
                [
                    "-initSpheres",
                    str(init_spheres),
                    "-minSpheres",
                    str(min_spheres),
                    "-erFact",
                    "2",
                    "-maxOptLevel",
                    "1",
                ]
            )
        if expand:
            command.append("-expand")
        if merge:
            command.append("-merge")
        command.extend(["-balExcess", "0.05"])
        if method != "medial" and optimise:
            command.extend(["-optimise", "simplex"])
        command.append(str(obj))
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(
                f"Foam failed with exit {completed.returncode}:\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        output_path = obj.with_name(f"input-{method}.sph")
        if not output_path.is_file():
            raise RuntimeError(
                "Foam returned success without the expected sphere file. "
                f"files={sorted(path.name for path in Path(directory).iterdir())}; "
            f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
            )
        levels = _read_sph(output_path, offset)
    selected = min(
        levels,
        key=lambda value: (abs(len(value["spheres"]) - target), -value["level"]),
    )
    for sphere in selected["spheres"]:
        sphere[3] += inflation
    return selected, command[:-1], {
        "input_faces": int(len(mesh.faces)),
        "processed_faces": int(len(processed.faces)),
    }


def _coverage(mesh, spheres, sample_count, seed):
    np.random.seed(seed)
    points, _ = trimesh.sample.sample_surface(mesh, sample_count)
    spheres = np.asarray(spheres, dtype=float)
    minimum = np.full(sample_count, np.inf)
    for start in range(0, sample_count, 4096):
        block = points[start : start + 4096]
        signed = np.linalg.norm(
            block[:, None, :] - spheres[None, :, :3], axis=-1
        ) - spheres[None, :, 3]
        minimum[start : start + len(block)] = signed.min(axis=1)
    return {
        "surface_samples": sample_count,
        "uncovered_ratio": float(np.mean(minimum > 0)),
        "maximum_uncovered_m": float(np.maximum(minimum, 0).max()),
        "mean_signed_excess_m": float(minimum.mean()),
    }


def _parent_bounds(sphere_config):
    bounds = {}
    for name, entries in sphere_config.items():
        if name == "self_collision":
            continue
        values = np.asarray(entries, dtype=float)
        lower = (values[:, :3] - values[:, 3, None]).min(axis=0)
        upper = (values[:, :3] + values[:, 3, None]).max(axis=0)
        center = 0.5 * (lower + upper)
        radius = np.max(np.linalg.norm(values[:, :3] - center, axis=1) + values[:, 3])
        bounds[name] = [
            {
                "center": center.tolist(),
                "radius": float(radius + 1e-6),
                "source_sphere_indices": list(range(len(entries))),
            }
        ]
    return {
        "metadata": {
            "method": "aabb_center_exact_sphere_cover",
            "safety_padding": 1e-6,
            "geometry_profile": "foam_pika_100",
        },
        "parent_bounds": bounds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foam-root", type=Path, default=Path("/home/eric/Projects/foam"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--method", choices=("medial", "grid", "spawn", "octree"), default="grid"
    )
    parser.add_argument("--inflation", type=float, default=0.006)
    parser.add_argument("--surface-samples", type=int, default=20000)
    parser.add_argument("--num-cover", type=int, default=500)
    parser.add_argument("--init-spheres", type=int, default=200)
    parser.add_argument("--min-spheres", type=int, default=20)
    parser.add_argument("--manifold-leaves", type=int, default=1000)
    parser.add_argument("--simplify-ratio", type=float, default=0.2)
    parser.add_argument("--expand", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument(
        "--optimise",
        action="store_true",
        help="Run Foam's expensive simplex post-optimisation (disabled by default).",
    )
    parser.add_argument(
        "--verify-mesh",
        action="store_true",
        help="Ask Foam to run its expensive input-mesh verification pass.",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if (
        args.inflation < 0
        or args.surface_samples < 100
        or args.manifold_leaves < 10
        or not 0 < args.simplify_ratio <= 1
    ):
        raise SystemExit("invalid coverage, inflation, or mesh preprocessing parameter")
    executable = (
        args.foam_root.resolve()
        / "foam/external"
        / {
            "medial": "makeTreeMedial",
            "grid": "makeTreeGrid",
            "spawn": "makeTreeSpawn",
            "octree": "makeTreeOctree",
        }[args.method]
    )
    if not executable.is_file():
        raise FileNotFoundError(executable)

    production = yaml.safe_load((PRODUCTION / "collision_spheres.yaml").read_text())
    reduced = {
        key: value
        for key, value in production.items()
        if key not in TOOL_LINK_SPECS and key != "self_collision"
    }
    manifest_links = {}
    command_template = None
    foam_cache = {}
    for index, (name, (mesh, sources)) in enumerate(_load_link_meshes().items()):
        spec = TOOL_LINK_SPECS[name]
        cache_key = (
            tuple(source["sha256"] for source in sources),
            spec["target"],
            spec["branch"],
            spec["depth"],
            args.inflation,
            args.method,
            args.num_cover,
            args.init_spheres,
            args.min_spheres,
            args.optimise,
            args.verify_mesh,
            args.manifold_leaves,
            args.simplify_ratio,
            args.expand,
            args.merge,
        )
        if cache_key not in foam_cache:
            foam_cache[cache_key] = _run_foam(
                mesh,
                foam_root=args.foam_root.resolve(),
                executable=executable,
                method=args.method,
                branch=spec["branch"],
                depth=spec["depth"],
                target=spec["target"],
                inflation=args.inflation,
                num_cover=args.num_cover,
                init_spheres=args.init_spheres,
                min_spheres=args.min_spheres,
                optimise=args.optimise,
                verify=args.verify_mesh,
                manifold_leaves=args.manifold_leaves,
                simplify_ratio=args.simplify_ratio,
                expand=args.expand,
                merge=args.merge,
            )
        selected, command_template, preprocessing = copy.deepcopy(foam_cache[cache_key])
        reduced[name] = selected["spheres"]
        manifest_links[name] = {
            "target_spheres": spec["target"],
            "foam_branch": spec["branch"],
            "foam_depth": spec["depth"],
            "actual_spheres": len(selected["spheres"]),
            "selected_level": selected["level"],
            "foam_error": {
                key: selected[key]
                for key in ("mean_error", "best_error", "worst_error")
            },
            "coverage": _coverage(
                mesh, selected["spheres"], args.surface_samples, 47 + index
            ),
            "mesh_sources": sources,
            "preprocessing": preprocessing,
        }
    reduced["self_collision"] = production["self_collision"]
    pika_count = sum(len(reduced[name]) for name in TOOL_LINK_SPECS)
    if not 90 <= pika_count <= 110:
        raise RuntimeError(
            f"foam_pika_100 must contain 90--110 Pika spheres, got {pika_count}: "
            f"{ {name: len(reduced[name]) for name in TOOL_LINK_SPECS} }"
        )

    foam_commit = subprocess.check_output(
        ["git", "-C", str(args.foam_root), "rev-parse", "HEAD"], text=True
    ).strip()
    outputs = {
        "collision_spheres.yaml": yaml.safe_dump(reduced, sort_keys=False, width=110),
        "collision_parent_bounds.yaml": yaml.safe_dump(
            _parent_bounds(reduced), sort_keys=False, width=110
        ),
        "self_collision_pairs.yaml": (PRODUCTION / "self_collision_pairs.yaml").read_text(),
    }
    manifest = {
        "schema": "marvin_foam_guide_geometry/v1",
        "profile": "foam_pika_100",
        "guidance_only": True,
        "production_validator_profile": "pika",
        "foam_root": str(args.foam_root.resolve()),
        "foam_commit": foam_commit,
        "foam_executable_sha256": _sha256(executable),
        "method": args.method,
        "link_specs": TOOL_LINK_SPECS,
        "inflation_m": args.inflation,
        "num_cover": args.num_cover,
        "init_spheres": args.init_spheres,
        "min_spheres": args.min_spheres,
        "optimise": args.optimise,
        "verify_mesh": args.verify_mesh,
        "manifold_leaves": args.manifold_leaves,
        "simplify_ratio": args.simplify_ratio,
        "expand": args.expand,
        "merge": args.merge,
        "command_template": [str(value) for value in command_template],
        "pika_sphere_count": pika_count,
        "total_sphere_count": sum(
            len(value) for key, value in reduced.items() if key != "self_collision"
        ),
        "links": manifest_links,
    }
    outputs["guide_geometry_manifest.yaml"] = yaml.safe_dump(
        manifest, sort_keys=False, width=110
    )

    changed = [
        name
        for name, contents in outputs.items()
        if not (args.output_dir / name).is_file()
        or (args.output_dir / name).read_text() != contents
    ]
    if args.check:
        if changed:
            raise SystemExit("Foam guide geometry drift:\n" + "\n".join(changed))
        print(
            json.dumps(
                {
                    "status": "pass",
                    "pika_spheres": pika_count,
                    "total_spheres": manifest["total_sphere_count"],
                },
                sort_keys=True,
            )
        )
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, contents in outputs.items():
        (args.output_dir / name).write_text(contents)
    print(
        json.dumps(
            {
                "status": "generated",
                "changed": changed,
                "pika_spheres": pika_count,
                "total_spheres": manifest["total_sphere_count"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
