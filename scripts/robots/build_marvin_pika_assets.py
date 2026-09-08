#!/usr/bin/env python3
"""Build/check a ROS-independent 14-axis Marvin model with fixed dual Pika.

Run in the ROS xacro environment. --check generates in memory and compares
every managed output without modifying the checkout. Original arm-only assets
remain available. Collision envelopes include the entire finger travel.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from scripts.robots.import_marvin_ros_assets import convert_collision, convert_joint_limits

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('mpd/torch_robotics/torch_robotics/data')
MODEL = DATA / 'urdf/robots/marvin'
CONFIG = DATA / 'configs/marvin/pika'
JOINTS = [f'Joint{i}_{side}' for side in ('L', 'R') for i in range(1, 8)]
PACKAGES = {
    'marvin_description': 'src/embodiments/robots/marvin/marvin_description',
    'pika_gripper_description': 'src/embodiments/end_effectors/pika_gripper/pika_gripper_description',
    'marvin_mpd_bimanual_bringup': 'src/apps/marvin_mpd_bimanual_bringup',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def xml_bytes(root):
    # ElementTree discards comments including workstation-specific Xacro paths.
    return ET.tostring(root, encoding='utf-8', xml_declaration=True) + b'\n'


def yaml_bytes(value):
    return yaml.safe_dump(value, sort_keys=False, width=110).encode()


def rpy_rotation(value):
    r, p, y = map(float, value.split())
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def stl_vertices(data):
    """Read binary STL without requiring a geometry engine in the ROS env."""
    count = int.from_bytes(data[80:84], 'little')
    if len(data) != 84 + 50 * count:
        raise ValueError('Expected a binary STL with complete triangles')
    dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attr', '<u2')])
    return np.frombuffer(data, dtype=dtype, offset=84)['vertices'].reshape(-1, 3).astype(float)


def freeze_fingers(root, finger_travel):
    """Bake q and mimic(q) into the origin, then remove movable-joint tags."""
    frozen = {}
    for side in ('left', 'right'):
        for finger in ('left', 'right'):
            name = f'{side}_gripper_{finger}_joint'
            joint = root.find(f"joint[@name='{name}']")
            if joint is None or joint.get('type') != 'prismatic':
                raise ValueError(f'Missing expected prismatic joint: {name}')
            limit = joint.find('limit')
            lower, upper = float(limit.get('lower')), float(limit.get('upper'))
            q = float(finger_travel)
            mimic = joint.find('mimic')
            if mimic is not None:
                expected = f'{side}_gripper_left_joint'
                if mimic.get('joint') != expected:
                    raise ValueError(f'Unexpected mimic target on {name}')
                q = q * float(mimic.get('multiplier', 1)) + float(mimic.get('offset', 0))
            if not math.isfinite(q) or not lower <= q <= upper:
                raise ValueError(f'{name}: q={q} outside [{lower}, {upper}]')
            origin = joint.find('origin')
            rotation = rpy_rotation(origin.get('rpy', '0 0 0'))
            axis = np.array(list(map(float, joint.find('axis').get('xyz').split())))
            xyz = np.array(list(map(float, origin.get('xyz').split()))) + rotation @ (axis*q)
            origin.set('xyz', ' '.join(format(v, '.17g') for v in xyz))
            joint.set('type', 'fixed')
            for child in list(joint):
                if child.tag in ('axis', 'limit', 'mimic', 'dynamics', 'safety_controller'):
                    joint.remove(child)
            frozen[joint.find('child').get('link')] = {
                'joint': name, 'q': q, 'lower': lower, 'upper': upper, 'axis': axis.tolist(),
            }
    actual = [j.get('name') for j in root.findall('joint') if j.get('type') != 'fixed']
    if actual != JOINTS:
        raise ValueError(f'Expected 14 canonical arm joints, got {actual}')
    return frozen


def envelope_spheres(lower, upper, cell_size=0.025):
    """Cover the entire AABB volume by circumspheres of a regular cell grid."""
    counts = np.maximum(1, np.ceil((upper-lower)/cell_size).astype(int))
    step = (upper-lower)/counts
    radius = float(np.linalg.norm(step/2) + 1e-6)
    return [[*map(float, lower+(np.array(index)+0.5)*step), radius]
            for index in itertools.product(*(range(int(n)) for n in counts))]


def build(ros_root, finger_travel=0.045):
    packages = {name: ros_root / rel for name, rel in PACKAGES.items()}
    source = packages['marvin_mpd_bimanual_bringup'] / 'urdf/marvin_pika_bimanual.urdf.xacro'
    # Resolve $(find ...) against exactly the requested source trees, even when
    # the shell has a stale ROS install overlay sourced.
    with tempfile.TemporaryDirectory(prefix='marvin-xacro-') as tmp:
        prefix = Path(tmp)
        index = prefix / 'share/ament_index/resource_index/packages'
        index.mkdir(parents=True)
        for name, package in packages.items():
            if not package.is_dir():
                raise FileNotFoundError(package)
            (index / name).touch()
            (prefix / 'share' / name).symlink_to(package.resolve(), target_is_directory=True)
        env = dict(os.environ, AMENT_PREFIX_PATH=str(prefix))
        expanded = subprocess.run(['xacro', str(source), 'ros2_control:=false'], env=env,
                                  check=True, capture_output=True, text=True).stdout
    reference = ET.fromstring(expanded)
    if reference.findall('ros2_control'):
        raise ValueError('Planning description must not contain hardware plugins')
    outputs = {}
    origins = {}
    provenance = {}
    # Archive source descriptions/configs and existing upstream attribution.
    for name, package in packages.items():
        paths = {package / 'package.xml'}
        for directory, pattern in (('urdf', '*.xacro'), ('config', '*.yaml'), ('config', '*.yml')):
            paths.update((package / directory).rglob(pattern))
        paths.update(p for p in package.glob('*') if p.is_file() and
                     (p.name.upper().startswith(('LICENSE', 'NOTICE', 'COPYING'))))
        if name == 'pika_gripper_description':
            paths.add(package / 'docs/MESH_SOURCES.md')
        source_hashes = {}
        for path in sorted(paths):
            relative = path.relative_to(package).as_posix()
            data = path.read_bytes()
            outputs[MODEL / 'sources' / name / relative] = data
            source_hashes[relative] = sha(data)
        package_xml = ET.parse(package / 'package.xml').getroot()
        commit = subprocess.check_output(['git', '-C', str(package), 'rev-parse', 'HEAD'], text=True).strip()
        repository = subprocess.check_output(['git', '-C', str(package), 'remote', 'get-url', 'origin'], text=True).strip()
        provenance[name] = {'repository': repository, 'commit': commit,
                            'declared_license': [e.text for e in package_xml.findall('license')],
                            'source_sha256': source_hashes}
    for mesh in reference.findall('.//mesh'):
        uri = mesh.get('filename')
        if not uri.startswith('package://'):
            raise ValueError(f'Expected source package URI, got {uri}')
        name, rel = uri[len('package://'):].split('/', 1)
        if name not in packages or '..' in Path(rel).parts:
            raise ValueError(f'Unsupported mesh source: {uri}')
        data = (packages[name] / rel).read_bytes()
        target = Path('meshes') / name / Path(rel).relative_to('meshes')
        outputs[MODEL / target] = data
        origins[target.as_posix()] = {'package': name, 'path': rel, 'sha256': sha(data)}
        mesh.set('filename', target.as_posix())
    outputs[MODEL / 'sources/ros_marvin_pika_expanded.urdf'] = xml_bytes(reference)
    planning = copy.deepcopy(reference)
    frozen = freeze_fingers(planning, finger_travel)
    planning.insert(0, ET.Comment(
        ' Modified for MPD: local mesh paths, no hardware, fixed Pika fingers. '
        'See licenses/SOURCES.md and pika_assets.lock.yaml. '))
    outputs[MODEL / 'marvin_pika_bimanual_mpd.urdf'] = xml_bytes(planning)
    native = packages['marvin_description'] / 'config'
    spheres, pairs = convert_collision(native / 'curobo/marvin.yml')
    original_links = [name for name in spheres if name != 'self_collision']
    envelopes = {}
    tools = []
    for link in planning.findall('link'):
        name = link.get('name')
        if not name.startswith(('left_', 'right_')) or link.find('collision') is None:
            continue
        points = []
        for collision in link.findall('collision'):
            mesh = collision.find('geometry/mesh')
            if mesh is None:
                raise ValueError(f'Expected mesh collision geometry on {name}')
            vertices = stl_vertices(outputs[MODEL / mesh.get('filename')])
            vertices *= np.array(list(map(float, mesh.get('scale', '1 1 1').split())))
            origin = collision.find('origin')
            if origin is not None:
                vertices = vertices @ rpy_rotation(origin.get('rpy', '0 0 0')).T
                vertices += np.array(list(map(float, origin.get('xyz', '0 0 0').split())))
            points.append(vertices)
        vertices = np.concatenate(points)
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        if name in frozen:
            entry = frozen[name]
            delta1 = np.array(entry['axis']) * (entry['lower']-entry['q'])
            delta2 = np.array(entry['axis']) * (entry['upper']-entry['q'])
            lower += np.minimum(delta1, delta2)
            upper += np.maximum(delta1, delta2)
        spheres[name] = envelope_spheres(lower, upper)
        envelopes[name] = {'lower': lower.tolist(), 'upper': upper.tolist()}
        tools.append(name)
    # New tools checked against all articulated body links except their rigid
    # mounting link (Link7). Same-side parts form one fixed assembly and are
    # deliberately not tested against one another; all cross-arm pairs remain.
    exclusions = []
    for i, tool in enumerate(tools):
        suffix = 'L' if tool.startswith('left_') else 'R'
        peers = []
        for other in original_links + tools[i+1:]:
            if other == f'Link7_{suffix}' or other.startswith(tool.split('_')[0]+'_'):
                exclusions.append([tool, other])
            else:
                peers.append(other)
        pairs[tool] = peers
    spheres['self_collision'] = pairs
    outputs[CONFIG / 'collision_spheres.yaml'] = yaml_bytes(spheres)
    outputs[CONFIG / 'self_collision_pairs.yaml'] = yaml_bytes(
        {'pairs': [[a, b] for a, peers in pairs.items() for b in peers]})
    outputs[CONFIG / 'joint_limits.yaml'] = yaml_bytes(convert_joint_limits(native / 'joint_limits.yaml'))
    bounds = {}
    for name, entries in spheres.items():
        if name == 'self_collision':
            continue
        arr = np.array(entries)
        center = ((arr[:, :3]-arr[:, 3:]).min(axis=0)+(arr[:, :3]+arr[:, 3:]).max(axis=0))/2
        radius = np.max(np.linalg.norm(arr[:, :3]-center, axis=1)+arr[:, 3])+1e-6
        bounds[name] = [{'center': center.tolist(), 'radius': float(radius),
                         'source_sphere_indices': list(range(len(entries)))}]
    outputs[CONFIG / 'collision_parent_bounds.yaml'] = yaml_bytes({
        'metadata': {'method': 'aabb_center_exact_sphere_cover', 'safety_padding': 1e-6},
        'parent_bounds': bounds})
    # Include migrated license texts in the same integrity gate.
    license_dir = ROOT / MODEL / 'licenses'
    for name in ('Apache-2.0.txt', 'pika_ros-BSD-3-Clause.txt', 'SOURCES.md'):
        if not (license_dir / name).is_file():
            raise FileNotFoundError(license_dir / name)
    for path in sorted(license_dir.glob('*')):
        if path.is_file():
            outputs[MODEL / 'licenses' / path.name] = path.read_bytes()
    manifest = {
        'schema': 'marvin_pika_assets/v1', 'packages': provenance, 'mesh_sources': origins,
        'joint_names': JOINTS, 'ee_links': ['left_pika_gripper_tcp', 'right_pika_gripper_tcp'],
        'frozen_fingers': frozen, 'tcp_calibrated': False,
        'collision': {'method': 'full_aabb_grid_circumspheres', 'cell_size_m': 0.025,
                      'finger_travel_envelope': True, 'link_envelopes': envelopes,
                      'excluded_rigid_mount_pairs': exclusions},
        'files': {path.as_posix(): sha(data) for path, data in sorted(outputs.items())},
    }
    manifest['asset_sha256'] = sha(json.dumps(manifest['files'], sort_keys=True).encode())
    outputs[MODEL / 'pika_assets.lock.yaml'] = yaml_bytes(manifest)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ros-root', type=Path, required=True)
    parser.add_argument('--finger-travel', type=float, default=0.045,
                        help='Single-finger travel in metres; visual pose, not total opening width')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    outputs = build(args.ros_root.resolve(), args.finger_travel)
    changed = [str(p) for p, data in outputs.items() if not (ROOT/p).is_file() or (ROOT/p).read_bytes() != data]
    if args.check:
        if changed:
            raise SystemExit('ROS/MPD asset drift:\n'+'\n'.join(changed))
        print(f'PASS: {len(outputs)} managed files match current ROS source and generated geometry')
        return
    # Nothing is published until expansion, limits, mesh loading and generation succeed.
    for relative, data in outputs.items():
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(f'Generated {len(outputs)} managed files ({len(changed)} changed)')


if __name__ == '__main__':
    main()
