from pathlib import Path

import numpy as np
import yaml

from mpd.datasets.spacetime_schema import (
    SCHEMA_VERSION,
    build_manifest,
    load_panda_robot_bundle,
    materialize_dataset_contract,
    sha256_file,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_panda_bundle_uses_urdf_joint_order_and_runtime_limits():
    bundle = load_panda_robot_bundle(REPOSITORY_ROOT)

    assert bundle.active_joint_names == tuple(f"panda_joint{i}" for i in range(1, 8))
    assert bundle.dof == 7
    np.testing.assert_allclose(bundle.dq_max, [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
    np.testing.assert_allclose(bundle.ddq_max, [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
    assert np.all(bundle.q_min < bundle.q_max)


def test_materialized_contract_contains_hash_stable_robot_bundle(tmp_path):
    bundle = load_panda_robot_bundle(REPOSITORY_ROOT)
    manifest = build_manifest(
        bundle,
        spatial_num_control_points=29,
        spatial_degree=5,
        spatial_num_phase_points=128,
        timing_num_control_points=8,
        timing_degree=3,
        timing_num_phase_points=128,
        timing_u_min=0.05,
        timing_duration_min=2.0,
        timing_duration_max=15.0,
        source_dataset="warehouse.hdf5",
        variants=[{"name": "toppra"}],
    )

    materialize_dataset_contract(tmp_path, bundle, manifest)

    loaded = yaml.safe_load((tmp_path / "manifest.yaml").read_text(encoding="utf-8"))
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["spatial_spline"]["num_control_points"] == 29
    assert loaded["robot"]["active_joint_names"] == list(bundle.active_joint_names)
    assert sha256_file(tmp_path / "robot" / "robot.urdf") == loaded["robot"]["urdf_sha256"]
