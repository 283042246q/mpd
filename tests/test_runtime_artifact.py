import zipfile

import numpy as np

from scripts.runtime.infer_dynamic_server import _build_parser
from scripts.runtime.infer_space_time_server import _build_parser as _build_space_time_parser
from scripts.runtime.infer_once import _atomic_write_npz


def test_uncompressed_npz_uses_stored_zip_entries(tmp_path):
    path = tmp_path / "trajectory.npz"
    _atomic_write_npz(
        path,
        compressed=False,
        collision_spheres=np.zeros((8, 128, 56, 3), dtype=np.float32),
    )
    with zipfile.ZipFile(path) as archive:
        assert {item.compress_type for item in archive.infolist()} == {
            zipfile.ZIP_STORED
        }


def test_dynamic_artifact_optimizations_are_independently_switchable():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--socket",
            "/tmp/test.sock",
            "--output-root",
            "/tmp/test-output",
            "--no-capacity-buckets",
            "--shape-grouping",
            "--no-time-table-cache",
            "--fused-reduction",
            "--no-dynamic-guide-pruning",
            "--trajectory-schema-version",
            "2",
            "--trajectory-compression",
            "zlib",
            "--no-collision-spheres-float32",
            "--no-deduplicate-best-trajectory",
        ]
    )
    assert not args.capacity_buckets
    assert args.shape_grouping
    assert not args.time_table_cache
    assert args.fused_reduction
    assert not args.dynamic_guide_pruning
    assert args.trajectory_schema_version == 2
    assert args.trajectory_compression == "zlib"
    assert not args.collision_spheres_float32
    assert not args.deduplicate_best_trajectory


def test_phase5_server_has_separate_mode_and_timing_bounds():
    parser = _build_space_time_parser()
    args = parser.parse_args(
        [
            "--socket",
            "/tmp/phase5.sock",
            "--output-root",
            "/tmp/phase5-output",
            "--timing-mode",
            "phase5_timing_only",
            "--duration-min",
            "7.0",
            "--duration-max",
            "13.0",
        ]
    )
    assert args.timing_mode == "phase5_timing_only"
    assert args.duration_min == 7.0
    assert args.duration_max == 13.0
