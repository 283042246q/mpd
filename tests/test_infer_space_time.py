import json
from pathlib import Path

import numpy as np

from scripts.inference.infer_space_time import _build_parser, run
from scripts.runtime.runtime_engine import PlanArtifacts


class FakeEngine:
    instances = []

    def __init__(self, **options):
        self.options = options
        self.world = None
        self.requests = []
        self.instances.append(self)

    def update_world(self, world):
        self.world = world
        return world["world_version"]

    def health(self):
        return {
            "space_time": {
                "enabled": True,
                "mode": self.options["timing_mode"],
                "trajectory_schema_version": 3,
                "timing_schema_version": 1,
                "candidate_specific_time": True,
            }
        }

    def plan(self, request):
        self.requests.append(request)
        times = np.asarray([[0.0, 5.0, 10.0], [0.0, 4.0, 9.0]])
        zeros = np.zeros((2, 3, 7), dtype=np.float64)
        return PlanArtifacts(
            result_payload={
                "schema_version": 1,
                "status": "success",
                "trajectory": {
                    "duration_s": float(times[0, -1]),
                    "minimum_environment_clearance_m": 0.2,
                    "minimum_self_clearance_m": 0.1,
                },
                "candidates": {
                    "generated": 2,
                    "dense_checked": 2,
                    "dense_complete": True,
                    "valid": 2,
                },
                "best_trajectory_diagnostics": {"path_length": 1.5},
                "timing": {"guide_sec": 0.1, "dense_validation_sec": 0.2},
            },
            trajectory_arrays={
                "artifact_schema_version": np.asarray(3, dtype=np.int64),
                "timing_schema_version": np.asarray(1, dtype=np.int64),
                "topk_time_from_start": times,
                "topk_positions": zeros,
                "topk_velocities": zeros.copy(),
                "topk_accelerations": zeros.copy(),
            },
        )


def _request(path: Path):
    path.write_text(
        json.dumps({"request_id": "standalone", "seed": 10}), encoding="utf-8"
    )


def test_parser_exposes_all_phase5_modes_and_repeat_controls(tmp_path):
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--request",
            str(tmp_path / "request.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--timing-mode",
            "phase5_timing_only",
            "--repeats",
            "3",
            "--seed-step",
            "2",
        ]
    )
    assert args.timing_mode == "phase5_timing_only"
    assert args.repeats == 3
    assert args.seed_step == 2
    assert args.static_spatial_pruning
    assert not args.dynamic_space_time_pruning


def test_standalone_runner_reuses_engine_and_exports_each_repeat(tmp_path):
    request_path = tmp_path / "request.json"
    _request(request_path)
    output_dir = tmp_path / "output"
    args = _build_parser().parse_args(
        [
            "--request",
            str(request_path),
            "--output-dir",
            str(output_dir),
            "--timing-mode",
            "phase5_joint",
            "--repeats",
            "2",
            "--seed-step",
            "3",
        ]
    )

    summary_path = run(args, engine_factory=FakeEngine)

    engine = FakeEngine.instances[-1]
    assert engine.options["static_spatial_pruning_enabled"]
    assert not engine.options["dynamic_space_time_pruning_enabled"]
    assert engine.world["objects"] == []
    assert [request["seed"] for request in engine.requests] == [10, 13]
    assert all(request["_dynamic_world_version"] == 1 for request in engine.requests)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["timing_mode"] == "phase5_joint"
    assert summary["repeats"] == 2
    assert summary["aggregate"]["duration_s"]["mean"] == 10.0
    for index in range(2):
        run_dir = output_dir / f"run-{index:04d}"
        assert (run_dir / "result.json").is_file()
        assert (run_dir / "trajectory.npz").is_file()
        with np.load(run_dir / "trajectory.npz", allow_pickle=False) as data:
            assert int(data["artifact_schema_version"]) == 3
            assert data["topk_time_from_start"].shape == (2, 3)


def test_standalone_runner_accepts_embedded_dynamic_world(tmp_path):
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "request_id": "embedded-world",
                "seed": 10,
                "dynamic_world": {
                    "world_version": 7,
                    "frame_id": "fr3_link0",
                    "stamp_unix_ns": 1,
                    "valid_until_unix_ns": 20_000_000_001,
                    "objects": [
                        {
                            "id": "moving-box",
                            "local_sdf": {
                                "type": "box",
                                "size_xyz": [0.16, 0.12, 0.18],
                            },
                            "pose": {
                                "position": [0.4, 0.4, 0.38],
                                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                            },
                            "linear_velocity": [-0.045, 0.0, 0.0],
                            "inflation": {
                                "mode": "linear",
                                "base_m": 0.03,
                                "horizon_rate_m_s": 0.02,
                            },
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    args = _build_parser().parse_args(
        ["--request", str(request_path), "--output-dir", str(output_dir)]
    )

    summary_path = run(args, engine_factory=FakeEngine)

    engine = FakeEngine.instances[-1]
    assert engine.world["world_version"] == 7
    assert engine.world["objects"][0]["id"] == "moving-box"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["world_source"] == "request.dynamic_world"
    assert summary["dynamic_object_count"] == 1
