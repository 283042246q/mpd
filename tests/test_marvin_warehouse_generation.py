from collections import Counter
from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml

from scripts.generate_data.generate_marvin_warehouse_bimanual import (
    DEFAULT_CONFIG,
    MarvinWarehouseGenerator,
    task_spec,
    validate_config,
)
from scripts.train.train_marvin_bimanual import validate_config as validate_train


def test_shard_schedule_preserves_both_ratios():
    for start in (0, 10, 500):
        specs = [task_spec(i) for i in range(start, start + 500)]
        assert Counter(d for _, d in specs) == {"random_to_placement": 250, "placement_to_placement": 250}
        for direction in ("random_to_placement", "placement_to_placement"):
            assert Counter(m for m, d in specs if d == direction) == {
                "dual_independent": 150,
                "left_only": 50,
                "right_only": 50,
            }


def test_region_schema_rejects_missing_orientation_and_invalid_bounds():
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    validate_config(config)
    broken = deepcopy(config)
    del broken["placement_regions"]["left_table"]["rotation"]
    with pytest.raises(ValueError):
        validate_config(broken)
    broken = deepcopy(config)
    broken["placement_regions"]["right_cabinet"]["translation"]["z"] = [[0.4, 0.1]]
    with pytest.raises(ValueError):
        validate_config(broken)


def test_production_planner_defaults_use_benchmark_winner_without_simplifier():
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    assert config["state_validity_resolution"] == 0.002
    assert config["planner_range"] == 0.35
    assert config["simplify_path"] is False
    assert config["pre_rrt_filter"] == "none"
    assert config["worker_lifetime_trajectories"] == 10
    assert config["max_worker_restarts_per_shard"] == 3
    validate_config(config)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("state_validity_resolution", 0.0, "state_validity_resolution"),
        ("planner_range", 0.0, "planner_range"),
        ("simplify_path", "false", "simplify_path"),
    ],
)
def test_invalid_production_planner_settings_are_rejected(key, value, message):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    config[key] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


@pytest.mark.parametrize(
    "value",
    ["bad", "clearance", "sparse_line"],
)
def test_invalid_pre_rrt_filter_is_rejected(value):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    config["pre_rrt_filter"] = value
    with pytest.raises(ValueError, match="pre_rrt_filter"):
        validate_config(config)


def test_pre_rrt_filters_apply_clearance_before_sparse_line():
    generator = MarvinWarehouseGenerator.__new__(MarvinWarehouseGenerator)
    generator.stats = Counter()
    generator.config = {
        "pre_rrt_filter": "endpoint_clearance",
        "pre_rrt_endpoint_environment_clearance": 0.005,
        "pre_rrt_endpoint_self_clearance": 0.0015,
        "pre_rrt_sparse_line_min_valid_fraction": 0.5,
    }
    calls = []

    def diagnostics(q_start, q_goal, include_sparse_line):
        calls.append(include_sparse_line)
        return {"endpoint_environment_clearance": 0.004, "endpoint_self_clearance": 0.02}

    generator.pre_rrt_diagnostics = diagnostics
    accepted, result = generator.pre_rrt_accept(np.zeros(14), np.ones(14))
    assert not accepted and result["pre_rrt_rejection_reason"] == "endpoint_environment_clearance"
    assert calls == [False]

    generator.config["pre_rrt_filter"] = "endpoint_clearance_and_sparse_line"
    generator.pre_rrt_diagnostics = lambda *args, **kwargs: {
        "endpoint_environment_clearance": 0.02,
        "endpoint_self_clearance": 0.02,
        "sparse_line_valid_fraction": 4 / 9,
    }
    accepted, result = generator.pre_rrt_accept(np.zeros(14), np.ones(14))
    assert not accepted and result["pre_rrt_rejection_reason"] == "sparse_line"


def test_pose_filter_rejects_wrong_orientation_even_inside_region():
    from scipy.spatial.transform import Rotation
    from types import SimpleNamespace

    generator = MarvinWarehouseGenerator.__new__(MarvinWarehouseGenerator)
    generator.regions = yaml.safe_load(DEFAULT_CONFIG.read_text())["placement_regions"]
    region = generator.regions["right_table"]
    center = np.array([np.mean(region["translation"][a]) for a in "xyz"])
    rotation = np.array(region["rotation"]["base"])
    generator._pose = lambda q, arm: SimpleNamespace(translation=center, rotation=rotation)
    assert generator.pose_in_region(np.zeros(14), "right", "right_table")
    rotation = rotation @ Rotation.from_euler("x", 30, degrees=True).as_matrix()
    assert not generator.pose_in_region(np.zeros(14), "right", "right_table")


def test_dense_path_filter_checks_between_waypoints():
    generator = MarvinWarehouseGenerator.__new__(MarvinWarehouseGenerator)
    generator.config = {"collision_max_joint_step": 0.025}
    generator._torch_state_valid = lambda q: not np.any((q[:, 0] > 0.45) & (q[:, 0] < 0.55))
    # Endpoint-only checking would accept this segment through a thin obstacle.
    assert not generator.path_valid(np.stack([np.zeros(14), np.ones(14)]))


def test_failed_rrt_discards_pair_and_samples_new_endpoints():
    generator = MarvinWarehouseGenerator.__new__(MarvinWarehouseGenerator)
    generator.config = {"max_attempts_per_trajectory": 3}
    generator.stats = Counter()
    sampled, planned = [], []

    def sample(mode, direction):
        start = np.full(14, len(sampled), dtype=float)
        sampled.append(start)
        return start, start + 1, {}, {}

    def plan(start, goal, mode):
        planned.append(start.copy())
        return None if len(planned) == 1 else np.stack([start, goal])

    generator._sample_task = sample
    generator.plan_once = plan
    generator.validated_spline = lambda path: (np.zeros(28), np.zeros((14, 22)), 5)
    generator.dual_ee_goal_pose = lambda q_goal: np.concatenate(
        (np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)), np.zeros((2, 3, 1), dtype=np.float32)),
        axis=-1,
    )
    paths, metadata = generator.generate(1)
    assert len(paths) == 1 and len(planned) == 2
    assert not np.array_equal(planned[0], planned[1])


def test_bimanual_rejects_single_tcp_training_context_and_accepts_dual_slots():
    with pytest.raises(ValueError, match="dual-slot"):
        validate_train(dict(robot_model="marvin_bimanual", task_family="independent", context_ee_goal_pose=True))
    config = validate_train(
        dict(
            robot_model="marvin_bimanual",
            task_family="independent",
            context_qs=True,
            context_ee_goal_pose=True,
            context_ee_goal_pose_bimanual=True,
            state_dim=14,
            context_q_dim=14,
            raw_context_dim=40,
        )
    )
    assert config["raw_context_dim"] == 40


def test_dual_slot_context_and_17_point_unet_backward():
    from mpd.models import TemporalUnet, UNET_DIM_MULTS
    from mpd.models.diffusion_models.context_models import ContextModelMarvinDualEE

    context_model = ContextModelMarvinDualEE(out_dim=64, n_layers=1)
    q_start = torch.randn(2, 14, requires_grad=True)
    orientation = torch.randn(2, 2, 9, requires_grad=True)
    position = torch.randn(2, 2, 3, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    raw_context = context_model.build_raw_context(q_start, orientation, position, mask)
    assert raw_context.shape == (2, 40)
    assert torch.equal(raw_context[:, :14], q_start)
    assert torch.equal(raw_context[:, -2:], mask)
    context = context_model(
        qs_normalized=q_start,
        ee_goal_orientation_normalized=orientation,
        ee_goal_position_normalized=position,
        active_ee_mask=mask,
    )
    model = TemporalUnet(
        state_dim=14,
        n_support_points=17,
        unet_input_dim=32,
        dim_mults=UNET_DIM_MULTS[1],
        conditioning_type="default",
        conditioning_embed_dim=64,
    )
    x = torch.randn(2, 17, 14, requires_grad=True)
    output = model(x, torch.tensor([1, 40]), context)
    assert output.shape == x.shape
    output.square().mean().backward()
    assert all(torch.isfinite(value.grad).all() for value in (x, q_start, orientation, position))


def test_coupled_14d_denoiser_backward_reaches_both_arms():
    from mpd.models import TemporalUnet, UNET_DIM_MULTS
    from mpd.models.diffusion_models.context_models import ContextModelQs

    torch.manual_seed(1)
    context = ContextModelQs(in_dim=28, out_dim=128, n_layers=2)
    model = TemporalUnet(
        state_dim=14,
        n_support_points=16,
        unet_input_dim=32,
        dim_mults=UNET_DIM_MULTS[1],
        conditioning_type="default",
        conditioning_embed_dim=128,
    )
    x = torch.randn(2, 16, 14, requires_grad=True)
    qs = torch.randn(2, 28, requires_grad=True)
    output = model(x, torch.tensor([1, 40]), context(qs))
    assert output.shape == x.shape
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(qs.grad).all()
    assert x.grad[..., :7].abs().sum() > 0 and x.grad[..., 7:].abs().sum() > 0
    assert qs.grad[:, :14].abs().sum() > 0 and qs.grad[:, 14:].abs().sum() > 0


def test_small_dataset_training_epoch_budget_counts_partial_batches():
    from mpd.trainer.trainer import get_num_epochs

    assert get_num_epochs(2, 128, 9) == 2
    assert get_num_epochs(100, 128, 257) == 34


def test_fast_sphere_centers_match_full_canonical_fk():
    from torch_robotics.robots import RobotMarvinBimanual

    robot = RobotMarvinBimanual(tensor_args={"device": "cpu", "dtype": torch.float32})
    generator = MarvinWarehouseGenerator.__new__(MarvinWarehouseGenerator)
    generator.torch_robot = robot
    torch.manual_seed(7)
    q = robot.q_pos_min + torch.rand(16, 14) * (robot.q_pos_max - robot.q_pos_min)
    assert torch.allclose(generator.collision_positions(q), robot.fk_map_collision(q), atol=5e-6)


def test_random_state_samples_both_arms_without_zero_fallback():
    from types import SimpleNamespace

    generator = MarvinWarehouseGenerator.__new__(MarvinWarehouseGenerator)
    generator.config = {"state_sample_tries": 10}
    generator.rng = np.random.default_rng(1)
    generator.robot = SimpleNamespace(joint_bounds_low_np=np.ones(14), joint_bounds_high_np=np.full(14, 2.0))
    generator.stats = Counter()
    generator.deadline = float("inf")
    generator._pose = lambda q, arm: SimpleNamespace(translation=np.zeros(3))
    checked = []
    generator.valid = lambda q: checked.append(q.copy()) or True
    q = generator._random_valid_state()
    assert np.all(q >= 1) and np.all(q <= 2)
    assert len(checked) == 2 and generator.stats["random_candidates"] == 2
    generator.valid = lambda q: False
    assert generator._random_valid_state() is None


def test_shard_resume_and_merge_keep_global_ids_and_detect_corruption(tmp_path):
    from scripts.generate_data.generate_marvin_warehouse_bimanual import _write_dataset
    from scripts.generate_data.launch_generate_marvin_warehouse_bimanual import (
        merge_shards,
        quarantine_incomplete_shard,
        shard_complete,
    )
    import h5py

    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    shards = []
    for start in (0, 10):
        shard = tmp_path / "shards" / f"{start:09d}"
        metadata = []
        for i in range(start, start + 10):
            mode, direction = task_spec(i)
            item = dict(
                task_id=i,
                task_mode=mode,
                direction=direction,
                q_start=np.zeros(14),
                q_goal=np.zeros(14),
                planning_time=0.0,
                bspline=(np.zeros(28), np.zeros((14, 22)), 5),
                ee_goal_pose=np.concatenate(
                    (
                        np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
                        np.zeros((2, 3, 1), dtype=np.float32),
                    ),
                    axis=-1,
                ),
            )
            for arm in ("left", "right"):
                item[f"source_region_{arm}"] = "random"
                item[f"goal_region_{arm}"] = f"{arm}_table"
            metadata.append(item)
        # Synthetic serialization fixtures, not physically validated paths.
        _write_dataset(shard, config, [np.zeros((128, 14))] * 10, metadata, start)
        assert shard_complete(shard, config, start, 10)
        assert not shard_complete(shard, config, start, 20)
        shards.append(shard)
    merge_shards(tmp_path, shards[::-1], config)
    with h5py.File(tmp_path / "dataset_merged.hdf5", "r") as data:
        assert np.array_equal(data["task_id"][:], np.arange(20))
        assert data["bspline_params_cc"].shape == (20, 14, 22)
        assert data["ee_goal_pose"].shape == (20, 2, 3, 4)
        assert data["active_ee_mask"].shape == (20, 2)
        assert Counter(data["direction"].asstr()[:]) == {"random_to_placement": 10, "placement_to_placement": 10}
    merged_manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert merged_manifest["region_counts"]["goal_region_left"] == {"left_table": 20}
    assert merged_manifest["region_counts"]["goal_region_right"] == {"right_table": 20}
    with pytest.raises(FileExistsError):
        merge_shards(tmp_path, shards, config)
    with h5py.File(shards[0] / "dataset_merged.hdf5", "r+") as data:
        data["sol_path"][0, 0, 0] = 0.5
    assert not shard_complete(shards[0], config, 0, 10)
    quarantine = quarantine_incomplete_shard(shards[0])
    assert quarantine.is_dir() and not shards[0].exists()

    empty = tmp_path / "shards" / "000000020"
    empty.mkdir()
    (empty / "manifest.yaml").write_text("")
    (empty / "generation_config.yaml").write_text("")
    assert not shard_complete(empty, config, 20, 10)


def test_shard_layout_uses_workers_without_breaking_ten_task_quotas():
    from scripts.generate_data.launch_generate_marvin_warehouse_bimanual import build_shards

    assert build_shards(20, 10, 3) == [(0, 10), (10, 10)]
    assert build_shards(1000, 500, 3) == [(0, 340), (340, 330), (670, 330)]
    assert build_shards(1500, 500, 3) == [(0, 500), (500, 500), (1000, 500)]
    for start, count in build_shards(1010, 500, 3):
        assert start % 10 == 0 and count % 10 == 0 and 0 < count <= 500


def test_shard_scheduler_rolls_a_fresh_process_without_waiting_for_other_slots(tmp_path):
    from scripts.generate_data.launch_generate_marvin_warehouse_bimanual import run_shards_resilient

    events = []
    completed = set()
    completion_order = iter((10, 0, 20, 30))

    class FakeFuture:
        def __init__(self, start):
            self.start = start

        def result(self):
            events.append(("finish", self.start))
            completed.add(self.start)
            return str(tmp_path / "shards" / f"{self.start:09d}")

    class FakeExecutor:
        def __init__(self, max_workers, mp_context):
            assert max_workers == 1
            assert mp_context.get_start_method() == "spawn"
            events.append(("new_executor", None))

        def submit(self, function, config, root, start, count):
            assert count == 10
            events.append(("start", start))
            return FakeFuture(start)

        def shutdown(self, wait):
            assert wait is True
            events.append(("shutdown", None))

    def fake_wait(futures, return_when):
        start = next(completion_order)
        future = next(future for future in futures if future.start == start)
        return {future}, set(futures).difference({future})

    def fake_shard_complete(path, config, start, count):
        return start in completed

    shards = [(0, 10), (10, 10), (20, 10), (30, 10)]
    paths = run_shards_resilient(
        {},
        tmp_path,
        shards,
        workers=3,
        max_restarts=0,
        _executor_factory=FakeExecutor,
        _wait=fake_wait,
        _shard_complete=fake_shard_complete,
        _run_shard=lambda *args: None,
    )

    # Shard 10 finishes first and shard 30 starts while shards 0 and 20 are
    # still in flight.  A wave/barrier scheduler would finish both first.
    assert events.index(("start", 30)) < events.index(("finish", 0))
    assert [event for event in events if event[0] == "new_executor"] == [
        ("new_executor", None)
    ] * len(shards)
    assert paths == [str(tmp_path / "shards" / f"{start:09d}") for start, _ in shards]
