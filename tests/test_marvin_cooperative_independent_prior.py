import json
from pathlib import Path

import torch
import yaml
from types import SimpleNamespace

from mpd.bimanual.cost_guide import (
    CostTaskSpaceCooperativeClosure,
    CostTaskSpaceObjectGoalPose,
    CostTaskSpacePayloadCollision,
)
from mpd.bimanual.cooperative_sources import request_from_config_source
from mpd.bimanual.runtime_contract import BimanualRequest
from mpd.models.diffusion_models.diffusion_model_base import GaussianDiffusionModel
from scripts.inference import inference_marvin_bimanual, inference_marvin_cooperative


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-cooperative-independent-prior.yaml"


class ZeroDenoiser(torch.nn.Module):
    state_dim = 2

    def forward(self, x, time, context=None):
        return torch.zeros_like(x)


class ZeroContext(torch.nn.Module):
    out_dim = 1

    def forward(self, **kwargs):
        batch = next(iter(kwargs.values())).shape[0]
        return torch.zeros(batch, 1)


def test_reference_warm_start_uses_a_shorter_reverse_schedule():
    model = GaussianDiffusionModel(
        denoise_fn=ZeroDenoiser(),
        n_diffusion_steps=20,
        predict_epsilon=True,
        context_model=ZeroContext(),
    )
    initial = torch.zeros(1, 5, 2)
    chain = model.run_inference(
        context_d={"value": torch.zeros(1, 1)},
        hard_conds={},
        n_samples=3,
        horizon=5,
        return_chain=True,
        method="ddim",
        ddim_sampling_timesteps=10,
        ddim_eta=0.0,
        initial_x=initial,
        initial_noise_timestep=7,
    )
    assert chain.shape[1:] == (3, 5, 2)
    assert 2 <= chain.shape[0] < 11


def test_cooperative_runtime_config_keeps_independent_checkpoint_contract():
    config = yaml.safe_load(CONFIG.read_text())
    inference_marvin_bimanual._validate_runtime_config(config)
    assert config["task_family"] == "independent"
    assert config["task_mode"] == "cooperative_rigid"
    assert config["cooperative_inference"]["prior_mode"] in {
        "direct_project",
        "reference_residual",
    }
    assert isinstance(
        config["cooperative_inference"]["closed_chain_projection"], bool
    )
    assert "CostTaskSpacePayloadCollision" in config["costs"]
    assert "CostTaskSpaceCooperativeClosure" in config["costs"]
    assert "CostTaskSpaceObjectGoalPosition" in config["costs"]


def test_cooperative_costs_return_path_wide_14d_gradients():
    class Field:
        def object_signed_distances(self, points, get_gradient=False):
            values = torch.zeros(*points.shape[:-2], 1, points.shape[-2])
            gradients = torch.ones(*points.shape[:-2], 1, points.shape[-2], 3)
            return (values, gradients) if get_gradient else values

    left_grasp = torch.eye(4)
    right_grasp = torch.eye(4)
    left_grasp[1, 3] = 0.12
    right_grasp[1, 3] = -0.12
    task = SimpleNamespace(
        object_to_left_grasp=left_grasp,
        object_to_right_grasp=right_grasp,
        object_goal_pose=torch.eye(4)[:3],
        active_ee_mask=torch.ones(2),
        tensor_args={"device": torch.device("cpu"), "dtype": torch.float32},
        parametric_trajectory=None,
        robot=SimpleNamespace(task_space_dim=3),
        env=None,
        get_collision_objects_field=lambda: Field(),
    )
    poses = torch.eye(4).view(1, 1, 1, 4, 4).repeat(2, 5, 2, 1, 1)
    poses[..., 0, :3, 3] = left_grasp[:3, 3]
    poses[..., 1, :3, 3] = right_grasp[:3, 3]
    poses = poses[..., :3, :]
    jacobians = torch.randn(2, 5, 2, 6, 14)
    zeros = torch.zeros(2, 5, 14)

    closure = CostTaskSpaceCooperativeClosure(task)
    closure_cost, closure_grad = closure.compute_cost_grad_wrt_q(
        None, zeros, zeros, zeros, None, None, poses, jacobians
    )
    assert closure_cost.shape == (2, 5)
    assert closure_grad["pos"].shape == (2, 5, 14)
    assert torch.allclose(closure_cost, torch.zeros_like(closure_cost), atol=1e-5)

    object_goal = CostTaskSpaceObjectGoalPose(task)
    goal_cost, goal_grad = object_goal.compute_cost_grad_wrt_q(
        None, zeros, zeros, zeros, None, None, poses, jacobians
    )
    assert goal_cost.shape == (2, 5)
    assert goal_grad["pos"].shape == (2, 5, 14)
    assert torch.allclose(goal_cost, torch.zeros_like(goal_cost), atol=1e-5)

    payload = CostTaskSpacePayloadCollision(task)
    payload_cost, payload_grad = payload.compute_cost_grad_wrt_q(
        None, zeros, zeros, zeros, None, None, poses, jacobians
    )
    assert payload_cost.shape == (2, 5)
    assert payload_grad["pos"].shape == (2, 5, 14)


def test_fixed_seed_source_builds_a_strict_cooperative_request():
    request = request_from_config_source(
        CONFIG,
        source="seed_file",
        source_path=None,
        sample_index=1,
        seed=17,
        request_id="cooperative-seed-1",
    )
    parsed = BimanualRequest.from_dict(request)
    assert parsed.task_mode == "cooperative_rigid"
    assert parsed.grasp_profile == "default_box"
    assert parsed.object_goal_pose is not None
    assert parsed.left_goal_pose is not None and parsed.right_goal_pose is not None
    assert request["scene"]["start_goal_source"]["index"] == 1


def test_cooperative_entrypoint_dry_run_exposes_prior_and_projection_switches(
    tmp_path, capsys
):
    output = tmp_path / "artifact"
    result = inference_marvin_cooperative.main(
        [
            "--config",
            str(CONFIG),
            "--start-goal-source",
            "seed_file",
            "--sample-index",
            "0",
            "--seed",
            "5",
            "--output-dir",
            str(output),
            "--prior-mode",
            "direct_project",
            "--no-closed-chain-projection",
            "--dry-run",
        ]
    )
    assert result == 0
    report = yaml.safe_load(capsys.readouterr().out)
    assert report["prior_mode"] == "direct_project"
    assert report["closed_chain_projection"] is False
    request = json.loads((output / "request.json").read_text())
    assert request["task_mode"] == "cooperative_rigid"


def test_independent_runtime_config_and_entrypoint_remain_unchanged():
    independent = yaml.safe_load(
        (ROOT / "scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-independent-runtime.yaml").read_text()
    )
    inference_marvin_bimanual._validate_runtime_config(independent)
    assert independent["task_mode"] == "dual_independent"
    assert inference_marvin_bimanual.DEFAULT_CONFIG.name.endswith(
        "independent-runtime.yaml"
    )
