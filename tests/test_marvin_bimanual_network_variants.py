import io

import pytest
import torch
import torch.nn as nn

from mpd.models import CoupledBimanualTemporalUnet, TemporalUnet, UNET_DIM_MULTS
from mpd.models.diffusion_models.diffusion_model_base import GaussianDiffusionModel
from mpd.models.diffusion_models.context_models import (
    ContextModelMarvinCrossArmEE,
    ContextModelMarvinCrossArmTokens,
    ContextModelMarvinDualEE,
    ContextModelMarvinStructuredEE,
)
from scripts.train.train_marvin_bimanual import results_dir_for_variant, validate_config


def _context_inputs(batch_size=2):
    return dict(
        qs_normalized=torch.randn(batch_size, 14, requires_grad=True),
        ee_goal_orientation_normalized=torch.randn(batch_size, 2, 9, requires_grad=True),
        ee_goal_position_normalized=torch.randn(batch_size, 2, 3, requires_grad=True),
        active_ee_mask=torch.tensor([[1.0, 1.0], [1.0, 0.0]])[:batch_size],
    )


@pytest.mark.parametrize(
    "model_class,expected_dim",
    [
        (ContextModelMarvinDualEE, 32),
        (ContextModelMarvinStructuredEE, 32),
        (ContextModelMarvinCrossArmEE, 32),
        (ContextModelMarvinCrossArmTokens, 96),
    ],
)
def test_bimanual_context_variant_shapes(model_class, expected_dim):
    model = model_class(out_dim=32, n_layers=1)
    output = model(**_context_inputs())
    assert output.shape == (2, expected_dim)
    assert model.out_dim == expected_dim
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize(
    "model_class",
    [ContextModelMarvinStructuredEE, ContextModelMarvinCrossArmEE, ContextModelMarvinCrossArmTokens],
)
def test_structured_context_ignores_pose_in_inactive_slot(model_class):
    torch.manual_seed(4)
    model = model_class(out_dim=32, n_layers=1).eval()
    inputs = _context_inputs()
    original = model(**inputs)[1]
    changed_inputs = dict(inputs)
    changed_inputs["ee_goal_orientation_normalized"] = inputs[
        "ee_goal_orientation_normalized"
    ].detach().clone()
    changed_inputs["ee_goal_position_normalized"] = inputs["ee_goal_position_normalized"].detach().clone()
    changed_inputs["ee_goal_orientation_normalized"][1, 1] += 100.0
    changed_inputs["ee_goal_position_normalized"][1, 1] -= 100.0
    changed = model(**changed_inputs)[1]
    assert torch.equal(original, changed)


def test_ablation_context_attention_boundary():
    model_b = ContextModelMarvinStructuredEE(out_dim=32, n_layers=1)
    model_c = ContextModelMarvinCrossArmEE(out_dim=32, n_layers=1, attention_layers=2)
    assert not any(isinstance(module, nn.MultiheadAttention) for module in model_b.modules())
    assert len(model_c.cross_arm_attention.layers) == 2


def test_variant_d_17_point_backward_has_explicit_cross_arm_coupling():
    torch.manual_seed(5)
    context_model = ContextModelMarvinCrossArmTokens(out_dim=32, n_layers=1)
    context_inputs = _context_inputs()
    context = context_model(**context_inputs)
    model = CoupledBimanualTemporalUnet(
        state_dim=14,
        n_support_points=17,
        unet_input_dim=8,
        dim_mults=UNET_DIM_MULTS[0],
        conditioning_type="default",
        conditioning_embed_dim=96,
        context_token_dim=32,
        attention_num_heads=4,
    )
    x = torch.randn(2, 17, 14, requires_grad=True)
    output = model(x, torch.tensor([1, 40]), context)
    assert output.shape == x.shape
    # A left-arm output must have a gradient path through the right-arm stream.
    output[..., :7].square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[..., :7].abs().sum() > 0
    assert x.grad[..., 7:].abs().sum() > 0
    assert context_inputs["qs_normalized"].grad[:, :7].abs().sum() > 0
    assert context_inputs["qs_normalized"].grad[:, 7:].abs().sum() > 0


def test_variant_d_runs_through_existing_gaussian_diffusion_wrapper():
    context_model = ContextModelMarvinCrossArmTokens(out_dim=32, n_layers=1)
    denoiser = CoupledBimanualTemporalUnet(
        state_dim=14,
        n_support_points=17,
        unet_input_dim=8,
        dim_mults=UNET_DIM_MULTS[0],
        conditioning_type="default",
        conditioning_embed_dim=context_model.out_dim,
        context_token_dim=32,
        attention_num_heads=4,
    )
    diffusion = GaussianDiffusionModel(
        denoise_fn=denoiser,
        context_model=context_model,
        n_diffusion_steps=4,
        predict_epsilon=True,
    )
    x = torch.randn(2, 17, 14)
    loss, _ = diffusion.loss(x, _context_inputs(), {})
    assert torch.isfinite(loss)
    loss.backward()
    diffusion.warmup(x.shape, device="cpu")
    checkpoint = io.BytesIO()
    torch.save(diffusion, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, map_location="cpu")
    restored.warmup(x.shape, device="cpu")
    assert isinstance(restored.model.module, CoupledBimanualTemporalUnet)


def test_joint_unet_contract_stays_available_for_variants_a_to_c():
    model = TemporalUnet(
        state_dim=14,
        n_support_points=17,
        unet_input_dim=8,
        dim_mults=UNET_DIM_MULTS[0],
        conditioning_type="default",
        conditioning_embed_dim=32,
    )
    x = torch.randn(2, 17, 14)
    output = model(x, torch.tensor([1, 40]), torch.randn(2, 32))
    assert output.shape == x.shape


def test_variant_config_validation_and_checkpoint_directory_separation():
    base = dict(
        robot_model="marvin_bimanual",
        task_family="independent",
        state_dim=14,
        context_qs=True,
        context_q_dim=14,
        context_ee_goal_pose=True,
        context_ee_goal_pose_bimanual=True,
        raw_context_dim=40,
    )
    for variant in "ABCD":
        assert validate_config(dict(base, bimanual_network_variant=variant))["bimanual_network_variant"] == variant
        assert results_dir_for_variant("logs/run_variant_A", variant) == f"logs/run_variant_{variant}"
    with pytest.raises(ValueError, match="one of A"):
        validate_config(dict(base, bimanual_network_variant="E"))
