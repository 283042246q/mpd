import numpy as np
import pytest
from scipy.interpolate import BSpline
import torch

from mpd.datasets.spacetime_legacy import open_uniform_knots
from mpd.inference.learned_timing import LearnedTimingCodec, load_timing_checkpoint, adapt_timing_path_basis
from mpd.timing_training.model import TimingDenoiser, TimingDiffusion
from mpd.parametric_trajectory.normalized_timing import NormalizedTimingSplineNumpy, minimum_duration_for_path
from mpd.parametric_trajectory.timing_fitting import TimingSplineNumpy, decode_timing_latent


def codec(representation):
    return LearnedTimingCodec(dict(representation=representation, path_mean=[0., 0.],
        path_std=[1., 1.], target_mean=[0.] * 6, target_std=[1.] * 6),
        velocity_limits=[2., 2.], acceleration_limits=[4., 4.]).double()


@pytest.mark.parametrize("representation", ["c", "tau_r"])
def test_codec_matches_teacher_and_has_path_and_timing_gradients(representation):
    c = codec(representation)
    points = np.linspace(0., 1., 12)[:, None] * np.array([[1., .7]])
    spline = BSpline(open_uniform_knots(12, 5), points, 5)
    q, qs, qss = [torch.tensor(spline.derivative(k)(c.phase.numpy()))[None] for k in range(3)]
    qs.requires_grad_()
    latent = torch.tensor([[3., .4, -.2, .7, .1, .5]], dtype=torch.float64, requires_grad=True)
    ev, tmin = c.evaluate(latent, q, qs, qss)
    if representation == "c":
        ref = TimingSplineNumpy().evaluate(decode_timing_latent(latent.detach().numpy()[0]))
        expected = ref.time_from_start
    else:
        ref = NormalizedTimingSplineNumpy().evaluate(latent.detach().numpy()[0, 1:])
        minimum = minimum_duration_for_path(spline, ref, dq_max=np.array([2.,2.]),
            ddq_max=np.array([4.,4.]), duration_floor=2.)
        assert tmin.item() == pytest.approx(minimum.value, rel=1e-5)
        duration = minimum.value + (14. - minimum.value) / (1. + np.exp(-3.))
        expected = duration * ref.normalized_time
    np.testing.assert_allclose(ev.time_from_start.detach().numpy()[0], expected, rtol=2e-6, atol=2e-6)
    assert torch.all(torch.diff(ev.time_from_start) > 0)
    gradients = torch.autograd.grad(ev.dq.square().sum() + ev.duration.sum(), (qs, latent))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)


def test_infeasible_shape_is_not_falsely_clamped_to_deadline():
    c = codec("tau_r")
    q = torch.zeros(1, 128, 2, dtype=torch.float64)
    ev, minimum = c.evaluate(torch.zeros(1, 6, dtype=torch.float64), q, q + 100., q)
    assert minimum.item() > 14.
    assert ev.duration.item() > 14.


def test_nan_checkpoint_fails_before_inference(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save(dict(checkpoint_version="timing_diffusion_checkpoint_v1", representation="tau_r",
        ema_model_state_dict={"weight": torch.tensor(float("nan"))}), path)
    with pytest.raises(ValueError, match="NaN/Inf"):
        load_timing_checkpoint(path)


def test_explicit_basis_adaptation_is_exact_and_does_not_change_weights():
    model = TimingDiffusion(TimingDenoiser(dof=2, num_spatial_control_points=29,
        spatial_degree=5, path_width=8, path_embedding_dim=8, hidden_dim=8,
        time_embedding_dim=8, num_residual_blocks=1)).double()
    parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
    adapt_timing_path_basis(model, 21)
    encoder = model.denoiser.path_encoder
    points = np.random.default_rng(7).normal(size=(21, 2))
    spline = BSpline(open_uniform_knots(21, 5), points, 5)
    for matrix, order in ((encoder.basis, 0), (encoder.basis_d1, 1), (encoder.basis_d2, 2)):
        predicted = matrix @ torch.tensor(points)
        np.testing.assert_allclose(predicted.numpy(), spline.derivative(order)(np.linspace(0, 1, 64)), atol=1e-10)
    assert encoder(torch.tensor(points)[None]).shape == (1, 8)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, parameters[name], rtol=0., atol=0.)


def test_old_spatial_unet_without_horizon_attribute_keeps_output():
    from mpd.models.diffusion_models.models import TemporalUnet
    model = TemporalUnet(n_support_points=16, state_dim=2, unet_input_dim=8,
        dim_mults=(1, 2), conditioning_type="default", conditioning_embed_dim=8).eval()
    x, time, context = torch.randn(2, 16, 2), torch.tensor([4, 5]), torch.randn(2, 8)
    with torch.no_grad():
        expected = model(x, time, context)
        del model.horizon_multiple
        actual = model(x, time, context)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
