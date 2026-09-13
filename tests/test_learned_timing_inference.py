import numpy as np
import pytest
from scipy.interpolate import BSpline
import torch

from mpd.datasets.spacetime_legacy import open_uniform_knots
from mpd.inference.learned_timing import LearnedTimingCodec, load_timing_checkpoint
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
