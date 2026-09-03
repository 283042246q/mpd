import torch

from mpd.models.timing_diffusion import (
    SpatialPathEncoder,
    TimingDenoiser,
    TimingDiffusion,
    cosine_beta_schedule,
)


def _small_denoiser():
    return TimingDenoiser(
        dof=2,
        num_spatial_control_points=12,
        spatial_degree=5,
        num_phase_points=16,
        path_width=16,
        path_embedding_dim=24,
        hidden_dim=32,
        time_embedding_dim=16,
        num_residual_blocks=2,
    )


def test_cosine_schedule_is_finite_and_in_range():
    betas = cosine_beta_schedule(20)

    assert betas.shape == (20,)
    assert torch.all(torch.isfinite(betas))
    assert torch.all((betas > 0.0) & (betas < 1.0))


def test_path_encoder_keeps_batch_and_embedding_dimensions():
    encoder = SpatialPathEncoder(
        dof=2,
        num_control_points=12,
        degree=5,
        num_phase_points=16,
        width=16,
        embedding_dim=24,
    )
    path = torch.randn(3, 12, 2)

    encoded = encoder(path)

    assert encoded.shape == (3, 24)
    assert torch.all(torch.isfinite(encoded))


def test_training_loss_backpropagates_through_path_conditioned_denoiser():
    torch.manual_seed(7)
    model = TimingDiffusion(_small_denoiser(), num_diffusion_steps=10)
    path = torch.randn(4, 12, 2)
    clean = torch.randn(4, 6)
    timesteps = torch.tensor([0, 2, 5, 9], dtype=torch.long)
    noise = torch.randn_like(clean)

    loss, metrics = model.training_loss(
        path, clean, timesteps=timesteps, noise=noise
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert metrics["noise_mse_per_dimension"].shape == (6,)
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0.0)
        for parameter in model.parameters()
    )


def test_q_sample_and_ancestral_sample_shapes():
    torch.manual_seed(11)
    model = TimingDiffusion(_small_denoiser(), num_diffusion_steps=4)
    clean = torch.randn(2, 6)
    noise = torch.randn_like(clean)
    noisy = model.q_sample(clean, torch.tensor([0, 3]), noise)

    assert noisy.shape == clean.shape
    samples = model.sample(torch.randn(2, 12, 2), num_candidates=3)
    assert samples.shape == (2, 3, 6)
    assert torch.all(torch.isfinite(samples))
