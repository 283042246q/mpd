import unittest

import torch

from mpd.models.diffusion_models.diffusion_model_base import GaussianDiffusionModel
from mpd.models.diffusion_models.sample_functions import guide_gradient_steps


class _FakeModel:
    def predict_x_recon(self, x, t, context_d):
        return x + 1.0


class CleanXReconGuidanceTest(unittest.TestCase):
    def test_guidance_updates_clean_prediction_iteratively(self):
        seen = []

        def guide(x, **kwargs):
            seen.append(x.detach().clone())
            return -0.5 * torch.ones_like(x)

        x = torch.zeros(1, 3, 2)
        result = guide_gradient_steps(
            x,
            t=torch.zeros(1, dtype=torch.long),
            model=_FakeModel(),
            hard_conds={},
            context_d={},
            guide=guide,
            guide_lr=1.0,
            n_guide_steps=2,
            max_perturb_x=0.75,
            compute_costs_with_xrecon=True,
        )

        self.assertEqual(len(seen), 2)
        torch.testing.assert_close(seen[0], torch.ones_like(x))
        torch.testing.assert_close(seen[1], torch.full_like(x, 0.5))
        # The second update would reach zero, but the trust region keeps it
        # within 0.75 of the original clean prediction.
        torch.testing.assert_close(result, torch.full_like(x, 0.25))


class _ConstantDenoiser(torch.nn.Module):
    state_dim = 1

    def forward(self, x, t, context):
        return torch.full_like(x, 0.25)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by the model DataParallel wrapper")
class CleanXReconDdimTest(unittest.TestCase):
    def test_zero_prior_weight_does_not_discard_clean_prediction(self):
        model = GaussianDiffusionModel(
            denoise_fn=_ConstantDenoiser(),
            n_diffusion_steps=4,
            predict_epsilon=False,
        ).cuda()
        seen = []

        def guide(x, **kwargs):
            seen.append(x.detach().clone())
            return torch.zeros_like(x)

        model.ddim_sample_loop(
            (1, 3, 1),
            {},
            guide=guide,
            ddim_sampling_timesteps=2,
            t_start_guide=2,
            n_guide_steps=1,
            compute_costs_with_xrecon=True,
            ddim_scale_grad_prior=0.0,
        )

        self.assertTrue(seen)
        for value in seen:
            torch.testing.assert_close(value, torch.full_like(value, 0.25))

    def test_a1_matches_unguided_update_for_epsilon_model_and_zero_cost_gradient(self):
        model = GaussianDiffusionModel(
            denoise_fn=_ConstantDenoiser(),
            n_diffusion_steps=4,
            predict_epsilon=True,
        ).cuda()

        def zero_guide(x, **kwargs):
            return torch.zeros_like(x)

        common = {
            "ddim_sampling_timesteps": 2,
            "t_start_guide": 2,
            "n_guide_steps": 1,
            "ddim_eta": 0.0,
            "ddim_scale_grad_prior": 1.0,
            "max_perturb_x": 100.0,
        }
        torch.manual_seed(123)
        legacy, = model.ddim_sample_loop(
            (1, 3, 1),
            {},
            guide=zero_guide,
            compute_costs_with_xrecon=False,
            **common,
        )
        torch.manual_seed(123)
        clean_a1, = model.ddim_sample_loop(
            (1, 3, 1),
            {},
            guide=zero_guide,
            compute_costs_with_xrecon=True,
            **common,
        )

        # With a zero cost gradient, A1 must be exactly the normal DDIM update:
        # guided x0 is unchanged and the original denoiser epsilon is retained.
        torch.testing.assert_close(clean_a1, legacy, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
