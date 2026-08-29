import torch

from scripts.runtime.dynamic_runtime_engine import _phase4_aligned_selection_score


def test_phase4_aligned_selection_uses_spatial_and_normalized_dynamic_risk():
    spatial = torch.tensor([0.2, 0.1], dtype=torch.float64)
    dynamic = torch.tensor([0.0, 2.0], dtype=torch.float64)

    score, components = _phase4_aligned_selection_score(spatial, dynamic)

    torch.testing.assert_close(
        score,
        torch.tensor([0.2, 0.3], dtype=torch.float64),
    )
    torch.testing.assert_close(
        components["normalized_dynamic_risk"],
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )


def test_phase4_aligned_tied_dynamic_risk_preserves_spatial_order():
    spatial = torch.tensor([0.4, 0.1], dtype=torch.float64)
    dynamic = torch.ones(2, dtype=torch.float64)

    score, components = _phase4_aligned_selection_score(spatial, dynamic)

    assert score[1] < score[0]
    assert not torch.count_nonzero(components["normalized_dynamic_risk"]).item()
