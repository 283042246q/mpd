import torch

from scripts.runtime.space_time_runtime_engine import _phase5_selection_score


def test_phase5_selection_score_uses_requested_normalized_weights():
    spatial = torch.tensor([0.2, 0.1], dtype=torch.float64)
    dynamic = torch.tensor([0.0, 2.0], dtype=torch.float64)
    duration = torch.tensor([10.0, 6.0], dtype=torch.float64)
    smoothness = torch.tensor([4.0, 0.0], dtype=torch.float64)

    score, components = _phase5_selection_score(
        spatial,
        dynamic,
        duration,
        smoothness,
        duration_min=6.0,
        duration_max=14.0,
    )

    torch.testing.assert_close(
        score,
        torch.tensor([0.35, 0.3], dtype=torch.float64),
    )
    torch.testing.assert_close(
        components["normalized_dynamic_risk"],
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        components["normalized_duration"],
        torch.tensor([0.5, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        components["normalized_timing_smoothness"],
        torch.tensor([1.0, 0.0], dtype=torch.float64),
    )


def test_phase5_selection_score_preserves_spatial_ranking_when_risks_tie():
    spatial = torch.tensor([0.1, 0.4], dtype=torch.float64)
    tied = torch.ones(2, dtype=torch.float64)

    score, components = _phase5_selection_score(
        spatial,
        tied,
        torch.full((2,), 10.0, dtype=torch.float64),
        tied,
        duration_min=6.0,
        duration_max=14.0,
    )

    assert score[0] < score[1]
    assert torch.count_nonzero(components["normalized_dynamic_risk"]) == 0
    assert torch.count_nonzero(components["normalized_timing_smoothness"]) == 0
