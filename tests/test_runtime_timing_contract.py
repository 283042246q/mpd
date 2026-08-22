import numpy as np
import pytest

from scripts.runtime.runtime_engine import PlanArtifacts
from scripts.runtime.timing_contract import (
    TIMING_SCHEMA_VERSION,
    TimingContractError,
    attach_candidate_timing,
    validate_candidate_times,
)


def _artifacts():
    return PlanArtifacts(
        result_payload={
            "trajectory": {"duration_s": 10.0},
            "trajectory_artifact": {
                "schema_version": 2,
                "best_trajectory_topk_index": 0,
            },
        },
        trajectory_arrays={
            "topk_positions": np.zeros((2, 4, 7), dtype=np.float64),
            "time_from_start": np.linspace(0.0, 10.0, 4),
        },
    )


def test_attach_candidate_timing_upgrades_artifact_without_inference():
    artifacts = _artifacts()
    times = np.asarray([[0.0, 1.0, 3.0, 6.0], [0.0, 2.0, 5.0, 9.0]])
    control_points = np.zeros((2, 8), dtype=np.float64)

    result = attach_candidate_timing(
        artifacts,
        topk_time_from_start=times,
        timing_mode="phase5_joint",
        duration_min=2.0,
        duration_max=12.0,
        timing_control_points=control_points,
        best_trajectory_topk_index=1,
    )

    assert int(result.trajectory_arrays["artifact_schema_version"]) == 3
    assert int(result.trajectory_arrays["timing_schema_version"]) == TIMING_SCHEMA_VERSION
    np.testing.assert_array_equal(result.trajectory_arrays["topk_time_from_start"], times)
    np.testing.assert_array_equal(result.trajectory_arrays["time_from_start"], times[1])
    assert result.result_payload["trajectory"]["duration_s"] == 9.0
    assert result.result_payload["trajectory"]["timing_mode"] == "phase5_joint"
    assert result.result_payload["trajectory_artifact"]["candidate_specific_time"] is True
    assert result.result_payload["top_k_timing"]["durations_s"] == [6.0, 9.0]


@pytest.mark.parametrize(
    "times, message",
    [
        (np.zeros((1, 4)), "shape"),
        (np.asarray([[0.1, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]]), "begin"),
        (np.asarray([[0.0, 1.0, 1.0, 3.0], [0.0, 1.0, 2.0, 3.0]]), "strictly"),
        (np.asarray([[0.0, 1.0, 2.0, 20.0], [0.0, 1.0, 2.0, 3.0]]), "durations"),
    ],
)
def test_candidate_timing_contract_rejects_implicit_or_invalid_time(times, message):
    with pytest.raises(TimingContractError, match=message):
        validate_candidate_times(
            times,
            expected_candidates=2,
            expected_horizon=4,
            duration_min=2.0,
            duration_max=12.0,
        )


def test_phase4_artifact_remains_unchanged_until_explicit_upgrade():
    artifacts = _artifacts()
    assert "topk_time_from_start" not in artifacts.trajectory_arrays
    assert artifacts.result_payload["trajectory_artifact"]["schema_version"] == 2
