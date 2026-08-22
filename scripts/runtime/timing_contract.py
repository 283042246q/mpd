"""Phase-5 candidate-specific timing artifact contract.

Phase-4 schema v2 keeps one shared ``time_from_start`` array.  Phase-5 schema
v3 requires one explicit array per returned candidate and never infers timing
from horizon or a nominal duration.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.runtime.runtime_engine import PlanArtifacts


TIMING_SCHEMA_VERSION = 1
VARIABLE_TIMING_TRAJECTORY_SCHEMA_VERSION = 3
TIMING_MODES = {
    "phase5_scalar_duration",
    "phase5_timing_only",
    "phase5_joint",
}


class TimingContractError(ValueError):
    """Candidate timing or artifact metadata violates the Phase-5 schema."""


def validate_candidate_times(
    candidate_times: Any,
    *,
    expected_candidates: int,
    expected_horizon: int,
    duration_min: float,
    duration_max: float,
) -> np.ndarray:
    times = np.asarray(candidate_times, dtype=np.float64)
    expected_shape = (int(expected_candidates), int(expected_horizon))
    if times.shape != expected_shape:
        raise TimingContractError(
            f"candidate time arrays must have shape {expected_shape}, got {times.shape}"
        )
    if not np.isfinite(times).all():
        raise TimingContractError("candidate time arrays contain NaN or Inf")
    if not np.allclose(times[:, 0], 0.0, rtol=0.0, atol=1e-8):
        raise TimingContractError("every candidate time array must begin at zero")
    if not (np.diff(times, axis=-1) > 0.0).all():
        raise TimingContractError("candidate time arrays must be strictly increasing")
    durations = times[:, -1]
    if (durations < duration_min - 1e-8).any() or (durations > duration_max + 1e-8).any():
        raise TimingContractError(
            f"candidate durations must remain in [{duration_min}, {duration_max}] seconds"
        )
    return times


def attach_candidate_timing(
    artifacts: PlanArtifacts,
    *,
    topk_time_from_start: Any,
    timing_mode: str,
    duration_min: float,
    duration_max: float,
    timing_control_points: Any | None = None,
    best_trajectory_topk_index: int = 0,
) -> PlanArtifacts:
    """Upgrade mutable plan artifacts to trajectory schema v3."""

    if timing_mode not in TIMING_MODES:
        raise TimingContractError(f"unsupported Phase-5 timing mode {timing_mode!r}")
    topk_positions = np.asarray(artifacts.trajectory_arrays.get("topk_positions"))
    if topk_positions.ndim != 3:
        raise TimingContractError("topk_positions must have shape [candidate,time,joint]")
    candidate_count, horizon = topk_positions.shape[:2]
    times = validate_candidate_times(
        topk_time_from_start,
        expected_candidates=candidate_count,
        expected_horizon=horizon,
        duration_min=duration_min,
        duration_max=duration_max,
    )
    if not 0 <= best_trajectory_topk_index < candidate_count:
        raise TimingContractError("best_trajectory_topk_index is out of range")

    arrays = artifacts.trajectory_arrays
    arrays.update(
        artifact_schema_version=np.asarray(
            VARIABLE_TIMING_TRAJECTORY_SCHEMA_VERSION, dtype=np.int64
        ),
        timing_schema_version=np.asarray(TIMING_SCHEMA_VERSION, dtype=np.int64),
        best_trajectory_topk_index=np.asarray(best_trajectory_topk_index, dtype=np.int64),
        topk_time_from_start=times,
        time_from_start=times[best_trajectory_topk_index],
    )
    if timing_control_points is not None:
        control_points = np.asarray(timing_control_points, dtype=np.float64)
        if control_points.ndim != 2 or control_points.shape[0] != candidate_count:
            raise TimingContractError(
                "timing_control_points must have shape [candidate,timing_control_point]"
            )
        if not np.isfinite(control_points).all():
            raise TimingContractError("timing_control_points contain NaN or Inf")
        arrays["timing_control_points"] = control_points

    artifact_metadata = artifacts.result_payload.setdefault("trajectory_artifact", {})
    artifact_metadata.update(
        schema_version=VARIABLE_TIMING_TRAJECTORY_SCHEMA_VERSION,
        timing_schema_version=TIMING_SCHEMA_VERSION,
        candidate_specific_time=True,
        best_trajectory_topk_index=best_trajectory_topk_index,
    )
    trajectory_metadata = artifacts.result_payload.setdefault("trajectory", {})
    trajectory_metadata.update(
        duration_s=float(times[best_trajectory_topk_index, -1]),
        duration_min_s=float(duration_min),
        duration_max_s=float(duration_max),
        timing_mode=timing_mode,
        timing_schema_version=TIMING_SCHEMA_VERSION,
    )
    artifacts.result_payload["top_k_timing"] = {
        "durations_s": times[:, -1].tolist(),
        "candidate_specific_time": True,
    }
    return artifacts
