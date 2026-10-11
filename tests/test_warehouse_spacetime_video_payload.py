import json

import numpy as np
import pytest

from scripts.isaaclab.warehouse_spacetime_payload import frame_times, interpolate, load_case, validate_suite


def test_replay_uses_seconds_not_waypoint_index():
    times = np.array([0., .2, 1.])
    values = np.array([[0., 0.], [2., 4.], [10., 20.]])
    np.testing.assert_allclose(interpolate(times, values, .6), [6., 12.])
    np.testing.assert_allclose(interpolate(times, values, 1.), [10., 20.])
    with pytest.raises(ValueError, match="outside"):
        interpolate(times, values, 1.1)


def test_video_preserves_terminal_pose_without_forecast_extrapolation():
    times = frame_times(1.05, 10)
    assert times[-1] == 1.05
    assert np.all(np.diff(times) > 0)
    assert len(times) == 12


def test_payload_rejects_out_of_execution_obstacle_events(tmp_path):
    path = tmp_path / "case.npz"
    values = dict(execution_time=np.array([0., 2.]), forecast_time=np.array([0., 3.]),
                  q=np.zeros((2, 7)), shape=np.array([0, 1, 2]),
                  size=np.array([[.05, 0., 0.], [.05, .06, .07], [.05, .09, 0.]]),
                  obstacle_positions=np.zeros((2, 3, 3)), event_times=np.array([.5, 1., 1.5]),
                  metadata_json=np.asarray(json.dumps({"mode": "linear"})))
    np.savez(path, **values)
    assert load_case(path)["metadata"]["mode"] == "linear"
    values["event_times"][-1] = 2.5
    np.savez(path, **values)
    with pytest.raises(ValueError, match="during arm execution"):
        load_case(path)


def test_partial_or_duplicate_suites_cannot_claim_all_variants():
    suite = {"schema": "warehouse_spacetime_video_suite_v1", "modes": ["linear"],
             "variants_per_mode": 2, "expected_cases": 2,
             "cases": [{"case_id": "linear-01", "mode": "linear", "variant": 1, "signature": "a"},
                       {"case_id": "linear-02", "mode": "linear", "variant": 2, "signature": "b"}]}
    validate_suite(suite)
    suite["cases"][1]["signature"] = "a"
    with pytest.raises(ValueError, match="duplicate worlds"):
        validate_suite(suite)
    suite["cases"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        validate_suite(suite)
