import numpy as np

from contextlib import contextmanager


@contextmanager
def _raises(error_type, match):
    message = match
    try:
        yield
    except error_type as error:
        assert message in str(error)
    else:
        raise AssertionError(f"Expected {error_type.__name__}")


from scripts.runtime.infer_once import (
    EXPECTED_JOINT_NAMES,
    RequestValidationError,
    _pose_xyzw_to_transform,
    validate_request,
)


DEFAULT_POSE = [
    0.4322542996381046,
    0.16375043690143717,
    0.6717085498613047,
    0.8765521159636589,
    0.47117624388215046,
    0.06455630619812767,
    -0.07403930395941108,
]
DEFAULT_Q = [0.2, -0.3, 0.1, -1.8, 0.2, 1.6, 0.1]


def _base_request():
    return {
        "schema_version": 1,
        "request_id": "test-request",
        "joint_names": list(EXPECTED_JOINT_NAMES),
        "q_pos_start": [0.0, -0.7, 0.0, -2.2, 0.0, 1.5, 0.7],
        "scene_id": "EnvWarehouseExtraObjectsV00",
        "seed": 123,
    }


def test_cartesian_is_default_goal_type():
    request = _base_request()
    request["ee_pose_goal"] = DEFAULT_POSE

    validated = validate_request(request)

    assert validated["goal_type"] == "cartesian"
    np.testing.assert_allclose(validated["ee_pose_goal"], DEFAULT_POSE)
    assert validated["q_pos_goal"] is None
    assert validated["ik_candidates"] == 0
    assert validated["ik_max_iters"] == 300


def test_explicit_joint_goal_type():
    request = _base_request()
    request.update(goal_type="joint", q_pos_goal=DEFAULT_Q)

    validated = validate_request(request)

    assert validated["goal_type"] == "joint"
    np.testing.assert_allclose(validated["q_pos_goal"], DEFAULT_Q)
    assert validated["ee_pose_goal"] is None


def test_legacy_q_only_request_is_joint_goal():
    request = _base_request()
    request["q_pos_goal"] = DEFAULT_Q

    assert validate_request(request)["goal_type"] == "joint"


def test_rejects_ambiguous_goal_fields():
    request = _base_request()
    request.update(
        goal_type="cartesian",
        ee_pose_goal=DEFAULT_POSE,
        q_pos_goal=DEFAULT_Q,
    )

    with _raises(RequestValidationError, match="q_pos_goal cannot"):
        validate_request(request)


def test_rejects_non_unit_quaternion():
    request = _base_request()
    request["ee_pose_goal"] = [0.4, 0.1, 0.6, 0.0, 0.0, 0.0, 0.5]

    with _raises(RequestValidationError, match="unit norm"):
        validate_request(request)


def test_accepts_custom_ik_settings():
    request = _base_request()
    request.update(
        ee_pose_goal=DEFAULT_POSE,
        ik_candidates=16,
        ik_max_iters=200,
    )

    validated = validate_request(request)

    assert validated["ik_candidates"] == 16
    assert validated["ik_max_iters"] == 200


def test_rejects_invalid_ik_settings():
    request = _base_request()
    request.update(ee_pose_goal=DEFAULT_POSE, ik_candidates=-1)

    with _raises(RequestValidationError, match="ik_candidates must be in"):
        validate_request(request)


def test_pose_xyzw_to_transform_identity_orientation():
    transform = _pose_xyzw_to_transform(np.asarray([0.4, 0.1, 0.6, 0.0, 0.0, 0.0, 1.0]))

    np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(transform[:3, 3], [0.4, 0.1, 0.6], atol=1e-12)
    np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} runtime contract tests passed")
