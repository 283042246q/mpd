"""Fail-fast metadata checks for Marvin dual-EE checkpoints."""

from __future__ import annotations

from collections.abc import Mapping


EXPECTED = {
    "robot_model": "marvin_bimanual",
    "task_family": "independent",
    "context_qs": True,
    "context_ee_goal_pose": True,
    "context_ee_goal_pose_bimanual": True,
    "state_dim": 14,
    "context_q_dim": 14,
    "raw_context_dim": 40,
    "parametric_trajectory_class": "ParametricTrajectoryBspline",
    "bspline_num_control_points_desired": 22,
    "bspline_num_control_points_exact": True,
}
NETWORK_VARIANTS = frozenset({"A", "B", "C", "D"})


def validate_checkpoint_args(
    args: Mapping,
    *,
    expected_dataset_subdir: str | None = None,
    expected_variant: str | None = None,
) -> dict:
    """Return normalized args or raise before a model is deserialized."""
    if not isinstance(args, Mapping):
        raise ValueError("checkpoint args must be a mapping")
    normalized = dict(args)
    for key, expected in EXPECTED.items():
        actual = normalized.get(key)
        if actual != expected:
            raise ValueError(
                f"checkpoint {key} must be {expected!r}, got {actual!r}"
            )
    variant = str(normalized.get("bimanual_network_variant", "A")).upper()
    if variant not in NETWORK_VARIANTS:
        raise ValueError("checkpoint bimanual_network_variant must be A, B, C, or D")
    if expected_variant is not None and variant != str(expected_variant).upper():
        raise ValueError(
            f"checkpoint variant {variant} does not match runtime variant "
            f"{str(expected_variant).upper()}"
        )
    normalized["bimanual_network_variant"] = variant
    if (
        expected_dataset_subdir is not None
        and normalized.get("dataset_subdir") != expected_dataset_subdir
    ):
        raise ValueError(
            "checkpoint dataset_subdir does not match the runtime configuration"
        )
    learnable = 17
    if int(normalized.get("n_learnable_control_points", learnable)) != learnable:
        raise ValueError("checkpoint must expose exactly 17 learnable control points")
    return normalized
