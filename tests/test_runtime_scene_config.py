from types import SimpleNamespace

import pytest

from scripts.runtime.infer_once import ConfigurationError, _validate_runtime_config


class RuntimeConfig(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)


def _config(scene_id="EnvWarehouseExtraObjectsV00", **overrides):
    values = {
        "start_goal_source": "states_file",
        "model_selection": "bspline",
        "planner_alg": "mpd",
        "diffusion_sampling_method": "ddim",
        "planning_env_id": None,
        "env_id_replace": scene_id,
        "num_T_pts": 128,
        "n_trajectory_samples": 100,
        "trajectory_duration": 10.0,
        "runtime": {
            "schema_version": 1,
            "scene_id": scene_id,
            "joint_names": [f"fr3_joint{index}" for index in range(1, 8)],
        },
    }
    values.update(overrides)
    return RuntimeConfig(**values)


def test_runtime_scene_can_be_selected_explicitly():
    _validate_runtime_config(
        _config(
            "EnvOpenDrawerShelf",
            planning_env_id="EnvOpenDrawerShelf",
            env_id_replace=None,
        )
    )


def test_runtime_scene_must_match_planning_environment():
    with pytest.raises(ConfigurationError, match="must match"):
        _validate_runtime_config(
            _config(
                "EnvOpenDrawerShelf",
                planning_env_id="EnvOpenDrawerShelf",
                env_id_replace=None,
                runtime={
                    "schema_version": 1,
                    "scene_id": "EnvWarehouseExtraObjectsV00",
                    "joint_names": [f"fr3_joint{index}" for index in range(1, 8)],
                },
            )
        )


def test_runtime_rejects_ambiguous_environment_selection():
    with pytest.raises(ConfigurationError, match="cannot both"):
        _validate_runtime_config(
            _config(
                "EnvOpenDrawerShelf",
                planning_env_id="EnvOpenDrawerShelf",
            )
        )
