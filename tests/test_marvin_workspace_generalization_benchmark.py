from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import yaml

from scripts.inference.benchmark_marvin_workspace_generalization import (
    BUILTIN_CASES,
    DEFAULT_REGIONS,
    build_safe_mpd_config,
    expand_atomic_translation_regions,
    resolve_cases,
    _sample_mapping,
    summarize,
    _write_json,
)


def _regions():
    rotation = {
        "base": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "x": [[0, 0]],
        "y": [[0, 0]],
        "z": [[0, 0]],
    }
    regions = {}
    for side in ("left", "right"):
        sign = 1 if side == "left" else -1
        regions[f"{side}_table"] = {
            "translation": {
                "x": [[0.3, 0.8]],
                "y": [[0.0, 0.5]] if sign > 0 else [[-0.5, 0.0]],
                "z": [[-0.1, 0.1]],
            },
            "rotation": deepcopy(rotation),
        }
        regions[f"{side}_cabinet"] = {
            "translation": {
                "x": [[0.25, 0.75]],
                "y": [[0.62, 0.9]] if sign > 0 else [[-0.9, -0.62]],
                "z": [[0.1, 0.3], [0.45, 0.6]],
            },
            "rotation": deepcopy(rotation),
        }
    return regions


def test_atomic_region_expansion_separates_both_shelf_levels():
    expanded, metadata = expand_atomic_translation_regions(_regions())
    assert set(expanded) == {
        "left_table",
        "right_table",
        "left_cabinet__z0",
        "left_cabinet__z1",
        "right_cabinet__z0",
        "right_cabinet__z1",
    }
    assert expanded["left_cabinet__z0"]["translation"]["z"] == [[0.1, 0.3]]
    assert expanded["left_cabinet__z1"]["translation"]["z"] == [[0.45, 0.6]]
    cases = resolve_cases(BUILTIN_CASES, metadata)
    assert cases["table_to_shelf_upper"]["goal"]["left"] == "left_cabinet__z1"
    assert cases["left_cross_to_right_table"]["goal"]["left"] == "right_table"


def test_checked_in_generalization_regions_have_two_shelf_levels():
    config = yaml.safe_load(DEFAULT_REGIONS.read_text())
    expanded, metadata = expand_atomic_translation_regions(config["placement_regions"])
    assert len(expanded) == 6
    assert resolve_cases(BUILTIN_CASES, metadata)["shelf_lower_to_upper"]["goal"] == {
        "left": "left_cabinet__z1",
        "right": "right_cabinet__z1",
    }


def test_safe_mpd_config_enables_memory_controls_without_weakening_validator():
    base = yaml.safe_load(
        """
n_trajectory_samples: 64
runtime_top_k_valid_trajectories: 8
collision_optimization: {}
gradient_pruning: {spatial: {}}
dense_validation:
  check_environment: true
  check_self_collision: true
"""
    )
    configured = build_safe_mpd_config(base, 32)
    assert configured["n_trajectory_samples"] == 32
    assert configured["collision_optimization"]["pair_streaming"]["enabled"]
    assert configured["collision_optimization"]["reduced_guide_geometry"]["enabled"]
    assert configured["dense_validation"]["chunking"]["enabled"]
    assert configured["dense_validation"]["check_environment"]
    assert configured["dense_validation"]["check_self_collision"]


def test_endpoint_pool_rechecks_interarm_collision():
    generator = SimpleNamespace(
        rng=np.random.default_rng(7),
        endpoint_pool={"left:a": [np.ones(14)], "right:b": [np.full(14, 2.0)]},
        valid=lambda q: False,
    )
    assert _sample_mapping(generator, np.zeros(14), {"left": "a", "right": "b"}) is None
    generator.valid = lambda q: True
    np.testing.assert_array_equal(
        _sample_mapping(generator, np.zeros(14), {"left": "a", "right": "b"}),
        [1.0] * 7 + [2.0] * 7,
    )


def test_infrastructure_failure_is_not_planning_failure(tmp_path):
    manifest = {"pairs": [{"case": "test", "pair_index": 0}]}
    _write_json(tmp_path / "mpd/test/pair-000/benchmark-result.json", {
        "status": "fault", "wall_seconds": 1.0,
    })
    row = summarize(tmp_path, manifest)["rows"][0]
    assert row["mpd_success_rate"] is None
    assert row["mpd_status_counts"] == {"fault": 1}
