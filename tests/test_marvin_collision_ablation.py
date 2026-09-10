from types import SimpleNamespace

from scripts.inference.benchmark_marvin_collision_optimizations import (
    CASES,
    _classify,
    build_case_config,
)


def _args():
    return SimpleNamespace(
        pair_chunk_size=2048,
        validator_candidate_chunk_size=2,
        validator_time_chunk_size=8,
        validator_pair_chunk_size=1024,
        parent_environment_margin=0.2,
        parent_self_margin=0.1,
    )


def test_ablation_cases_have_independent_expected_switches():
    base = {
        "n_trajectory_samples": 32,
        "runtime_top_k_valid_trajectories": 8,
        "collision_optimization": {},
        "dense_validation": {},
        "gradient_pruning": {"spatial": {}},
    }
    for case_id, expected in CASES.items():
        config = build_case_config(base, case_id, 4, _args())
        actual = (
            config["collision_optimization"]["pair_streaming"]["enabled"],
            config["dense_validation"]["chunking"]["enabled"],
            config["gradient_pruning"]["spatial"]["link_broad_phase"]["enabled"],
            config["collision_optimization"]["reduced_guide_geometry"]["enabled"],
        )
        assert actual == expected
        assert config["n_trajectory_samples"] == 4
        assert config["runtime_top_k_valid_trajectories"] == 4
    assert base["n_trajectory_samples"] == 32


def test_ablation_classifies_cuda_oom_from_structured_or_process_output():
    assert (
        _classify(
            5,
            {"status": "fault", "error": {"message": "CUDA out of memory"}},
            "",
            "",
        )
        == "cuda_oom"
    )
    assert _classify(0, {"status": "success"}, "", "") == "success"
    assert _classify(5, {}, "", "failure") == "process_error"
