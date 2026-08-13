"""Configuration helpers for gradient pruning and independent validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


DEFAULT_GRADIENT_PRUNING_CONFIG = {
    # Python default is the cacheless B3 production path. YAML can still opt
    # into the exact legacy implementation with gradient_pruning.enabled=false.
    "enabled": True,
    "force_all_active": False,
    "profile": False,
    "profile_per_guide_call": False,
    "record_active_statistics": False,
    "endpoint": {
        "ee_only_last_point": True,
    },
    "candidate": {
        # Skip the expensive collision guide for candidates whose conservative
        # FK/SDF scan contains no risky phase.  This is intentionally separate
        # from temporal sparsity so both effects can be ablated independently.
        "enabled": False,
    },
    "preselection": {
        # Independent from link broad phase: candidate/temporal selection can
        # scan conservative parent bounds while exact guidance keeps fine spheres.
        "parent_bounds_scan": False,
    },
    "span_certificate": {
        # Experimental continuous-time B-spline certificate layered directly
        # on B3.  It uses the original fine collision spheres, performs no
        # parent-envelope scan, and never reuses scan poses or SDF values.
        "enabled": False,
        "max_subdivision_depth": 3,
        "environment_safe_margin": 0.08,
        "self_safe_margin": 0.06,
        # Component-wise derivative/Jacobian-column bounds are tighter than a
        # single Frobenius bound while remaining configuration independent.
        "jacobian_bound_mode": "componentwise",
        "exact_sdf_for_certificate": True,
        "grid_error_scale": 1.0,
        "profile_stages": False,
    },
    "temporal": {
        # B3 is the default pruned path: temporal sparsity stays off unless it
        # is explicitly requested, or the conditional gate below accepts it.
        "enabled": False,
        "conditional_enabled": False,
        "conditional_active_ratio_threshold": 0.35,
        "coarse_points": 32,
        "probe_midpoints": True,
        "coarse_scan": True,
        "reuse_selection_within_ddim_step": False,
        "buckets": [32, 64, 128],
        "environment_refine_margin": 0.08,
        "self_refine_margin": 0.06,
        "q_delta_threshold": None,
        "neighbor_dilation": 2,
        "always_keep_endpoints": True,
    },
    "spatial": {
        "parent_link_kinematics": True,
        "dense_parent_fast_path": True,
        # Pack only sphere/link entries with a non-zero task-space collision
        # gradient for the J^T g projection. FK and distance evaluation remain
        # conservative and unchanged.
        "active_link_pruning": True,
        # Optional second-stage pruning. C1 defaults to the original full-horizon
        # fine-sphere FK-only/SDF scan, then precise guidance evaluates J-FK/SDF
        # only for selected physical parents.
        "link_broad_phase": {
            "enabled": False,
            "full_scan": True,
            # parent_bounds remains available as a research comparison; C2 uses
            # it independently through preselection.parent_bounds_scan.
            "scan_geometry": "fine_spheres",
            # Reuse the full-scan TorchKin poses, fine-sphere centers, and SDF
            # values in the first precise C1 guide iteration.
            "reuse_scan_cache": False,
            "environment_margin": 0.20,
            "self_margin": 0.10,
        },
        "environment_link_broad_phase": False,
        "self_link_pair_broad_phase": False,
    },
    "mapping": {
        # Fuse the B-spline control-point mapping with phase integration.
        # The legacy materialized [B, H, K, D] path remains available as an
        # explicit equivalence fallback.
        "fused_bspline_integration": False,
        # Use the compact B-spline support (degree + 1 entries per phase) and
        # scatter directly into control-point gradients.
        "sparse_bspline_support": False,
    },
    "scheduling": {
        "enabled": False,
        "skip_safe_candidates": False,
        "promote_on_stalled_cost": True,
    },
}


DEFAULT_DENSE_VALIDATION_CONFIG = {
    "enabled": False,
    "runtime_points": 128,
    "benchmark_points": 128,
    "check_environment": True,
    "check_self_collision": True,
    "check_joint_limits": True,
    "check_joint_position": True,
    "check_joint_velocity": True,
    "check_joint_acceleration": True,
    "reject_invalid": True,
    "ranked_early_exit": {
        # Runtime-only latency optimization. Benchmark/evaluation configs keep
        # this disabled so validity statistics still cover every candidate.
        "enabled": False,
        # Progressive fixed GPU shapes: top-8, then 16, then 32, then the
        # remaining candidates in one or more 64-slot buffers.  A partial last
        # bucket is padded and excluded by a static slot mask.
        "batch_buckets": [8, 16, 32, 64],
        "preallocate_buffers": True,
        # CUDA Graph capture is intentionally opt-in because a replay is only
        # safe while the validator scene tensors keep stable addresses.
        "cuda_graph": False,
    },
}


def _as_plain_mapping(value: Any) -> dict:
    if value is None:
        return {}
    if hasattr(value, "toDict"):
        value = value.toDict()
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a mapping, got {type(value).__name__}.")
    return {
        key: _as_plain_mapping(item) if isinstance(item, Mapping) or hasattr(item, "toDict") else item
        for key, item in value.items()
    }


def _deep_merge(defaults: dict, override: Mapping[str, Any]) -> dict:
    result = deepcopy(defaults)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_gradient_pruning_config(args_inference) -> dict:
    """Resolve cacheless B3 defaults while preserving an explicit legacy switch."""

    root = _as_plain_mapping(args_inference)
    config = _deep_merge(DEFAULT_GRADIENT_PRUNING_CONFIG, _as_plain_mapping(root.get("gradient_pruning")))
    config["enabled"] = bool(config["enabled"])

    config["candidate"]["enabled"] = bool(config["candidate"]["enabled"])
    config["preselection"]["parent_bounds_scan"] = bool(
        config["preselection"]["parent_bounds_scan"]
    )
    span_certificate = config["span_certificate"]
    span_certificate["enabled"] = bool(span_certificate["enabled"])
    span_certificate["max_subdivision_depth"] = int(
        span_certificate["max_subdivision_depth"]
    )
    span_certificate["environment_safe_margin"] = float(
        span_certificate["environment_safe_margin"]
    )
    span_certificate["self_safe_margin"] = float(
        span_certificate["self_safe_margin"]
    )
    span_certificate["jacobian_bound_mode"] = str(
        span_certificate["jacobian_bound_mode"]
    )
    span_certificate["grid_error_scale"] = float(
        span_certificate["grid_error_scale"]
    )
    span_certificate["profile_stages"] = bool(
        span_certificate["profile_stages"]
    )
    span_certificate["exact_sdf_for_certificate"] = bool(
        span_certificate["exact_sdf_for_certificate"]
    )
    if span_certificate["max_subdivision_depth"] < 0:
        raise ValueError(
            "gradient_pruning.span_certificate.max_subdivision_depth cannot be negative."
        )
    if (
        span_certificate["environment_safe_margin"] < 0
        or span_certificate["self_safe_margin"] < 0
        or span_certificate["grid_error_scale"] < 0
    ):
        raise ValueError("gradient_pruning.span_certificate margins/scales cannot be negative.")
    if span_certificate["jacobian_bound_mode"] not in {"componentwise", "frobenius"}:
        raise ValueError(
            "gradient_pruning.span_certificate.jacobian_bound_mode must be "
            "'componentwise' or 'frobenius'."
        )

    temporal = config["temporal"]
    temporal["enabled"] = bool(temporal["enabled"])
    temporal["conditional_enabled"] = bool(temporal["conditional_enabled"])
    temporal["conditional_active_ratio_threshold"] = float(
        temporal["conditional_active_ratio_threshold"]
    )
    temporal["coarse_points"] = int(temporal["coarse_points"])
    temporal["neighbor_dilation"] = int(temporal["neighbor_dilation"])
    temporal["buckets"] = sorted({int(bucket) for bucket in temporal["buckets"]})
    if temporal["coarse_points"] < 2:
        raise ValueError("gradient_pruning.temporal.coarse_points must be at least 2.")
    if not temporal["buckets"] or any(bucket < 1 for bucket in temporal["buckets"]):
        raise ValueError("gradient_pruning.temporal.buckets must contain positive integers.")
    if temporal["neighbor_dilation"] < 0:
        raise ValueError("gradient_pruning.temporal.neighbor_dilation cannot be negative.")
    if not 0.0 <= temporal["conditional_active_ratio_threshold"] <= 1.0:
        raise ValueError(
            "gradient_pruning.temporal.conditional_active_ratio_threshold must be in [0, 1]."
        )

    config["mapping"]["fused_bspline_integration"] = bool(
        config["mapping"]["fused_bspline_integration"]
    )
    config["mapping"]["sparse_bspline_support"] = bool(
        config["mapping"]["sparse_bspline_support"]
    )
    config["spatial"]["active_link_pruning"] = bool(
        config["spatial"]["active_link_pruning"]
    )
    link_broad_phase = config["spatial"]["link_broad_phase"]
    link_broad_phase["enabled"] = bool(link_broad_phase["enabled"])
    link_broad_phase["full_scan"] = bool(link_broad_phase["full_scan"])
    link_broad_phase["scan_geometry"] = str(link_broad_phase["scan_geometry"])
    link_broad_phase["reuse_scan_cache"] = bool(
        link_broad_phase["reuse_scan_cache"]
    )
    if link_broad_phase["scan_geometry"] not in {"parent_bounds", "fine_spheres"}:
        raise ValueError(
            "gradient_pruning.spatial.link_broad_phase.scan_geometry must be "
            "'parent_bounds' or 'fine_spheres'."
        )
    link_broad_phase["environment_margin"] = float(
        link_broad_phase["environment_margin"]
    )
    link_broad_phase["self_margin"] = float(link_broad_phase["self_margin"])
    if link_broad_phase["environment_margin"] < 0 or link_broad_phase["self_margin"] < 0:
        raise ValueError("gradient_pruning.spatial.link_broad_phase margins cannot be negative.")
    if link_broad_phase["enabled"] and not bool(
        config["spatial"]["parent_link_kinematics"]
    ):
        raise ValueError(
            "gradient_pruning.spatial.link_broad_phase requires parent_link_kinematics=true."
        )
    if config["preselection"]["parent_bounds_scan"] and not bool(
        config["spatial"]["parent_link_kinematics"]
    ):
        raise ValueError(
            "gradient_pruning.preselection.parent_bounds_scan requires "
            "parent_link_kinematics=true."
        )
    if span_certificate["enabled"]:
        if not bool(config["spatial"]["parent_link_kinematics"]):
            raise ValueError(
                "gradient_pruning.span_certificate requires parent_link_kinematics=true."
            )
        if config["temporal"]["enabled"] or config["temporal"]["conditional_enabled"]:
            raise ValueError(
                "span_certificate is an independent B3 temporal path; disable temporal and "
                "conditional_temporal."
            )
        if config["spatial"]["link_broad_phase"]["enabled"]:
            raise ValueError(
                "span_certificate validation must not be combined with link broad phase."
            )
        if config["preselection"]["parent_bounds_scan"]:
            raise ValueError(
                "span_certificate validation uses fine spheres, not parent bounds."
            )

    # These optimizations require separate equivalence work and must never be
    # silently accepted while the implementation still uses sphere-link FK.
    unsupported_spatial = [
        key
        for key in ("environment_link_broad_phase", "self_link_pair_broad_phase")
        if bool(config["spatial"].get(key))
    ]
    if config["enabled"] and unsupported_spatial:
        names = ", ".join(f"gradient_pruning.spatial.{key}" for key in unsupported_spatial)
        raise NotImplementedError(f"Unsupported spatial pruning option(s): {names}.")

    if config["enabled"] and bool(config["scheduling"]["enabled"]):
        raise NotImplementedError(
            "gradient_pruning.scheduling.enabled is not available yet; keep it false."
        )
    return config


def resolve_dense_validation_config(args_inference) -> dict:
    root = _as_plain_mapping(args_inference)
    config = _deep_merge(DEFAULT_DENSE_VALIDATION_CONFIG, _as_plain_mapping(root.get("dense_validation")))
    config["enabled"] = bool(config["enabled"])
    config["runtime_points"] = int(config["runtime_points"])
    config["benchmark_points"] = int(config["benchmark_points"])
    ranked_early_exit = config["ranked_early_exit"]
    ranked_early_exit["enabled"] = bool(ranked_early_exit["enabled"])
    ranked_early_exit["preallocate_buffers"] = bool(
        ranked_early_exit["preallocate_buffers"]
    )
    ranked_early_exit["cuda_graph"] = bool(ranked_early_exit["cuda_graph"])
    ranked_early_exit["batch_buckets"] = [
        int(value) for value in ranked_early_exit["batch_buckets"]
    ]
    if config["runtime_points"] != 128 or config["benchmark_points"] != 128:
        raise ValueError(
            "dense_validation is unified at 128 points; set both runtime_points "
            "and benchmark_points to 128."
        )
    if (
        not ranked_early_exit["batch_buckets"]
        or any(value < 1 for value in ranked_early_exit["batch_buckets"])
        or ranked_early_exit["batch_buckets"]
        != sorted(set(ranked_early_exit["batch_buckets"]))
    ):
        raise ValueError(
            "dense_validation.ranked_early_exit.batch_buckets must contain "
            "unique positive integers in ascending order."
        )
    if ranked_early_exit["enabled"] and not bool(config["reject_invalid"]):
        raise ValueError(
            "dense_validation.ranked_early_exit requires reject_invalid=true."
        )
    return config
