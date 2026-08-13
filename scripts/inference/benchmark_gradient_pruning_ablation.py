#!/usr/bin/env python3
"""Run incremental gradient-pruning ablations on deterministic scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from benchmark_gradient_pruning import (
    DEFAULT_PRUNING,
    SCRIPT_DIR,
    generate_suite,
    load_yaml,
    parse_inference_report,
    resolve_manifest,
    write_yaml,
)


VARIANTS = OrderedDict(
    [
        (
            "a0_legacy",
            {
                "enabled": False,
                "profile": True,
                "profile_per_guide_call": True,
                "record_active_statistics": True,
            },
        ),
        (
            "a1_pruned_full",
            {
                **deepcopy(DEFAULT_PRUNING),
                "endpoint": {"ee_only_last_point": False},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": False},
                "spatial": {
                    **deepcopy(DEFAULT_PRUNING["spatial"]),
                    "parent_link_kinematics": False,
                },
            },
        ),
        (
            "a2_endpoint",
            {
                **deepcopy(DEFAULT_PRUNING),
                "endpoint": {"ee_only_last_point": True},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": False},
                "spatial": {
                    **deepcopy(DEFAULT_PRUNING["spatial"]),
                    "parent_link_kinematics": False,
                },
            },
        ),
        (
            "a2p_parent_link",
            {
                **deepcopy(DEFAULT_PRUNING),
                "endpoint": {"ee_only_last_point": True},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": False},
                "spatial": {
                    **deepcopy(DEFAULT_PRUNING["spatial"]),
                    "parent_link_kinematics": True,
                    "dense_parent_fast_path": False,
                },
            },
        ),
        (
            "a2p_parent_fast",
            {
                **deepcopy(DEFAULT_PRUNING),
                "endpoint": {"ee_only_last_point": True},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": False},
                "spatial": {
                    **deepcopy(DEFAULT_PRUNING["spatial"]),
                    "parent_link_kinematics": True,
                    "dense_parent_fast_path": True,
                },
            },
        ),
        (
            "a2pf_clean_x0",
            {
                **deepcopy(DEFAULT_PRUNING),
                "endpoint": {"ee_only_last_point": True},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": False},
                "spatial": {
                    **deepcopy(DEFAULT_PRUNING["spatial"]),
                    "parent_link_kinematics": True,
                    "dense_parent_fast_path": True,
                },
            },
        ),
        (
            "a2pf_clean_x0_2steps",
            {
                **deepcopy(DEFAULT_PRUNING),
                "endpoint": {"ee_only_last_point": True},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": False},
                "spatial": {
                    **deepcopy(DEFAULT_PRUNING["spatial"]),
                    "parent_link_kinematics": True,
                    "dense_parent_fast_path": True,
                },
            },
        ),
        (
            "a3r_temporal_parent",
            {
                **deepcopy(DEFAULT_PRUNING),
                "candidate": {"enabled": True},
                "endpoint": {"ee_only_last_point": True},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": True},
                "spatial": {
                    **deepcopy(DEFAULT_PRUNING["spatial"]),
                    "parent_link_kinematics": True,
                    "dense_parent_fast_path": True,
                },
            },
        ),
        (
            "a3_temporal_diagnostic",
            {
                **deepcopy(DEFAULT_PRUNING),
                "candidate": {"enabled": True},
                "endpoint": {"ee_only_last_point": True},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": True},
                "spatial": {
                    **deepcopy(DEFAULT_PRUNING["spatial"]),
                    "parent_link_kinematics": False,
                    "dense_parent_fast_path": False,
                },
            },
        ),
        (
            "a5_temporal_full_scan",
            {
                **deepcopy(DEFAULT_PRUNING),
                "candidate": {"enabled": True},
                "endpoint": {"ee_only_last_point": True},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": True, "coarse_scan": False},
                "spatial": {**deepcopy(DEFAULT_PRUNING["spatial"]), "parent_link_kinematics": False},
            },
        ),
        (
            "b0_a2pfast_materialized",
            {
                **deepcopy(DEFAULT_PRUNING),
                "candidate": {"enabled": False},
                "endpoint": {"ee_only_last_point": True},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": False},
                "spatial": {
                    **deepcopy(DEFAULT_PRUNING["spatial"]),
                    "parent_link_kinematics": True,
                    "dense_parent_fast_path": True,
                    "active_link_pruning": False,
                },
                "mapping": {
                    **deepcopy(DEFAULT_PRUNING["mapping"]),
                    "fused_bspline_integration": False,
                    "sparse_bspline_support": False,
                },
            },
        ),
        (
            "b1_candidate_sparse",
            {
                **deepcopy(DEFAULT_PRUNING),
                "candidate": {"enabled": True},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": False},
                "mapping": {
                    **deepcopy(DEFAULT_PRUNING["mapping"]),
                    "fused_bspline_integration": False,
                    "sparse_bspline_support": False,
                },
            },
        ),
        (
            "b2_time_sparse",
            {
                **deepcopy(DEFAULT_PRUNING),
                "candidate": {"enabled": False},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": True},
                "mapping": {
                    **deepcopy(DEFAULT_PRUNING["mapping"]),
                    "fused_bspline_integration": False,
                    "sparse_bspline_support": False,
                },
            },
        ),
        (
            "b3_link_sparse",
            {
                **deepcopy(DEFAULT_PRUNING),
                "candidate": {"enabled": False},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": False},
                "spatial": {
                    **deepcopy(DEFAULT_PRUNING["spatial"]),
                    "active_link_pruning": True,
                },
                "mapping": {
                    **deepcopy(DEFAULT_PRUNING["mapping"]),
                    "fused_bspline_integration": False,
                    "sparse_bspline_support": False,
                },
            },
        ),
        (
            "b4_control_point_sparse",
            {
                **deepcopy(DEFAULT_PRUNING),
                "candidate": {"enabled": False},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": False},
                "mapping": {
                    **deepcopy(DEFAULT_PRUNING["mapping"]),
                    "fused_bspline_integration": False,
                    "sparse_bspline_support": True,
                },
            },
        ),
        (
            "b5_candidate_time",
            {
                **deepcopy(DEFAULT_PRUNING),
                "candidate": {"enabled": True},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": True},
                "mapping": {
                    **deepcopy(DEFAULT_PRUNING["mapping"]),
                    "fused_bspline_integration": False,
                    "sparse_bspline_support": False,
                },
            },
        ),
        (
            "b6_candidate_time_link",
            {
                **deepcopy(DEFAULT_PRUNING),
                "candidate": {"enabled": True},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": True},
                "spatial": {
                    **deepcopy(DEFAULT_PRUNING["spatial"]),
                    "active_link_pruning": True,
                },
                "mapping": {
                    **deepcopy(DEFAULT_PRUNING["mapping"]),
                    "fused_bspline_integration": False,
                    "sparse_bspline_support": False,
                },
            },
        ),
        (
            "b7_all_sparse",
            {
                **deepcopy(DEFAULT_PRUNING),
                "candidate": {"enabled": True},
                "temporal": {**deepcopy(DEFAULT_PRUNING["temporal"]), "enabled": True},
                "spatial": {
                    **deepcopy(DEFAULT_PRUNING["spatial"]),
                    "active_link_pruning": True,
                },
                "mapping": {
                    **deepcopy(DEFAULT_PRUNING["mapping"]),
                    "fused_bspline_integration": False,
                    "sparse_bspline_support": True,
                },
            },
        ),
        (
            "c1_link_broad_phase",
            {
                **deepcopy(DEFAULT_PRUNING),
                "candidate": {"enabled": False},
                "temporal": {
                    **deepcopy(DEFAULT_PRUNING["temporal"]),
                    "enabled": False,
                    "conditional_enabled": False,
                },
                "spatial": {
                    **deepcopy(DEFAULT_PRUNING["spatial"]),
                    "parent_link_kinematics": True,
                    "dense_parent_fast_path": True,
                    "active_link_pruning": True,
                    "link_broad_phase": {
                        **deepcopy(DEFAULT_PRUNING["spatial"]["link_broad_phase"]),
                        "enabled": True,
                    },
                },
                "mapping": {
                    **deepcopy(DEFAULT_PRUNING["mapping"]),
                    "fused_bspline_integration": False,
                    "sparse_bspline_support": False,
                },
            },
        ),
        (
            "c2_conditional_temporal",
            {
                **deepcopy(DEFAULT_PRUNING),
                "candidate": {"enabled": False},
                "preselection": {"parent_bounds_scan": True},
                "temporal": {
                    **deepcopy(DEFAULT_PRUNING["temporal"]),
                    "enabled": False,
                    "conditional_enabled": True,
                    "conditional_active_ratio_threshold": 0.35,
                },
                "spatial": {
                    **deepcopy(DEFAULT_PRUNING["spatial"]),
                    "parent_link_kinematics": True,
                    "dense_parent_fast_path": True,
                    "active_link_pruning": True,
                    "link_broad_phase": {
                        **deepcopy(DEFAULT_PRUNING["spatial"]["link_broad_phase"]),
                        "enabled": False,
                    },
                },
                "mapping": {
                    **deepcopy(DEFAULT_PRUNING["mapping"]),
                    "fused_bspline_integration": False,
                    "sparse_bspline_support": False,
                },
            },
        ),
        (
            "c3_broad_phase_conditional",
            {
                **deepcopy(DEFAULT_PRUNING),
                "candidate": {"enabled": False},
                "temporal": {
                    **deepcopy(DEFAULT_PRUNING["temporal"]),
                    "enabled": False,
                    "conditional_enabled": True,
                    "conditional_active_ratio_threshold": 0.35,
                },
                "spatial": {
                    **deepcopy(DEFAULT_PRUNING["spatial"]),
                    "parent_link_kinematics": True,
                    "dense_parent_fast_path": True,
                    "active_link_pruning": True,
                    "link_broad_phase": {
                        **deepcopy(DEFAULT_PRUNING["spatial"]["link_broad_phase"]),
                        "enabled": True,
                    },
                },
                "mapping": {
                    **deepcopy(DEFAULT_PRUNING["mapping"]),
                    "fused_bspline_integration": False,
                    "sparse_bspline_support": False,
                },
            },
        ),
        (
            "d1_span_certificate",
            {
                **deepcopy(DEFAULT_PRUNING),
                "candidate": {"enabled": False},
                "preselection": {"parent_bounds_scan": False},
                "span_certificate": {
                    **deepcopy(DEFAULT_PRUNING["span_certificate"]),
                    "enabled": True,
                    "profile_stages": False,
                },
                "temporal": {
                    **deepcopy(DEFAULT_PRUNING["temporal"]),
                    "enabled": False,
                    "conditional_enabled": False,
                    "reuse_selection_within_ddim_step": False,
                },
                "spatial": {
                    **deepcopy(DEFAULT_PRUNING["spatial"]),
                    "parent_link_kinematics": True,
                    "dense_parent_fast_path": True,
                    "active_link_pruning": True,
                    "link_broad_phase": {
                        **deepcopy(DEFAULT_PRUNING["spatial"]["link_broad_phase"]),
                        "enabled": False,
                        "reuse_scan_cache": False,
                    },
                },
                "mapping": {
                    **deepcopy(DEFAULT_PRUNING["mapping"]),
                    "fused_bspline_integration": False,
                    "sparse_bspline_support": False,
                },
            },
        ),
    ]
)

VARIANT_LABELS = {
    "a0_legacy": "Legacy",
    "a1_pruned_full": "新路径/全量Jac",
    "a2_endpoint": "+EE末端",
    "a2p_parent_link": "+Parent-link/通用全量路径",
    "a2p_parent_fast": "+Parent-link dense fast",
    "a2pf_clean_x0": "+clean-x0 A1 on A2P-fast",
    "a2pf_clean_x0_2steps": "+clean-x0 A1/2 steps on A2P-fast",
    "a3r_temporal_parent": "+Temporal on A2P-fast",
    "a3_temporal_diagnostic": "旧顺序Temporal诊断",
    "a5_temporal_full_scan": "时间分桶(完整扫描对照)",
    "b0_a2pfast_materialized": "B0 A2P-fast/no-grad/原映射",
    "b1_candidate_sparse": "B1 候选稀疏",
    "b2_time_sparse": "B2 时间稀疏",
    "b3_link_sparse": "B3 Link稀疏",
    "b4_control_point_sparse": "B4 控制点稀疏",
    "b5_candidate_time": "B5 候选+时间",
    "b6_candidate_time_link": "B6 候选+时间+Link",
    "b7_all_sparse": "B7 全稀疏",
    "c1_link_broad_phase": "C1 B3 + Link broad phase",
    "c2_conditional_temporal": "C2 Conditional temporal + Parent-bounds scan",
    "c3_broad_phase_conditional": "C3 B3 + 两者",
    "d1_span_certificate": "D1 B3 + span certificate",
}


INCREMENTAL_PARENT = {
    "a0_legacy": None,
    "a1_pruned_full": "a0_legacy",
    "a2_endpoint": "a1_pruned_full",
    "a2p_parent_link": "a2_endpoint",
    "a2p_parent_fast": "a2p_parent_link",
    "a2pf_clean_x0": "a2p_parent_fast",
    "a2pf_clean_x0_2steps": "a2pf_clean_x0",
    "a3r_temporal_parent": "a2p_parent_fast",
    "a3_temporal_diagnostic": "a2_endpoint",
    "a5_temporal_full_scan": "a3_temporal_diagnostic",
    "b0_a2pfast_materialized": None,
    "b1_candidate_sparse": "b0_a2pfast_materialized",
    "b2_time_sparse": "b0_a2pfast_materialized",
    "b3_link_sparse": "b0_a2pfast_materialized",
    "b4_control_point_sparse": "b0_a2pfast_materialized",
    "b5_candidate_time": "b0_a2pfast_materialized",
    "b6_candidate_time_link": "b0_a2pfast_materialized",
    "b7_all_sparse": "b0_a2pfast_materialized",
    "c1_link_broad_phase": "b3_link_sparse",
    "c2_conditional_temporal": "b3_link_sparse",
    "c3_broad_phase_conditional": "b3_link_sparse",
    "d1_span_certificate": "b3_link_sparse",
}


def generate_ablation_suite(
    manifest_path,
    output_dir,
    base_config_path=None,
    seed=None,
    dense_points=128,
    candidates=20,
    scenario_ids=None,
):
    index = generate_suite(
        manifest_path,
        output_dir,
        base_config_path=base_config_path,
        seed=seed,
        dense_points=dense_points,
        candidates=candidates,
    )
    selected = set(scenario_ids or ())
    if selected:
        known = {run["scenario_id"] for run in index["runs"]}
        missing = sorted(selected - known)
        if missing:
            raise ValueError(f"Unknown scenario id(s): {', '.join(missing)}")
        index["runs"] = [run for run in index["runs"] if run["scenario_id"] in selected]

    output_dir = Path(output_dir)
    for run in index["runs"]:
        shared = load_yaml(run["pruning_config"])
        run["variant_configs"] = {}
        for variant, pruning_config in VARIANTS.items():
            config = deepcopy(shared)
            timing_pruning_config = deepcopy(pruning_config)
            # CUDA synchronization around every profiler section distorts the
            # incremental timing comparison because the pruned path has more
            # sections. Keep active statistics, but benchmark latency without
            # per-section synchronization.
            timing_pruning_config["profile"] = False
            timing_pruning_config["profile_per_guide_call"] = False
            timing_pruning_config["record_active_statistics"] = True
            config["gradient_pruning"] = timing_pruning_config
            if variant in {"a2pf_clean_x0", "a2pf_clean_x0_2steps"}:
                config["compute_costs_with_xrecon"] = True
            if variant == "a2pf_clean_x0_2steps":
                config.setdefault("ddim", {})["n_guide_steps"] = 2
            path = output_dir / "configs" / f"{run['scenario_id']}-{variant}.yaml"
            write_yaml(config, path)
            run["variant_configs"][variant] = path.resolve().as_posix()
    write_yaml(index, output_dir / "generated-ablation-suite.yaml")
    return index


def resolve_variants(variants=None):
    if variants is None:
        return list(VARIANTS)
    selected = list(dict.fromkeys(variants))
    unknown = [variant for variant in selected if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variant(s): {', '.join(unknown)}")
    return selected


def expected_executions(index, repeats, variants=None):
    executions = []
    variants = resolve_variants(variants)
    output_dir = Path(index["runs"][0]["variant_configs"]["a0_legacy"]).parents[1]
    for run in index["runs"]:
        for variant in variants:
            config = run["variant_configs"][variant]
            for repeat in range(repeats):
                results_dir = output_dir / variant / run["scenario_id"] / f"repeat-{repeat:02d}"
                executions.append(
                    {
                        "scenario_id": run["scenario_id"],
                        "variant": variant,
                        "repeat": repeat,
                        "results_dir": results_dir.as_posix(),
                        "config": config,
                        "seed": run["seed"],
                    }
                )
    return executions


def execute_ablation_suite(index, device, contexts, repeats, resume=False, variants=None):
    executions = expected_executions(index, repeats, variants=variants)
    inference_script = SCRIPT_DIR / "inference.py"
    for execution in executions:
        results_dir = Path(execution["results_dir"])
        complete = bool(list(results_dir.rglob("inference-report-*.txt")))
        if resume and complete:
            execution["command"] = None
            continue
        command = [
            sys.executable,
            str(inference_script),
            "--cfg_inference_path",
            execution["config"],
            "--n_start_goal_states",
            str(contexts),
            "--device",
            device,
            "--seed",
            str(execution["seed"]),
            "--results_dir",
            str(results_dir),
            "--render_joint_space_time_iters",
            "false",
        ]
        subprocess.run(command, check=True, cwd=SCRIPT_DIR, env=dict(os.environ))
        execution["command"] = command
    return executions


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator else None


def _format(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _read_active_rows(results_dir):
    rows = []
    for path in sorted(Path(results_dir).rglob("active-statistics.csv")):
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def build_ablation_report(index, executions, output_dir, variants=None):
    output_dir = Path(output_dir)
    variants = resolve_variants(
        variants
        if variants is not None
        else [variant for variant in VARIANTS if any(row["variant"] == variant for row in executions)]
    )
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    scenario_metadata = {}
    for run in index["runs"]:
        scenario = load_yaml(run["scenario"])
        scenario_metadata[run["scenario_id"]] = {
            "difficulty": scenario.get("difficulty", "unknown"),
            "type": scenario.get("type", "unknown"),
        }

    reports = []
    active = []
    for execution in executions:
        for path in sorted(Path(execution["results_dir"]).rglob("inference-report-*.txt")):
            reports.append(
                {
                    **{key: value for key, value in execution.items() if key != "command"},
                    "context": int(path.stem.rsplit("-", 1)[-1]),
                    **parse_inference_report(path),
                }
            )
        for row in _read_active_rows(execution["results_dir"]):
            active.append(
                {
                    "scenario_id": execution["scenario_id"],
                    "variant": execution["variant"],
                    "repeat": execution["repeat"],
                    **row,
                }
            )

    summary_rows = []
    for scenario_id in [run["scenario_id"] for run in index["runs"]]:
        scenario_reports = [row for row in reports if row["scenario_id"] == scenario_id]
        scenario_active = [row for row in active if row["scenario_id"] == scenario_id]
        by_variant = {}
        for variant in variants:
            variant_reports = [row for row in scenario_reports if row["variant"] == variant]
            variant_active = [row for row in scenario_active if row["variant"] == variant]
            n_active = sum(_float(row.get("n_time_points_active")) or 0 for row in variant_active)
            n_total = sum(_float(row.get("n_time_points_total")) or 0 for row in variant_active)
            bucket_total = sum(
                sum(_float(row.get(f"bucket_{bucket}")) or 0 for bucket in (0, 32, 64, 128))
                for row in variant_active
            )
            data = {
                "scenario_id": scenario_id,
                **scenario_metadata[scenario_id],
                "variant": variant,
                "label": VARIANT_LABELS[variant],
                "guide_p50_s": _median(
                    [_float(row.get("timing.guide_sec")) for row in variant_reports]
                ),
                "generator_p50_s": _median(
                    [_float(row.get("timing.generator_sec")) for row in variant_reports]
                ),
                "inference_p50_s": _median(
                    [_float(row.get("timing.inference_total_sec")) for row in variant_reports]
                ),
                "dense_p50_s": _median(
                    [_float(row.get("timing.dense_validation_sec")) for row in variant_reports]
                ),
                "total_with_dense_p50_s": _median(
                    [
                        _float(row.get("timing.inference_total_with_dense_validation_sec"))
                        for row in variant_reports
                    ]
                ),
                "valid_rate_mean": _mean(
                    [_float(row.get("mpd_validity.valid_rate")) for row in variant_reports]
                ),
                "collision_rate_mean": _mean(
                    [_float(row.get("mpd_validity.collision_rate")) for row in variant_reports]
                ),
                "ee_position_all_mean_m": _mean(
                    [
                        _float(
                            row.get(
                                "end-effector_goal_error_-_all_trajectories.position_error_mean_m"
                            )
                        )
                        for row in variant_reports
                    ]
                ),
                "ee_orientation_all_mean_deg": _mean(
                    [
                        _float(
                            row.get(
                                "end-effector_goal_error_-_all_trajectories.orientation_error_mean_deg"
                            )
                        )
                        for row in variant_reports
                    ]
                ),
                "valid_path_length_mean": _mean(
                    [
                        _float(
                            row.get(
                                "path_length_-_joint_space.valid_trajectories_path_length_mean"
                            )
                        )
                        for row in variant_reports
                    ]
                ),
                "valid_smoothness_median": _mean(
                    [
                        _float(
                            row.get(
                                "valid_trajectory_metrics_-_median.smoothness_median"
                            )
                        )
                        for row in variant_reports
                    ]
                ),
                "active_time_ratio": _ratio(n_active, n_total),
            }
            for bucket in (0, 32, 64, 128):
                data[f"bucket_{bucket}_ratio"] = _ratio(
                    sum(_float(row.get(f"bucket_{bucket}")) or 0 for row in variant_active),
                    bucket_total,
                )
            by_variant[variant] = data

        baseline = by_variant.get("a0_legacy")
        for variant in variants:
            data = by_variant[variant]
            data["guide_speedup_vs_legacy"] = _ratio(
                baseline["guide_p50_s"] if baseline else None, data["guide_p50_s"]
            )
            data["inference_speedup_vs_legacy"] = _ratio(
                baseline["inference_p50_s"] if baseline else None, data["inference_p50_s"]
            )
            data["total_with_dense_speedup_vs_legacy"] = _ratio(
                baseline["total_with_dense_p50_s"] if baseline else None,
                data["total_with_dense_p50_s"],
            )
            parent_variant = INCREMENTAL_PARENT[variant]
            data["incremental_parent"] = parent_variant
            parent = None if parent_variant is None else by_variant.get(parent_variant)
            data["guide_incremental_speedup"] = (
                None if parent is None else _ratio(parent["guide_p50_s"], data["guide_p50_s"])
            )
            summary_rows.append(data)

    fieldnames = list(summary_rows[0]) if summary_rows else []
    with (reports_dir / "ablation-summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# Gradient pruning incremental ablation",
        "",
        "| 场景 | 难度 | 阶段 | Guide p50 (s) | 增量加速 | 相对 Legacy | 推理 p50 (s) | 含 dense 总加速 | Active 点比例 | K0/K32/K64/K128 | Valid rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        buckets = "/".join(
            _format(100 * row[f"bucket_{bucket}_ratio"], 1)
            if row[f"bucket_{bucket}_ratio"] is not None
            else "—"
            for bucket in (0, 32, 64, 128)
        )
        lines.append(
            "| {scenario_id} | {difficulty} | {label} | {guide} | {incremental} | {legacy} | "
            "{inference} | {total} | {active} | {buckets} | {valid} |".format(
                scenario_id=row["scenario_id"],
                difficulty=row["difficulty"],
                label=row["label"],
                guide=_format(row["guide_p50_s"]),
                incremental=_format(row["guide_incremental_speedup"]) + "×",
                legacy=_format(row["guide_speedup_vs_legacy"]) + "×",
                inference=_format(row["inference_p50_s"]),
                total=_format(row["total_with_dense_speedup_vs_legacy"]) + "×",
                active=(
                    _format(100 * row["active_time_ratio"], 1) + "%"
                    if row["active_time_ratio"] is not None
                    else "—"
                ),
                buckets=buckets,
                valid=(
                    _format(100 * row["valid_rate_mean"], 1) + "%"
                    if row["valid_rate_mean"] is not None
                    else "—"
                ),
            )
        )
    lines.extend(
        [
            "",
            "说明：所有阶段使用相同 checkpoint、seed、候选数和 dense checker；",
            "`含 dense 总加速` 使用 `inference + dense_validation`，bucket 列依次为百分比。",
        ]
    )
    (reports_dir / "ablation-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="warehouse_panda")
    parser.add_argument("--base-config")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dense-points", type=int, choices=(128,), default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contexts", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scenario-ids", help="Comma-separated subset of manifest scenario ids.")
    parser.add_argument(
        "--variants",
        help="Comma-separated subset of ablation variants to execute and report.",
    )
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip runs that already have a report.")
    parser.add_argument("--report-only", action="store_true", help="Rebuild reports from existing runs.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest_path = resolve_manifest(args.suite)
    output_dir = Path(
        args.output_dir
        or f"/tmp/mpd-gradient-pruning-ablation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ).resolve()
    scenario_ids = (
        [value.strip() for value in args.scenario_ids.split(",") if value.strip()]
        if args.scenario_ids
        else None
    )
    variants = resolve_variants(
        [value.strip() for value in args.variants.split(",") if value.strip()]
        if args.variants
        else None
    )
    index = generate_ablation_suite(
        manifest_path,
        output_dir,
        base_config_path=args.base_config,
        seed=args.seed,
        dense_points=args.dense_points,
        candidates=args.candidates,
        scenario_ids=scenario_ids,
    )
    executions = []
    if not args.generate_only:
        executions = (
            expected_executions(index, args.repeats, variants=variants)
            if args.report_only
            else execute_ablation_suite(
                index,
                args.device,
                args.contexts,
                args.repeats,
                resume=args.resume,
                variants=variants,
            )
        )
        build_ablation_report(index, executions, output_dir, variants=variants)
        (output_dir / "reports" / "executed-commands.json").write_text(
            json.dumps(executions, indent=2),
            encoding="utf-8",
        )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
