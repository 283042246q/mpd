#!/usr/bin/env python3
"""Generate and optionally execute paired gradient-pruning benchmarks.

The generator is deterministic and intentionally contains no autoset or
threshold-tuning behavior. Baseline and pruning configs differ only in their
``gradient_pruning`` section.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SCENARIO_DIR = SCRIPT_DIR / "cfgs" / "gradient_pruning_scenarios"
TORCH_ROBOTICS_CONFIG_DIR = (
    SCRIPT_DIR.parents[1] / "mpd" / "torch_robotics" / "torch_robotics" / "data" / "configs"
)
DEFAULT_PRUNING = {
    "enabled": True,
    "force_all_active": False,
    "profile": True,
    "profile_per_guide_call": True,
    "record_active_statistics": True,
    "endpoint": {"ee_only_last_point": True},
    "candidate": {"enabled": False},
    "preselection": {"parent_bounds_scan": False},
    "span_certificate": {
        "enabled": False,
        "max_subdivision_depth": 3,
        "environment_safe_margin": 0.08,
        "self_safe_margin": 0.06,
        "jacobian_bound_mode": "componentwise",
        "exact_sdf_for_certificate": True,
        "grid_error_scale": 1.0,
        "profile_stages": False,
    },
    "temporal": {
        "enabled": True,
        "conditional_enabled": False,
        "conditional_active_ratio_threshold": 0.35,
        "coarse_points": 32,
        "probe_midpoints": True,
        "coarse_scan": True,
        "reuse_selection_within_ddim_step": True,
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
        "active_link_pruning": False,
        "link_broad_phase": {
            "enabled": False,
            "full_scan": True,
            "scan_geometry": "fine_spheres",
            "reuse_scan_cache": True,
            "environment_margin": 0.20,
            "self_margin": 0.10,
        },
        "environment_link_broad_phase": False,
        "self_link_pair_broad_phase": False,
    },
    "mapping": {
        "fused_bspline_integration": True,
        "sparse_bspline_support": False,
    },
    "scheduling": {
        "enabled": False,
        "skip_safe_candidates": False,
        "promote_on_stalled_cost": True,
    },
}


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_yaml(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)


def resolve_manifest(suite):
    candidate = Path(suite)
    if candidate.is_file():
        return candidate.resolve()
    candidate = SCENARIO_DIR / f"{suite}.yaml"
    if not candidate.is_file():
        known = ", ".join(path.stem for path in sorted(SCENARIO_DIR.glob("*.yaml")))
        raise ValueError(f"Unknown suite {suite!r}; expected one of: {known}")
    return candidate


def validate_manifest(manifest):
    if manifest.get("schema") != "mpd_gradient_pruning_scenarios" or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported gradient-pruning scenario manifest schema.")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Scenario manifest must contain a non-empty scenarios list.")
    ids = [scenario.get("id") for scenario in scenarios]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Scenario ids must be non-empty and unique.")


def normalize_manifest_document(document):
    if document.get("schema") == "mpd_gradient_pruning_scenarios":
        return document
    if document.get("id") and document.get("type") and document.get("base_config"):
        return {
            "schema": "mpd_gradient_pruning_scenarios",
            "schema_version": 1,
            "suite_name": document.get("suite_name", "replayed_gradient_pruning_scenario"),
            "seed": int(document.get("scenario_seed", 0)),
            "base_config": document["base_config"],
            "scenarios": [document],
        }
    return document


def paired_configs(base_config, scenario, seed, dense_points=128, scenario_path=None):
    shared = deepcopy(base_config)
    shared["seed"] = int(seed)
    shared["planner_alg"] = "mpd"
    shared["dense_validation"] = {
        "enabled": True,
        "runtime_points": int(dense_points),
        "benchmark_points": int(dense_points),
        "check_environment": True,
        "check_self_collision": True,
        "check_joint_limits": True,
        "check_joint_position": True,
        "check_joint_velocity": True,
        "check_joint_acceleration": True,
        "reject_invalid": True,
    }
    if scenario["type"].startswith("warehouse"):
        shared["env_id_replace"] = "EnvWarehouseExtraBoxes"
        shared["extra_boxes_yaml"] = scenario["extra_boxes_yaml"]
    elif scenario["type"] in {"simple_2d", "narrow_2d"}:
        shared["env_id_replace"] = "EnvGradientPruning2DTest"
        shared["gradient_pruning_scenario_yaml"] = str(scenario_path or scenario.get("source_path"))
    if "start_goal_states_path" in scenario:
        shared["start_goal_source"] = "states_file"
        shared["selection_start_goal"] = scenario["start_goal_states_path"]

    baseline = deepcopy(shared)
    baseline["gradient_pruning"] = {
        "enabled": False,
        "profile": True,
        "profile_per_guide_call": True,
        "record_active_statistics": True,
    }
    pruning = deepcopy(shared)
    pruning["gradient_pruning"] = deepcopy(DEFAULT_PRUNING)
    return baseline, pruning


def generate_suite(
    manifest_path,
    output_dir,
    base_config_path=None,
    seed=None,
    dense_points=128,
    candidates=None,
):
    manifest_path = Path(manifest_path).resolve()
    manifest = normalize_manifest_document(load_yaml(manifest_path))
    validate_manifest(manifest)
    manifest_seed = int(manifest.get("seed", 0) if seed is None else seed)
    if base_config_path is None:
        base_config_path = (manifest_path.parent / manifest["base_config"]).resolve()
    else:
        base_config_path = Path(base_config_path).resolve()
    base_config = load_yaml(base_config_path)
    # Derived configs are written under /tmp; resolve paths that were relative
    # to the source config before copying them so special Panda environments
    # keep their region/state files.
    for key in ("start_goal_regions_path", "selection_start_goal"):
        value = base_config.get(key)
        if value and not Path(str(value)).is_absolute():
            candidate = (base_config_path.parent / str(value)).resolve()
            if candidate.exists():
                base_config[key] = candidate.as_posix()
    if candidates is not None:
        base_config["n_trajectory_samples"] = int(candidates)

    output_dir = Path(output_dir).resolve()
    (output_dir / "configs").mkdir(parents=True, exist_ok=True)
    (output_dir / "scenarios").mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (output_dir / "pruning").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    generated = []
    for scenario_index, scenario in enumerate(manifest["scenarios"]):
        scenario_seed = int(scenario.get("seed", manifest_seed + scenario_index))
        exact_scenario = deepcopy(scenario)
        exact_scenario.update(
            suite_name=manifest["suite_name"],
            scenario_seed=scenario_seed,
            dense_checker_points=int(dense_points),
            base_config=base_config_path.as_posix(),
            checkpoint={
                key: base_config.get(key)
                for key in ("model_dir_ddpm_bspline", "model_dir_cvae_bspline", "model_dir_ddpm_waypoints")
                if base_config.get(key) is not None
            },
        )
        extra_boxes_yaml = exact_scenario.get("extra_boxes_yaml")
        if extra_boxes_yaml:
            boxes_path = Path(extra_boxes_yaml)
            if not boxes_path.is_absolute():
                boxes_path = TORCH_ROBOTICS_CONFIG_DIR / boxes_path
            boxes_config = load_yaml(boxes_path)
            exact_scenario["resolved_extra_boxes_yaml"] = boxes_path.resolve().as_posix()
            exact_scenario["reference_frame"] = boxes_config.get("reference_frame")
            exact_scenario["boxes"] = boxes_config.get("boxes", [])
        scenario_path = output_dir / "scenarios" / f"{scenario['id']}.yaml"
        write_yaml(exact_scenario, scenario_path)
        if "q_pos_start" in exact_scenario and "q_pos_goal" in exact_scenario:
            start_goal_path = output_dir / "scenarios" / f"{scenario['id']}-start-goal.yaml"
            ee_pose_goal = exact_scenario.get(
                "ee_pose_goal",
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            )
            write_yaml(
                [
                    {
                        "q_pos_start": exact_scenario["q_pos_start"],
                        "q_pos_goal": exact_scenario["q_pos_goal"],
                        "ee_pose_goal": ee_pose_goal,
                    }
                ],
                start_goal_path,
            )
            exact_scenario["start_goal_states_path"] = start_goal_path.as_posix()
            write_yaml(exact_scenario, scenario_path)
        baseline, pruning = paired_configs(
            base_config,
            exact_scenario,
            scenario_seed,
            dense_points,
            scenario_path=scenario_path,
        )
        baseline_path = output_dir / "configs" / f"{scenario['id']}-baseline.yaml"
        pruning_path = output_dir / "configs" / f"{scenario['id']}-pruning.yaml"
        write_yaml(baseline, baseline_path)
        write_yaml(pruning, pruning_path)
        generated.append(
            {
                "scenario_id": scenario["id"],
                "scenario": scenario_path.as_posix(),
                "baseline_config": baseline_path.as_posix(),
                "pruning_config": pruning_path.as_posix(),
                "seed": scenario_seed,
            }
        )

    index = {
        "schema": "mpd_gradient_pruning_generated_suite",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite_name": manifest["suite_name"],
        "manifest": manifest_path.as_posix(),
        "base_config": base_config_path.as_posix(),
        "dense_points": int(dense_points),
        "runs": generated,
    }
    write_yaml(index, output_dir / "generated-suite.yaml")
    return index


def execute_suite(index, device, contexts, candidates, repeats):
    executions = []
    inference_script = SCRIPT_DIR / "inference.py"
    for run in index["runs"]:
        for variant in ("baseline", "pruning"):
            config = run[f"{variant}_config"]
            for repeat in range(repeats):
                results_dir = Path(config).parents[1] / variant / run["scenario_id"] / f"repeat-{repeat:02d}"
                command = [
                    sys.executable,
                    str(inference_script),
                    "--cfg_inference_path",
                    config,
                    "--n_start_goal_states",
                    str(contexts),
                    "--device",
                    device,
                    "--seed",
                    str(run["seed"]),
                    "--results_dir",
                    str(results_dir),
                    "--render_joint_space_time_iters",
                    "false",
                ]
                env = dict(os.environ, MPD_GRADIENT_PRUNING_CANDIDATES=str(candidates))
                subprocess.run(command, check=True, cwd=SCRIPT_DIR, env=env)
                executions.append(
                    {
                        "scenario_id": run["scenario_id"],
                        "variant": variant,
                        "repeat": repeat,
                        "results_dir": results_dir.as_posix(),
                        "command": command,
                    }
                )
    return executions


def _parse_scalar(value):
    value = value.strip().split(" (", 1)[0]
    if value in {"N/A", "None", "nan"}:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def parse_inference_report(path):
    section = "root"
    parsed = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].lower().replace(" ", "_")
        elif ":" in line:
            key, value = line.split(":", 1)
            parsed[f"{section}.{key.strip()}"] = _parse_scalar(value)
    return parsed


def _write_csv(rows, path):
    path = Path(path)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_reports(executions, output_dir, fail_on_regression=False):
    output_dir = Path(output_dir)
    paired_rows = []
    timing_rows = []
    for execution in executions:
        results_dir = Path(execution["results_dir"])
        for report_path in sorted(results_dir.rglob("inference-report-*.txt")):
            row = {
                key: value
                for key, value in execution.items()
                if key != "command"
            }
            row["context"] = int(report_path.stem.rsplit("-", 1)[-1])
            row.update(parse_inference_report(report_path))
            paired_rows.append(row)
        for active_statistics in sorted(results_dir.rglob("active-statistics.csv")):
            with active_statistics.open("r", encoding="utf-8", newline="") as stream:
                for active_row in csv.DictReader(stream):
                    timing_rows.append(
                        {
                            "scenario_id": execution["scenario_id"],
                            "variant": execution["variant"],
                            "repeat": execution["repeat"],
                            **active_row,
                        }
                    )

    reports_dir = output_dir / "reports"
    _write_csv(paired_rows, reports_dir / "paired-results.csv")
    _write_csv(timing_rows, reports_dir / "timing-breakdown.csv")
    _write_csv(timing_rows, reports_dir / "active-statistics.csv")

    by_key = {
        (row["scenario_id"], row["repeat"], row["context"], row["variant"]): row
        for row in paired_rows
    }
    failures = []
    paired = []
    base_keys = sorted({key[:3] for key in by_key})
    for key in base_keys:
        baseline = by_key.get((*key, "baseline"))
        pruning = by_key.get((*key, "pruning"))
        if baseline is None or pruning is None:
            failures.append({"scenario_id": key[0], "repeat": key[1], "context": key[2], "reason": "missing_pair"})
            continue
        paired.append((baseline, pruning))
        baseline_valid = baseline.get("mpd_validity.valid_trajectories") or 0
        pruning_valid = pruning.get("mpd_validity.valid_trajectories") or 0
        if baseline_valid > 0 and pruning_valid == 0:
            failures.append({"scenario_id": key[0], "repeat": key[1], "context": key[2], "reason": "coverage_regression"})
    write_yaml(failures, reports_dir / "failed-contexts.yaml")

    def values(variant_index, field):
        return [pair[variant_index][field] for pair in paired if isinstance(pair[variant_index].get(field), float)]

    baseline_valid = values(0, "mpd_validity.valid_rate")
    pruning_valid = values(1, "mpd_validity.valid_rate")
    baseline_total = values(0, "timing.inference_total_with_dense_validation_sec")
    pruning_total = values(1, "timing.inference_total_with_dense_validation_sec")
    baseline_guide = values(0, "timing.guide_sec")
    pruning_guide = values(1, "timing.guide_sec")
    baseline_position = values(0, "end-effector_goal_error_-_valid_trajectories.position_error_mean_m")
    pruning_position = values(1, "end-effector_goal_error_-_valid_trajectories.position_error_mean_m")
    baseline_orientation = values(0, "end-effector_goal_error_-_valid_trajectories.orientation_error_mean_deg")
    pruning_orientation = values(1, "end-effector_goal_error_-_valid_trajectories.orientation_error_mean_deg")

    summary = {
        "paired_contexts": len(paired),
        "coverage_regressions": sum(item.get("reason") == "coverage_regression" for item in failures),
        "baseline_valid_rate_mean": statistics.fmean(baseline_valid) if baseline_valid else None,
        "pruning_valid_rate_mean": statistics.fmean(pruning_valid) if pruning_valid else None,
        "total_p50_speedup": (
            statistics.median(baseline_total) / statistics.median(pruning_total)
            if baseline_total and pruning_total and statistics.median(pruning_total) > 0
            else None
        ),
        "guidance_p50_speedup": (
            statistics.median(baseline_guide) / statistics.median(pruning_guide)
            if baseline_guide and pruning_guide and statistics.median(pruning_guide) > 0
            else None
        ),
        "baseline_ee_position_mean_m": statistics.fmean(baseline_position) if baseline_position else None,
        "pruning_ee_position_mean_m": statistics.fmean(pruning_position) if pruning_position else None,
        "baseline_ee_orientation_mean_deg": statistics.fmean(baseline_orientation) if baseline_orientation else None,
        "pruning_ee_orientation_mean_deg": statistics.fmean(pruning_orientation) if pruning_orientation else None,
    }
    lines = ["# Gradient pruning paired benchmark", ""] + [
        f"- {key}: {value}" for key, value in summary.items()
    ]
    (reports_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    regressions = []
    if summary["coverage_regressions"]:
        regressions.append("context coverage decreased")
    if baseline_valid and pruning_valid and summary["baseline_valid_rate_mean"] - summary["pruning_valid_rate_mean"] > 0.01:
        regressions.append("mean valid rate decreased by more than 0.01")
    if baseline_position and pruning_position and summary["pruning_ee_position_mean_m"] - summary["baseline_ee_position_mean_m"] > 0.002:
        regressions.append("EE position error increased by more than 0.002 m")
    if baseline_orientation and pruning_orientation and summary["pruning_ee_orientation_mean_deg"] - summary["baseline_ee_orientation_mean_deg"] > 0.2:
        regressions.append("EE orientation error increased by more than 0.2 deg")
    if summary["guidance_p50_speedup"] is not None and summary["guidance_p50_speedup"] < 1.8:
        regressions.append("guidance p50 speedup is below 1.8x")
    if summary["total_p50_speedup"] is not None and summary["total_p50_speedup"] < 1.5:
        regressions.append("total p50 speedup is below 1.5x")
    if fail_on_regression and regressions:
        raise RuntimeError("Gradient pruning regression: " + "; ".join(regressions))
    return summary, failures


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="warehouse_panda")
    parser.add_argument("--base-config")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dense-points", type=int, choices=(128,), default=128)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contexts", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--replay-scenario")
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.fail_on_regression and args.generate_only:
        raise ValueError("--fail-on-regression requires executed benchmark results.")
    suite = args.replay_scenario or args.suite
    manifest_path = resolve_manifest(suite)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or f"/tmp/mpd-gradient-pruning-{run_id}"
    index = generate_suite(
        manifest_path,
        output_dir,
        base_config_path=args.base_config,
        seed=args.seed,
        dense_points=args.dense_points,
        candidates=args.candidates,
    )
    if not args.generate_only:
        executions = execute_suite(index, args.device, args.contexts, args.candidates, args.repeats)
        summary, failures = build_reports(executions, output_dir, args.fail_on_regression)
        report = {"suite": index, "executions": executions, "summary": summary, "failures": failures}
        report_path = Path(output_dir) / "reports" / "executed-commands.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(Path(output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
