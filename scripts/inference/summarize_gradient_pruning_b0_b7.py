#!/usr/bin/env python3
"""Summarize the fixed B0--B7 multi-scene gradient-pruning benchmark.

The expected result layout is::

    ROOT/<suite>/runs/<scenario>/<variant>/repeat-XX/2/

This script never runs inference.  It reads the inference reports, active-set
CSV files and saved tensors, then emits machine-readable per-scene and
aggregate summaries under ``ROOT/reports``.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import torch

from benchmark_gradient_pruning import parse_inference_report


VARIANTS = [
    "b0_a2pfast_materialized",
    "b1_candidate_sparse",
    "b2_time_sparse",
    "b3_link_sparse",
    "b4_control_point_sparse",
    "b5_candidate_time",
    "b6_candidate_time_link",
    "b7_all_sparse",
]
LABELS = {
    "b0_a2pfast_materialized": "B0 baseline",
    "b1_candidate_sparse": "B1 candidate",
    "b2_time_sparse": "B2 time",
    "b3_link_sparse": "B3 link",
    "b4_control_point_sparse": "B4 control-point",
    "b5_candidate_time": "B5 candidate+time",
    "b6_candidate_time_link": "B6 candidate+time+link",
    "b7_all_sparse": "B7 all",
}
DIFFICULTIES = {
    "open_clearance": "simple",
    "single_obstacle": "medium",
    "narrow_020": "hard",
    "narrow_014": "hard",
    "three_pillars_regions": "hard",
    "drawer_to_shelf": "hard",
    "to_drawer": "medium",
}
SCENE_ORDER = list(DIFFICULTIES)
TIMING_KEYS = {
    "generator_p50_s": "timing.generator_sec",
    "guide_p50_s": "timing.guide_sec",
    "inference_p50_s": "timing.inference_total_sec",
    "dense_p50_s": "timing.dense_validation_sec",
    "total_p50_s": "timing.inference_total_with_dense_validation_sec",
}


def median(values):
    return statistics.median(float(value) for value in values)


def mean(values):
    return statistics.fmean(float(value) for value in values)


def geometric_mean(values):
    values = list(values)
    return math.exp(statistics.fmean(math.log(value) for value in values))


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_runs(root):
    rows = []
    pattern = "*/runs/*/*/repeat-*/2/inference-report-000.txt"
    for report_path in sorted(root.glob(pattern)):
        suite, _, scenario, variant, repeat = report_path.relative_to(root).parts[:5]
        if variant not in VARIANTS:
            continue
        rows.append(
            {
                "suite": suite,
                "scenario": scenario,
                "variant": variant,
                "repeat": int(repeat.rsplit("-", 1)[-1]),
                "report_path": report_path.as_posix(),
                **parse_inference_report(report_path),
            }
        )
    return rows


def read_active_statistics(root, suite, scenario, variant):
    rows = []
    pattern = f"{suite}/runs/{scenario}/{variant}/repeat-*/2/active-statistics.csv"
    for path in sorted(root.glob(pattern)):
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def trajectory_delta(root, suite, scenario, variant, repeats):
    if variant == VARIANTS[0]:
        return {
            "final_cp_max_abs_vs_b0": 0.0,
            "final_q_max_abs_vs_b0": 0.0,
            "valid_mask_diff_max_vs_b0": 0,
            "collision_mask_diff_max_vs_b0": 0,
        }
    cp_deltas = []
    q_deltas = []
    valid_diffs = []
    collision_diffs = []
    for repeat in repeats:
        rel = Path(suite) / "runs" / scenario
        suffix = Path(f"repeat-{repeat:02d}") / "2" / "results_single_plan-000.pt"
        baseline = torch.load(root / rel / VARIANTS[0] / suffix, map_location="cpu")
        current = torch.load(root / rel / variant / suffix, map_location="cpu")
        cp_deltas.append(
            (baseline.control_points_iters[-1] - current.control_points_iters[-1])
            .abs()
            .max()
            .item()
        )
        q_deltas.append(
            (baseline.q_trajs_pos_iters[-1] - current.q_trajs_pos_iters[-1])
            .abs()
            .max()
            .item()
        )
        valid_diffs.append(
            int((baseline.valid_trajectory_mask != current.valid_trajectory_mask).sum().item())
        )
        collision_diffs.append(
            int(
                (
                    baseline.collision_trajectory_mask
                    != current.collision_trajectory_mask
                )
                .sum()
                .item()
            )
        )
    return {
        "final_cp_max_abs_vs_b0": max(cp_deltas),
        "final_q_max_abs_vs_b0": max(q_deltas),
        "valid_mask_diff_max_vs_b0": max(valid_diffs),
        "collision_mask_diff_max_vs_b0": max(collision_diffs),
    }


def build_scene_summary(root, run_rows):
    output = []
    for scenario in SCENE_ORDER:
        scene_rows = [row for row in run_rows if row["scenario"] == scenario]
        if not scene_rows:
            continue
        suite = scene_rows[0]["suite"]
        repeats = sorted({row["repeat"] for row in scene_rows})
        baseline_rows = [row for row in scene_rows if row["variant"] == VARIANTS[0]]
        baseline_timing = {
            output_key: median(row[report_key] for row in baseline_rows)
            for output_key, report_key in TIMING_KEYS.items()
        }
        for variant in VARIANTS:
            rows = [row for row in scene_rows if row["variant"] == variant]
            active_rows = read_active_statistics(root, suite, scenario, variant)
            active_points = sum(float(row["n_time_points_active"]) for row in active_rows)
            total_points = sum(float(row["n_time_points_total"]) for row in active_rows)
            bucket_counts = {
                bucket: sum(float(row[f"bucket_{bucket}"]) for row in active_rows)
                for bucket in (0, 32, 64, 128)
            }
            bucket_total = sum(bucket_counts.values())
            timing = {
                output_key: median(row[report_key] for row in rows)
                for output_key, report_key in TIMING_KEYS.items()
            }
            output.append(
                {
                    "suite": suite,
                    "scenario": scenario,
                    "difficulty": DIFFICULTIES[scenario],
                    "variant": variant,
                    "label": LABELS[variant],
                    **timing,
                    "guide_speedup_vs_b0": baseline_timing["guide_p50_s"]
                    / timing["guide_p50_s"],
                    "inference_speedup_vs_b0": baseline_timing["inference_p50_s"]
                    / timing["inference_p50_s"],
                    "total_speedup_vs_b0": baseline_timing["total_p50_s"]
                    / timing["total_p50_s"],
                    "active_time_ratio": active_points / total_points,
                    **{
                        f"bucket_{bucket}_ratio": bucket_counts[bucket] / bucket_total
                        for bucket in (0, 32, 64, 128)
                    },
                    "selection_cache_hit_ratio": sum(
                        row.get("temporal_selection_cache_hit") == "True"
                        for row in active_rows
                    )
                    / len(active_rows),
                    "valid_rate_mean": mean(
                        row["mpd_validity.valid_rate"] for row in rows
                    ),
                    "collision_rate_mean": mean(
                        row["mpd_validity.collision_rate"] for row in rows
                    ),
                    "ee_position_error_mean_m": mean(
                        row[
                            "end-effector_goal_error_-_all_trajectories.position_error_mean_m"
                        ]
                        for row in rows
                    ),
                    "ee_orientation_error_mean_deg": mean(
                        row[
                            "end-effector_goal_error_-_all_trajectories.orientation_error_mean_deg"
                        ]
                        for row in rows
                    ),
                    "valid_path_length_mean": mean(
                        row[
                            "path_length_-_joint_space.valid_trajectories_path_length_mean"
                        ]
                        for row in rows
                        if row[
                            "path_length_-_joint_space.valid_trajectories_path_length_mean"
                        ]
                        is not None
                    )
                    if any(
                        row[
                            "path_length_-_joint_space.valid_trajectories_path_length_mean"
                        ]
                        is not None
                        for row in rows
                    )
                    else None,
                    **trajectory_delta(root, suite, scenario, variant, repeats),
                }
            )
    return output


def build_aggregate(scene_rows):
    output = []
    for variant in VARIANTS:
        rows = [row for row in scene_rows if row["variant"] == variant]
        output.append(
            {
                "variant": variant,
                "label": LABELS[variant],
                **{
                    key.replace("p50", "scene_p50_mean"): mean(row[key] for row in rows)
                    for key in TIMING_KEYS
                },
                "guide_speedup_geomean_vs_b0": geometric_mean(
                    row["guide_speedup_vs_b0"] for row in rows
                ),
                "inference_speedup_geomean_vs_b0": geometric_mean(
                    row["inference_speedup_vs_b0"] for row in rows
                ),
                "total_speedup_geomean_vs_b0": geometric_mean(
                    row["total_speedup_vs_b0"] for row in rows
                ),
                "valid_rate_scene_mean": mean(row["valid_rate_mean"] for row in rows),
                "valid_rate_delta_vs_b0": mean(
                    row["valid_rate_mean"]
                    - next(
                        baseline["valid_rate_mean"]
                        for baseline in scene_rows
                        if baseline["scenario"] == row["scenario"]
                        and baseline["variant"] == VARIANTS[0]
                    )
                    for row in rows
                ),
                "max_valid_mask_diff_vs_b0": max(
                    row["valid_mask_diff_max_vs_b0"] for row in rows
                ),
            }
        )
    return output


def write_markdown(scene_rows, aggregate_rows, path):
    lines = [
        "# B0--B7 gradient-pruning benchmark summary",
        "",
        "## Aggregate over seven scenes",
        "",
        "| Variant | Guide mean (s) | Guide speedup | Inference speedup | Total speedup | Valid delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['label']} | {row['guide_scene_p50_mean_s']:.6f} | "
            f"{row['guide_speedup_geomean_vs_b0']:.3f}x | "
            f"{row['inference_speedup_geomean_vs_b0']:.3f}x | "
            f"{row['total_speedup_geomean_vs_b0']:.3f}x | "
            f"{100 * row['valid_rate_delta_vs_b0']:+.2f} pp |"
        )
    lines += [
        "",
        "## Per-scene guide timing and validity",
        "",
        "| Scene | Variant | Guide p50 (s) | Speedup | Active time | Valid | Mask diff |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in scene_rows:
        lines.append(
            f"| {row['scenario']} | {row['label']} | {row['guide_p50_s']:.6f} | "
            f"{row['guide_speedup_vs_b0']:.3f}x | {100 * row['active_time_ratio']:.1f}% | "
            f"{100 * row['valid_rate_mean']:.1f}% | {row['valid_mask_diff_max_vs_b0']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "reports").resolve()
    run_rows = read_runs(root)
    if len(run_rows) != 168:
        raise RuntimeError(f"Expected 168 reports, found {len(run_rows)}")
    scene_rows = build_scene_summary(root, run_rows)
    aggregate_rows = build_aggregate(scene_rows)
    write_csv(scene_rows, output_dir / "b0-b7-scene-summary.csv")
    write_csv(aggregate_rows, output_dir / "b0-b7-aggregate-summary.csv")
    write_markdown(scene_rows, aggregate_rows, output_dir / "b0-b7-summary.md")
    print(output_dir)


if __name__ == "__main__":
    main()
