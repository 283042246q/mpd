#!/usr/bin/env python3
"""Summarize B3 vs link-broad-phase and conditional-temporal benchmarks."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import torch

from benchmark_gradient_pruning import parse_inference_report


VARIANTS = [
    "b3_link_sparse",
    "c1_link_broad_phase",
    "c2_conditional_temporal",
    "c3_broad_phase_conditional",
]
LABELS = {
    "b3_link_sparse": "B3 sparse J^Tg",
    "c1_link_broad_phase": "C1 B3+broad phase",
    "c2_conditional_temporal": "C2 B3+conditional temporal",
    "c3_broad_phase_conditional": "C3 B3+both",
}
SCENES = [
    "open_clearance",
    "single_obstacle",
    "narrow_020",
    "narrow_014",
    "three_pillars_regions",
    "drawer_to_shelf",
    "to_drawer",
]
TIMINGS = {
    "generator_p50_s": "timing.generator_sec",
    "guide_p50_s": "timing.guide_sec",
    "inference_p50_s": "timing.inference_total_sec",
    "dense_p50_s": "timing.dense_validation_sec",
    "total_p50_s": "timing.inference_total_with_dense_validation_sec",
}


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_report_rows(root):
    rows = []
    for path in sorted(root.glob("*/runs/*/*/repeat-*/2/inference-report-000.txt")):
        suite, _, scenario, variant, repeat = path.relative_to(root).parts[:5]
        if variant in VARIANTS:
            rows.append(
                {
                    "suite": suite,
                    "scenario": scenario,
                    "variant": variant,
                    "repeat": int(repeat.rsplit("-", 1)[-1]),
                    **parse_inference_report(path),
                }
            )
    return rows


def active_statistics(root, suite, scenario, variant):
    rows = []
    pattern = f"{suite}/runs/{scenario}/{variant}/repeat-*/2/active-statistics.csv"
    for path in sorted(root.glob(pattern)):
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def ratio(rows, numerator, denominator):
    return sum(float(row[numerator]) for row in rows) / sum(
        float(row[denominator]) for row in rows
    )


def trajectory_delta(root, suite, scenario, variant):
    if variant == VARIANTS[0]:
        return 0.0, 0.0, 0, 0
    q_max = cp_max = 0.0
    valid_diff = collision_diff = 0
    for repeat in range(3):
        prefix = root / suite / "runs" / scenario
        suffix = Path(f"repeat-{repeat:02d}/2/results_single_plan-000.pt")
        baseline = torch.load(prefix / VARIANTS[0] / suffix, map_location="cpu")
        current = torch.load(prefix / variant / suffix, map_location="cpu")
        q_max = max(
            q_max,
            float(
                (baseline.q_trajs_pos_iters[-1] - current.q_trajs_pos_iters[-1])
                .abs()
                .max()
            ),
        )
        cp_max = max(
            cp_max,
            float(
                (baseline.control_points_iters[-1] - current.control_points_iters[-1])
                .abs()
                .max()
            ),
        )
        valid_diff = max(
            valid_diff,
            int((baseline.valid_trajectory_mask != current.valid_trajectory_mask).sum()),
        )
        collision_diff = max(
            collision_diff,
            int(
                (
                    baseline.collision_trajectory_mask
                    != current.collision_trajectory_mask
                ).sum()
            ),
        )
    return q_max, cp_max, valid_diff, collision_diff


def build(root):
    reports = read_report_rows(root)
    if len(reports) != 84:
        raise RuntimeError(f"Expected 84 inference reports, found {len(reports)}")
    scene_rows = []
    for scenario in SCENES:
        source = [row for row in reports if row["scenario"] == scenario]
        suite = source[0]["suite"]
        baseline = [row for row in source if row["variant"] == VARIANTS[0]]
        baseline_timing = {
            output: statistics.median(float(row[key]) for row in baseline)
            for output, key in TIMINGS.items()
        }
        baseline_valid = statistics.fmean(
            float(row["mpd_validity.valid_rate"]) for row in baseline
        )
        for variant in VARIANTS:
            rows = [row for row in source if row["variant"] == variant]
            active = active_statistics(root, suite, scenario, variant)
            timing = {
                output: statistics.median(float(row[key]) for row in rows)
                for output, key in TIMINGS.items()
            }
            q_max, cp_max, valid_mask_diff, collision_mask_diff = trajectory_delta(
                root, suite, scenario, variant
            )
            valid_rate = statistics.fmean(
                float(row["mpd_validity.valid_rate"]) for row in rows
            )
            scene_rows.append(
                {
                    "suite": suite,
                    "scenario": scenario,
                    "variant": variant,
                    "label": LABELS[variant],
                    **timing,
                    "guide_speedup_vs_b3": baseline_timing["guide_p50_s"]
                    / timing["guide_p50_s"],
                    "inference_speedup_vs_b3": baseline_timing["inference_p50_s"]
                    / timing["inference_p50_s"],
                    "total_speedup_vs_b3": baseline_timing["total_p50_s"]
                    / timing["total_p50_s"],
                    "active_time_ratio": ratio(
                        active, "n_time_points_active", "n_time_points_total"
                    ),
                    "active_sphere_ratio": ratio(
                        active, "n_spheres_active", "n_spheres_total"
                    ),
                    "active_self_pair_ratio": ratio(
                        active, "n_self_pairs_active", "n_self_pairs_total"
                    ),
                    "conditional_applied_call_ratio": sum(
                        row.get("conditional_temporal_applied") == "True"
                        for row in active
                    )
                    / len(active),
                    "predicted_active_ratio_mean": statistics.fmean(
                        float(row.get("predicted_active_ratio", 1.0)) for row in active
                    ),
                    "valid_rate_mean": valid_rate,
                    "valid_rate_delta_vs_b3": valid_rate - baseline_valid,
                    "collision_rate_mean": statistics.fmean(
                        float(row["mpd_validity.collision_rate"]) for row in rows
                    ),
                    "final_q_max_abs_vs_b3": q_max,
                    "final_cp_max_abs_vs_b3": cp_max,
                    "valid_mask_diff_max_vs_b3": valid_mask_diff,
                    "collision_mask_diff_max_vs_b3": collision_mask_diff,
                }
            )

    aggregate_rows = []
    for variant in VARIANTS:
        rows = [row for row in scene_rows if row["variant"] == variant]
        aggregate_rows.append(
            {
                "variant": variant,
                "label": LABELS[variant],
                "guide_scene_p50_mean_s": statistics.fmean(
                    row["guide_p50_s"] for row in rows
                ),
                "guide_speedup_geomean_vs_b3": math.exp(
                    statistics.fmean(math.log(row["guide_speedup_vs_b3"]) for row in rows)
                ),
                "inference_speedup_geomean_vs_b3": math.exp(
                    statistics.fmean(
                        math.log(row["inference_speedup_vs_b3"]) for row in rows
                    )
                ),
                "total_speedup_geomean_vs_b3": math.exp(
                    statistics.fmean(math.log(row["total_speedup_vs_b3"]) for row in rows)
                ),
                "valid_rate_delta_scene_mean_vs_b3": statistics.fmean(
                    row["valid_rate_delta_vs_b3"] for row in rows
                ),
                "max_valid_mask_diff_vs_b3": max(
                    row["valid_mask_diff_max_vs_b3"] for row in rows
                ),
            }
        )
    return scene_rows, aggregate_rows


def write_markdown(scene_rows, aggregate_rows, path):
    lines = [
        "# B3 broad-phase / conditional-temporal summary",
        "",
        "| Variant | Guide mean (s) | Guide speedup | Inference speedup | Total speedup | Valid delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['label']} | {row['guide_scene_p50_mean_s']:.6f} | "
            f"{row['guide_speedup_geomean_vs_b3']:.3f}x | "
            f"{row['inference_speedup_geomean_vs_b3']:.3f}x | "
            f"{row['total_speedup_geomean_vs_b3']:.3f}x | "
            f"{100 * row['valid_rate_delta_scene_mean_vs_b3']:+.2f} pp |"
        )
    lines += [
        "",
        "| Scene | Variant | Guide p50 | Speedup | Time | Spheres | Pairs | Conditional | Valid | Mask diff |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scene_rows:
        lines.append(
            f"| {row['scenario']} | {row['label']} | {row['guide_p50_s']:.6f} | "
            f"{row['guide_speedup_vs_b3']:.3f}x | "
            f"{100 * row['active_time_ratio']:.1f}% | "
            f"{100 * row['active_sphere_ratio']:.1f}% | "
            f"{100 * row['active_self_pair_ratio']:.1f}% | "
            f"{100 * row['conditional_applied_call_ratio']:.1f}% | "
            f"{100 * row['valid_rate_mean']:.1f}% | "
            f"{row['valid_mask_diff_max_vs_b3']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    reports_dir = root / "reports"
    scene_rows, aggregate_rows = build(root)
    write_csv(scene_rows, reports_dir / "b3-broad-conditional-scene-summary.csv")
    write_csv(aggregate_rows, reports_dir / "b3-broad-conditional-aggregate-summary.csv")
    write_markdown(scene_rows, aggregate_rows, reports_dir / "b3-broad-conditional-summary.md")
    print(reports_dir)


if __name__ == "__main__":
    main()
