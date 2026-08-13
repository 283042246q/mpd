#!/usr/bin/env python3
"""Benchmark the conservative B-spline span certificate against B3.

Latency runs keep all synchronized profilers off. A separate one-repeat
diagnostic pass enables synchronized guide sections and span sub-sections, so
the reported decomposition never contaminates the latency p50 comparison.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from benchmark_gradient_pruning import load_yaml, resolve_manifest, write_yaml
from benchmark_gradient_pruning_ablation import (
    build_ablation_report,
    execute_ablation_suite,
    generate_ablation_suite,
)


LATENCY_VARIANTS = ["b3_link_sparse", "d1_span_certificate"]


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _median(rows, key):
    values = [_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def _mean(rows, key):
    values = [_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _active_rows(executions):
    rows = []
    for execution in executions:
        for path in sorted(Path(execution["results_dir"]).rglob("active-statistics.csv")):
            with path.open("r", encoding="utf-8", newline="") as stream:
                rows.extend(
                    {
                        "scenario_id": execution["scenario_id"],
                        "variant": execution["variant"],
                        "repeat": execution["repeat"],
                        **row,
                    }
                    for row in csv.DictReader(stream)
                )
    return rows


def _profile_index(latency_index, output_dir):
    profile_dir = Path(output_dir) / "profile-diagnostic"
    index = generate_ablation_suite(
        latency_index["manifest"],
        profile_dir,
        base_config_path=latency_index["base_config"],
        candidates=latency_index.get("candidates", 100),
        scenario_ids=[run["scenario_id"] for run in latency_index["runs"]],
    )
    for run in index["runs"]:
        for variant in LATENCY_VARIANTS:
            path = Path(run["variant_configs"][variant])
            config = load_yaml(path)
            pruning = config["gradient_pruning"]
            pruning["profile"] = True
            pruning["profile_per_guide_call"] = True
            pruning["record_active_statistics"] = True
            if variant == "d1_span_certificate":
                pruning["span_certificate"]["profile_stages"] = True
            write_yaml(config, path)
    return index


def summarize(latency_rows, profile_rows, output_dir):
    reports_dir = Path(output_dir) / "reports"
    latency_by_key = {
        (row["scenario_id"], row["variant"]): row for row in latency_rows
    }
    scenarios = sorted({key[0] for key in latency_by_key})
    summary = []
    for scenario in scenarios:
        b3 = latency_by_key[(scenario, "b3_link_sparse")]
        span = latency_by_key[(scenario, "d1_span_certificate")]
        active = [
            row
            for row in profile_rows
            if row["scenario_id"] == scenario
            and row["variant"] == "d1_span_certificate"
        ]
        b3_profile = [
            row
            for row in profile_rows
            if row["scenario_id"] == scenario and row["variant"] == "b3_link_sparse"
        ]
        depth_remaining = [
            _mean(active, f"span_depth_{depth}_remaining_measure_ratio")
            for depth in range(4)
        ]
        b3_jfk = _median(b3_profile, "time_collision_jacobian_s")
        span_jfk = _median(active, "time_collision_jacobian_s")
        b3_sdf = sum(
            value or 0.0
            for value in (
                _median(b3_profile, "time_environment_sdf_query_s"),
                _median(b3_profile, "time_self_collision_s"),
            )
        )
        span_sdf = sum(
            value or 0.0
            for value in (
                _median(active, "time_environment_sdf_query_s"),
                _median(active, "time_self_collision_s"),
            )
        )
        row = {
            "scenario_id": scenario,
            "b3_guide_p50_s": b3["guide_p50_s"],
            "span_guide_p50_s": span["guide_p50_s"],
            "guide_speedup_b3_over_span": (
                b3["guide_p50_s"] / span["guide_p50_s"]
                if span["guide_p50_s"]
                else None
            ),
            "b3_inference_p50_s": b3["inference_p50_s"],
            "span_inference_p50_s": span["inference_p50_s"],
            "b3_valid_rate": b3["valid_rate_mean"],
            "span_valid_rate": span["valid_rate_mean"],
            "b3_collision_rate": b3["collision_rate_mean"],
            "span_collision_rate": span["collision_rate_mean"],
            "span_active_time_ratio": span["active_time_ratio"],
            "span_first_level_certification_ratio": _mean(
                active, "span_first_level_certification_ratio"
            ),
            **{
                f"span_depth_{depth}_remaining_measure_ratio": value
                for depth, value in enumerate(depth_remaining)
            },
            "span_derivative_bound_p50_s": _median(
                active, "time_span_derivative_bound_s"
            ),
            "span_midpoint_fk_p50_s": _median(active, "time_span_midpoint_fk_s"),
            "span_midpoint_sdf_p50_s": _median(active, "time_span_midpoint_sdf_s"),
            "span_bound_arithmetic_p50_s": _median(
                active, "time_span_bound_arithmetic_s"
            ),
            "b3_jfk_p50_s": b3_jfk,
            "span_active_jfk_p50_s": span_jfk,
            "jfk_saved_p50_s": (
                b3_jfk - span_jfk
                if b3_jfk is not None and span_jfk is not None
                else None
            ),
            "b3_active_sdf_p50_s": b3_sdf,
            "span_active_sdf_p50_s": span_sdf,
            "active_sdf_saved_p50_s": b3_sdf - span_sdf,
        }
        summary.append(row)

    fields = sorted({key for row in summary for key in row})
    with (reports_dir / "span-certificate-summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# B3 vs conservative B-spline span certificate",
        "",
        "Latency uses unsynchronized runs; stage decomposition uses a separate synchronized diagnostic run.",
        "TorchKin exposes joint FK+Jacobian as one J-FK operation, so that saving is reported jointly.",
        "",
        "| Scene | B3/Span guide (s) | Speedup | B3/Span valid | First certified | Remaining d0/d1/d2/d3 | Active points | Mid FK/SDF/bound (ms) | J-FK saved (ms) | Active SDF saved (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        remaining = "/".join(
            "—" if row[f"span_depth_{depth}_remaining_measure_ratio"] is None else f"{100 * row[f'span_depth_{depth}_remaining_measure_ratio']:.1f}%"
            for depth in range(4)
        )
        stage = "/".join(
            "—" if row[key] is None else f"{1000 * row[key]:.2f}"
            for key in (
                "span_midpoint_fk_p50_s",
                "span_midpoint_sdf_p50_s",
                "span_bound_arithmetic_p50_s",
            )
        )
        lines.append(
            f"| {row['scenario_id']} | {row['b3_guide_p50_s']:.4f}/{row['span_guide_p50_s']:.4f} | "
            f"{row['guide_speedup_b3_over_span']:.3f}x | "
            f"{100 * row['b3_valid_rate']:.1f}%/{100 * row['span_valid_rate']:.1f}% | "
            f"{100 * row['span_first_level_certification_ratio']:.1f}% | {remaining} | "
            f"{100 * row['span_active_time_ratio']:.1f}% | {stage} | "
            f"{1000 * row['jfk_saved_p50_s']:.2f} | {1000 * row['active_sdf_saved_p50_s']:.2f} |"
        )
    (reports_dir / "span-certificate-summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="warehouse_panda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--contexts", type=int, default=1)
    parser.add_argument("--scenario-ids")
    args = parser.parse_args(argv)
    scenario_ids = (
        [value.strip() for value in args.scenario_ids.split(",") if value.strip()]
        if args.scenario_ids
        else None
    )
    output_dir = Path(args.output_dir).resolve()
    index = generate_ablation_suite(
        resolve_manifest(args.suite),
        output_dir,
        candidates=args.candidates,
        scenario_ids=scenario_ids,
    )
    index["candidates"] = args.candidates
    latency_executions = execute_ablation_suite(
        index,
        args.device,
        args.contexts,
        args.repeats,
        variants=LATENCY_VARIANTS,
    )
    latency_rows = build_ablation_report(
        index, latency_executions, output_dir, variants=LATENCY_VARIANTS
    )
    profile_index = _profile_index(index, output_dir)
    profile_executions = execute_ablation_suite(
        profile_index,
        args.device,
        args.contexts,
        1,
        variants=LATENCY_VARIANTS,
    )
    summarize(latency_rows, _active_rows(profile_executions), output_dir)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
