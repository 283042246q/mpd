#!/usr/bin/env python3
"""Compare full dense validation with ranked fixed-shape GPU buckets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from benchmark_gradient_pruning import load_yaml, parse_inference_report


SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_SCRIPT = SCRIPT_DIR / "inference.py"
VARIANTS = {
    "full_dense": "false",
    "fixed_buckets": "true",
}


def _number(value, default=None):
    return value if isinstance(value, (int, float)) else default


def _report_path(results_dir):
    reports = sorted(Path(results_dir).rglob("inference-report-*.txt"))
    if len(reports) != 1:
        raise RuntimeError(
            f"Expected one inference report under {results_dir}, found {len(reports)}."
        )
    return reports[0]


def _run_one(config, results_dir, seed, device, variant):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--cfg_inference_path",
        str(config),
        "--n_start_goal_states",
        "1",
        "--device",
        device,
        "--seed",
        str(seed),
        "--results_dir",
        str(results_dir),
        "--dense_ranked_early_exit_override",
        VARIANTS[variant],
        "--save_results_single_plan_low_mem",
        "true",
        "--render_joint_space_time_iters",
        "false",
    ]
    with (results_dir / "run.log").open("w", encoding="utf-8") as log_stream:
        subprocess.run(
            command,
            check=True,
            cwd=SCRIPT_DIR,
            env=os.environ.copy(),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
    parsed = parse_inference_report(_report_path(results_dir))
    ranking = _number(parsed.get("timing.trajectory_ranking_sec"), 0.0)
    dense = _number(parsed.get("timing.dense_validation_sec"), 0.0)
    return {
        "variant": variant,
        "results_dir": str(results_dir),
        "dense_sec": dense,
        "ranking_sec": ranking,
        "postprocess_sec": dense + ranking,
        "total_sec": _number(
            parsed.get("timing.inference_total_with_dense_validation_sec")
        ),
        "candidates_checked": _number(
            parsed.get("root.dense_validation_candidates_checked")
        ),
        "batches_evaluated": _number(
            parsed.get("root.dense_validation_batches_evaluated")
        ),
        "padding_slots": _number(
            parsed.get("root.dense_validation_padding_slots"), 0
        ),
        "valid_trajectories": _number(
            parsed.get("mpd_validity.valid_trajectories"), 0
        ),
        "selected_candidate_index": _number(
            parsed.get("best_trajectory_selection.selected_candidate_index")
        ),
        "selection_score": _number(
            parsed.get("best_trajectory_selection.score")
        ),
        "valid_position_error_mean_m": _number(
            parsed.get(
                "end-effector_goal_error_-_valid_trajectories.position_error_mean_m"
            )
        ),
        "valid_orientation_error_mean_deg": _number(
            parsed.get(
                "end-effector_goal_error_-_valid_trajectories.orientation_error_mean_deg"
            )
        ),
    }


def execute(generated_suites, output_dir, repeats, device):
    rows = []
    for generated_suite in generated_suites:
        index = load_yaml(generated_suite)
        suite_name = index["suite_name"]
        for run in index["runs"]:
            scenario_id = run["scenario_id"]
            # The pruning config supplies one common trajectory-generation
            # pipeline.  The CLI override changes only ranked dense validation.
            config = run["pruning_config"]
            for repeat in range(repeats):
                order = (
                    ("full_dense", "fixed_buckets")
                    if repeat % 2 == 0
                    else ("fixed_buckets", "full_dense")
                )
                for variant in order:
                    results_dir = (
                        output_dir
                        / suite_name
                        / scenario_id
                        / f"repeat-{repeat:02d}"
                        / variant
                    )
                    row = _run_one(
                        config,
                        results_dir,
                        run["seed"],
                        device,
                        variant,
                    )
                    row.update(
                        suite=suite_name,
                        scenario=scenario_id,
                        repeat=repeat,
                        seed=run["seed"],
                    )
                    rows.append(row)
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    paired = defaultdict(dict)
    for row in rows:
        grouped[(row["suite"], row["scenario"], row["variant"])].append(row)
        paired[(row["suite"], row["scenario"], row["repeat"])][
            row["variant"]
        ] = row

    summaries = []
    scenario_keys = sorted({key[:2] for key in grouped})
    for suite, scenario in scenario_keys:
        full = grouped[(suite, scenario, "full_dense")]
        fixed = grouped[(suite, scenario, "fixed_buckets")]
        full_post = statistics.median(row["postprocess_sec"] for row in full)
        fixed_post = statistics.median(row["postprocess_sec"] for row in fixed)
        full_total = statistics.median(row["total_sec"] for row in full)
        fixed_total = statistics.median(row["total_sec"] for row in fixed)
        scenario_pairs = [
            value
            for key, value in paired.items()
            if key[:2] == (suite, scenario)
            and set(value) == set(VARIANTS)
        ]
        selection_matches = sum(
            value["full_dense"]["selected_candidate_index"]
            == value["fixed_buckets"]["selected_candidate_index"]
            for value in scenario_pairs
        )
        coverage_regressions = sum(
            value["full_dense"]["valid_trajectories"] > 0
            and value["fixed_buckets"]["valid_trajectories"] == 0
            for value in scenario_pairs
        )
        summaries.append(
            {
                "suite": suite,
                "scenario": scenario,
                "repeats": len(scenario_pairs),
                "full_dense_p50_ms": 1000.0
                * statistics.median(row["dense_sec"] for row in full),
                "fixed_ranking_p50_ms": 1000.0
                * statistics.median(row["ranking_sec"] for row in fixed),
                "fixed_dense_p50_ms": 1000.0
                * statistics.median(row["dense_sec"] for row in fixed),
                "fixed_postprocess_p50_ms": 1000.0 * fixed_post,
                "postprocess_speedup": full_post / fixed_post,
                "total_speedup": full_total / fixed_total,
                "fixed_checked_p50": statistics.median(
                    row["candidates_checked"] for row in fixed
                ),
                "fixed_batches_p50": statistics.median(
                    row["batches_evaluated"] for row in fixed
                ),
                "selection_matches": selection_matches,
                "coverage_regressions": coverage_regressions,
            }
        )
    return summaries


def write_outputs(rows, summaries, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("runs.csv", rows), ("summary.csv", summaries)):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (output_dir / "runs.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    lines = [
        "# Fixed-bucket dense validation benchmark",
        "",
        "100 candidates, dense=128, fixed buckets 8/16/32/64; p50 across repeats.",
        "",
        "| Scene | Full dense ms | Rank ms | Bucket dense ms | Rank+dense ms | Speedup | Checked | Batches | Selection match | Coverage regression |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['scenario']} | {row['full_dense_p50_ms']:.3f} | "
            f"{row['fixed_ranking_p50_ms']:.3f} | {row['fixed_dense_p50_ms']:.3f} | "
            f"{row['fixed_postprocess_p50_ms']:.3f} | {row['postprocess_speedup']:.3f}x | "
            f"{row['fixed_checked_p50']:.0f} | {row['fixed_batches_p50']:.0f} | "
            f"{row['selection_matches']}/{row['repeats']} | {row['coverage_regressions']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-suites",
        required=True,
        help="Comma-separated generated-suite.yaml paths.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    generated_suites = [
        Path(value).resolve()
        for value in args.generated_suites.split(",")
        if value.strip()
    ]
    output_dir = Path(args.output_dir).resolve()
    rows = execute(generated_suites, output_dir, args.repeats, args.device)
    summaries = summarize(rows)
    write_outputs(rows, summaries, output_dir)
    print(output_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
