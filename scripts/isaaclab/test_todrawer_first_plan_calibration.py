#!/usr/bin/env python3
"""Summarize first-plan timing and test median-based ToDrawer crossing calibration.

This is an offline analysis: it reads a completed benchmark report and creates
calibrated scenario JSON. It never starts ROS, Isaac Lab, or the planner.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from scripts.isaaclab.benchmark_todrawer_random import (
    DIFFICULTY_BY_CATEGORY,
    MODE_SPECS,
    _calibrate_crossing_from_report,
    materialize_suite,
)


def _timing_value(row: dict[str, str]) -> float | None:
    raw = row.get("first_plan_completed_from_world_s")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def _summarize(rows: list[dict[str, str]], modes: list[str], threshold_s: float,
               expected_attempts: int | None = None) -> dict:
    output = {}
    for mode in modes:
        hard_rows = [row for row in rows if row.get("mode") == mode and row.get("difficulty") == "hard"]
        measured = [(row, value) for row in hard_rows if (value := _timing_value(row)) is not None]
        values = [value for _, value in measured]
        categories = {}
        for category in sorted({row.get("category", "unknown") for row in hard_rows}):
            category_rows = [row for row in hard_rows if row.get("category", "unknown") == category]
            category_values = [value for row, value in measured if row.get("category", "unknown") == category]
            categories[category] = {
                "attempts": len(category_rows),
                "returned_results": len(category_values),
                "missing_results": len(category_rows) - len(category_values),
                "median_s": statistics.median(category_values) if category_values else None,
                "over_threshold_count": sum(value > threshold_s for value in category_values),
            }
        status_counts: dict[str, int] = {}
        for row, _ in measured:
            status = row.get("first_plan_status") or "unknown"
            status_counts[status] = status_counts.get(status, 0) + 1
        output[mode] = {
            "expected_hard_scene_attempts": expected_attempts,
            "hard_scene_attempts": len(hard_rows),
            "unattempted": (max(0, expected_attempts - len(hard_rows))
                            if expected_attempts is not None else 0),
            "returned_results": len(values),
            "missing_results": (max(0, expected_attempts - len(values))
                                if expected_attempts is not None else len(hard_rows) - len(values)),
            "median_s": statistics.median(values) if values else None,
            "mean_s": statistics.mean(values) if values else None,
            "over_threshold_count": sum(value > threshold_s for value in values),
            "planning_failed_results": sum(
                count for status, count in status_counts.items() if status != "success"
            ),
            "first_plan_status_counts": status_counts,
            "by_category": categories,
        }
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=tuple(MODE_SPECS))
    parser.add_argument("--threshold-sec", type=float, default=2.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not math.isfinite(args.threshold_sec) or args.threshold_sec <= 0.0:
        raise SystemExit("--threshold-sec must be a positive finite number")
    report_path = args.runs_csv.resolve()
    source_suite_path = report_path.parent.parent / "suite.json"
    with report_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    source_suite = json.loads(source_suite_path.read_text(encoding="utf-8"))
    modes = list(dict.fromkeys(args.modes or source_suite["modes"]))
    hard_scenario_count = sum(
        scenario.get("difficulty", DIFFICULTY_BY_CATEGORY.get(scenario.get("category"))) == "hard"
        for scenario in source_suite["scenarios"]
    )
    protocol_count = len(source_suite.get("timing_protocols") or [source_suite["timing_protocol"]])
    expected_attempts = hard_scenario_count * int(source_suite["planner_repeats"]) * protocol_count
    summary = _summarize(rows, modes, args.threshold_sec, expected_attempts)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "todrawer_first_plan_median_calibration_test_v1",
        "source_runs_csv": report_path.as_posix(),
        "threshold_s": args.threshold_sec,
        "timing_scope": "hard scenes; earliest completed planner result regardless of planning status",
        "modes": summary,
        "calibration": None,
        "calibrated_suite": None,
        "calibration_error": None,
    }
    missing_modes = [mode for mode, item in summary.items() if item["missing_results"] > 0]
    if not missing_modes:
        try:
            calibration = _calibrate_crossing_from_report(report_path, modes)
            result["calibration"] = calibration
            suite = materialize_suite(
                output_dir,
                int(source_suite["environment_count_per_category"]),
                int(source_suite["suite_seed"]),
                timing_protocol="motion_aligned",
                modes=modes,
                planner_repeats=int(source_suite["planner_repeats"]),
            )
            result["calibrated_suite"] = (output_dir / "suite.json").as_posix()
            result["calibrated_scenario_count"] = suite["scenario_count"]
        except (ValueError, RuntimeError) as error:
            result["calibration_error"] = str(error)
    result_path = output_dir / "first-plan-calibration-test.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("mode | median s | > threshold / returned | failed results | missing results")
    for mode, item in summary.items():
        median = "—" if item["median_s"] is None else f"{item['median_s']:.3f}"
        print(
            f"{mode} | {median} | {item['over_threshold_count']} / "
            f"{item['returned_results']} | {item['planning_failed_results']} | "
            f"{item['missing_results']}"
        )
    print(f"result: {result_path}")
    if missing_modes:
        print("calibration not applied: incomplete first-plan results for " + ", ".join(missing_modes))
        return 2
    if result["calibration_error"] is not None:
        print("calibration not applied: " + result["calibration_error"])
        return 2
    print(f"calibrated suite: {result['calibrated_suite']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
