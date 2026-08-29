#!/usr/bin/env python3
"""Run paired Phase-4/Phase-5 fake-hardware benchmarks on random ToDrawer worlds."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = REPO_ROOT / "scripts" / "isaaclab" / "run_dynamic_demo_pipeline.sh"
DEFAULT_AIRUNTIME_ROOT = Path("/home/eric/Projects/physical_ai_runtime")
MODE_SPECS = {
    "phase4": ("phase4", None),
    "scalar_duration": ("phase5", "phase5_scalar_duration"),
    "timing_only": ("phase5", "phase5_timing_only"),
    "joint": ("phase5", "phase5_joint"),
}
CATEGORIES = (
    "single_crossing",
    "staggered_multi",
    "simultaneous_multi",
    "fast_crossing",
    "inflated_dense",
    "safe_control",
)
BASE_CROSSINGS = (
    ((-0.68831415, -1.22503140, 0.65347225), (0.61993897, 0.78465003, 0.0)),
    ((-0.02146948, 0.40876295, 0.46108561), (0.99858189, -0.05323737, 0.0)),
    ((-0.2, -0.16, 0.5), (0.0, 0.0, 1.0)),
)
REPORT_FIELDS = (
    "scenario_id",
    "category",
    "repeat",
    "mode",
    "planner_seed",
    "pipeline_returncode",
    "pipeline_completed",
    "goal_reached",
    "goal_time_s",
    "brake_count",
    "guard_dynamic_collision_rejections",
    "accepted_nonpositive_clearance_count",
    "episode_duration_s",
    "execution_duration_s",
    "executed_plan_count",
    "joint_l2_path_rad",
    "joint_l1_travel_rad",
    "planned_duration_mean_s",
    "hard_minimum_clearance_m",
    "common_window_minimum_clearance_m",
    "clearance_mean_cost",
    "clearance_cvar_cost",
    "dense_environment_clearance_m",
    "dense_self_clearance_m",
    "inference_total_mean_s",
    "inference_total_p95_s",
    "maximum_command_gap_s",
    "no_valid_trajectory_count",
    "error",
    "attempt_dir",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rotate_xy(direction: tuple[float, float, float], angle: float) -> list[float]:
    if abs(direction[2]) > 0.5:
        return [0.0, 0.0, direction[2]]
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        direction[0] * cosine - direction[1] * sine,
        direction[0] * sine + direction[1] * cosine,
        0.0,
    ]


def _random_object(
    rng: random.Random,
    *,
    scenario_index: int,
    object_index: int,
    category: str,
    crossing_time_s: float,
) -> dict[str, Any]:
    anchor_base, direction_base = BASE_CROSSINGS[object_index % len(BASE_CROSSINGS)]
    anchor_jitter = 0.025 if category != "inflated_dense" else 0.05
    anchor = [value + rng.uniform(-anchor_jitter, anchor_jitter) for value in anchor_base]
    if abs(direction_base[2]) > 0.5:
        direction = [0.0, 0.0, rng.choice((-1.0, 1.0))]
    else:
        direction = _rotate_xy(direction_base, rng.uniform(-0.22, 0.22))
        if rng.random() < 0.5:
            direction = [-value for value in direction]

    if category == "fast_crossing":
        speed = rng.uniform(0.24, 0.38)
    elif category == "safe_control":
        speed = rng.uniform(0.08, 0.14)
    else:
        speed = rng.uniform(0.10, 0.26)
    if category == "inflated_dense":
        size = [rng.uniform(0.14, 0.22), rng.uniform(0.12, 0.20), rng.uniform(0.18, 0.30)]
        base_inflation = rng.uniform(0.025, 0.05)
        horizon_rate = rng.uniform(0.015, 0.03)
        position_std = rng.uniform(0.012, 0.025)
    else:
        size = [rng.uniform(0.08, 0.18), rng.uniform(0.08, 0.16), rng.uniform(0.12, 0.24)]
        base_inflation = rng.uniform(0.01, 0.03)
        horizon_rate = rng.uniform(0.005, 0.018)
        position_std = rng.uniform(0.005, 0.015)
    variance = position_std * position_std
    return {
        "id": f"random-{scenario_index:03d}-{object_index:02d}",
        "local_sdf": {"type": "box", "size_xyz": size},
        "anchor_position": anchor,
        "direction": direction,
        "crossing_time_s": crossing_time_s,
        "speed_m_s": speed,
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "position_covariance_3x3": [
            variance,
            0.0,
            0.0,
            0.0,
            variance,
            0.0,
            0.0,
            0.0,
            variance,
        ],
        "inflation": {
            "mode": "linear",
            "base_m": base_inflation,
            "horizon_rate_m_s": horizon_rate,
        },
    }


def generate_suite(count: int, seed: int) -> dict[str, Any]:
    if count < 1:
        raise ValueError("scenario count must be positive")
    rng = random.Random(seed)
    scenarios = []
    for index in range(count):
        category = CATEGORIES[index % len(CATEGORIES)]
        if category == "single_crossing":
            object_count, crossing_times = 1, [rng.uniform(9.0, 17.0)]
        elif category == "staggered_multi":
            object_count = rng.randint(2, 4)
            start = rng.uniform(8.0, 11.0)
            crossing_times = [start + 2.5 * item + rng.uniform(-0.4, 0.4) for item in range(object_count)]
        elif category == "simultaneous_multi":
            object_count = rng.randint(2, 4)
            center = rng.uniform(11.0, 15.0)
            crossing_times = [center + rng.uniform(-0.6, 0.6) for _ in range(object_count)]
        elif category == "fast_crossing":
            object_count = rng.randint(2, 3)
            crossing_times = [rng.uniform(9.0, 17.0) for _ in range(object_count)]
        elif category == "inflated_dense":
            object_count = rng.randint(4, 5)
            crossing_times = [rng.uniform(9.0, 17.0) for _ in range(object_count)]
        else:
            object_count = rng.randint(1, 2)
            crossing_times = [rng.uniform(55.0, 70.0) for _ in range(object_count)]
        objects = [
            _random_object(
                rng,
                scenario_index=index,
                object_index=object_index,
                category=category,
                crossing_time_s=crossing_times[object_index],
            )
            for object_index in range(object_count)
        ]
        scenarios.append(
            {
                "schema": "mpd_todrawer_dynamic_scenario",
                "schema_version": 1,
                "id": f"scenario-{index:03d}",
                "category": category,
                "frame_id": "fr3_link0",
                "objects": objects,
            }
        )
    return {
        "schema": "mpd_todrawer_random_suite",
        "schema_version": 1,
        "suite_seed": seed,
        "scenario_count": count,
        "categories": list(CATEGORIES),
        "scenarios": scenarios,
    }


def materialize_suite(output_dir: Path, count: int, seed: int) -> dict[str, Any]:
    suite_path = output_dir / "suite.json"
    if suite_path.is_file():
        suite = _read_json(suite_path)
        if suite.get("suite_seed") != seed or suite.get("scenario_count") != count:
            raise ValueError(
                "existing suite.json does not match --suite-seed/--scenario-count; use a new output directory"
            )
        return suite
    suite = generate_suite(count, seed)
    for scenario in suite["scenarios"]:
        _write_json(output_dir / "scenarios" / f"{scenario['id']}.json", scenario)
    _write_json(suite_path, suite)
    return suite


def _finite(values: list[Any]) -> list[float]:
    result = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def _describe(values: list[Any]) -> dict[str, Any]:
    numbers = np.asarray(_finite(values), dtype=np.float64)
    if not len(numbers):
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(len(numbers)),
        "mean": float(numbers.mean()),
        "median": float(np.median(numbers)),
        "p95": float(np.percentile(numbers, 95)),
        "min": float(numbers.min()),
        "max": float(numbers.max()),
    }


def _trajectory_segment_metrics(manifest_path: Path, plans: list[dict[str, Any]]) -> tuple[float, float]:
    total_l2 = 0.0
    total_l1 = 0.0
    for plan in plans:
        if "active_from_s" not in plan or "active_until_s" not in plan:
            continue
        archive = manifest_path.parent / plan["trajectory"]
        if not archive.is_file():
            continue
        with np.load(archive, allow_pickle=False) as data:
            if "positions" not in data:
                continue
            positions = np.asarray(data["positions"], dtype=np.float64)
            times = np.asarray(data["time_from_start"], dtype=np.float64)
        relative_start = max(float(times[0]), float(plan["active_from_s"]) - float(plan["start_s"]))
        relative_end = min(float(times[-1]), float(plan["active_until_s"]) - float(plan["start_s"]))
        if relative_end <= relative_start:
            continue
        interior = (times > relative_start) & (times < relative_end)
        sample_times = np.concatenate(([relative_start], times[interior], [relative_end]))
        samples = np.column_stack(
            [np.interp(sample_times, times, positions[:, joint]) for joint in range(positions.shape[1])]
        )
        delta = np.diff(samples, axis=0)
        total_l2 += float(np.linalg.norm(delta, axis=1).sum())
        total_l1 += float(np.abs(delta).sum())
    return total_l2, total_l1


def _selected_clearance(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for plan in plans:
        if "active_from_s" not in plan:
            continue
        diagnostics = plan.get("candidate_clearance_diagnostics") or []
        finite = [item for item in diagnostics if item.get("composite_cost") is not None]
        if finite:
            selected.append(min(finite, key=lambda item: float(item["composite_cost"])))
    return selected


def _parse_ros_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    start_match = re.search(r"\[(\d+\.\d+)\].*dynamic MPD replanner started", text)
    goal_match = re.search(r"\[(\d+\.\d+)\].*goal reached; holding position", text)
    goal_time = None
    if start_match and goal_match:
        goal_time = float(goal_match.group(1)) - float(start_match.group(1))
    reasons: dict[str, int] = {}
    for reason in re.findall(r'"reason":\s*"([^"]+)"', text):
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "goal_reached": goal_match is not None,
        "goal_time_s": goal_time,
        "candidate_rejection_reasons": reasons,
        "no_valid_trajectory_count": text.count("NoValidTrajectoryError"),
        "jtc_error_count": len(re.findall(r"JTC dynamic plan .* entered", text)),
    }


def extract_run_metrics(attempt_dir: Path, run_spec: dict[str, Any], returncode: int) -> dict[str, Any]:
    manifest_path = attempt_dir / "episode" / "replay-manifest.json"
    metrics: dict[str, Any] = {
        **run_spec,
        "pipeline_returncode": int(returncode),
        "pipeline_completed": returncode == 0 and manifest_path.is_file(),
        "attempt_dir": attempt_dir.as_posix(),
        "error": None,
    }
    if not manifest_path.is_file():
        metrics["error"] = "replay manifest missing"
        return metrics
    manifest = _read_json(manifest_path)
    plans = manifest.get("plans", [])
    executed = [plan for plan in plans if "active_from_s" in plan and "active_until_s" in plan]
    events = manifest.get("events", [])
    timing_path = attempt_dir / "to_drawer-replan-timing.json"
    timing = _read_json(timing_path) if timing_path.is_file() else {}
    ros = _parse_ros_log(attempt_dir / "ros-replan.log")
    joint_l2, joint_l1 = _trajectory_segment_metrics(manifest_path, executed)
    selected_clearance = _selected_clearance(executed)

    result_payloads = []
    for result_path in sorted((attempt_dir / "planner-results").glob("*/result.json")):
        try:
            payload = _read_json(result_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("status") == "success":
            result_payloads.append(payload)
    inference_times = _finite(
        [payload.get("timing", {}).get("inference_total_sec") for payload in result_payloads]
    )
    planned_durations = _finite(
        [plan.get("phase_timing", {}).get("mpd_suffix_s") for plan in executed]
    )
    hard_clearance = _finite(
        [item.get("hard_minimum_clearance_m") for item in selected_clearance]
    )
    common_clearance = _finite(
        [item.get("common_window_minimum_clearance_m") for item in selected_clearance]
    )
    dense_environment = _finite(
        [payload.get("trajectory", {}).get("minimum_environment_clearance_m") for payload in result_payloads]
    )
    dense_self = _finite(
        [payload.get("trajectory", {}).get("minimum_self_clearance_m") for payload in result_payloads]
    )
    metrics.update(
        goal_reached=ros["goal_reached"],
        goal_time_s=ros["goal_time_s"],
        brake_count=sum(event.get("type") == "brake" for event in events),
        guard_dynamic_collision_rejections=ros["candidate_rejection_reasons"].get(
            "dynamic_collision", 0
        ),
        candidate_rejection_reasons=ros["candidate_rejection_reasons"],
        accepted_nonpositive_clearance_count=sum(value <= 0.0 for value in hard_clearance),
        episode_duration_s=float(manifest.get("duration_s", 0.0)),
        execution_duration_s=sum(
            float(plan["active_until_s"]) - float(plan["active_from_s"]) for plan in executed
        ),
        executed_plan_count=len(executed),
        plan_record_count=len(plans),
        joint_l2_path_rad=joint_l2,
        joint_l1_travel_rad=joint_l1,
        planned_duration_mean_s=(float(np.mean(planned_durations)) if planned_durations else None),
        hard_minimum_clearance_m=min(hard_clearance, default=None),
        common_window_minimum_clearance_m=min(common_clearance, default=None),
        clearance_mean_cost=(
            float(np.mean(_finite([item.get("clearance_mean_cost") for item in selected_clearance])))
            if _finite([item.get("clearance_mean_cost") for item in selected_clearance])
            else None
        ),
        clearance_cvar_cost=(
            float(np.mean(_finite([item.get("clearance_cvar_cost") for item in selected_clearance])))
            if _finite([item.get("clearance_cvar_cost") for item in selected_clearance])
            else None
        ),
        dense_environment_clearance_m=min(dense_environment, default=None),
        dense_self_clearance_m=min(dense_self, default=None),
        inference_total_mean_s=(float(np.mean(inference_times)) if inference_times else None),
        inference_total_p95_s=(float(np.percentile(inference_times, 95)) if inference_times else None),
        maximum_command_gap_s=timing.get("maximum_command_gap_s"),
        no_valid_trajectory_count=ros["no_valid_trajectory_count"],
        jtc_error_count=ros["jtc_error_count"],
    )
    return metrics


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runs": len(rows),
        "completed": sum(bool(row.get("pipeline_completed")) for row in rows),
        "goal_reached": sum(bool(row.get("goal_reached")) for row in rows),
        "brake_runs": sum(int(row.get("brake_count") or 0) > 0 for row in rows),
        "brake_events": sum(int(row.get("brake_count") or 0) for row in rows),
        "guard_dynamic_collision_rejections": sum(
            int(row.get("guard_dynamic_collision_rejections") or 0) for row in rows
        ),
        "accepted_nonpositive_clearance_count": sum(
            int(row.get("accepted_nonpositive_clearance_count") or 0) for row in rows
        ),
        "goal_time_s": _describe([row.get("goal_time_s") for row in rows]),
        "episode_duration_s": _describe([row.get("episode_duration_s") for row in rows]),
        "execution_duration_s": _describe([row.get("execution_duration_s") for row in rows]),
        "planned_duration_s": _describe([row.get("planned_duration_mean_s") for row in rows]),
        "joint_l2_path_rad": _describe([row.get("joint_l2_path_rad") for row in rows]),
        "joint_l1_travel_rad": _describe([row.get("joint_l1_travel_rad") for row in rows]),
        "hard_minimum_clearance_m": _describe(
            [row.get("hard_minimum_clearance_m") for row in rows]
        ),
        "common_window_minimum_clearance_m": _describe(
            [row.get("common_window_minimum_clearance_m") for row in rows]
        ),
        "clearance_mean_cost": _describe([row.get("clearance_mean_cost") for row in rows]),
        "clearance_cvar_cost": _describe([row.get("clearance_cvar_cost") for row in rows]),
        "dense_environment_clearance_m": _describe(
            [row.get("dense_environment_clearance_m") for row in rows]
        ),
        "dense_self_clearance_m": _describe(
            [row.get("dense_self_clearance_m") for row in rows]
        ),
        "inference_total_s": _describe([row.get("inference_total_mean_s") for row in rows]),
        "maximum_command_gap_s": _describe(
            [row.get("maximum_command_gap_s") for row in rows]
        ),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def write_reports(output_dir: Path, rows: list[dict[str, Any]], suite: dict[str, Any]) -> None:
    reports = output_dir / "report"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    by_mode = {
        mode: _aggregate([row for row in rows if row.get("mode") == mode])
        for mode in MODE_SPECS
    }
    by_category = {
        category: {
            mode: _aggregate(
                [
                    row
                    for row in rows
                    if row.get("category") == category and row.get("mode") == mode
                ]
            )
            for mode in MODE_SPECS
        }
        for category in CATEGORIES
    }
    summary = {
        "schema": "mpd_todrawer_random_benchmark_report",
        "schema_version": 1,
        "suite_seed": suite["suite_seed"],
        "scenario_count": suite["scenario_count"],
        "run_count": len(rows),
        "metric_semantics": {
            "collision": "guard/DenseCheck prediction; physical contact is not measured by passive replay",
            "joint_l2_path_rad": "sum of Euclidean joint increments over realized command intervals",
            "clearance": "selected candidate guard clearance plus successful MPD DenseCheck clearance",
        },
        "by_mode": by_mode,
        "by_category": by_category,
        "runs": rows,
    }
    _write_json(reports / "summary.json", summary)

    lines = [
        "# ToDrawer 随机动态重规划基准报告",
        "",
        f"- suite seed：`{suite['suite_seed']}`",
        f"- 场景数：{suite['scenario_count']}",
        f"- 已发现运行：{len(rows)}",
        "- 模式：Phase 4、scalar duration、timing only、joint",
        "",
        "## 指标口径",
        "",
        "`碰撞`统计 guard/DenseCheck 的预测碰撞拒绝；被接受轨迹出现非正 hard clearance 会单独计数。被动 replay 不测量真实物理接触，因此报告不会把预测碰撞写成实际接触。总路径为实际生效命令区间的关节空间路径。",
        "",
        "## 安全与完成情况",
        "",
        "| 模式 | 完成/总数 | 到达目标 | brake runs | brake events | 动态碰撞拒绝 | 非正 clearance | 最大命令间隙 mean s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, data in by_mode.items():
        lines.append(
            "| "
            + " | ".join(
                (
                    mode,
                    f"{data['completed']}/{data['runs']}",
                    str(data["goal_reached"]),
                    str(data["brake_runs"]),
                    str(data["brake_events"]),
                    str(data["guard_dynamic_collision_rejections"]),
                    str(data["accepted_nonpositive_clearance_count"]),
                    _fmt(data["maximum_command_gap_s"]["mean"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 时长、路径与推理耗时",
            "",
            "| 模式 | goal time mean s | episode mean s | 执行时长 mean s | 规划轨迹时长 mean s | path L2 mean rad | joint L1 mean rad | inference mean s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, data in by_mode.items():
        lines.append(
            f"| {mode} | {_fmt(data['goal_time_s']['mean'])} | "
            f"{_fmt(data['episode_duration_s']['mean'])} | "
            f"{_fmt(data['execution_duration_s']['mean'])} | "
            f"{_fmt(data['planned_duration_s']['mean'])} | "
            f"{_fmt(data['joint_l2_path_rad']['mean'])} | "
            f"{_fmt(data['joint_l1_travel_rad']['mean'])} | "
            f"{_fmt(data['inference_total_s']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Clearance 汇总",
            "",
            "| 模式 | hard min m | common-window min m | mean cost mean | CVaR cost mean | DenseCheck env min m | DenseCheck self min m |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, data in by_mode.items():
        lines.append(
            f"| {mode} | {_fmt(data['hard_minimum_clearance_m']['min'])} | "
            f"{_fmt(data['common_window_minimum_clearance_m']['min'])} | "
            f"{_fmt(data['clearance_mean_cost']['mean'])} | "
            f"{_fmt(data['clearance_cvar_cost']['mean'])} | "
            f"{_fmt(data['dense_environment_clearance_m']['min'])} | "
            f"{_fmt(data['dense_self_clearance_m']['min'])} |"
        )
    lines.extend(
        [
            "",
            "## 分类结果",
            "",
            "| 场景类型 | 模式 | 完成/总数 | 到达目标 | brake runs | path L2 mean rad | hard clearance min m |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for category in CATEGORIES:
        for mode, data in by_category[category].items():
            if not data["runs"]:
                continue
            lines.append(
                f"| {category} | {mode} | {data['completed']}/{data['runs']} | "
                f"{data['goal_reached']} | {data['brake_runs']} | "
                f"{_fmt(data['joint_l2_path_rad']['mean'])} | "
                f"{_fmt(data['hard_minimum_clearance_m']['min'])} |"
            )
    completed_modes = {mode: data for mode, data in by_mode.items() if data["completed"]}
    lines.extend(["", "## 描述性结论", ""])
    if completed_modes:
        goal_best = max(
            completed_modes,
            key=lambda mode: completed_modes[mode]["goal_reached"] / completed_modes[mode]["completed"],
        )
        brake_best = min(
            completed_modes,
            key=lambda mode: completed_modes[mode]["brake_runs"] / completed_modes[mode]["completed"],
        )
        lines.append(f"- 当前样本目标到达率最高：`{goal_best}`。")
        lines.append(f"- 当前样本 brake run 比例最低：`{brake_best}`。")
        clearance_modes = {
            mode: data
            for mode, data in completed_modes.items()
            if data["hard_minimum_clearance_m"]["min"] is not None
        }
        if clearance_modes:
            clearance_best = max(
                clearance_modes,
                key=lambda mode: clearance_modes[mode]["hard_minimum_clearance_m"]["min"],
            )
            lines.append(f"- 当前样本最差 hard clearance 最大：`{clearance_best}`。")
    lines.append("- 以上是描述性统计；场景/重复数较小时不代表统计显著性。")
    failures = [row for row in rows if not row.get("pipeline_completed")]
    if failures:
        lines.extend(["", "## 未完成运行", ""])
        for row in failures:
            lines.append(
                f"- `{row.get('scenario_id')}/{row.get('repeat')}/{row.get('mode')}`: "
                f"{row.get('error') or 'pipeline failed'} (`{row.get('attempt_dir')}`)"
            )
    (reports / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _attempt_dir(output_dir: Path, scenario_id: str, repeat: int, mode: str) -> Path:
    root = output_dir / "runs" / scenario_id / f"repeat-{repeat:02d}" / mode
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(root.glob("attempt-*"))
    return root / f"attempt-{len(existing) + 1:03d}"


def _existing_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_dir.glob("runs/**/run-metrics.json")):
        try:
            rows.append(_read_json(path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("scenario_id"), row.get("repeat"), row.get("mode"))
        latest[key] = row
    return sorted(
        latest.values(),
        key=lambda row: (str(row.get("scenario_id")), int(row.get("repeat", 0)), str(row.get("mode"))),
    )


def _run_command(command: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        return int(process.wait())


def run_benchmark(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = materialize_suite(output_dir, args.scenario_count, args.suite_seed)
    if args.report_only:
        rows = _existing_rows(output_dir)
        write_reports(output_dir, rows, suite)
        print(output_dir / "report" / "report.md")
        return 0

    airuntime_root = Path(os.environ.get("AIRUNTIME_ROOT", DEFAULT_AIRUNTIME_ROOT)).resolve()
    if not args.skip_build:
        build_log = output_dir / "ros-build.log"
        print("[setup] building mpd_dynamic_planner_adapter", flush=True)
        status = _run_command(
            ["pixi", "run", "build", "--packages-up-to", "mpd_dynamic_planner_adapter"],
            airuntime_root,
            build_log,
        )
        if status != 0:
            print(f"ROS build failed; see {build_log}", file=sys.stderr)
            return status

    completed_keys = {
        (row.get("scenario_id"), int(row.get("repeat", 0)), row.get("mode"))
        for row in _existing_rows(output_dir)
        if row.get("pipeline_completed")
    }
    failures = 0
    for repeat in range(args.repeats):
        for scenario_index, scenario in enumerate(suite["scenarios"]):
            scenario_path = output_dir / "scenarios" / f"{scenario['id']}.json"
            modes = list(args.modes)
            shift = (scenario_index + repeat) % len(modes)
            modes = modes[shift:] + modes[:shift]
            planner_seed = (args.suite_seed + 1009 * scenario_index + 9176 * repeat) % 2147483647
            for mode in modes:
                key = (scenario["id"], repeat, mode)
                if key in completed_keys:
                    print(f"[skip] completed {scenario['id']} repeat={repeat} mode={mode}")
                    continue
                attempt_dir = _attempt_dir(output_dir, scenario["id"], repeat, mode)
                phase, timing_mode = MODE_SPECS[mode]
                run_spec = {
                    "scenario_id": scenario["id"],
                    "category": scenario["category"],
                    "repeat": repeat,
                    "mode": mode,
                    "phase": phase,
                    "timing_mode": timing_mode,
                    "planner_seed": planner_seed,
                    "scenario_file": scenario_path.as_posix(),
                }
                command = [
                    PIPELINE.as_posix(),
                    "--profile",
                    "to_drawer",
                    "--phase",
                    phase,
                    "--world-scenario-file",
                    scenario_path.as_posix(),
                    "--planner-seed",
                    str(planner_seed),
                    "--duration-sec",
                    str(args.duration_sec),
                    "--plan-rate-hz",
                    str(args.plan_rate_hz),
                    "--output-dir",
                    attempt_dir.as_posix(),
                    "--allow-brake",
                    "--skip-build",
                ]
                if timing_mode is not None:
                    command.extend(("--timing-mode", timing_mode))
                if not args.render:
                    command.append("--skip-render")
                _write_json(attempt_dir / "run-spec.json", {**run_spec, "command": command})
                print(
                    f"[run] {scenario['id']} category={scenario['category']} "
                    f"repeat={repeat} mode={mode} seed={planner_seed}",
                    flush=True,
                )
                started = time.time()
                returncode = _run_command(command, REPO_ROOT, attempt_dir / "pipeline.log")
                metrics = extract_run_metrics(attempt_dir, run_spec, returncode)
                metrics["wall_time_s"] = time.time() - started
                _write_json(attempt_dir / "run-metrics.json", metrics)
                rows = _existing_rows(output_dir)
                write_reports(output_dir, rows, suite)
                if returncode != 0:
                    failures += 1
                    if args.fail_fast:
                        return returncode
    print(f"report: {output_dir / 'report' / 'report.md'}")
    return 1 if failures else 0


def _parser() -> argparse.ArgumentParser:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "scripts" / "isaaclab" / "logs" / "todrawer-random-benchmark" / timestamp,
    )
    parser.add_argument("--scenario-count", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--suite-seed", type=int, default=20260829)
    parser.add_argument("--duration-sec", type=float, default=35.0)
    parser.add_argument("--plan-rate-hz", type=float, default=1.0)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODE_SPECS),
        default=list(MODE_SPECS),
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--render", action="store_true", help="Render every episode; disabled by default")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeats < 1 or args.duration_sec <= 0.0 or args.plan_rate_hz <= 0.0:
        raise SystemExit("repeats, duration-sec, and plan-rate-hz must be positive")
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
