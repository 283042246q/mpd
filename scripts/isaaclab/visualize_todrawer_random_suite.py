#!/usr/bin/env python3
"""Render spatial and temporal audit plots for a ToDrawer random suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.isaaclab.benchmark_todrawer_random import BASE_CROSSINGS, SAFE_CONTROL_CROSSINGS
from scripts.isaaclab.todrawer_scenario_validation import load_static_environment_boxes
from scripts.isaaclab.validate_todrawer_random_suite import validate_suite


ANCHOR_COLORS = {
    "A0": "tab:blue",
    "A1": "tab:orange",
    "A2": "tab:green",
    "S0": "tab:purple",
    "S1": "tab:brown",
}


def _spatial_plot(
    payload: dict,
    output: Path,
    axes: tuple[int, int],
    labels: tuple[str, str],
    static_boxes,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for box in static_boxes:
        lower = box.center[axes[0]] - 0.5 * box.size[axes[0]]
        bottom = box.center[axes[1]] - 0.5 * box.size[axes[1]]
        axis.add_patch(
            plt.Rectangle(
                (lower, bottom),
                box.size[axes[0]],
                box.size[axes[1]],
                facecolor="0.75",
                edgecolor="0.45",
                alpha=0.25,
            )
        )
    for scenario in payload["scenarios"]:
        for item in scenario["objects"]:
            anchor_id = item["anchor_id"]
            anchor = item["anchor_position"]
            axis.scatter(
                anchor[axes[0]],
                anchor[axes[1]],
                s=13,
                alpha=0.40,
                color=ANCHOR_COLORS[anchor_id],
            )
    for spec in BASE_CROSSINGS + SAFE_CONTROL_CROSSINGS:
        anchor = spec["anchor"]
        axis.scatter(
            anchor[axes[0]],
            anchor[axes[1]],
            marker="x",
            s=90,
            linewidths=2.0,
            color=ANCHOR_COLORS[spec["id"]],
            label=f"{spec['id']} {spec['name']}",
        )
    axis.set_xlabel(f"{labels[0]} [m]")
    axis.set_ylabel(f"{labels[1]} [m]")
    axis.set_title(f"ToDrawer anchor samples: {labels[0]}–{labels[1]}")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    axis.set_aspect("equal", adjustable="box")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _first_manifest_times(path: Path | None) -> list[tuple[str, float]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    executed = [plan for plan in payload.get("plans", []) if "active_from_s" in plan]
    if not executed:
        return []
    timing = executed[0].get("phase_timing", {})
    return [
        (label, float(timing[key]))
        for label, key in (
            ("first submit", "planning_submitted_s"),
            ("first command", "command_start_s"),
            ("first bridge", "bridge_start_s"),
            ("first handoff", "handoff_s"),
        )
        if key in timing
    ]


def _timeline_plot(payload: dict, output: Path, manifest: Path | None) -> None:
    categories = list(payload.get("categories", []))
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    y_by_category = {category: index for index, category in enumerate(categories)}
    seen_labels = set()
    for scenario in payload["scenarios"]:
        y = y_by_category[scenario["category"]]
        for item in scenario["objects"]:
            anchor_id = item["anchor_id"]
            label = anchor_id if anchor_id not in seen_labels else None
            seen_labels.add(anchor_id)
            axis.scatter(
                item["crossing_time_s"],
                y,
                s=28,
                alpha=0.55,
                color=ANCHOR_COLORS[anchor_id],
                label=label,
            )
    axis.axvspan(4.0, 6.5, color="tab:blue", alpha=0.06, label="primary window")
    for label, value in _first_manifest_times(manifest):
        axis.axvline(value, linestyle="--", linewidth=1.2, label=label)
    axis.set_yticks(range(len(categories)), categories)
    axis.set_xlabel("world-clock time [s]")
    axis.set_title("Scheduled crossing timeline by category")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--static-scene", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.suite.read_text(encoding="utf-8"))
    validate_suite(payload, static_scene=args.static_scene)
    static_boxes = load_static_environment_boxes(args.static_scene)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _spatial_plot(
        payload,
        args.output_dir / "anchors_xy.png",
        (0, 1),
        ("x", "y"),
        static_boxes,
    )
    _spatial_plot(
        payload,
        args.output_dir / "anchors_yz.png",
        (1, 2),
        ("y", "z"),
        static_boxes,
    )
    _timeline_plot(payload, args.output_dir / "scenario_timeline.png", args.manifest)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
