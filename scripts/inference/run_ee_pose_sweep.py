from __future__ import annotations

import argparse
import csv
from itertools import product
import os
from pathlib import Path
import statistics
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = REPO_ROOT / "scripts" / "inference"
INFERENCE_SCRIPT = INFERENCE_DIR / "inference.py"
SWEEP_CONFIG = INFERENCE_DIR / "cfgs" / "config_EnvWarehouse-RobotPanda-paper_sweep.yaml"


def inference_subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    allocator_config = environment.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:" not in allocator_config:
        allocator_config = ",".join(option for option in (allocator_config, "expandable_segments:True") if option)
    environment["PYTORCH_CUDA_ALLOC_CONF"] = allocator_config
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the lightweight MPD EE-pose parameter sweep.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "log" / "test2")
    parser.add_argument("--contexts", type=int, default=3)
    parser.add_argument(
        "--contexts-per-process",
        type=int,
        default=3,
        help="Maximum contexts evaluated by one inference subprocess.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="One-based experiment index to start running; earlier groups are loaded from existing reports.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def experiment_grid():
    ee_weight_options = (
        ("constant", 1.0, 1.0, "1"),
        ("constant", 3.0, 3.0, "3"),
        ("linear", 1.0, 3.0, "1to3"),
    )
    prior_weight_options = (
        ("constant", 0.25, 0.25, "0.25"),
        ("constant", 0.5, 0.5, "0.5"),
        ("linear", 0.5, 0.25, "0.5to0.25"),
    )
    for guide_steps, gradients_per_step, ee_option, prior_option in product(
        (3, 5),
        (4, 6),
        ee_weight_options,
        prior_weight_options,
    ):
        ee_schedule, ee_weight_start, ee_weight_end, ee_label = ee_option
        prior_schedule, prior_weight_start, prior_weight_end, prior_label = prior_option
        yield {
            "ee_pose_goal_weight_schedule": ee_schedule,
            "ee_pose_goal_weight": ee_weight_start,
            "ee_pose_goal_weight_end": ee_weight_end,
            "guide_steps": guide_steps,
            "guide_fraction": 0.2 if guide_steps == 3 else 0.333,
            "n_guide_steps": gradients_per_step,
            "prior_weight_schedule": prior_schedule,
            "prior_weight": prior_weight_start,
            "prior_weight_end": prior_weight_end,
            "total_cost_gradients": guide_steps * gradients_per_step,
            "name": f"w{ee_label}-i{guide_steps}-m{gradients_per_step}-p{prior_label}",
        }


def parse_optional_float(value: str):
    return None if value == "N/A" else float(value)


def mean_available(context_results: list[dict], key: str):
    available_values = [result[key] for result in context_results if result.get(key) is not None]
    return statistics.fmean(available_values) if available_values else None


def parse_report(report_path: Path) -> dict:
    values = {}
    section = ""
    for raw_line in report_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line
            continue
        if ":" not in line:
            continue
        key, value = (item.strip() for item in line.split(":", 1))
        if section == "[END-EFFECTOR GOAL ERROR - VALID TRAJECTORIES]":
            if key == "position_error_mean_m":
                values["position_error_mean_m"] = parse_optional_float(value)
            elif key == "orientation_error_mean_deg":
                values["orientation_error_mean_deg"] = parse_optional_float(value)
        elif section == "[MPD VALIDITY]" and key == "valid_rate":
            values["valid_rate"] = float(value.split()[0])
        elif section == "[TIMING]" and key == "inference_total_sec":
            values["inference_total_sec"] = float(value)
    return values


def load_existing_experiment(args: argparse.Namespace, experiment: dict) -> list[dict]:
    output_dir = args.output_root / experiment["name"]
    context_results = []
    for batch_start in range(0, args.contexts, args.contexts_per_process):
        batch_size = min(args.contexts_per_process, args.contexts - batch_start)
        batch_seed = args.seed + batch_start
        report_dir = output_dir / f"batch-{batch_start:03d}" / str(batch_seed)
        report_paths = sorted(report_dir.glob("inference-report-*.txt"))
        if len(report_paths) != batch_size:
            raise RuntimeError(
                f"Cannot reuse experiment {experiment['name']}: found {len(report_paths)} reports in "
                f"{report_dir}; expected {batch_size}. Use an earlier --start-index or rerun from 1."
            )
        context_results.extend(parse_report(path) for path in report_paths)
    return context_results


def run_experiment(args: argparse.Namespace, experiment: dict) -> list[dict]:
    output_dir = args.output_root / experiment["name"]
    context_results = []
    for batch_start in range(0, args.contexts, args.contexts_per_process):
        batch_size = min(args.contexts_per_process, args.contexts - batch_start)
        batch_seed = args.seed + batch_start
        batch_output_dir = output_dir / f"batch-{batch_start:03d}"
        print(
            f"    contexts {batch_start + 1}-{batch_start + batch_size}/{args.contexts}",
            flush=True,
        )
        command = [
            sys.executable,
            str(INFERENCE_SCRIPT),
            "--cfg_inference_path",
            str(SWEEP_CONFIG),
            "--selection_start_goal",
            "validation",
            "--n_start_goal_states",
            str(batch_size),
            "--ee_pose_goal_weight_override",
            str(experiment["ee_pose_goal_weight"]),
            "--t_start_guide_steps_fraction_override",
            str(experiment["guide_fraction"]),
            "--n_guide_steps_override",
            str(experiment["n_guide_steps"]),
            "--ddim_scale_grad_prior_override",
            str(experiment["prior_weight"]),
            "--sim_backend",
            "none",
            "--run_evaluation_issac_gym",
            "False",
            "--run_evaluation_isaac_lab",
            "False",
            "--isaaclab_replay",
            "False",
            "--render_joint_space_time_iters",
            "False",
            "--render_joint_space_env_iters",
            "False",
            "--render_env_robot_opt_iters",
            "False",
            "--render_env_robot_trajectories",
            "False",
            "--render_pybullet",
            "False",
            "--render_isaacgym_viewer",
            "False",
            "--render_isaacgym_movie",
            "False",
            "--save_args_inference",
            "False",
            "--save_results_single_plan",
            "False",
            "--lightweight_output",
            "True",
            "--device",
            args.device,
            "--seed",
            str(batch_seed),
            "--results_dir",
            str(batch_output_dir),
        ]
        if experiment["ee_pose_goal_weight_schedule"] == "linear":
            command.extend(
                [
                    "--ee_pose_goal_weight_end_override",
                    str(experiment["ee_pose_goal_weight_end"]),
                ]
            )
        if experiment["prior_weight_schedule"] == "linear":
            command.extend(
                [
                    "--ddim_scale_grad_prior_end_override",
                    str(experiment["prior_weight_end"]),
                ]
            )
        completed = subprocess.run(
            command,
            cwd=INFERENCE_DIR,
            env=inference_subprocess_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-80:])
            raise RuntimeError(
                f"Experiment {experiment['name']} failed in contexts "
                f"{batch_start + 1}-{batch_start + batch_size}:\n{tail}"
            )

        report_dir = batch_output_dir / str(batch_seed)
        report_paths = sorted(report_dir.glob("inference-report-*.txt"))
        if len(report_paths) != batch_size:
            raise RuntimeError(
                f"Experiment {experiment['name']} produced {len(report_paths)} reports in "
                f"batch {batch_start}; expected {batch_size}."
            )
        context_results.extend(parse_report(path) for path in report_paths)
    return context_results


def main() -> None:
    args = parse_args()
    if args.contexts < 1 or args.contexts_per_process < 1:
        raise ValueError("--contexts and --contexts-per-process must both be positive.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    experiments = list(experiment_grid())
    if not 1 <= args.start_index <= len(experiments):
        raise ValueError(f"--start-index must be between 1 and {len(experiments)}.")
    for index, experiment in enumerate(experiments, start=1):
        reuse_existing = index < args.start_index
        mode = "reuse existing" if reuse_existing else "run"
        print(f"[{index:02d}/{len(experiments):02d}] {experiment['name']} ({mode})", flush=True)
        if reuse_existing:
            context_results = load_existing_experiment(args, experiment)
        else:
            context_results = run_experiment(args, experiment)
        contexts_with_pose_metrics = sum(
            result.get("position_error_mean_m") is not None and result.get("orientation_error_mean_deg") is not None
            for result in context_results
        )
        row = {
            **experiment,
            "contexts": args.contexts,
            "contexts_per_process": args.contexts_per_process,
            "contexts_with_pose_metrics": contexts_with_pose_metrics,
            "pose_metric_coverage": contexts_with_pose_metrics / args.contexts,
            "position_error_mean_m": mean_available(context_results, "position_error_mean_m"),
            "orientation_error_mean_deg": mean_available(context_results, "orientation_error_mean_deg"),
            "valid_rate_mean": statistics.fmean(result["valid_rate"] for result in context_results),
            "inference_total_sec_mean": statistics.fmean(result["inference_total_sec"] for result in context_results),
        }
        rows.append(row)
        position_text = "N/A" if row["position_error_mean_m"] is None else f"{row['position_error_mean_m']:.6f} m"
        orientation_text = (
            "N/A" if row["orientation_error_mean_deg"] is None else f"{row['orientation_error_mean_deg']:.6f} deg"
        )
        print(
            f"  pos={position_text}, ori={orientation_text}, "
            f"valid={row['valid_rate_mean']:.3f}, "
            f"pose_contexts={contexts_with_pose_metrics}/{args.contexts}",
            flush=True,
        )

    fieldnames = list(rows[0].keys())
    summary_path = args.output_root / "sweep-summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ranked_path = args.output_root / "sweep-ranked.csv"
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            -row["pose_metric_coverage"],
            float("inf") if row["position_error_mean_m"] is None else row["position_error_mean_m"],
            float("inf") if row["orientation_error_mean_deg"] is None else row["orientation_error_mean_deg"],
            -row["valid_rate_mean"],
        ),
    )
    with ranked_path.open("w", encoding="utf-8", newline="") as ranked_file:
        writer = csv.DictWriter(ranked_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked_rows)

    print(f"Summary: {summary_path}", flush=True)
    print(f"Ranked:  {ranked_path}", flush=True)


if __name__ == "__main__":
    main()
