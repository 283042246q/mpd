"""Measure final teacher acceptance at a fixed attempt budget per mode/count.

Each scene uses one motion mode for every obstacle. Exit code 2 from the
generator is expected: the positive target deliberately exceeds the budget.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from scripts.spacetime_data.generate_warehouse_spacetime import REPOSITORY, SOURCE, atomic_json, sha256_file
from scripts.spacetime_data.warehouse_world import MOTION_MODES


def summarize_case(report, *, mode, obstacles, attempts):
    if report["attempts"] != attempts:
        raise ValueError("incomplete fixed-attempt case: %s/n%d" % (mode, obstacles))
    failures = report["failure_counts"]
    stats = report["session_pipeline_statistics"]
    accepted = report["accepted"]
    source_failed = failures.get("source_static_self_or_limits", 0)
    world_failed = failures.get("world_anchor_or_initial_collision", 0)
    no_solution = failures.get("teacher_no_solution", 0)
    # Report failures are cumulative across committed shards, whereas pipeline
    # counters describe only the latest session after a resume.
    submitted = attempts - source_failed - world_failed
    solved = submitted - no_solution
    if not (accepted + sum(failures.values()) == attempts and
            0 <= accepted <= solved <= submitted <= attempts):
        raise ValueError("inconsistent attempt/solver accounting")
    if stats.get("prepared", attempts) == attempts and not (
        stats["submitted"] == stats["completed"] == submitted and stats["raw_solved_queries"] == solved
    ):
        raise ValueError("inconsistent attempt/solver accounting")
    if report["tau_r_valid_samples"] != accepted:
        raise ValueError("benchmark requires independently valid c and tau-r")
    return dict(mode=mode, obstacles=obstacles, attempts=attempts, accepted=accepted,
                attempt_success_rate=accepted / attempts, source_failed=source_failed,
                world_failed=world_failed, submitted=submitted, raw_solved=solved,
                planner_no_solution=no_solution, postprocess_rejected=solved - accepted,
                acceptance_given_submitted=accepted / submitted if submitted else None,
                raw_solution_rate=solved / submitted if submitted else None,
                failure_counts=failures, session_wall_s=report["session_wall_s"],
                config_hash=report["config_hash"])


def command_for_case(args, mode, obstacles, output):
    return [sys.executable, "-m", "scripts.spacetime_data.generate_warehouse_spacetime",
            "--backend", "torch-forest", "--device", args.device,
            "--source", str(args.source), "--output-root", str(output),
            "--num-trajectories", str(args.attempts + 1), "--max-attempts", str(args.attempts),
            "--query-batch-size", str(args.query_batch_size), "--torch-batch-policy", "fill",
            "--edges-per-query", "16", "--goal-roots", "4", "--max-iterations", "512",
            "--solve-seconds", str(args.solve_seconds), "--collision-chunk", "512",
            "--seed", str(args.seed), "--start-index", str(args.start_index),
            "--max-world-attempts", str(args.max_world_attempts),
            "--source-group-size", "2", "--obstacles", str(obstacles),
            "--motion-modes", mode, "--require-tau-r"]


def verify_case_config(config, args, mode, obstacles, shared_config=None):
    # Historical generators used exactly 12 samples without a CLI field. This
    # inference permits read-only summaries; software hashes stay unchanged.
    config = dict(config)
    config.setdefault("max_world_attempts", 12)
    expected = dict(backend="torch-forest", device=args.device, source=str(args.source.resolve()),
                    seed=args.seed, start_index=args.start_index, source_group_size=2,
                    query_batch_size=args.query_batch_size, solve_seconds=args.solve_seconds,
                    torch_batch_policy="fill", require_tau_r=True, edges_per_query=16,
                    goal_roots=4, max_iterations=512, collision_chunk=512,
                    motion_modes=[mode], obstacles=obstacles, max_world_attempts=args.max_world_attempts)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("case configuration does not match benchmark parameters")
    normalized = {key: value for key, value in config.items() if key not in ("motion_modes", "obstacles")}
    if shared_config is not None and normalized != shared_config:
        raise ValueError("source/assets/software/common parameters changed between cases")
    return normalized


def publish(root, rows, args):
    atomic_json(root / "results.json", dict(
        backend="torch-forest", counts=[3, 5], modes=list(MOTION_MODES),
        attempts_per_case=args.attempts, seed=args.seed, start_index=args.start_index,
        max_world_attempts=args.max_world_attempts,
        query_batch_size=args.query_batch_size, solve_seconds=args.solve_seconds,
        complete=len(rows) == 2 * len(MOTION_MODES), cases=rows,
        success_definition="final accepted c AND tau-r / all attempts, including preparation failures",
        protocol="one mode per scene; fixed attempt budget; same task prefix and seed; randomized shapes/events",
        caveat="One task prefix and seed; modes/counts may differ in accepted tasks and GPU batch composition."))
    fields = ["mode", "obstacles", "attempts", "accepted", "attempt_success_rate", "source_failed",
              "world_failed", "submitted", "raw_solved", "planner_no_solution", "postprocess_rejected",
              "acceptance_given_submitted", "raw_solution_rate", "session_wall_s"]
    temporary = root / "results.csv.inprogress"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "results.csv")
    by_key = {(row["mode"], row["obstacles"]): row for row in rows}
    lines = ["# Warehouse 单 mode attempt 成功率", "",
             "后端 torch-forest；每格固定 %d 次 attempt；成功要求 c 和 tau-r 均通过最终验收。" % args.attempts,
             "每个 attempt 最多采样 %d 个动态场景，参考轨迹起始索引为 %d。" % (args.max_world_attempts, args.start_index),
             "", "| mode | 3 个障碍 | 5 个障碍 | 差值（5−3，百分点） |", "| --- | ---: | ---: | ---: |"]
    for mode in MOTION_MODES:
        a, b = by_key.get((mode, 3)), by_key.get((mode, 5))
        def cell(row):
            return "%d/%d = %.1f%%" % (row["accepted"], row["attempts"], row["attempt_success_rate"] * 100) if row else "待完成"
        delta = "%.1f" % (100 * (b["attempt_success_rate"] - a["attempt_success_rate"])) if a and b else "—"
        lines.append("| %s | %s | %s | %s |" % (mode, cell(a), cell(b), delta))
    lines += ["", "仅代表本次任务区间、seed 和搜索预算；障碍集合不严格嵌套。",
              "所有模式的障碍均按该单一 mode 运动。混合模式场景不能由此直接推断。",
              "详细失败分类和条件成功率见 results.json / results.csv。", ""]
    (root / "results.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--attempts", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-world-attempts", type=int, default=12)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--solve-seconds", type=float, default=5.)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.attempts, args.query_batch_size, args.solve_seconds, args.max_world_attempts) <= 0 or min(args.seed, args.start_index) < 0:
        parser.error("invalid attempt budget, batch size, time, seed or index")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    rows, shared_config = [], None
    for mode in MOTION_MODES:
        for obstacles in (3, 5):
            output = root / "datasets" / ("n%d" % obstacles) / mode
            report_path = output / "report.json"
            complete = report_path.exists() and json.loads(report_path.read_text())["attempts"] >= args.attempts
            if args.summarize_only and not complete:
                continue
            if not args.summarize_only and not complete:
                if shared_config is not None and shared_config["software"]["generate_warehouse_spacetime.py"] != sha256_file(
                    REPOSITORY / "scripts/spacetime_data/generate_warehouse_spacetime.py"
                ):
                    raise ValueError("generator version changed: resume this frozen experiment with its original version, "
                                     "or choose a new output-root; summarize-only can read the old results")
                command = command_for_case(args, mode, obstacles, output)
                if (output / "config.json").exists():
                    command.append("--resume")
                print("Running %s with %d obstacles, %d attempts" % (mode, obstacles, args.attempts), flush=True)
                with (root / "logs" / ("%s-n%d.log" % (mode, obstacles))).open("a") as log:
                    status = subprocess.run(command, cwd=str(REPOSITORY), stdout=log, stderr=subprocess.STDOUT).returncode
                if status != 2:
                    raise RuntimeError("fixed-attempt generator exited %d; inspect logs" % status)
            config = json.loads((output / "config.json").read_text())
            shared_config = verify_case_config(config, args, mode, obstacles, shared_config)
            row = summarize_case(json.loads((output / "report.json").read_text()),
                                 mode=mode, obstacles=obstacles, attempts=args.attempts)
            row["dataset"] = str(output.relative_to(root))
            rows.append(row)
            publish(root, rows, args)
            print("%s n%d: %d/%d = %.1f%%; valid queries=%d" % (
                mode, obstacles, row["accepted"], args.attempts, 100 * row["attempt_success_rate"], row["submitted"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
