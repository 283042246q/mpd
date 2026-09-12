# Marvin Warehouse Workspace-Generalization Benchmark

## Purpose

This benchmark separates three questions that should not be conflated:

1. Can one Marvin arm solve IK for a sampled TCP pose outside the training
   region, before collision rejection?
2. Can two independently sampled arm solutions form collision-valid 14-DoF
   start and goal states?
3. Given the exact same accepted endpoint pair, can MPD and RRTConnect produce
   a validated path?

The benchmark regions come from the dedicated
`scripts/inference/cfgs/start_goal_regions/EnvWarehouse-RobotMarvinBimanual-generalization-regions.yaml`.
This keeps the out-of-distribution benchmark independent of the mutable runtime
region selection file.
Every disjoint XYZ interval is expanded into an atomic box, so the two cabinet
height intervals are measured separately. The original data-generation region
file is loaded only to label samples as inside or outside the training bounds.

## Built-in cases

- same-side table to lower middle shelf;
- same-side table to upper middle shelf;
- lower shelf to upper shelf;
- left arm crossing into `right_table` while the right arm moves to its shelf;
- the mirrored right-arm crossing case;
- both arms exchanging table sides, as an intentionally hard case.

All cases keep both arms active. Endpoint sampling uses system entropy and the
accepted requests are saved before either planner runs. RRT therefore receives
the exact `q_start/q_goal` used by MPD. MPD remains dual-EE-goal conditioned;
RRTConnect targets the exact sampled joint goal. This distinction is retained
in the report rather than presenting their endpoint objectives as identical.

## Stages

Default density: 100 poses for each of 2 arms × 6 atomic regions (1,200
poses total), at most 10 restarts per fixed pose. Every accepted 14D solution
is saved in an endpoint pool. Twenty distinct endpoint pairs per case are
drawn from those pools, checking collisions again after combining arm solutions.
Missing pools are sampling shortfalls, not planner failures.

Generate the IK audit and endpoint pairs:

```bash
conda run --no-capture-output -n mpd-splines-public \
python scripts/inference/benchmark_marvin_workspace_generalization.py \
  --stage generate \
  --output-dir scripts/inference/logs/marvin-workspace-generalization
```

Run only RRTConnect on the saved pairs:

```bash
conda run --no-capture-output -n mpd-splines-public \
python scripts/inference/benchmark_marvin_workspace_generalization.py \
  --stage rrt --resume \
  --output-dir scripts/inference/logs/marvin-workspace-generalization
```

Run only MPD on the same pairs:

```bash
conda run --no-capture-output -n mpd-splines-public \
python scripts/inference/benchmark_marvin_workspace_generalization.py \
  --stage mpd --resume --device cuda:0 \
  --output-dir scripts/inference/logs/marvin-workspace-generalization
```

Or execute all stages in sequence with a small pilot:

```bash
conda run --no-capture-output -n mpd-splines-public \
python scripts/inference/benchmark_marvin_workspace_generalization.py \
  --stage all --ik-attempts 5 --pairs-per-case 1 \
  --output-dir scripts/inference/logs/marvin-workspace-pilot \
  --device cuda:0
```

By default the benchmark derives a 32-candidate A5 MPD configuration: pair
streaming, validator chunking, parent-bound broad phase, and foam guide geometry
are enabled. Full production collision geometry remains enabled in the dense
validator. `--raw-mpd-config` disables these memory-safety overrides.

## Artifacts and interpretation

- `ik-reachability.json`: per arm and atomic region, raw IK-tolerance hit rate,
  collision-valid rate, and out-of-training-bound counts;
- `pair-manifest.json` and `pairs/**/request.json`: immutable comparison pairs;
- `rrt/**`: exact-solution, post-validation, and fitted-spline results;
- `mpd/**`: ordinary inference artifacts plus benchmark process status;
- `summary.json` and `summary.csv`: per-case MPD/RRT success and timing totals.

RRT reports both raw exact-path success and spline success because its raw
piecewise path is not directly equivalent to MPD's smooth B-spline. MPD reports
production-validator clearance and goal accuracy. A successful RRT result does
not currently imply velocity/acceleration feasibility because this benchmark
does not time-parameterize the RRT spline.

IK failure means the bounded search did not find a solution, not that the
position is unreachable for all orientations or redundant joint solutions.
The other arm is held at a collision-valid random reference during each audit.
The measured collision-valid rate therefore includes that reference's influence.

`--resume` retains completed IK region groups and saved endpoint pairs. Region
configuration snapshots are saved under `configs/`. A changed MPD configuration
is rejected in an existing run directory, preventing CPU/batch-8 results from
overwriting an A5-b32 experiment. CUDA availability is checked before execution;
there is no automatic CPU fallback. Infrastructure faults/OOM are counted
separately from `success` and `no_valid_trajectory` in planning success rates.

For the dense comparison, run `--stage all --ik-attempts 100 --ik-restarts 10
--pairs-per-case 20 --batch-size 32 --device cuda:0` with a fresh output directory.
MPD wall time includes subprocess startup and checkpoint loading; RRT timings
exclude generator construction. These are not equivalent latency measurements.

The RRT validity gate uses the generator's mesh and sphere checks with
`min_distance_robot_env=0.02`; MPD uses its configured production dense
validator (including velocity/acceleration and EE tolerances). Thus the report
compares the existing planning pipelines, not identical feasibility oracles.
Do not interpret an MPD success / RRT failure as proof that MPD solves a harder
identical problem: MPD can choose a different redundant joint goal and has a
different environment margin. Endpoint sampling remains collision-validated
for both planners before the comparison.

Verified CUDA smoke result (2026-09-11): the saved OOD request in
`marvin-workspace-pilot-v2/pairs/table_to_shelf_lower/pair-000/request.json`
produced 1/32 valid candidates with A5-b32 CUDA, 5.352 s core inference,
1.422 s dense validation, and 3.593 GiB peak allocated GPU memory. Output:
`scripts/inference/logs/marvin-workspace-cuda-b32-smoke/`. The previous RRT run
on that request had no exact solution in 10 s. This is a functional smoke test,
not a statistical comparison; another GPU job was running concurrently.
