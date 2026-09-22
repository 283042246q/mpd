# Marvin region × model benchmark

This benchmark compares selected checkpoints of reversed-data variants A, B,
C, and D on exactly the same frozen start/goal requests.

## What is covered

The checked-in 22-scenario catalog contains four difficulty strata:

- `easy`: exact dataset rows and local in-distribution table/single-shelf moves;
- `medium`: dual shelf and training-random-to-placement moves;
- `hard`: in-distribution cross-arm table moves, structured position OOD, and
  single/dual strict-random extrapolation;
- `extreme`: dual reach-boundary OOD, shared-centerline goals, and deliberately
  sparse shelf-lateral OOD.

The transition tags explicitly cover ID→ID, ID→structured-OOD,
structured-OOD→ID, structured-OOD→structured-OOD, ID→random-OOD,
random-OOD→ID, random-OOD→random-OOD, mixed-OOD, and ID→shared-OOD.

`shared_centerline_left`, `shared_centerline_right`,
`shared_centerline_dual`, and `id_cross_swap` have
`workspace_overlap: true`.  They explicitly exercise inter-arm collision and
shared-workspace behavior.  The endpoint sampler still performs the complete
environment, intra-arm, and inter-arm collision check before a request is
admitted.

Endpoint generation and MPD are separate populations in the reports.  A
scenario's `shortfall` means a collision-valid IK endpoint pair was not sampled
within its bounded budget; it is not counted as an MPD planning failure.

## Checkpoint set

The model registry pins these full EMA checkpoints:

| Variant | checkpoint 1 | checkpoint 2 |
| --- | ---: | ---: |
| A | 835k | 660k |
| B | 955k | 715k |
| C | 715k | 835k |
| D | 305k | 665k (plus requested 600k) |

They are the two lowest finite validation-loss saved checkpoints found in each
reversed-data run when the registry was created.  The benchmark measures their
actual planning validity; validation loss is only the selection proxy.

Every generated inference config enables the A5-tuned memory controls:
pair-streaming with chunk 1024, validator chunking, parent-bound broad phase,
and `foam_pika_100` guide geometry.  The default inference budget is one batch
of 32 candidates.  Use four batches for the 4 × 32 quality setting.

## One-click execution

Run all 22 scenarios, one frozen request per scenario, nine checkpoints, and one
32-candidate batch:

```bash
cd /home/eric/Projects/MotionPlanningDiffusion/mpd
conda run --no-capture-output -n mpd-splines-public \
  python scripts/inference/benchmark_marvin_region_models.py \
  --stage all \
  --output-dir scripts/inference/logs/marvin-region-model-benchmark \
  --device cuda:0
```

For the final 4 × 32 comparison, add:

```text
--candidate-batches 4 --candidates-per-batch 32
```

Each request/seed trial stops at its first successful batch.  A failed trial
runs all four batches, so the maximum is 792 subprocesses for the configured 22
tasks × 9 checkpoints × 1 seed × 4 batches. Sampling shortfalls reduce the
actual count.

It is often more convenient to freeze tasks first and run later.  Use the same
arguments and output directory for both commands:

```bash
conda run --no-capture-output -n mpd-splines-public \
  python scripts/inference/benchmark_marvin_region_models.py \
  --stage prepare \
  --output-dir scripts/inference/logs/marvin-region-model-benchmark \
  --candidate-batches 4 --device cuda:0

conda run --no-capture-output -n mpd-splines-public \
  python scripts/inference/benchmark_marvin_region_models.py \
  --stage run \
  --output-dir scripts/inference/logs/marvin-region-model-benchmark \
  --candidate-batches 4 --device cuda:0
```

Runs are resumable.  Existing per-batch reports are reused, while immutable
settings prevent an accidental mixture of different tasks, checkpoints,
candidate budgets, or seeds in one output directory.

The manifest is also safely extensible: newly registered checkpoint IDs are
appended to an existing experiment, but existing checkpoint definitions and
the frozen task/budget fields cannot change. Reusing the original output
directory therefore runs only missing checkpoint batches and retains the exact
same requests.

For the existing 4 × 32 result directory, this command reuses all completed
B/C/D reports and runs only A-835k, A-660k, and D-600k:

```bash
conda run --no-capture-output -n mpd-splines-public \
  python scripts/inference/benchmark_marvin_region_models.py \
  --stage run \
  --output-dir scripts/inference/logs/marvin-region-model-benchmark-4x32 \
  --candidate-batches 4 --candidates-per-batch 32 --device cuda:0
```

Useful smaller runs include:

```bash
# Only shared-workspace cases.
conda run --no-capture-output -n mpd-splines-public \
  python scripts/inference/benchmark_marvin_region_models.py \
  --stage all --output-dir scripts/inference/logs/marvin-region-shared \
  --scenarios id_cross_swap shared_centerline_left \
              shared_centerline_right shared_centerline_dual \
  --candidate-batches 4 --device cuda:0

# Sampling/config smoke test without inference.
conda run --no-capture-output -n mpd-splines-public \
  python scripts/inference/benchmark_marvin_region_models.py \
  --stage prepare --output-dir /tmp/marvin-region-smoke \
  --scenarios dataset_exact id_table_local --device cpu
```

## Outputs

The selected output directory contains:

- `tasks/tasks.json`: accepted counts, attempts, errors, and shortfalls;
- `tasks/requests/*.json`: frozen runtime requests shared by every model;
- `configs/*.yaml`: concrete A5 inference config for each checkpoint;
- `manifest.json`: checkpoint fingerprints and complete experiment budget;
- `runs/<model>/<scenario>/...`: logs, MPD result, and GPU memory per batch;
- `summary.json`: model, difficulty, support, scenario, and shared-workspace
  aggregates;
- `summary-by-model-scenario.csv`: a flat comparison table.

`planning_success_rate` (also exposed as `success_rate`) uses only planning-
evaluable trials: a success, or a trial whose batches all returned
`no_valid_trajectory`.  `end_to_end_success_rate` includes infrastructure and
contract failures in its denominator.  A trial is complete after its first
success or after exhausting all candidate batches.  `fault`, `timeout`, and
`cuda_oom` remain visible in status counts and are not silently treated as
normal no-solution results.

`mean_wall_seconds` is the average end-to-end wall time for one
model × frozen request × base-seed trial. It sums every attempted 32-candidate
subprocess until the first successful batch, or all four subprocesses when the
trial fails. It includes model/dataset/environment startup and dense validation,
but excludes endpoint sampling and other checkpoints. It is therefore the
latency of the current early-stop benchmark policy, not pure network inference
time.

## Variant selection

Variant A is enabled. Its `top_validation/top_k: 2` policy resolves 835k and
660k from the completed run. All variants are selected by default; an explicit
subset remains available, for example:

```text
--variants A D
```

Use `--strict-models` when a missing/mismatched run should abort preparation;
without it, unavailable models are recorded in `skipped_models`.
