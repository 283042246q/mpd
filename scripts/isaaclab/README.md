# IsaacLab MPD Evaluation

Stage B adds a standalone IsaacLab evaluator that runs outside the legacy MPD Python 3.8 environment.

Expected input is a `torch.save` file containing either a tensor `q_trajs_pos` or a dictionary with:

- `q_trajs_pos`: tensor shaped `[H, B, D]`
- `q_pos_starts`: optional tensor shaped `[B, D]`
- `q_pos_goal`: optional tensor shaped `[D]`
- `robot_name`: optional metadata, currently only `panda`
- `env_name`: optional metadata
- `dt`: optional simulation dt

Example:

```bash
/home/eric/IsaacLab/isaaclab.sh -p scripts/isaaclab/evaluate_mpd_trajectories.py \
  --input logs/trajectories.pt \
  --output logs/isaaclab_statistics.json \
  --headless
```

By default the evaluator uses Isaac Sim's current
`Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd` asset. This avoids the obsolete IsaacLab
`Robots/FrankaEmika/panda_instanceable.usd` URL, which is unavailable in the Isaac 6.0 asset bundle. Pass
`--robot_usd /absolute/path/to/franka.usd` to use a local or mirrored asset instead.

The first implementation replays Panda joint-space trajectories and writes a statistics JSON compatible with the old
IsaacGym result fields. Full MPD obstacle export, videos, and batch launch integration are handled by later migration
stages.

## Phase-4 multi-replan video

`replay_mpd_trajectory.py` keeps the original `--input *.pt` mode and adds a separate
`--manifest` mode for a recorded dynamic-planning session. This mode is a deterministic
visual replay: it does not run MPD, collision checking, ROS 2, or a controller while the
video is being rendered. The manifest records what was known and what decision was made
at each time, while each plan references the original Phase-4 `trajectory.npz`.

The rendering convention is:

- gray line: superseded/obsolete plan;
- blue line: plan being executed at the current replay time;
- green line: newest accepted plan waiting for its handoff;
- yellow translucent spheres: constant-velocity obstacle prediction with the recorded
  linear/covariance inflation;
- red line or segment: rejected plan or recorded collision interval;
- purple sphere: recorded handoff position (or the hand position calculated from the
  referenced plan);
- flashing red frame: recorded safe-brake interval.

An editable manifest is provided at
`scripts/isaaclab/examples/dynamic_replay_manifest.example.json`. Trajectory paths are
resolved relative to the manifest. Quaternion fields named `orientation_xyzw` use
`[qx, qy, qz, qw]`. The optional legacy static-scene `orientation` field uses IsaacLab's
`[qw, qx, qy, qz]` order.

Render a 1080p video and final frame:

```bash
cd /home/eric/Projects/MotionPlanningDiffusion/mpd

conda run -n env_isaaclab /home/eric/IsaacLab/isaaclab.sh -p \
  scripts/isaaclab/replay_mpd_trajectory.py \
  --manifest /absolute/path/to/dynamic-replay.json \
  --output_video /absolute/path/to/dynamic-replay.mp4 \
  --screenshot_path /absolute/path/to/dynamic-replay-final.png \
  --output_json /absolute/path/to/dynamic-replay-summary.json \
  --video_fps 30 \
  --width 1920 \
  --height 1080 \
  --prediction_horizon_s 3.0 \
  --prediction_samples 10 \
  --enable_cameras
```

If `env_isaaclab` is already activated, call `isaaclab.sh` directly. Current IsaacLab
runs headless when `--viz kit` is absent; older installations may still require
`--headless`.

The renderer interpolates the command at video-frame time and teleports the Panda to that
joint state. Consequently the output shows the exact MPD command timeline without ROS
tracking noise. Use the fake-hardware/real-robot logs to populate `active_from_s`,
`active_until_s`, `collision_segments_s`, `world_snapshots`, and `events`; inventing these
fields is useful for a presentation demo but must not be treated as an experiment trace.

### One-command ToDrawer recording and replay

The orchestration script keeps the planner, ROS adapter, recorder, and IsaacLab renderer
as separate processes. It exports the selected MPD static environment, starts one resident
GPU planner, launches Franka fake hardware plus a moving known obstacle, checkpoints a
replay manifest, and then renders MP4/PNG/summary artifacts after ROS exits:

```bash
cd /home/eric/Projects/MotionPlanningDiffusion/mpd

scripts/isaaclab/run_dynamic_demo_pipeline.sh \
  --profile to_drawer \
  --output-dir /tmp/mpd-to-drawer-demo \
  --duration-sec 35
```

The pipeline defaults to Phase 5 joint space-time replanning. Select the preserved
Phase 4 fixed-timing path, or another Phase 5 ablation mode, explicitly:

```bash
# Default; --phase phase5 may be omitted.
scripts/isaaclab/run_dynamic_demo_pipeline.sh \
  --profile to_drawer \
  --phase phase5 \
  --timing-mode phase5_joint

# Preserved Phase 4 fixed-timing pipeline.
scripts/isaaclab/run_dynamic_demo_pipeline.sh \
  --profile to_drawer \
  --phase phase4
```

Valid Phase 5 timing modes are `phase5_joint`, `phase5_timing_only`, and
`phase5_scalar_duration`. The fixed-time aligned comparison is selected with
`--phase phase4aligned` (the canonical spelling is `phase4_aligned`).
`--timing-mode` is rejected with either Phase 4 path so a requested ablation
cannot be silently ignored.

### Paired random ToDrawer benchmark

`benchmark_todrawer_random.py` freezes every random world to a repository artifact and
runs the same scenario/planner seed against Phase 4, Phase 4 aligned, and all three
Phase-5 timing modes. The default 50 scenarios x 5 repeats x 5 modes is 1250 sequential
GPU/ROS runs and is intended as the large comparison suite. It gives each of the ten
environment categories five independently generated worlds and each frozen world five
independent planner seeds. Start a smaller smoke suite before launching the full matrix:

```bash
cd /home/eric/Projects/MotionPlanningDiffusion/mpd

/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  scripts/isaaclab/benchmark_todrawer_random.py \
  --output-dir scripts/isaaclab/logs/todrawer-random-smoke \
  --scenario-count 10 \
  --repeats 1 \
  --duration-sec 20 \
  --categories curved_crossing \
  --modes phase4 phase4_aligned joint
```

Full benchmark:

```bash
/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  scripts/isaaclab/benchmark_todrawer_random.py \
  --output-dir scripts/isaaclab/logs/todrawer-random-50x5x5 \
  --scenario-count 50 \
  --repeats 5 \
  --duration-sec 35 \
  --suite-seed 20260829
```

Phase-4 aligned one-factor-off ablation (default: 40 frozen scenarios x 2
planner-seed repeats x 8 modes = 640 paired runs):

```bash
/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  scripts/isaaclab/benchmark_phase4_aligned_ablation.py \
  --output-dir scripts/isaaclab/logs/phase4-aligned-ablation-40x2x8 \
  --duration-sec 35 \
  --suite-seed 20260829
```

The eight modes are plain Phase 4, all Phase-4-aligned changes, and six
one-factor-off variants for ROS deviation, relative hysteresis, motion/terminal-hold
clearance split, 1:4 tail kinematic weighting, MPD mean+CVaR dynamic guidance, and
MPD dynamic-risk selection. Use `--dry-run` to freeze the suite and inspect every
pipeline command without starting CUDA or ROS.

The generator retains horizontal and vertical crossings and uses five deterministic
continuous motion laws: constant velocity, constant longitudinal acceleration,
sinusoidal curves, smooth speed variation, and curved motion with speed variation.
The last two freeze their phase/frequency/amplitude in `suite.json`; their stated speed
and acceleration standard deviations therefore produce repeatable model mismatch for
the worker's constant-velocity predictor. Motions are combined with single, staggered,
simultaneous, fast, high-inflation, and mixed multi-object interactions.

Scenarios are tagged `easy`, `moderate`, or `hard`. Hard scenes may use larger objects,
up to 23 cm prediction-horizon inflation, two near-simultaneous crossings, or a third
delayed obstacle. They still retain a spatial corridor or later time gap, avoiding an
obvious permanent wall without making the benchmark artificially easy. This is a
construction criterion, not a guarantee that every planner run succeeds.
Use `--categories` and/or `--modes` to run a resumable slice of the frozen large
suite without changing `suite.json`; omitted filters select all ten categories and all
five modes.

Completed mode/scenario/repeat triples are skipped when the same output directory is
resumed. Failed attempts are retained as `attempt-NNN`; nothing is deleted. Reports are
regenerated after every run under `report/report.md`, `report/summary.json`, and
`report/runs.csv`. Use `--report-only` to rebuild reports without starting ROS. Video
rendering is disabled for batch runs; pass `--render` only when every episode needs an
MP4/PNG.

Retry only CycloneDDS startup failures without replacing algorithm or continuity
failures:

```bash
/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  scripts/isaaclab/benchmark_todrawer_random.py \
  --output-dir scripts/isaaclab/logs/todrawer-random-50x5x5 \
  --scenario-count 50 \
  --repeats 5 \
  --duration-sec 35 \
  --skip-build \
  --retry-failure-class dds_startup \
  --ros-domain-id 221
```

The report retains historical infrastructure-failure run/attempt counts while all
algorithm metrics use the latest attempt. It also separates manifest availability from
pipeline validation, reports goal-plus-brake and no-goal-plus-brake, gives path and
execution time conditional on reaching the goal, and includes a strictly paired table
where every mode has a manifest for the same scenario/repeat.

The report distinguishes guard/DenseCheck collision predictions from physical contact.
Passive replay does not measure contact forces. It reports dynamic collision rejection,
brake events, goal/episode/execution duration, realized joint-space path length, selected
hard/common-window clearance, mean/CVaR clearance cost, dense environment/self
clearance, and inference latency.

Every invocation selects an isolated ROS 2 DDS domain after Pixi activation so stale
transient-local `/robot_description` publishers cannot switch a fake-hardware run onto
the real Franka interface. Use `--ros-domain-id 146` only when a repeatable explicit
domain is needed. The CLI spelling `to_drawer_bridge-crossing` is accepted for the
three-object scenario and normalized to the ROS node's
`to_drawer_bridge_crossing` identifier.

Omit `--output-dir` for a timestamped directory below
`scripts/inference/logs/dynamic-replay-to_drawer-phase5/`. Explicit Phase 4 keeps its
legacy `scripts/inference/logs/dynamic-replay-to_drawer/` location. Use `--skip-build` only after the
ROS workspace has already been rebuilt. The current profile maps to:

- static MPD/IsaacLab scene: `EnvOpenDrawerShelf`;
- runtime config: `config_EnvOpenDrawerShelf-RobotPanda-runtime-to-drawer.yaml`;
- dynamic observation scenario: `to_drawer_crossing`;
- recorded facts: filtered world versions, exact submitted JTC commands, handoffs, and
  controlled-braking events.

The profile `case` in `run_dynamic_demo_pipeline.sh` is the intended extension point for
another environment. Add its environment name, runtime config, world scenario, target,
and a matching static-scene exporter profile. No MPD inference or collision-cost code
needs to be changed.

### Manifest contract

The top-level schema is `mpd_dynamic_replay`, version `1`:

- `duration_s`: replay duration;
- `initial_q`: optional 7- or 9-DoF initial joint state;
- `plans`: planning results ordered by `created_s`. An accepted/superseded plan can have
  an execution interval, while a replacement scheduled after recording ends has none;
  a rejected plan cannot have one;
- `world_snapshots`: strictly time/version-increasing world states, starting at
  `time_s=0`. Each snapshot carries a validity deadline and known dynamic objects;
- `events`: `handoff` and `brake` facts recorded by the execution manager;
- `static_scene`: optional legacy sphere/box scene description.

For every dynamic object, `pose` is its filtered pose at snapshot time,
`linear_velocity` is the constant-velocity Kalman estimate, and `local_sdf` is a known
sphere/box/capsule. `inflation.mode=linear` uses
`base_m + horizon_rate_m_s * dt`; `mode=covariance` propagates the supplied 6x6
position/velocity covariance and draws a conservative bounding sphere. Orientation is
held constant, matching the Phase-4 collision model.
