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

F1/F2/F3 use the same Phase-5 ROS execution and safety contract through the
separate `factorized` worker. A checkpoint is deliberately required instead of
silently choosing one, and the 29-control-point timing condition must be
explicitly adapted for the 21-control-point ToDrawer spatial model:

```bash
scripts/isaaclab/run_dynamic_demo_pipeline.sh \
  --profile to_drawer \
  --phase factorized \
  --factorized-method f1 \
  --factorized-timing-checkpoint \
    data_trained_models/timing_diffusion/EnvWarehouse/tau_r/warehouse-tau-r-v2/checkpoints/step-00060000.pt \
  --factorized-adapt-spatial-basis
```

Corridor A is an independent, default-off timing refinement for `phase5_joint`
and factorized `f1` (both `c` and `tau_r` checkpoints). Enable it on either
pipeline with `--corridor-a`; optionally set `--corridor-a-weight 0.1`.
For example:

```bash
scripts/isaaclab/run_dynamic_demo_pipeline.sh \
  --profile to_drawer --phase phase5 --timing-mode phase5_joint \
  --corridor-a --output-dir /tmp/mpd-phase5-corridor-a

scripts/isaaclab/run_dynamic_demo_pipeline.sh \
  --profile to_drawer --phase factorized --factorized-method f1 \
  --factorized-timing-checkpoint /absolute/path/to/timing-checkpoint.pt \
  --factorized-adapt-spatial-basis --corridor-a \
  --output-dir /tmp/mpd-f1-corridor-a
```

The worker samples a full-body dynamic clearance grid for each fixed spatial
candidate, selects reachable early/late safe-interval branches, and adjusts
only its timing parameters. It keeps the trained F1 six-dimensional latent and
the original timing control-point count. The complete candidate-specific
DenseCheck remains authoritative; an originally valid candidate is restored
if refinement makes it invalid. `result.json` records the switch, branch
counts, timing, and DenseCheck fallbacks under `space_time_guidance.corridor_a`.
The grid is sampled, so it does not prove continuous clearance, and opt-in
latency increases with the number of candidates and branches. The direct
`infer_space_time.py` and `infer_factorized.py` CLIs expose the same switch
plus grid, margin, step-count, and learning-rate controls.

### Paired random ToDrawer benchmark

`benchmark_todrawer_random.py` freezes every random world to an output artifact and
uses paired planner seeds for eight modes: `phase4_aligned`, `joint`,
`joint_corridor_a`, `f1_tau_r`, `f1_tau_r_corridor_a`, `f1_c`,
`f1_c_corridor_a`, and `f3_tau_r`. Suffix `_corridor_a` enables the cost;
the matching mode without the suffix leaves it off. Each on/off pair receives
the same frozen obstacle geometry, direction, speed, size, motion law, planner
seed, and timing checkpoint. With `motion_aligned`, every mode—including the
three Corridor modes—uses its own measured first-plan completion profile. The
reference is the earliest completed planner result after world start, whether
that result is successful or a terminal failure such as `no_valid_trajectory`;
it does not wait for the first successful trajectory or significant robot
motion. Slower Corridor modes therefore receive later crossing times. With
`absolute_world_time`, all
modes share identical crossing times and the Corridor startup delay remains a
real online penalty. Use `--timing-protocol both` to materialize both protocols
from one accepted geometry sample and run/report them separately.
The default is 5 environments/category x 10 categories x 5 planner repeats x
8 modes = 2000 sequential GPU/ROS runs. Every environment independently resamples its
anchors, directions, speeds, sizes, motion laws, and crossing times. Planner repeats
keep that environment fixed and change only the planner seed; every mode receives the
same seed sequence. Start a smaller smoke suite before launching the full matrix:

```bash
cd /home/eric/Projects/MotionPlanningDiffusion/mpd

/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  scripts/isaaclab/benchmark_todrawer_random.py \
  --output-dir scripts/isaaclab/logs/todrawer-random-smoke \
  --environment-count-per-category 1 \
  --planner-repeats 1 \
  --timing-protocol motion_aligned \
  --duration-sec 20 \
  --categories single_crossing
```

Full benchmark:

```bash
/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  scripts/isaaclab/benchmark_todrawer_random.py \
  --output-dir scripts/isaaclab/logs/todrawer-corridor-a-50x5x8 \
  --environment-count-per-category 5 \
  --planner-repeats 5 \
  --timing-protocol motion_aligned \
  --duration-sec 35 \
  --suite-seed 20260829
```

Run both timing protocols from the same frozen geometry batch:

```bash
/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  scripts/isaaclab/benchmark_todrawer_random.py \
  --output-dir scripts/isaaclab/logs/todrawer-corridor-a-dual-protocol \
  --environment-count-per-category 5 \
  --planner-repeats 5 \
  --timing-protocol both \
  --duration-sec 35 \
  --suite-seed 20260829
```

This doubles the run count relative to one protocol. `summary.json` stores
separate `by_timing_protocol` aggregates, and `report.md` prints separate
`motion_aligned` and `absolute_world_time` tables before the compatibility
aggregate tables. Runs are stored below a protocol directory so attempts from
the two clocks cannot overwrite each other.

The earlier Phase 4, Phase 4 aligned, Phase 5 joint, and F1/F2/F3 comparison
under both `c` and `tau_r`
(50 x 5 x 9 = 2250 runs), select the nine modes explicitly:

```bash
/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  scripts/isaaclab/benchmark_todrawer_random.py \
  --output-dir scripts/isaaclab/logs/todrawer-factorized-c-tau-r-50x5x9 \
  --environment-count-per-category 5 \
  --planner-repeats 5 \
  --timing-protocol motion_aligned \
  --duration-sec 35 \
  --suite-seed 20260829 \
  --modes phase4 phase4_aligned joint \
    f1_c f2_c f3_c f1_tau_r f2_tau_r f3_tau_r \
  --factorized-c-checkpoint \
    data_trained_models/timing_diffusion/EnvWarehouse/c/warehouse-c-v2/checkpoints/step-00500000.pt \
  --factorized-tau-r-checkpoint \
    data_trained_models/timing_diffusion/EnvWarehouse/tau_r/warehouse-tau-r-v2/checkpoints/step-00060000.pt
```

These two best-known checkpoint paths are also the CLI defaults. The benchmark enables
explicit spatial-basis adaptation by default for factorized modes;
`--no-factorized-adapt-spatial-basis` is a fail-closed contract experiment and will not
run this 29-to-21 checkpoint/config pair. Within each representation, F1/F2/F3 share one
checkpoint so the comparison isolates sampler design. The generic legacy modes
`f1/f2/f3` remain available with `--factorized-timing-checkpoint`.

`motion_aligned` is the default timing protocol. It holds object geometry and motion
parameters fixed across modes, then shifts crossing times using each mode's measured
time to its first completed planner result, regardless of terminal status. This aligns
the modes at the end of the first planning attempt without waiting for a successful
path or visible motion. `absolute_world_time` shares crossing times exactly,
so every mode sees an identical world-clock trajectory. Both protocols perform bounded
rejection sampling before execution: robot-base clearance over the full design episode,
initial parked-Franka 56-sphere clearance through the expected-goal safety boundary,
anchor work-volume bounds, valid crossing windows, distinct multi-object direction
lines, and finite valid motion parameters. Static-furniture clearance is diagnostic
only: dynamic objects may pass through furniture at any time. Suites generated under
the older furniture-rejecting policy need a new output directory.

To run the ten scenario categories sequentially and retry each category until it reaches
the goal without an earlier controlled brake, use the until-success runner. Select one
of `phase4`, `phase4_aligned`, `joint`, `f1_c`, `f2_c`, `f3_c`, `f1_tau_r`,
`f2_tau_r`, or `f3_tau_r`; `f3_c` remains the default:

```bash
/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  scripts/isaaclab/run_todrawer_f3c_until_success.py \
  --mode joint \
  --output-dir scripts/isaaclab/logs/todrawer-until-success-joint \
  --duration-sec 35 \
  --plan-rate-hz 1.0 \
  --skip-build
```

Planner and geometry seeds change deterministically on every retry. Each attempt uses
the selected mode's measured first-plan completion profile to move the original crossing
schedule later, while retaining a 0.5 s reserve before the mode's expected goal. The
alignment reference is the earliest completed planner result, whether it succeeded or
failed; it does not wait for a valid path or significant robot motion. Anchor,
direction, speed, size, and crossing time are all resampled. The complete moving box is
checked from world time zero through the mode's expected-goal safety boundary against
all 56 collision spheres of the initial Franka pose; candidates must retain more than
5 mm clearance and remain
outside the robot-base exclusion for the full episode. Multi-object attempts additionally
require at least two motion-line orientations separated by 35 degrees or more; opposite
vectors on the same line count as one orientation. Old successful artifacts without this
policy revision are rerun instead of skipped. Every attempt with a replay
manifest is rendered, whether it passes or fails, under
`runs/<scenario>/<mode>/attempt-NNN/replay.mp4`. Only a passing attempt advances to the
next category. Successful replay copies are indexed under `videos/<mode>/`, while exact
scenario JSON, seeds, assessment, screenshot, and replay summary remain in the attempt
directory. Infrastructure failures that produce no replay manifest are recorded but
cannot be rendered.

After execution, the runner measures the actual 0.01 rad robot-motion start from the
recorded trajectory and audits every dynamic object against parked Franka up to that
measured time. A logged `q_pos_start is in collision` and a failed measured pre-motion
clearance audit have different outcomes: the former rejects that planning request and
the ROS node continues replanning, while the latter makes the attempt fail even if it
later reaches the goal without braking. The start-collision warning remains in the
attempt diagnostics.

Audit both first-plan completion time and significant-motion profiles from recorded
replays at 0.01 and 0.02 rad joint-displacement thresholds with:

```bash
/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  scripts/isaaclab/analyze_todrawer_mode_motion_start.py \
  --logs-root scripts/isaaclab/logs \
  --output /tmp/todrawer-mode-motion-start.json
```

Phase-4 aligned one-factor-off ablation (default: 4 environments/category x 10
categories x 2 planner repeats x 8 modes = 640 paired runs):

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
Use `--categories` to run a resumable slice of the frozen suite; the mode list is
part of `suite.json` and must remain the same when resuming. Omitted filters
select all ten categories and the eight default comparison modes. The F1/F3
modes use the two default checkpoint paths above; generic `f1/f2/f3` require
an explicit checkpoint.

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
  --output-dir scripts/isaaclab/logs/todrawer-corridor-a-50x5x8 \
  --environment-count-per-category 5 \
  --planner-repeats 5 \
  --timing-protocol motion_aligned \
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
For factorized runs it additionally records timing representation, checkpoint step/hash,
whether explicit basis adaptation was active, and spatial/timing denoiser NFE. Strictly
paired reporting uses exactly the modes selected for that benchmark invocation, rather
than requiring unrelated ablation modes.

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
