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
`phase5_scalar_duration`. `--timing-mode` is rejected with `--phase phase4` so a
requested ablation cannot be silently ignored.

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
  an execution interval, and a rejected plan cannot;
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
