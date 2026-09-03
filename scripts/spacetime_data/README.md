# Unified Space-Time `(P,c)` and `(P,tau,r)` data generator

The generator is independent from the legacy spatial training loader. It
streams `sol_path` from the warehouse HDF5, fits one canonical full spatial
B-spline `P`, generates multiple fixed-path timing teachers, fits the runtime
TimingSpline representation `c`, validates joint position/velocity/
acceleration limits, derives bounded `(tau,r)` targets, and writes all fields
to the same Space-Time MPD shards. A separate augmentation pass is not needed
for newly generated v2 datasets.

Use the repository environment without inheriting an unrelated `PYTHONPATH`:

```bash
conda activate mpd-splines-public
unset PYTHONPATH PYTHONHOME

# Small acceptance run: 10 spatial paths, six c modes plus expanded tau-r modes.
python scripts/spacetime_data/generate_spacetime_dataset.py \
  --max-paths 10 \
  --paths-per-shard 10 \
  --output-root /tmp/mpd-spacetime-smoke

# Full warehouse conversion. Existing shards are never overwritten.
python scripts/spacetime_data/generate_spacetime_dataset.py \
  --paths-per-shard 1000

# Independently recompute timing and derivative checks from written P/c.
python scripts/spacetime_data/validate_spacetime_dataset.py \
  data_trajectories_spacetime/EnvWarehouse-RobotPanda-RRTConnect-SpaceTime-v2
```

Each shard is first written with an `.inprogress` suffix and atomically renamed
after the source range is complete. An interrupted run therefore cannot make a
partial shard look final. Completed shards are never silently overwritten; to
resume, set `--start-index` to the first not-yet-generated shard boundary.

The default `c` variants are generated in this order:

- run TOPP-RA with the repository Panda limits;
- fit the runtime TimingSpline and globally slow it until it satisfies the
  sampled velocity and acceleration limits, producing `fast_anchor`;
- scale that feasible fitted anchor by 1.5, 2.0 and 2.5, then refit and
  revalidate each result;
- generate randomized local-slowdown and near-wait shapes from the feasible
  anchor, then refit and revalidate them.

The default duration floor is 2 seconds and the task horizon is 14 seconds.
Variants that cannot fit below 14 seconds are rejected rather than clipped.
Local modes limit their requested bump using the available horizon before
fitting; this preserves their shape without creating a pile-up exactly at 14
seconds.

Every accepted row stores full `P[29,7]`, full `c[8]`, the final teacher
`t_ref[128]`, task/base-path/variant IDs and quality metrics. The same HDF5 can
be viewed as spatial-only, factorized timing or paired joint data for F1, F2,
F3 and JointDual. The doubled warehouse source stores a forward path and its
reverse at adjacent IDs. The default `--source-group-size 2` hashes
`base_path_id // 2`, so both directions and all their timing variants remain in
one train/validation/test split. Use `--source-group-size 1` for a source that
does not contain adjacent reverse pairs.

## Duration/shape-decoupled labels

The unified generator writes the normalized fields before publishing each
shard. For each distinct timing shape (`fast_anchor`, `local_slowdown`, and
`near_wait`) it stores five bounded duration targets:

```text
y = (T - T_min) / (T_max - T_min)
y in {0.01, 0.05, 0.15, 0.35, 0.60}
T = T_min + y * (T_max - T_min)
tau = logit(y)
```

The `tau_r` dataset view expands each eligible shape row over those five
targets. Global `c` scale variants are deliberately not expanded, because they
share almost the same `r` and would overweight the anchor shape. Expanded
targets use `quality/tau_r_mode_valid`; scalar `duration_bounds_valid` only
governs legacy rows whose `tau` describes the duration of that row's `c`.

The old augmentation entry remains only for upgrading an existing v1 dataset
without rerunning RRT or TOPP-RA. Its duration floor now also defaults to 2
seconds:

```bash
python scripts/spacetime_data/augment_normalized_timing.py \
  data_trajectories_spacetime/EnvWarehouse-RobotPanda-RRTConnect-SpaceTime-v1 \
  --duration-max 14
```

`--duration-max` is the task horizon and defaults to 14 seconds. It can be
changed later by rerunning the command; derived fields are recomputed while all
canonical v1 fields are preserved. Each shard is updated through a same-folder
temporary copy and atomic rename. The script refuses to run while an
`.inprogress` marker exists, so finish dataset generation before running it.

The extension fits the five-dimensional normalized timing shape `r` from the
deployed `c` curve by default, computes the sampled velocity/acceleration lower
bound `T_min(P,r)`, and writes:

```text
timing/shape_control_points
timing/t_min
timing/t_min_velocity
timing/t_min_acceleration
timing/t_max
timing/duration_fraction
timing/tau
timing/duration_fraction_modes
timing/duration_modes
timing/tau_modes
quality/normalized_timing_fit_rmse
quality/normalized_timing_density_clip_fraction
quality/duration_logit_clipped
quality/duration_bounds_valid
quality/tau_r_mode_valid
```

Here scalar `tau` audits the duration represented by that row's `c`. The
five-value `tau_modes` is the balanced training grid used by the `tau_r`
loader. Values are clipped only to keep the scalar audit logit finite. Rows
that exceed 14 seconds, violate the computed lower
bound, have `T_min >= T_max`, or fail shape fitting remain auditable but have
`quality/duration_bounds_valid = false`; timing-model loaders must filter them.
Use `--shape-source reference_time` only when the desired target is the saved
TOPP-RA/retiming teacher rather than the actually deployed v1 TimingSpline.

`quality/static_clearance_min` is `NaN` in this converter because it does not
instantiate the warehouse collision world. `quality/accepted` means the
spatial spline stays inside URDF joint limits and the reconstructed TimingSpline
passes duration, velocity and acceleration validation. This limitation is
recorded in `manifest.yaml`.

The independent `P -> timing` training commands and network contract are in
[`scripts/train/TIMING_DIFFUSION.md`](../train/TIMING_DIFFUSION.md).
