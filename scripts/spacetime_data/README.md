# Space-Time `(P,c)` data generator

The generator is independent from the legacy spatial training loader. It
streams `sol_path` from the warehouse HDF5, fits one canonical full spatial
B-spline `P`, generates multiple fixed-path timing teachers, fits the runtime
TimingSpline representation `c`, validates joint position/velocity/
acceleration limits, and writes canonical Space-Time MPD v1 shards.

Use the repository environment without inheriting an unrelated `PYTHONPATH`:

```bash
conda activate mpd-splines-public
unset PYTHONPATH PYTHONHOME

# Small acceptance run: 10 spatial paths, seven retiming modes per path.
python scripts/spacetime_data/generate_spacetime_dataset.py \
  --max-paths 10 \
  --paths-per-shard 10 \
  --output-root /tmp/mpd-spacetime-smoke

# Full warehouse conversion. Existing shards are never overwritten.
python scripts/spacetime_data/generate_spacetime_dataset.py \
  --paths-per-shard 1000

# Independently recompute timing and derivative checks from written P/c.
python scripts/spacetime_data/validate_spacetime_dataset.py \
  data_trajectories_spacetime/EnvWarehouse-RobotPanda-RRTConnect-SpaceTime-v1
```

Each shard is first written with an `.inprogress` suffix and atomically renamed
after the source range is complete. An interrupted run therefore cannot make a
partial shard look final. Completed shards are never silently overwritten; to
resume, set `--start-index` to the first not-yet-generated shard boundary.

The default variants are:

- TOPP-RA anchor with the repository Panda limits;
- duration scales 1.2 and 1.5;
- TOPP-RA with `(velocity, acceleration)` limit scales `(0.85, 0.80)` and
  `(0.65, 0.70)`;
- randomized local slowdown;
- randomized near-wait.

Every accepted row stores full `P[29,7]`, full `c[8]`, the final teacher
`t_ref[128]`, task/base-path/variant IDs and quality metrics. The same HDF5 can
be viewed as spatial-only, factorized timing or paired joint data for F1, F2,
F3 and JointDual. Split files contain base-path IDs, so variants from one path
cannot leak across train/validation/test.

## Add duration/shape-decoupled labels

After generation has finished, augment the existing shards without rerunning
RRT or TOPP-RA:

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
quality/normalized_timing_fit_rmse
quality/normalized_timing_density_clip_fraction
quality/duration_logit_clipped
quality/duration_bounds_valid
```

Here `tau = logit((T-T_min)/(T_max-T_min))`. Values are clipped only to keep
the logit finite. Rows that exceed 14 seconds, violate the computed lower
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
