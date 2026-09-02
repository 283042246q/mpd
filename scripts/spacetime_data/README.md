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

`quality/static_clearance_min` is `NaN` in this converter because it does not
instantiate the warehouse collision world. `quality/accepted` means the
spatial spline stays inside URDF joint limits and the reconstructed TimingSpline
passes duration, velocity and acceleration validation. This limitation is
recorded in `manifest.yaml`.
