# Marvin rigid cooperative data generation

The production cooperative entry point is separate from the existing Panda,
single-arm, and Marvin independent generators:

```bash
conda run -n mpd-splines-public python \
  scripts/generate_data/generate_marvin_warehouse_cooperative.py --dry-run

conda run -n mpd-splines-public python \
  scripts/generate_data/launch_generate_marvin_warehouse_cooperative.py
```

The default configuration is
`data_generation_cfgs/EnvWarehouse-RobotMarvinBimanual-cooperative.yaml`. It
first generates 100 deterministic task contexts. Each context fixes an object
start/goal and paired 14-D joint endpoints, then targets four diverse
solutions with the same `task_id`.

## Pipeline

1. Sample the box pose from the configured common workspace and solve four
   collision-valid paired endpoint IK branches.
2. Propose object paths in 4-D `xyz+yaw` (3-D `xyz` is also supported): two
   lift-transfer-lower homotopies and two OMPL RRTConnect paths. Path
   simplification stays disabled because the existing Marvin benchmark found
   it both slow and capable of invalidating paths.
3. Lift each proposal through a continuous paired-IK beam using the previous
   solution and null-space perturbations.
4. Alternate degree-5, 22-control-point 14-D B-spline fitting with paired IK
   projection. Start/goal object poses and rigid closure are rejection-hard.
5. If proposals under-fill the quota, try OMPL `ProjectedStateSpace` plus
   `RRTConnect` directly on the six-constraint rigid-grasp manifold.
6. Audit the final spline at 512 samples and adaptively subdivide every segment
   beyond a 0.025-rad maximum joint step. Every dense state is checked by the
   Torch sphere model, PyBullet mesh model, payload/environment collision, and
   payload/robot collision. Payload contact is allowed only with configured
   finger links, and support contact is allowed only at `grasp`/`place`.

## Dataset contract

`sol_path` is `[N,128,14]`; `object_path` is `[N,128,7]` in
`xyz+quaternion_xyzw`; and B-spline controls are `[N,14,22]`. `task_id` repeats
for the solutions of one context and `solution_id` is local to that context.
The `context_*` datasets retain every attempted context, including zero-solution
failures and their terminal reason. `manifest.yaml` records stage timing,
failure counters, solution histograms, proposal counts, and content hashes.

The launcher derives the native-process lifetime from the target solution
quota. With four solutions/context and a lifetime of ten trajectories, one
fresh worker handles at most two contexts. Shards are written to staging
directories, fsynced, atomically published, checksum-validated on resume, and
then merged. Incomplete published shards are quarantined rather than deleted.

For the first scale gate, inspect `context_solution_histogram`,
`failed_context_reasons`, and the `*_seconds` counters before increasing
`dataset.num_contexts`. Do not silently resample failed contexts: retaining
them is necessary to measure workspace bias and proposal efficiency.
