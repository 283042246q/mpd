# Per-arm region inference

Use `scripts/inference/cfgs/start_goal_regions/EnvWarehouse-RobotMarvinBimanual-regions-matrix-generalization.yaml`.
Each arm has the seven current training placement regions: `table`,
`table_cross_y`, `table_cross_x`, `cabinet`, `cabinet_lower_xedge`,
`cabinet_lower_yedge`, `cabinet_upper`, prefixed with `left_` or `right_`.
Coordinates are copied from the matrix generation config, not the legacy
inference region template.

Five structured OOD regions per arm live under `placement_ood_regions`:
`table_near_edge_ood`, `table_approach_ood`,
`cabinet_front_approach_ood`, `cabinet_lower_lateral_ood`, and
`cabinet_upper_lateral_ood`. Every XYZ box has zero positive-volume overlap
with all 14 structured training cells. Table regions copy the training table
orientation; cabinet regions copy that arm's exact training cabinet orientation
(including the right cabinet's asymmetric Y range). Only position support is OOD.
The table near-edge and table approach cells are structured-placement OOD but
not full-training OOD, because the training random sampler covers much of that
workspace. They isolate task-conditioned placement generalization.

`random` uses the existing per-arm training random box. Strict random OOD names
are `left_`/`right_` + `random_ood_high`, `random_ood_forward`, and
`random_ood_centerline`. They use joint/FK rejection sampling with unrestricted
orientation and full collision checks, for either start or goal. Each box has
zero positive-volume overlap with both arms' training random boxes and all 14
structured placement cells. High tests vertical extrapolation, forward tests
the reach boundary, and centerline tests shared bimanual workspace.

Set four selections in the regions YAML:

```yaml
inference_selection:
  start: {left: left_table, right: right_table}
  goal: {left: left_cabinet_front_approach_ood, right: right_random_ood_centerline}
```

A YAML selection can also be a list of names, sampled uniformly. Coordinates
are always sampled with fresh system entropy, independent of diffusion seed
and sample index. CLI overrides replace only their corresponding selection.

From the repository root, using the separately prepared reversed-D config:

```bash
conda run --no-capture-output -n mpd-splines-public \
python scripts/inference/inference_marvin_bimanual.py \
  --config scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-reversed-D-305k.yaml \
  --start-goal-source regions \
  --start-goal-file scripts/inference/cfgs/start_goal_regions/EnvWarehouse-RobotMarvinBimanual-regions-matrix-generalization.yaml \
  --left-start-region left_table \
  --left-goal-region left_cabinet_front_approach_ood \
  --right-start-region right_table \
  --right-goal-region right_random_ood_centerline \
  --output-dir scripts/inference/logs/marvin-regions-custom \
  --device cuda:0 --sim-backend none
```

Omit the four region flags to use the YAML selections. Alternatively set the
runtime YAML's `start_goal_source: regions` and `start_goal_regions_path` to
`start_goal_regions/EnvWarehouse-RobotMarvinBimanual-regions-matrix-generalization.yaml`.
The explicit `--start-goal-file` resolves relative to the working directory;
the runtime YAML path resolves relative to that YAML's directory.

The four flags require region sourcing and cannot accompany `--request`.
Sampling failure reports an error; it never falls back to another region.
The generated request records the selected names and actual endpoint joints.

The loader rechecks the disjointness claims before constructing the robot and
rejects a mislabeled overlapping OOD box. Reachability probes are recorded only
after full collision validation; they demonstrate nonempty support rather than
estimating success rates. A contract-stub CLI smoke checks request construction,
not MPD planning success.

The 2026-09-22 endpoint smoke found valid structured endpoints for both arms'
near-edge, table-approach and cabinet-front cells, plus left upper-lateral.
It found none in 20 `_target_state` calls (up to roughly 600 target IK attempts)
for either lower-lateral cell or right upper-lateral. Keep these three as exact
requested stress boxes, but treat them as potentially empty or extremely sparse
until a dense reachability run says otherwise. All six strict random OOD boxes
produced a full-collision-valid endpoint in the first outer draw in this smoke.
