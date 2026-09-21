# Per-arm region inference

Use `scripts/inference/cfgs/start_goal_regions/EnvWarehouse-RobotMarvinBimanual-regions-matrix-generalization.yaml`.
Each arm has the seven current training placement regions: `table`,
`table_cross_y`, `table_cross_x`, `cabinet`, `cabinet_lower_xedge`,
`cabinet_lower_yedge`, `cabinet_upper`, prefixed with `left_` or `right_`.
Coordinates are copied from the matrix generation config, not the legacy
inference region template.

Three extra placement regions per arm probe adjacent space outside the current
placement union: `table_adjacent_above`, `cabinet_adjacent_front`,
`cross_adjacent_outer`. They retain the corresponding placement orientation
constraints. Their success rates have not been calibrated.

`random` uses the existing per-arm training random box. Named random regions
are `left_`/`right_` + `random_shelf_lower_ood`, `random_shelf_upper_ood`,
`random_cross_ood`. They use joint/FK rejection sampling with free orientation
and full collision checks, for either start or goal. Their supports are
strictly disjoint from the corresponding arm's training random XYZ box;
they can overlap placement training data. Reachable configurations do not
guarantee a valid pair with every other-arm configuration or a valid path.

Set four selections in the regions YAML:

```yaml
inference_selection:
  start: {left: left_table, right: right_table}
  goal: {left: left_cabinet_upper, right: right_random_shelf_lower_ood}
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
  --left-goal-region left_cabinet_upper \
  --right-start-region right_table \
  --right-goal-region right_random_shelf_lower_ood \
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

Verification (2026-09-22): all six named random supports produced at least one
full-collision-valid endpoint, using fresh reference states and up to 60 draws
of 20,000 joint candidates each (30-second limit per support). Actual draws
were 1/2/1 for left lower/upper/cross and 3/16/8 for right lower/upper/cross.
This demonstrates nonempty feasible support, not a reachability rate estimate.
A real CLI sampling smoke also constructed a dual-arm request from training
random starts to both named lower-shelf random goals. Its downstream backend
was `contract_stub`, so it did not test MPD planning success.
