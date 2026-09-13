# Marvin placement sampling

The independent generation YAML now selects `placement_region_weighting: mixture`.
For each arm and cell, `p = .55 * normalized(volume * collision_valid_rate)
+ .30 / 7 + .15 / 6` for hard cells (all except the main table). The historical
rates are initialization proxies from `marvin-workspace-dense-v3` and
`marvin-workspace-edges-2400`; they are not new measurements of the cropped cells.
Recalibrating the YAML rates recomputes all matrices automatically.

P->P uses a symmetric 7x7 joint distribution, in the YAML region-list order.
Its diagonal is `.12 * p`; off-diagonal preferences are balanced to row and
column marginals `.88 * p`. This reproduces the proposed matrices without using
rounded percentages. Other directions use `p` for each placement endpoint;
random endpoints retain their existing workspace and joint sampling.

Dual P->P draws one left-pair/right-pair combination from a balanced 49x49
distribution. Both-cross preference is .2, both-upper and cross-upper are .3;
overlapping penalties use the minimum. Balancing preserves both single-arm
pair marginals. These preferences are design choices, not feasibility proofs.

The matrix defines proposal probabilities, not exact finite accepted quotas.
The 45/35/10/10 direction schedule remains exact over complete 100-task periods.
One task fixes its mode, direction and regions before any retries. All retries
keep that contract; 30 failed attempts still terminate the shard. No cell is
silently substituted or threshold relaxed. `min_active_joint_delta` stays .08.
Selected, attempted and accepted pairs are counted separately in stats; completed
dataset manifests contain accepted pair counts and resolved matrices. Existing
volume/explicit-weight configurations without pair sampling retain their logic.

Inspect the exact probabilities without creating a robot or running RRT:

```bash
conda run --no-capture-output -n mpd-splines-public python -m \
  scripts.generate_data.generate_marvin_warehouse_bimanual \
  --config data_generation_cfgs/EnvWarehouse-RobotMarvinBimanual-independent.yaml \
  --dry-run
```

Use the existing launcher and a new output directory for generation. Do not mix
old and new sampling configurations in the same resumable dataset.

Calibration yielding a cell probability >= .5 is rejected by this diagonal
policy: its off-diagonal marginal is infeasible (or degenerate at equality).
Adjust the mixture or deliberately redesign the diagonal policy before running.
