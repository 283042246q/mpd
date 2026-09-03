# Standalone TimingDiffusion training

This path trains only the factorized timing prior. It directly reads canonical
Space-Time HDF5 shards and does not import the legacy trajectory dataset,
planning-task loader, spatial TemporalUnet, experiment launcher, or old
trainer.

## Representations

Both choices are standardized to a six-dimensional training target:

- `c`: `[c0,c2,c3,c4,c5,c7]` from the full endpoint-projected `c[8]`;
- `tau_r`: `[tau,r0,r1,r2,r3,r4]`, where `r` is normalized timing shape and
  `T=T_min(P,r)+(T_max-T_min(P,r))*sigmoid(tau)`.

`tau_r` requires the fields produced by
`scripts/spacetime_data/augment_normalized_timing.py`. Rows with
`quality/duration_bounds_valid=false` are excluded. Each representation gets a
separate normalizer and checkpoint.

## Environment

```bash
conda activate mpd-splines-public
unset PYTHONPATH PYTHONHOME
export MPLCONFIGDIR=/tmp/mpd-mpl
cd /home/eric/Projects/MotionPlanningDiffusion/mpd
```

Formal `c` training:

```bash
python scripts/train/train_timing_diffusion.py \
  --config scripts/train/cfgs/timing_diffusion_warehouse.yaml \
  --env EnvWarehouse \
  --representation c \
  --run-name c-v1
```

Formal decoupled timing training after dataset augmentation:

```bash
python scripts/spacetime_data/augment_normalized_timing.py \
  data_trajectories_spacetime/EnvWarehouse-RobotPanda-RRTConnect-SpaceTime-v1 \
  --duration-max 14

python scripts/train/train_timing_diffusion.py \
  --config scripts/train/cfgs/timing_diffusion_warehouse.yaml \
  --env EnvWarehouse \
  --representation tau_r \
  --run-name tau-r-v1
```

To select another environment, add its dataset roots under `training_data` in
the YAML and pass `--env`. To select data without editing YAML, repeat
`--dataset-root`; compatible roots are combined:

```bash
python scripts/train/train_timing_diffusion.py \
  --config scripts/train/cfgs/timing_diffusion_warehouse.yaml \
  --env EnvOpenDrawerShelf \
  --dataset-root /path/to/drawer/spacetime-v1 \
  --dataset-root /path/to/additional/compatible/spacetime-v1 \
  --representation c \
  --output-dir /path/to/output
```

`--env` is currently the dataset selector and checkpoint/experiment identity;
the first timing prior is conditioned on `P`, not on an obstacle-map encoding.
Train separate environment checkpoints unless multiple compatible datasets are
intentionally combined.

The current incomplete warehouse snapshot has no finalized split files. It may
only be used for a smoke test with an explicit fallback:

```bash
python scripts/train/train_timing_diffusion.py \
  --config scripts/train/cfgs/timing_diffusion_warehouse.yaml \
  --representation c \
  --device cpu --no-amp --max-steps 2 \
  --batch-size 8 --num-workers 0 \
  --max-train-samples 64 --max-val-samples 32 \
  --allow-hash-split-fallback \
  --output-dir /tmp/mpd-timing-smoke
```

Do not use fallback splits for final results. Formal generation writes grouped
split files so timing variants of one base path cannot leak across splits.

Resume with the same data/model settings and a larger `--max-steps`:

```bash
python scripts/train/train_timing_diffusion.py \
  --config scripts/train/cfgs/timing_diffusion_warehouse.yaml \
  --representation c \
  --run-name c-v1 \
  --max-steps 600000 \
  --resume data_trained_models/timing_diffusion/EnvWarehouse/c/c-v1/checkpoints/latest.pt
```

## Network

The path encoder evaluates the full spatial B-spline on a fixed phase grid and
forms the ordered feature sequence:

```text
G(P) = [q(s), q_s(s), q_ss(s), s],  s in 64 phase points
```

It uses a width-128 residual 1D CNN with dilations 1/2/4, followed by three
stride-2 convolutions and a 256-dimensional path embedding. The timing branch
takes the noisy six-vector, a 128-dimensional sinusoidal diffusion-step
embedding, and the path embedding. Six FiLM residual MLP blocks of width 256
predict epsilon in the standardized timing space. The default model has about
3.56 million parameters.

The DDPM uses 100 cosine-schedule steps. Training minimizes mean epsilon MSE;
each latent dimension is standardized from train data before noise is added.
The checkpoint contains raw and EMA weights, all constructor settings,
normalization, robot/spline identity, manifest hashes, optimizer/scaler state,
and RNG state.

Artifacts are written below:

```text
<output>/resolved_config.yaml
<output>/normalization.json
<output>/dataset_identity.json
<output>/metrics.jsonl
<output>/checkpoints/latest.pt
<output>/checkpoints/step-XXXXXXXX.pt
```
