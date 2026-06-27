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
/home/eric/IsaacLab_ori/isaaclab.sh -p scripts/isaaclab/evaluate_mpd_trajectories.py \
  --input logs/trajectories.pt \
  --output logs/isaaclab_statistics.json \
  --headless
```

The first implementation replays Panda joint-space trajectories with IsaacLab's built-in Panda asset and writes a
statistics JSON compatible with the old IsaacGym result fields. Full MPD obstacle export, videos, and batch launch
integration are handled by later migration stages.
