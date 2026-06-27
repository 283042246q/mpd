# IsaacLab Migration Progress

本文件记录 `ISAACGYM_TO_ISAACLAB_MIGRATION.md` 中各阶段的完成情况、改动范围和对应 push 版本。

## 阶段 A：解除 IsaacGym 强依赖

状态：已完成，等待 push 版本确认。

完成步骤：
- 移除 `scripts/train/train.py` 顶层 `import isaacgym`。
- 移除 `mpd/parametric_trajectory/trajectory_bspline.py` 顶层 `import isaacgym`。
- 移除 `mpd/parametric_trajectory/trajectory_waypoints.py` 顶层 `import isaacgym`。
- 移除 `mpd/torch_robotics/torch_robotics/visualizers/configuration_free_space.py` 中未使用的顶层 `import isaacgym`，避免 `torch_robotics.tasks.tasks` 默认 import 时加载旧库。
- 将 `scripts/inference/inference.py` 中旧 IsaacGym backend 的 import 移到 `run_evaluation_issac_gym=True` 分支内。
- 在 `scripts/train/train.py` 入口复用已有 `numpy_monkey_patch()`，保证新 NumPy 环境中训练入口能继续通过 `networkx` import。

改动说明：
- 不改变训练、采样、轨迹优化或指标计算逻辑。
- 旧 IsaacGym backend 保留；只有显式启用 `run_evaluation_issac_gym` 时才导入旧 backend。
- 目标是让 `run_evaluation_issac_gym=False` 的训练和 inference 路径不在 import 阶段加载 `isaacgym`、`gym_38.so` 或 `gymtorch`。

验证：
- 通过：`CUDA_VISIBLE_DEVICES='' conda run -n mpd-splines-public python -c "import scripts.train.train; import mpd.parametric_trajectory.trajectory_bspline; import mpd.parametric_trajectory.trajectory_waypoints; import scripts.inference.inference; print('phase_a_import_ok_cpu')"`。
- 通过：`CUDA_VISIBLE_DEVICES='' conda run -n mpd-splines-public python -c "...阻断 isaacgym import 后导入 train/trajectory/inference 入口..."`，输出 `isaacgym_blocked_import_ok_cpu`。
- 说明：直接暴露 RTX 5060 GPU 时，旧 `mpd-splines-public` 环境中的 PyTorch/Theseus 会在 import 阶段报 `no kernel image is available for execution on the device`；该失败发生在 `deps/theseus` 预建 CUDA 张量时，不是 IsaacGym 加载导致。

Push 版本：
- `isaaclab-migration-stage-a`。
