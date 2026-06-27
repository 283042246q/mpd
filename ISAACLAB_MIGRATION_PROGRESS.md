# IsaacLab Migration Progress

本文件记录 `ISAACGYM_TO_ISAACLAB_MIGRATION.md` 中各阶段的完成情况、改动范围和对应 push 版本。

## 阶段 A：解除 IsaacGym 强依赖

状态：已完成并已 push。

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
- `isaaclab-migration-stage-a`，commit `6b82eb3`。

## 阶段 B：IsaacLab 独立执行器

状态：已完成并已 push。

完成步骤：
- 新增 `scripts/isaaclab/evaluate_mpd_trajectories.py`。
- 新增 `scripts/isaaclab/README.md`。
- evaluator 支持读取 `torch.save` 的 tensor 或 dict 输入，其中核心字段为 `q_trajs_pos`，shape 为 `[H, B, D]`。
- evaluator 使用 `env_isaaclab_ori` 中的 IsaacLab/Isaac Sim，按 IsaacLab 要求先启动 `AppLauncher`，再导入 `isaaclab.sim`、scene、assets、sensor 等运行时模块。
- evaluator 使用 IsaacLab 内置 Panda high-PD asset，启用 robot contact sensors，按 MPD 的 7 DoF arm trajectory 补齐 Panda finger joints 为 `0.04`。
- evaluator 输出兼容旧 IsaacGym 统计字段的 JSON：`backend`、`n_trajectories_collision`、`n_trajectories_free`、`n_trajectories_free_fraction`、`collision_mask`、`first_collision_step` 等。
- 预留 `--make_video`、`--video_path` 参数；实际视频录制留到阶段 D。

改动说明：
- 该脚本独立于 MPD Python 3.8 环境运行，不向 MPD 训练、采样、优化或 metrics 路径引入 IsaacLab import。
- 第一版聚焦 Panda joint-space trajectory replay 和接触统计；完整 MPD 障碍物导出/映射留给后续阶段。
- 为兼容 IsaacLab `AppLauncher.add_app_launcher_args()` 的 `parse_known_args()` 逻辑，`--input` 和 `--output` 采用 parse 后手动校验，而不是 argparse 的 `required=True`。

验证：
- 通过：`conda run -n env_isaaclab_ori python -m py_compile scripts/isaaclab/evaluate_mpd_trajectories.py`。
- 通过：`conda run -n env_isaaclab_ori python scripts/isaaclab/evaluate_mpd_trajectories.py --help`，确认显示 evaluator 参数和 AppLauncher 参数。
- 通过：`conda run -n mpd-splines-public pre-commit run --files scripts/isaaclab/evaluate_mpd_trajectories.py scripts/isaaclab/README.md`。
- 通过：使用 `/tmp/mpd_isaaclab_smoke.pt` 的 `[4, 1, 7]` Panda 轨迹运行 headless smoke test，生成 `/tmp/mpd_isaaclab_smoke_stats.json`，结果为 1 条轨迹、0 碰撞、free fraction `1.0`。
- 说明：首次 headless smoke test 需要从 Omniverse S3 打开 IsaacLab Panda USD 资产，资产加载约 235 秒；后续 360 秒限时命令已生成 JSON，但进程被外层 `timeout` 截断在 Isaac Sim 退出/清理阶段。

Push 版本：
- `isaaclab-migration-stage-b`。
