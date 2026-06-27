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

## 阶段 C：接入 inference

状态：已完成并已 push。

完成步骤：
- 在 `scripts/inference/inference.py` 新增 `sim_backend`，支持 `none`、`isaacgym`、`isaaclab`。
- 保留旧参数 `run_evaluation_issac_gym`，并新增 `run_evaluation_isaac_lab` 作为兼容开关。
- 将 IsaacLab evaluation 接入为 subprocess 文件交换：MPD 保存 `isaaclab-trajectories-XXX.pt`，调用 `scripts/isaaclab/evaluate_mpd_trajectories.py`，再读取 `isaaclab-statistics-XXX.json`。
- 新增 IsaacLab 运行参数：`isaaclab_root`、`isaaclab_conda_env`、`isaaclab_device`、`isaaclab_headless`、`isaaclab_action_repeat`、`isaaclab_timeout_s`、`render_isaaclab_movie`。
- 继续保留 `results_single_plan.isaacgym_statistics` 兼容旧结果字段，同时新增 `results_single_plan.sim_statistics`。
- 在 low-memory 保存路径中同时保存 `isaacgym_statistics` 和 `sim_statistics`。
- 在 `scripts/inference/launch_inference-experiments.py` 中补齐默认 IsaacLab 参数，默认 `sim_backend="none"`，不改变现有批量实验行为。

改动说明：
- MPD inference 默认不运行任何物理仿真后端。
- `sim_backend="isaacgym"` 时仍走旧 IsaacGym lazy import 路径。
- `sim_backend="isaaclab"` 时 MPD 主进程不 import IsaacLab，只调用独立 IsaacLab 环境中的 evaluator。
- IsaacLab subprocess 日志写入 `isaaclab-evaluator-XXX.log`；如果 IsaacLab 在写出 JSON 后超时或非零退出，inference 会保留已生成 statistics 并记录 warning/log path。

验证：
- 通过：`conda run -n mpd-splines-public python -m py_compile scripts/inference/inference.py scripts/inference/launch_inference-experiments.py`。
- 通过：`CUDA_VISIBLE_DEVICES='' ... import scripts.inference.inference` 并阻断 `isaacgym` import，输出 `stage_c_import_ok`。
- 通过：`rg -n "^import isaaclab|^from isaaclab" scripts/inference/inference.py scripts/inference/launch_inference-experiments.py || true`，无直接 IsaacLab import。
- 通过：`conda run -n mpd-splines-public pre-commit run --files scripts/inference/inference.py scripts/inference/launch_inference-experiments.py ISAACLAB_MIGRATION_PROGRESS.md`。
- 说明：未跑完整模型 inference；当前旧 MPD 环境直接暴露 RTX 5060 时仍会在 Theseus/PyTorch CUDA import 阶段触发 `sm_120` 不兼容，阶段 C 验证使用 CPU import 路径隔离后端接入逻辑。

Push 版本：
- `isaaclab-migration-stage-c`。

## 阶段 D：可视化和批量实验支持

状态：已完成并已 push。

完成步骤：
- 新增 `scripts/isaaclab/subprocess_utils.py`，把调用 IsaacLab evaluator 的 subprocess 逻辑抽成纯 Python helper；该 helper 不 import IsaacLab，可在旧 MPD 环境中安全导入。
- 新增 `scripts/isaaclab/__init__.py`。
- `scripts/inference/inference.py` 改为复用公共 subprocess helper，保持阶段 C 行为不变。
- `scripts/generate_data/visualize_trajectories.py` 新增 `--sim-backend none|isaacgym|isaaclab`，默认 `none`，避免默认启动旧 IsaacGym。
- `visualize_trajectories.py` 中旧 IsaacGym 环境改为显式 `--sim-backend isaacgym` 时 lazy import。
- `visualize_trajectories.py` 中新增 `--sim-backend isaaclab` 路径：保存可视化轨迹 payload，并调用 IsaacLab evaluator 输出统计 JSON/log。
- `mpd/motion_planning_baselines/examples/panda_isaac_replay.py` 重构为 `main()`/argparse 入口，新增 `--sim_backend none|isaacgym|isaaclab`，默认 `isaaclab`。
- `panda_isaac_replay.py` 中旧 IsaacGym replay 改为 lazy import，并新增 IsaacLab evaluator replay 路径。
- `panda_isaac_replay.py` 显式传入 `ParametricTrajectoryWaypoints`，避免当前 `PlanningTask` 构造器缺少 parametric trajectory。
- 两个可视化入口都在项目 import 前调用已有 `numpy_monkey_patch()`，规避新 NumPy 与旧 networkx 的 `np.int` 兼容问题。

改动说明：
- 可视化和 replay 入口默认不再因为顶层 import 触发 IsaacGym。
- 批量 inference 在阶段 C 已补齐 IsaacLab 参数；阶段 D 把独立脚本/示例也接到同一 evaluator helper。
- IsaacLab evaluator 的 `--make_video`/`--video_path` 参数已从调用侧贯通，但实际相机视频采集仍是 evaluator 内的保留功能；当前稳定输出为碰撞统计 JSON 和 subprocess log。

验证：
- 通过：`conda run -n mpd-splines-public python -m py_compile scripts/isaaclab/subprocess_utils.py scripts/generate_data/visualize_trajectories.py mpd/motion_planning_baselines/examples/panda_isaac_replay.py scripts/inference/inference.py`。
- 通过：`CUDA_VISIBLE_DEVICES='' ... import scripts.generate_data.visualize_trajectories; import mpd.motion_planning_baselines.examples.panda_isaac_replay; import scripts.inference.inference` 并阻断 `isaacgym` import，输出 `stage_d_import_ok`。
- 通过：`CUDA_VISIBLE_DEVICES='' conda run -n mpd-splines-public python scripts/generate_data/visualize_trajectories.py --help`。
- 通过：`CUDA_VISIBLE_DEVICES='' conda run -n mpd-splines-public python mpd/motion_planning_baselines/examples/panda_isaac_replay.py --help`。
- 通过：`rg -n "^import isaacgym|^from isaacgym|^from torch_robotics\\.isaac_gym_envs" scripts/generate_data/visualize_trajectories.py mpd/motion_planning_baselines/examples/panda_isaac_replay.py scripts/inference/inference.py || true`，无顶层旧 backend import。
- 通过：`conda run -n mpd-splines-public pre-commit run --files scripts/isaaclab/__init__.py scripts/isaaclab/subprocess_utils.py scripts/generate_data/visualize_trajectories.py mpd/motion_planning_baselines/examples/panda_isaac_replay.py scripts/inference/inference.py ISAACLAB_MIGRATION_PROGRESS.md`。

Push 版本：
- `isaaclab-migration-stage-d`。

## 阶段 E：剩余入口清理

状态：已完成并已 push。

完成步骤：
- 移除 `scripts/generate_data/post_process_generated_dataset.py` 中未使用的顶层 `import isaacgym`。
- 移除 `mpd/motion_planning_baselines/examples/pointmass_2d_CHOMP.py` 中未使用的顶层 `import isaacgym`。
- 移除 `mpd/motion_planning_baselines/examples/panda_spheres_CHOMP.py` 中未使用的顶层 `import isaacgym`。
- 移除 `mpd/motion_planning_baselines/examples/planar_2_link_RRT.py` 中未使用的顶层 `import isaacgym`。
- 上述入口均补充已有 `numpy_monkey_patch()`，避免新 NumPy 环境中旧 networkx 的 `np.int` import 问题。

改动说明：
- 该阶段不改变 planner、dataset post-process 或 baseline 示例逻辑，只删除未使用的旧仿真库导入。
- 最终顶层 IsaacGym import 只保留在旧 backend 文件 `mpd/torch_robotics/torch_robotics/isaac_gym_envs/motion_planning_envs.py` 内，且只有显式选择 IsaacGym backend 时才会进入。

验证：
- 通过：`rg -n "^import isaacgym|^from isaacgym|^from torch_robotics\\.isaac_gym_envs" scripts mpd`，结果只剩旧 IsaacGym backend 文件本身。
- 通过：`conda run -n mpd-splines-public python -m py_compile scripts/generate_data/post_process_generated_dataset.py mpd/motion_planning_baselines/examples/pointmass_2d_CHOMP.py mpd/motion_planning_baselines/examples/panda_spheres_CHOMP.py mpd/motion_planning_baselines/examples/planar_2_link_RRT.py`。

Push 版本：
- `isaaclab-migration-stage-e`。
