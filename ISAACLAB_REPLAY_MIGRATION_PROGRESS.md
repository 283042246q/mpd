# IsaacLab Replay Migration Progress

本文档记录按 `problem.txt` 推荐优先级推进 IsaacLab 验证与 replay 的阶段结果。每个阶段完成后单独 commit 并 push。

## 阶段计划

1. 稳定当前 headless IsaacLab evaluator，并在未完整支持的场景上显式报错或 warning。
2. 导出 MPD 场景障碍物到 IsaacLab，先支持 `EnvSpheres3D` 的 sphere 障碍物。
3. 扩展 Warehouse/Table/Shelf 的 box 障碍物导出和 IsaacLab 加载。
4. 移除 inference 中的 `--render_isaaclab_movie` 视频语义，改为通过新的 IsaacLab replay 脚本直接导出视频；需要视觉验证的仿真步骤保留截图。

## 阶段 1：headless evaluator 边界保护

- Push 版本：`stage-1-headless-evaluator-guard`
- 状态：已完成并 push
- Commit：`22860db`
- 目标：
  - 保持当前 `EnvSpheres3D + RobotPanda` headless IsaacLab batch evaluator 的稳定路径。
  - 在 Warehouse 障碍物导出完成前，阻止 `sim_backend=isaaclab` 对未支持环境给出误导性统计。
- 改动：
  - 在 `scripts/inference/inference.py` 增加 `ISAACLAB_BATCH_SUPPORTED_ENVS` 白名单。
  - `sim_backend=isaaclab` 当前只允许 `EnvSpheres3D` 和 `EnvSpheres3DExtraObjectsV00`。
  - 其他环境会抛出 `NotImplementedError`，提示先用 `--sim_backend none` 或等待专用 replay/export 路径。
- 验证：
  - Python 编译检查。
  - pre-commit 检查。

## 阶段 2：EnvSpheres3D obstacle export

- Push 版本：`stage-2-spheres-obstacle-export`
- 状态：已完成，待 push
- 目标：
  - 将 MPD 的 sphere obstacle payload 写入 `isaaclab-trajectories-XXX.pt`。
  - IsaacLab evaluator 读取 payload 并生成静态 sphere collider。
  - 对需要视觉确认的仿真输出保存截图。
- 改动：
  - 新增 `scripts/isaaclab/scene_payload.py`，将 MPD `MultiSphereField` 导出为 IsaacLab scene payload。
  - `scripts/inference/inference.py` 在 `isaaclab-trajectories-XXX.pt` 中写入 `scene` 字段。
  - `scripts/isaaclab/evaluate_mpd_trajectories.py` 读取 `scene.obstacles` 并在每个 IsaacLab env 下生成静态 sphere collider。
  - IsaacLab statistics 增加 `n_obstacles`、`obstacle_types`、`n_unsupported_obstacles`、`scene_schema` 等字段。
- 验证：
  - `EnvSpheres3D` payload 导出轻量测试：10 个 sphere，0 个 unsupported。
  - 完整 MPD + IsaacLab headless 验证通过。
  - 验证输出目录：`scripts/inference/logs/stage2_spheres_export/1783111874`
  - IsaacLab statistics：`n_obstacles=10`，`obstacle_types=['sphere']`，`n_unsupported_obstacles=0`。

## 阶段 3：Warehouse box obstacle export

- Push 版本：`stage-3-warehouse-box-export`
- 状态：未开始
- 目标：
  - 将 Warehouse/Table/Shelf 中的 box obstacle payload 导出到 IsaacLab。
  - IsaacLab evaluator 生成静态 cuboid collider。
  - 解除 Warehouse 的 batch evaluator 保护，但只在 payload 中包含障碍物时允许。

## 阶段 4：IsaacLab replay 视频导出

- Push 版本：`stage-4-replay-video-export`
- 状态：未开始
- 目标：
  - 新增专用 `scripts/isaaclab/replay_mpd_trajectory.py`。
  - replay 脚本负责单条/少量轨迹的视觉验证、截图和 mp4 导出。
  - 从 inference 主流程移除 `--render_isaaclab_movie`，避免误以为 batch evaluator 会录视频。
