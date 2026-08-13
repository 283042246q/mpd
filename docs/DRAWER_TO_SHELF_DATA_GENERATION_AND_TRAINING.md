# Drawer-to-shelf 轨迹再生成与重新训练流程

本文说明如何为当前 `EnvOpenDrawerShelf + RobotPanda` 场景重新生成
drawer-to-shelf 轨迹、合并数据、训练 MPD，并列出真正实施时需要修改的文件。

本次已更新数据生成 YAML，并为 `generate_trajectories.py` 增加向后兼容的
`subregions` 采样；训练器和场景代码没有修改。

## 1. 先说结论

目标任务定义为：

1. 起点末端位姿来自打开的底层抽屉中央可达带，或靠前、靠 shelf 一侧且
   从抽屉壁内缩的角落。
2. 终点末端位姿位于相邻 shelf 的某一层。
3. 保留原生成定义：约一半是随机关节状态到 region，约一半是两个 region
   之间无顺序移动；后者同时包含 `drawer -> shelf` 和
   `shelf -> drawer`。
4. 规划、训练和推理都使用当前场景的最终世界坐标；不要修改
   `rotation_z_axis_deg`，也不要重新引入 region 随环境旋转的逻辑。

已新增的数据生成配置：

`data_generation_cfgs/EnvOpenDrawerShelf-RobotPanda-drawer-to-shelf.yaml`

当前几何下，可靠验证通过的是 shelf 下层和中层；两层使用不同 xy 范围。
上层隔间几何上存在，但其
高度和到 Panda 基座的距离使当前末端姿态无法可靠求得 IK。因此，“当前场景
完全不动，同时覆盖全部三层”不是一个可生成有效训练集的配置。

可执行的方案是先用 YAML 中的下层和中层生成数据；若必须覆盖全部三层，
需要把 shelf 向 Panda 移近和/或整体降低，再重新计算、验证所有坐标。

## 2. 当前 region 的具体范围

全部坐标单位为米，均在 Panda base/world frame 中。

| 区域 | x | y | z | 含义 |
|---|---:|---:|---:|---|
| drawer 中央起点 | `[-0.26, -0.18]` | `[0.50, 0.54]` | `[0.28, 0.32]` | 原“corner”范围，实际是稳定的中央可达带 |
| drawer 前方 shelf 侧内缩角落 | `[-0.10, -0.07]` | `[0.34, 0.40]` | `[0.32, 0.36]` | 1000-task pilot 后收紧 x；保留 y/z 覆盖和俯视姿态 |
| shelf 下层终点 | `[0.40, 0.46]` | `[0.52, 0.56]` | `[0.24, 0.27]` | 下层隔间、靠开口的中等范围 |
| shelf 中层终点 | `[0.36, 0.50]` | `[0.50, 0.60]` | `[0.53, 0.60]` | 中层隔间的较宽可达范围 |
| shelf 上层候选，当前禁用 | `[0.36, 0.50]` | `[0.50, 0.60]` | `[0.80, 0.84]` | 位于上层隔间内，但当前布局 IK 失败 |

drawer 中央起点和两个 shelf 终点使用相同的朝前基础旋转矩阵：

```yaml
base: [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
```

drawer 的姿态扰动为 x/y/z 各 `[-2°, 2°]`；shelf 的姿态扰动为
x/y `[-2°, 2°]`、z `[-3°, 3°]`。

靠墙的 drawer 角落如果沿用朝前姿态，夹爪或腕部容易与抽屉壁冲突。新增
角落采用俯视基础旋转矩阵：

```yaml
base: [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
```

其姿态扰动为 x/y/z 各 `[-2°, 2°]`。这个 region 特意从物理墙面内缩，
不应把范围直接扩到抽屉侧壁中心线。

场景几何给出的 shelf 内部大致范围是：

- 内部 x：`[0.225, 0.635]`
- 开口到背板前方的 y：约 `[0.46, 0.90]`
- 下层自由高度：约 `[0.06, 0.355]`
- 中层自由高度：约 `[0.405, 0.695]`
- 上层自由高度：约 `[0.745, 1.04]`

YAML 中只有 `drawer` 和 `shelf` 两个顶层 region。drawer 内含中央、内缩
角落两个 subregion；shelf 内含下层、中层两个 subregion。选择某个顶层
region 后，再在它的 subregions 中等概率采样。因此两种 drawer 子区域
理论上各占该 region 样本的 50%，下层和中层也各占 shelf 样本的 50%。
下层的 IK 和路径约束更强，所以 xy 小于中层。

验证记录：

- 最初保守的下层/中层共同范围：12 次采样、4 条路径全部通过。
- 扩大后的 shelf 下层：10 次采样无 IK 重采样，3 条路径通过稠密复核，
  最低物体间隙约 `0.0248 m`。
- 扩大后的 shelf 中层：10 次采样无 IK 重采样，3 条路径通过稠密复核，
  最低物体间隙约 `0.0219 m`。
- 把下层和中层强行共用较大的 xy 时，下层右后侧出现多次 IK 失败，因此
  最终拆成两个 region。
- drawer 前方两个内缩角落沿用朝前姿态时均连续 IK 失败。
- drawer 前方 shelf 侧内缩角落改为俯视姿态后，3 次采样和 1 条路径通过，
  但最多使用 3 次采样尝试；正式生成时需要单独统计它的成功率。
- 多个上层候选点、两种末端朝向均未得到可用 IK。

这只是小样本 region 验证，不代替大规模数据生成前的成功率统计。

## 3. subregion 定义及采样语义

`scripts/generate_data/generate_trajectories.py` 的
`get_random_pose_from_region()` 现在支持两种兼容 schema：

```yaml
pose_regions:
  drawer:
    subregions:
      center_reachable: {...}
      front_shelf_inset: {...}
  shelf:
    subregions:
      lower: {...}
      middle: {...}
```

没有 `subregions` 的旧 YAML 仍直接读取 `translation` 和 `rotation`。存在
`subregions` 时，采样器先等概率选择一个子区域，再按旧格式采样位姿。

`get_random_ee_pose_from_cfg_file()` 的顶层逻辑没有改变：

1. `move_between_pose_regions: true` 时，约 50% 进入 region-to-region 分支；
2. 该分支从两个顶层 region 无放回采样，因此端点一定分别属于 drawer 和
   shelf，但起终点顺序随机；
3. 其余约 50% 从随机无碰撞关节状态移动到随机选择的 drawer 或 shelf；
4. subregion 只决定顶层 region 内部的具体位置，不参与顶层分支选择。

这避免了 drawer-to-drawer 或 shelf-to-shelf 的 region-to-region 任务，同时
保留论文仓库原来的任务混合比例和无方向语义。

## 4. 真正实施时要改哪些文件

| 文件 | 是否必须 | 要做的事 |
|---|---|---|
| `data_generation_cfgs/EnvOpenDrawerShelf-RobotPanda-drawer-to-shelf.yaml` | 已完成 | 定义 drawer/shelf 两个顶层 region 及各自两个 subregion |
| `scripts/generate_data/generate_trajectories.py` | 已完成 | 等概率采样 subregion，同时兼容原有直接 region schema |
| `scripts/generate_data/launch_generate_trajectories.py` | 批量生成时必须 | 增加该环境的实验项、任务数、分片大小和唯一数据集名称 |
| `scripts/train/launch_train_diffusion_models-v00.py` | 使用 launcher 时必须 | 增加新 `dataset_subdir`；直接运行 `train.py` 时可不改 |
| `scripts/inference/cfgs/config_EnvOpenDrawerShelf-RobotPanda-regions-drawer-to-shelf.yaml` | 训练后必须 | 把 `model_dir_ddpm_bspline` 指向新 checkpoint |
| `mpd/torch_robotics/torch_robotics/environments/env_open_drawer_shelf.py` | 仅全三层时必须 | 移近/降低 shelf，使上层进入 Panda 工作空间 |
| `scripts/inference/cfgs/start_goal_regions/EnvOpenDrawerShelf-RobotPanda-regions-drawer-to-shelf.yaml` | 场景或范围变化时 | 同步推理 start/goal 世界坐标 |
| `scripts/inference/validate_constrained_panda_regions.py` 和相关测试 | 范围变化时 | 重新验证 IK、OMPL、稠密碰撞和 payload 一致性 |

不需要修改 `rotation_z_axis_deg` 或它与 region 的逻辑关系。当前场景和本
YAML 已直接使用旋转后的最终世界坐标。

原生成定义本身允许两个方向。基线训练可先读取 `dataset_merged.hdf5`；若要
沿用仓库的数据增强策略，可运行 `flip_solution_paths.py` 并训练
`dataset_merged_doubled.hdf5`。注意翻转也会把 joint-to-region 复制成
region-to-joint，因此应在数据统计中明确记录是否启用了该增强。

## 5. 完整执行流程

以下命令均从仓库根目录执行：

```bash
cd /home/eric/Projects/MotionPlanningDiffusion/mpd
```

### 5.1 固化任务定义并验证场景

先确认：

- `EnvOpenDrawerShelf` 已从 `torch_robotics.environments` 导出；
- PyBullet/torch-robotics 与 IsaacLab 使用相同的障碍物世界坐标；
- `rotation_z_axis_deg` 保持 `0`；
- 推理 region 没有通过 `rotate_with_environment` 再旋转一次；
- NVIDIA 驱动和 IsaacLab 可用性问题不影响 CPU 上的 OMPL 数据生成，但会
  影响后续 IsaacLab replay/evaluation。

现有推理 region 可用以下命令做 IK/OMPL 回归：

```bash
conda run --no-capture-output -n mpd-splines-public \
  python scripts/inference/validate_constrained_panda_regions.py \
  scripts/inference/cfgs/start_goal_regions/EnvOpenDrawerShelf-RobotPanda-regions-drawer-to-shelf.yaml \
  --samples 20 \
  --plans 5 \
  --planner-time 30 \
  --interpolate-num 256
```

正式扩展数据 YAML 的范围时，应另加一个读取 data-generation schema 的
validator，或临时转换成推理的 `start_regions/goal_regions` schema 后再跑
上述验证。

### 5.2 subregion 支持（已完成）

顶层任务选择仍由 `get_random_ee_pose_from_cfg_file()` 完成。它只看到
`drawer` 和 `shelf` 两个顶层 region。具体位姿由更新后的
`get_random_pose_from_region()` 采样：

```text
top_level_region = random choice(pose_regions)
if top_level_region has subregions:
    subregion = uniform random choice(top_level_region.subregions)
    pose = sample(subregion.translation, subregion.rotation)
else:
    pose = sample(top_level_region.translation, top_level_region.rotation)
```

当前实现会拒绝空 `subregions`，也会拒绝同一个 region 同时定义
`subregions` 和直接 `translation/rotation`，避免含糊配置。旧版 Warehouse
等 YAML 不含 `subregions`，行为保持不变。

### 5.3 先生成小样本

建议的数据集目录名：

```text
EnvOpenDrawerShelf-RobotPanda-drawer-to-shelf-reachable-levels-joint_joint-one-RRTConnect
```

先运行 100 个任务检查任务比例和各子区域成功率：

```bash
conda run --no-capture-output -n mpd-splines-public \
  python scripts/generate_data/generate_trajectories.py \
  --env_id EnvOpenDrawerShelf \
  --robot_id RobotPanda \
  --cfg_file EnvOpenDrawerShelf-RobotPanda-drawer-to-shelf.yaml \
  --start_task_id 0 \
  --num_tasks 100 \
  --num_trajectories_per_task 1 \
  --min_distance_robot_env 0.02 \
  --min_distance_q_pos_start_goal 0.5 \
  --planner RRTConnect \
  --planner_allowed_time 30 \
  --simplify_path True \
  --fit_bspline False \
  --interpolate_num 128 \
  --n_parallel_jobs 1 \
  --task_batch_size 1 \
  --debug False \
  --seed 7 \
  --results_dir data_trajectories/EnvOpenDrawerShelf-RobotPanda-drawer-to-shelf-reachable-levels-joint_joint-one-RRTConnect/smoke/0
```

小样本至少检查：

- `dataset.hdf5` 和 `args.yaml` 已生成；
- 生成成功率和失败原因；
- joint-to-region 与 region-to-region 数量接近 1:1；
- region-to-region 的端点分别属于 drawer 和 shelf，且两个方向均有样本；
- shelf 下层/中层在 shelf 样本中接近 1:1；
- drawer 中央/内缩角落在 drawer 样本中接近 1:1；
- 分别统计各 subregion 的 IK 重采样率和 OMPL 成功率；
- 稠密插值后的整条路径无环境碰撞和自碰撞；
- 随机抽取若干条在 PyBullet 中可视化。

不要因为 OMPL 返回 success 就跳过稠密碰撞检查；简化后的稀疏路径可能漏掉
相邻 waypoint 之间的碰撞。

### 5.4 批量生成生产数据

在 `launch_generate_trajectories.py` 中增加一个独立实验项，建议初始参数：

- `env_id=EnvOpenDrawerShelf`
- `robot_id=RobotPanda`
- `cfg_file=EnvOpenDrawerShelf-RobotPanda-drawer-to-shelf.yaml`
- `planner=RRTConnect`
- `num_trajectories_per_task=1`
- `min_distance_robot_env=0.02`
- `planner_allowed_time=30`
- `fit_bspline=False`
- `interpolate_num=128`
- `n_parallel_jobs=4`（当前 Python 3.8 + PyBullet/OMPL native 栈的实测稳定值；
  8 和 16 进程均可复现 `SIGSEGV`，因此不能按 CPU 核数直接设置）
- `task_batch_size=1`（每条轨迹独占一个 PyBullet/OMPL client，并在
  `finally` 中释放；同一个 PbOMPL planner 跨任务复用会触发原生
  `SIGSEGV`）
- `task_timeout_seconds=300`（单个任务超过 5 分钟时终止对应 worker，跳过
  当前任务并重新创建 worker，不阻塞其他任务）
- 设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1` 和
  `OPENBLAS_NUM_THREADS=1`，避免每个子进程再次创建整机数量的线程
- 先生成 10,000 条 pilot，再根据成功率扩到 100,000～500,000 条
- 该机器建议每个 shard 500 个任务，使用不重叠的
  `start_task_id` 和 seed，以限制单个 native 故障的重跑成本

`num_tasks=150000` 表示尝试 150,000 个任务，不保证最终恰好写入 150,000
条成功轨迹。应先用 pilot 统计 IK 与 OMPL 总成功率 `r`，正式任务数至少设置为
`ceil(150000 / r)`，合并后再检查
`num_trajectories_generated`，并用新的、不重叠的 task id 补齐缺口。

不要与现有 `config_file` 数据目录共用名字；launcher 启动前会提示可能覆盖
`data_trajectories`，应先确认最终目录。

生产配置可设置为尝试 `150_000` 个任务，并拆成每个 500 个任务的 300 个
shard。launcher 使用基于脚本位置解析的绝对 `DATA_TRAJECTORIES_DIR`，因此不论
从哪个工作目录启动，数据集根目录都是：

```text
/home/eric/Projects/MotionPlanningDiffusion/mpd/data_trajectories/EnvOpenDrawerShelf-RobotPanda-drawer-to-shelf-reachable-levels-joint_joint-one-RRTConnect
```

每个 shard 的完整 `results_dir` 还会附加 launcher 参数目录，例如
`EnvOpenDrawerShelf/RobotPanda/yes/False/one/RRTConnect`。输出根目录由
`launch_generate_trajectories.py` 中的 `DATA_TRAJECTORIES_DIR` 设置。

launcher 的覆盖确认只在主进程启动时询问一次。重新运行时，它会检查每个
shard 的 `args.yaml`、`timing_stats.pkl` 和 `dataset.hdf5`，并验证 HDF5 中的
期望轨迹数量；完整 shard 自动跳过，缺失、损坏或任务数不匹配的 shard 才会
使用相同 task id 重跑。不要把只有 `args.yaml` 或 `logfile` 的目录当作完成。

并行实现不再使用 joblib。主进程监督 4 个 `spawn` worker；目标位姿采样、
IK 和 OMPL 都在 worker 内并行执行。每个任务重新创建并在 `finally` 中释放
自己的 `GenerateDataOMPL`/PyBullet DIRECT client。若 native worker 发生
`SIGSEGV`，主进程把当前任务记为失败、重建该 worker，并继续其余 shard；
`timing_stats.pkl` 中的 `num_worker_failures` 可用于统计此类恢复次数。

批量启动入口：

```bash
cd scripts/generate_data
conda run --no-capture-output -n mpd-splines-public \
  python launch_generate_trajectories.py
cd ../..
```

### 5.5 合并 shard

```bash
conda run --no-capture-output -n mpd-splines-public \
  python scripts/generate_data/post_process_generated_dataset.py \
  --data_dir data_trajectories/EnvOpenDrawerShelf-RobotPanda-drawer-to-shelf-reachable-levels-joint_joint-one-RRTConnect
```

输出应包括：

```text
data_trajectories/<DATASET_DIR>/
├── args.yaml
├── dataset_merged.hdf5
└── <各生成 shard>/dataset.hdf5
```

检查 `args.yaml` 中的 `env_id`、`robot_id`、`cfg_file`、planner 和生成数量。
是否执行 path flip 取决于训练方案。先用 `dataset_merged.hdf5` 建立不增强
基线；需要与仓库既有 doubled 数据集对齐时，再明确运行 flip 并改用
`dataset_merged_doubled.hdf5`。

### 5.6 训练前做一个短跑

先用少量数据和少量 step 检查数据 loader、B-spline 拟合、loss 和 checkpoint：

```bash
conda run --no-capture-output -n mpd-splines-public \
  python scripts/train/train.py \
  --dataset_subdir /home/eric/Projects/MotionPlanningDiffusion/mpd/data_trajectories/EnvOpenDrawerShelf-RobotPanda-drawer-to-shelf-reachable-levels-joint_joint-one-RRTConnect/EnvOpenDrawerShelf \
  --dataset_file_merged dataset_merged_doubled.hdf5 \
  --reload_data False \
  --n_task_samples 1000 \
  --parametric_trajectory_class ParametricTrajectoryBspline \
  --bspline_degree 5 \
  --bspline_num_control_points_desired 22 \
  --num_T_pts 128 \
  --context_qs True \
  --context_ee_goal_pose True \
  --unet_input_dim 32 \
  --unet_dim_mults_option 1 \
  --batch_size 512 \
  --num_train_steps 1000 \
  --use_ema True \
  --clip_grad True \
  --device cuda:0 \
  --debug False \
  --steps_til_summary 200 \
  --steps_til_ckpt 500 \
  --results_dir data_trained_models/drawer_to_shelf_smoke
```

第一次读取新 HDF5 或 HDF5 发生变化时使用 `--reload_data True`，避免复用旧
reload cache。确认缓存和数据不再变化后，正式训练可设为 `False`。

### 5.7 正式训练

基线建议：

```bash
conda run --no-capture-output -n mpd-splines-public \
  python scripts/train/train.py \
  --dataset_subdir /home/eric/Projects/MotionPlanningDiffusion/mpd/data_trajectories/EnvOpenDrawerShelf-RobotPanda-drawer-to-shelf-reachable-levels-joint_joint-one-RRTConnect/EnvOpenDrawerShelf \
  --dataset_file_merged dataset_merged_doubled.hdf5 \
  --reload_data False \
  --n_task_samples -1 \
  --parametric_trajectory_class ParametricTrajectoryBspline \
  --bspline_degree 5 \
  --bspline_num_control_points_desired 22 \
  --num_T_pts 128 \
  --context_qs True \
  --context_qs_n_layers 2 \
  --context_q_out_dim 128 \
  --context_ee_goal_pose True \
  --context_ee_goal_pose_n_layers 2 \
  --context_ee_goal_pose_out_dim 128 \
  --context_combined_out_dim 128 \
  --generative_model_class GaussianDiffusionModel \
  --variance_schedule cosine \
  --n_diffusion_steps 100 \
  --predict_epsilon True \
  --conditioning_type default \
  --unet_input_dim 32 \
  --unet_dim_mults_option 1 \
  --batch_size 512 \
  --lr 0.0003 \
  --num_train_steps 3000000 \
  --use_ema True \
  --clip_grad True \
  --steps_til_summary 25000 \
  --steps_til_ckpt 50000 \
  --device cuda:0 \
  --debug False \
  --results_dir data_trained_models/drawer_to_shelf
```

`batch_size=128` 是保守起点；显存允许时再提高。训练集是目标位姿约束任务，
应保留 `context_qs=True` 和 `context_ee_goal_pose=True`。

### 5.8 接入推理和 IsaacLab

训练完成后：

1. 找到包含训练 `args.yaml` 和 EMA checkpoint 的 model directory。
2. 将 drawer-to-shelf 推理配置中的 `model_dir_ddpm_bspline` 指向它。
3. 保持 `planning_env_id=EnvOpenDrawerShelf`。
4. 保持现有 start/goal region 的最终世界坐标，不改
   `rotation_z_axis_deg`。
5. 先运行少量 MPD 推理并做 torch-robotics 稠密碰撞验证。
6. 再运行 IsaacLab evaluation/replay，确认仿真中的 drawer、shelf 和 Panda
   与规划环境一致。

如果训练数据只覆盖下层和中层，推理 region 也只能在这两个已训练分布内。
不要在未重新训练和验证的情况下把推理目标直接扩到上层。

## 6. 如果必须覆盖 shelf 全部三层

建议优先改场景，而不是继续扩大一个不可达的 region：

1. 在 `env_open_drawer_shelf.py` 中把 shelf 整体向 Panda 基座移近；
2. 视需要把 shelf 整体降低，使上层目标进入 Panda 的可靠工作空间；
3. 保持开口朝向和默认相机视角不变；
4. 同步修改所有 shelf box center，而不是只移动隔板；
5. 根据新几何重新计算三个隔间的自由高度；
6. 分层运行 IK 成功率测试；
7. 每层至少验证多条 OMPL 路径和稠密碰撞；
8. 再把第三个 z 区间加入数据 YAML；
9. 同步推理 region、IsaacLab payload 测试和场景回放。

不要通过修改 `rotation_z_axis_deg` 来“修复”上层可达性；那会改变坐标解释
和 region 选择逻辑，却不会解决真实的工作空间距离问题。

## 7. 推荐验收门槛

在开始长时间训练前，建议满足：

- 至少 10,000 个 pilot 任务；
- IK 重采样率、OMPL 失败率和各失败原因有统计；
- joint-to-region 与 region-to-region 比例接近 1:1；
- region-to-region 不出现 drawer-to-drawer 或 shelf-to-shelf；
- drawer-to-shelf 与 shelf-to-drawer 都有覆盖且数量大致平衡；
- shelf 下层/中层在 shelf 样本中的比例接近 1:1；
- drawer 中央/内缩角落在 drawer 样本中的比例接近 1:1；
- 各 subregion 有独立的成功率统计；
- 内缩角落的 IK 重采样率处于可接受范围；
- 稠密碰撞验证通过率达到预设门槛；
- PyBullet 与 IsaacLab 的障碍物坐标抽样核对一致；
- 短训练能稳定下降并成功保存、重新加载 EMA checkpoint；
- 新模型在独立 region 随机种子上评估，而不是只回放训练轨迹。

满足这些条件后，再扩大任务量或放宽 region。先扩大数据量、后发现方向或
坐标定义错误，会使整批 OMPL 数据和训练 checkpoint 都失去价值。
