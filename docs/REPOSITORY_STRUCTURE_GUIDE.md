# MPD 仓库结构与常见修改入口

本文档用于快速定位本仓库中训练、推理、数据生成、环境、机器人、SDF 和 cost guidance 的实现位置。它描述的是当前代码的实际组织方式，适合作为日常开发时的索引。

## 1. 从数据到推理的整体流程

```text
data_generation_cfgs/*.yaml
        |
        v
scripts/generate_data/
  生成 start/goal、调用规划器、保存轨迹
        |
        v
data_trajectories/<dataset>/
  args.yaml + merged HDF5
        |
        v
scripts/train/train.py
  构建 dataset、ContextModel、UNet、diffusion/CVAE
        |
        v
训练目录/
  args.yaml + checkpoints + train/val indices
        |
        v
scripts/inference/cfgs/*.yaml
  选择模型、环境替换、cost、DDIM/DDPM guidance 参数
        |
        v
scripts/inference/inference.py
        |
        v
mpd/inference/inference.py
  采样、cost guidance、轨迹选择和评估
```

理解这条链路很重要：数据生成时的环境和机器人信息会写入数据集目录中的 `args.yaml`；训练再把模型配置写入训练结果目录；推理同时读取训练目录的 `args.yaml` 和推理 YAML。

## 2. 顶层目录

| 路径 | 作用 |
| --- | --- |
| `scripts/` | 用户直接运行的训练、推理、数据生成和 Isaac Lab 脚本。大部分实验配置从这里进入。 |
| `mpd/` | MPD 核心 Python 包，包括模型、数据集、loss、训练器、推理 guidance、轨迹参数化和评估。 |
| `mpd/torch_robotics/` | 机器人、环境、运动学、碰撞检测、SDF、planning task 和可视化等基础设施。 |
| `mpd/motion_planning_baselines/` | RRT、CHOMP、GPMP、STOMP、MPPI 等传统规划 baseline。其 cost 系统与 MPD inference guidance 的 cost 系统不同。 |
| `data_generation_cfgs/` | 轨迹数据生成配置，主要定义 `env_id`、`robot_id`、末端位姿采样区域等。 |
| `data_public/` | 发布的数据集和预训练模型。通常通过符号链接映射成根目录下的 `data_trajectories/` 和 `data_trained_models/`。 |
| `deps/` | 外部依赖和子模块，例如 Isaac Gym、Theseus、PyBullet OMPL 和 experiment launcher。一般不在这里实现 MPD 业务逻辑。 |
| `tests/` | 自动化测试。目前主要覆盖 Isaac Lab 子进程工具，核心 MPD 模块的测试仍较少。 |
| `figures/` | README 和文档使用的 GIF、图片等静态资源。 |
| `*.md` | 项目安装、迁移、运行和扩展规划文档。 |

## 3. `scripts/`：实验入口

### 3.1 `scripts/train/`

| 文件 | 作用 |
| --- | --- |
| `scripts/train/train.py` | 单次训练入口；定义数据集、B-spline、context、diffusion/CVAE、UNet、batch size、学习率、训练步数、EMA 等参数。 |
| `scripts/train/launch_train_diffusion_models-v00.py` | 批量启动 diffusion 训练实验。适合配置多个环境或超参数组合。 |
| `scripts/train/launch_train_cvae_models-v00.py` | 批量启动 CVAE 训练实验。 |
| `scripts/train/launch_train_diffusion_models-EnvEmpty2D-v00.py` | 针对特定环境的 launch 示例。 |

修改训练超参数时，通常先改 `scripts/train/train.py` 中 `experiment(...)` 的参数，或者在 launch 文件中覆盖这些参数。

常见参数分组：

- 数据：`dataset_subdir`、`dataset_file_merged`、`reload_data`、`n_task_samples`。
- 轨迹：`parametric_trajectory_class`、`bspline_degree`、`bspline_num_control_points_desired`、`num_T_pts`。
- 条件：`context_qs`、`context_ee_goal_pose` 及各 context 输出维度。
- Diffusion：`variance_schedule`、`n_diffusion_steps`、`predict_epsilon`。
- 模型：`conditioning_type`、`unet_input_dim`、`unet_dim_mults_option`。
- 优化：`batch_size`、`lr`、`clip_grad`、`num_train_steps`、`use_ema`、`use_amp`。
- 输出：`results_dir`、`steps_til_summary`、`steps_til_ckpt`、`wandb_*`。

训练器的实际优化循环、AdamW、EMA、梯度裁剪、scheduler 和 checkpoint 保存位于 `mpd/trainer/trainer.py`。当前 `train.py` 调用训练器时固定使用 `lr_scheduler=False`；若要启用或更换 scheduler，需要同时查看这两个文件。

### 3.2 `scripts/inference/`

| 文件 | 作用 |
| --- | --- |
| `scripts/inference/inference.py` | 推理命令入口，选择推理 YAML，读取训练参数并启动 MPD inference。 |
| `scripts/inference/cfgs/*.yaml` | 每个环境/机器人对应的推理超参数。这里是修改采样和 guidance 参数的首选位置。 |
| `scripts/inference/launch_inference-experiments.py` | 批量评估多个模型、环境和 planner 配置。 |

推理 YAML 中常见参数：

- 模型：`model_dir_ddpm_bspline`、`model_dir_cvae_bspline`、`model_dir_ddpm_waypoints`、`model_selection`。
- 环境：`env_id_replace`、`rotation_z_axis_deg`、`grasped_object`。
- 碰撞余量：`obstacle_cutoff_margin_extra`、`margin_for_dense_collision_checking`。
- 轨迹：`num_T_pts`、`trajectory_duration`、`phase_time_class`。
- Cost：`costs` 下的 cost 类名、权重和特定参数。
- 采样：`n_trajectory_samples`、`planner_alg`、`diffusion_sampling_method`。
- DDIM guidance：`ddim_sampling_timesteps`、`t_start_guide_steps_fraction`、`n_guide_steps`、`guide_lr`、`ddim_scale_grad_prior`、`max_perturb_x` 和梯度裁剪参数。
- 后处理：`extra_mp_steps`、`best_trajectory_selection`。

需要修改推理流程本身，而不是参数时，主要看 `mpd/inference/inference.py`。Diffusion 的 DDPM/DDIM 反向采样和 guide 插入时机位于 `mpd/models/diffusion_models/diffusion_model_base.py`，通用的 guide 梯度更新位于 `mpd/models/diffusion_models/sample_functions.py`。

### 3.3 `scripts/generate_data/`

| 文件 | 作用 |
| --- | --- |
| `generate_trajectories.py` | 单个数据生成任务的核心入口，创建环境、机器人、start/goal，并调用规划器生成轨迹。 |
| `launch_generate_trajectories.py` | 批量或并行启动数据生成。数据量、环境、规划器和 seed 等实验组合通常在这里配置。 |
| `post_process_generated_dataset.py` | 合并、整理生成结果，形成训练使用的 HDF5。 |
| `flip_solution_paths.py` | 通过翻转起终点和轨迹做数据增强。 |
| `visualize_trajectories.py` | 检查生成数据和环境是否正确。 |
| `make_trajectories_gp_prior.py` | 生成或转换 GP prior 相关轨迹。 |

修改 Warehouse 数据生成范围时，通常需要同时调整：

1. `data_generation_cfgs/EnvWarehouse-RobotPanda*.yaml` 中的 `env_id`、`robot_id` 和 `pose_regions`。
2. `scripts/generate_data/launch_generate_trajectories.py` 中的数据量、规划器、配置文件和并行参数。
3. 如果要改变环境几何，再修改 `mpd/torch_robotics/torch_robotics/environments/env_warehouse.py`。

### 3.4 `scripts/isaaclab/`

用于把 MPD 轨迹送入 Isaac Lab 回放和评估：

- `replay_mpd_trajectory.py`：轨迹回放。
- `evaluate_mpd_trajectories.py`：仿真评估。
- `scene_payload.py`：场景数据传递。
- `subprocess_utils.py`：子进程启动与结果处理。

此目录服务于仿真验证，不负责 MPD 模型训练。

## 4. `mpd/`：核心模块

### 4.1 `mpd/models/`

| 路径 | 作用 |
| --- | --- |
| `diffusion_models/models.py` | Temporal UNet 等 diffusion denoiser 结构。 |
| `diffusion_models/diffusion_model_base.py` | 加噪、训练 loss 调用、DDPM/DDIM 采样和 inference guidance 插入。 |
| `diffusion_models/context_models.py` | `ContextModelQs`、`ContextModelEEPoseGoal` 和 `ContextModelCombined`。新增 world latent 时主要修改这里。 |
| `diffusion_models/sample_functions.py` | 通用采样函数和 `guide_gradient_steps`。 |
| `diffusion_models/helpers.py` | diffusion schedule、tensor 提取和 hard conditioning 等辅助函数。 |
| `cvae/` | CVAE prior 的编码器、解码器、loss 和采样。 |
| `layers/` | MLP、attention、等变层等底层网络组件。 |

常见操作：

- 改 UNet 宽度/层级：先改训练参数 `unet_input_dim`、`unet_dim_mults_option`；需要新结构时改 `models.py`。
- 改环境条件输入：改 `context_models.py`，同时扩展 dataset context 和 `scripts/train/train.py` 的构建逻辑。
- 改 diffusion 步数或噪声 schedule：优先改训练参数；算法级变化再改 `diffusion_model_base.py`。
- 改 guide 在采样中的时机或组合方式：改 `diffusion_model_base.py`。

### 4.2 `mpd/datasets/`

| 文件 | 作用 |
| --- | --- |
| `trajectories_dataset_bspline.py` | 读取 HDF5，构造 B-spline 控制点、normalization、context 和 hard conditions。 |
| `trajectories_dataset_waypoints.py` | Waypoint 版本的数据集和 hard conditions。 |
| `normalization.py` | 数据归一化实现。 |
| `utils.py` | 数据处理辅助函数。 |

若要把 `env_id`、obstacle parameters、SDF grid 或 world latent 加入训练输入，需要修改数据生成格式，并在这里把字段读入 `input_dict/context_d`。

`mpd/utils/loaders.py` 负责把数据集中的 `args.yaml` 转成环境、机器人、planning task、dataset 和 dataloader。环境类通过 `getattr(environments, env_id)` 获取，所以新环境还必须从 environments 包导出。

### 4.3 `mpd/parametric_trajectory/`

- `trajectory_bspline.py`：B-spline 控制点到连续轨迹的位置、速度和加速度。
- `trajectory_waypoints.py`：Waypoint 轨迹表示。
- `trajectory_base.py`：公共接口。
- `phase_time.py`：线性或 sigmoid phase-time 映射。

只改控制点数量、degree、采样点数和 trajectory duration 时，一般修改训练/推理参数即可。要改变参数化公式、边界条件或导数计算时才修改这里。

### 4.4 `mpd/inference/`

| 文件 | 作用 |
| --- | --- |
| `inference.py` | 模型加载、context 准备、planner 选择、采样、额外优化、轨迹筛选和结果组织。 |
| `cost_guides.py` | MPD inference 使用的 cost 类和从 task/joint space 到控制点的梯度链。 |

当前主要 cost 类包括：

- `CostTaskSpaceCollisionObjects`
- `CostTaskSpaceCollisionSelf`
- `CostTaskSpaceEEGoalPose`
- `CostJointSpaceJointLimits`
- `CostJointSpacePathLength`
- `CostJointSpaceVelocity`
- `CostJointSpaceAcceleration`

启用、禁用和调权重时，只需要改推理 YAML 的 `costs`。新增 cost 时，需要在 `mpd/inference/cost_guides.py` 中实现新类，并保证类名与 YAML 中的 key 一致。当前 `CostGuideManagerParametricTrajectory` 根据名称创建 cost，因此重命名会影响配置兼容性。

注意：`mpd/motion_planning_baselines/mp_baselines/planners/costs/cost_functions.py` 是 CHOMP/GPMP/STOMP 等 baseline 使用的另一套 cost；修改它不会自动改变 MPD 的 cost guidance。

### 4.5 `mpd/losses/`

- `gaussian_diffusion_loss.py`：diffusion 训练 loss 包装。
- `cvae_loss.py`：CVAE reconstruction 和 KL loss。

当前 diffusion 训练主要是 denoising loss，inference 中的 collision/goal cost 不会自动进入训练。若要做 cost-aware training，需要同时修改 loss、dataset 提供的信息以及模型训练流程，而不是只改推理 YAML。

### 4.6 `mpd/trainer/`

`trainer.py` 实现：

- AdamW optimizer。
- AMP。
- 梯度裁剪。
- EMA。
- validation、early stopping。
- checkpoint 和 loss 保存。
- 可选的 cosine warm restart scheduler。

修改学习率数值通常在 `scripts/train/train.py`；修改 optimizer 类型、参数组学习率、resume optimizer state 或 scheduler 行为则在 `trainer.py`。

### 4.7 其他 `mpd/` 目录

| 路径 | 作用 |
| --- | --- |
| `mpd/metrics/` | 成功率、路径质量或轨迹相关评估指标。 |
| `mpd/summaries/` | 训练期间的可视化和 summary。 |
| `mpd/plotting/` | 图像、视频和轨迹可视化工具。 |
| `mpd/utils/` | loader、checkpoint/model 加载、压缩、orientation 和通用工具。 |
| `mpd/paths.py` | 数据等路径的集中定义。路径或符号链接异常时优先检查这里和环境变量。 |

## 5. `mpd/torch_robotics/`：环境、机器人和碰撞

### 5.1 环境

路径：`mpd/torch_robotics/torch_robotics/environments/`

| 文件 | 作用 |
| --- | --- |
| `env_base.py` | 环境基类；管理 fixed/extra objects、SDF grid 和 occupancy map。 |
| `env_warehouse.py` | Warehouse 和 Warehouse extra objects 的具体几何。 |
| `env_*.py` | 其他 2D/3D 环境。 |
| `primitives.py` | sphere、box 等障碍物 primitive 及其 signed distance。 |
| `grid_map_sdf.py` | 从显式障碍物预计算 SDF grid 和梯度。 |
| `occupancy_map.py` | occupancy map。 |
| `__init__.py` | 导出环境类，供 loader 通过字符串 `env_id` 创建。 |

新增环境的常规步骤：

1. 新建 `env_xxx.py` 并继承 `EnvBase`。
2. 定义 workspace limits、fixed objects 和可选 extra objects。
3. 在 `environments/__init__.py` 导出新类。
4. 在 `data_generation_cfgs/*.yaml` 中使用新的 `env_id`。
5. 若只在推理替换环境，可在推理 YAML 中设置 `env_id_replace`。

`env_id_replace` 只会替换推理时的环境，不会让已训练 prior 自动获得环境条件。模型如果没有 world/environment context，仍然主要依靠 cost guidance 适应替换后的障碍物。

### 5.2 机器人

路径：`mpd/torch_robotics/torch_robotics/robots/`

- `robot_base.py`：机器人公共接口、collision fields 和运动学相关逻辑。
- `robot_panda.py`：Panda 定义。
- `robot_point_mass.py`、`robot_planar_link.py`：其他机器人。
- `torchkin_robot_wrapper.py`：运动学树包装。

机器人资源位于 `mpd/torch_robotics/torch_robotics/data/`：

- `urdf/`：URDF 和 mesh。
- `configs/*/joint_limits.yaml`：关节限制。
- `configs/*/*sphere_config.yaml`：collision sphere 配置。

修改机器人碰撞体半径或 link sphere 时，优先改 sphere config；修改运动范围时检查 joint limits 和机器人类。

### 5.3 Planning task、SDF 和 collision field

| 路径 | 作用 |
| --- | --- |
| `tasks/tasks.py` | 组合 environment、robot、parametric trajectory，创建 object/self/workspace collision fields。 |
| `torch_planning_objectives/fields/distance_fields.py` | 把 object signed distance 转成 collision cost 和梯度。 |
| `environments/grid_map_sdf.py` | 当前显式环境的 grid SDF。 |

修改 collision cost 的含义时，需要区分三个层次：

1. 推理 YAML：选择 cost、权重、安全 margin 和 guide 强度。
2. `mpd/inference/cost_guides.py`：定义整条轨迹的 cost，以及梯度如何映射回控制点。
3. `distance_fields.py` / SDF：定义单个机器人 link 位置相对障碍物的距离、margin 和局部梯度。

如果接入预训练 SDF，推荐保持 `compute_distance_field_cost_and_gradient(link_pos)` 接口不变，在 distance field 层新增 learned SDF provider，这样上层 cost guidance 可以复用。

## 6. 常见修改速查

| 想做的操作 | 首先修改 | 可能还需修改 |
| --- | --- | --- |
| 改训练学习率、batch size、训练步数 | `scripts/train/train.py` | 批量实验时改 `scripts/train/launch_train_*.py` |
| 改 optimizer、scheduler、EMA、resume | `mpd/trainer/trainer.py` | `scripts/train/train.py` 的参数和调用 |
| 换训练数据集 | `scripts/train/train.py` 的 `dataset_subdir` | 确认数据集目录下 `args.yaml` 和 HDF5 匹配 |
| 改 diffusion steps、schedule、预测目标 | `scripts/train/train.py` | 算法级变化改 `diffusion_model_base.py` |
| 改 UNet 规模 | `scripts/train/train.py` | 新结构改 `models.py`，需要重新训练 |
| 改 start/goal 或 EE goal context | `scripts/train/train.py` | `context_models.py`、dataset context、checkpoint 兼容性 |
| 加 world latent | `context_models.py` | 数据生成、dataset、`train.py`、推理 context 构造 |
| 改推理采样数量或 planner | `scripts/inference/cfgs/*.yaml` | 流程变化改 `mpd/inference/inference.py` |
| 改 DDIM/DDPM guidance 强度和步数 | `scripts/inference/cfgs/*.yaml` | 采样公式变化改 `diffusion_model_base.py` |
| 调整 cost 权重/启用 cost | 推理 YAML 的 `costs` | 无需重新训练 prior |
| 新增 MPD cost function | `mpd/inference/cost_guides.py` | 推理 YAML 添加同名 key |
| 改 SDF 或 collision margin 公式 | `distance_fields.py` | `env_base.py`、`grid_map_sdf.py`、推理 margin 参数 |
| 修改 Warehouse 几何 | `environments/env_warehouse.py` | 新类需要在 `environments/__init__.py` 导出 |
| 推理时换 extra 环境 | 推理 YAML 的 `env_id_replace` | 确保环境类已导出 |
| 改数据生成位姿区域 | `data_generation_cfgs/*.yaml` | `launch_generate_trajectories.py` |
| 改用于生成标签的规划器/数据量 | `launch_generate_trajectories.py` | `generate_trajectories.py`、baseline planner |
| 改 B-spline degree/控制点数 | `scripts/train/train.py` | 推理读取训练 `args.yaml`，结构变化需要重训 |
| 改轨迹展开点数/时长 | 训练和推理对应参数 | `parametric_trajectory/` 仅在公式变化时修改 |
| 改 Panda collision spheres | `data/configs/panda/panda_sphere_config.yaml` | `robot_panda.py` |
| 添加评估指标 | `mpd/metrics/metrics.py` | inference 或 summary 中调用 |
| 修改 Isaac Lab 回放/验证 | `scripts/isaaclab/` | 场景和机器人映射配置 |

## 7. 容易混淆的几点

### 7.1 训练超参数和推理超参数是分开的

`n_diffusion_steps` 是训练 diffusion 过程的步数，来自训练参数；`ddim_sampling_timesteps` 是推理时实际执行的 DDIM 步数，来自推理 YAML。后者可以比前者小，但不能把两者当成同一个参数。

### 7.2 修改 cost guidance 通常不需要重新训练

当前 collision、self-collision、joint limit、velocity 和 acceleration cost 主要在 inference 使用。改 YAML 中的 cost 权重或 guide 参数会改变推理结果，但不会改变已训练 diffusion prior。

### 7.3 修改 context 或模型维度需要重新训练

`context_qs`、`context_ee_goal_pose`、world latent、UNet 通道数和控制点数都会改变模型结构或数据表示。旧 checkpoint 通常不能严格加载；若做热启动，需要显式处理 partial state dict。

### 7.4 `env_id_replace` 不是多环境训练

它只在 loader 中把数据集原环境替换成另一个环境用于推理/评估。若训练数据和 ContextModel 中没有环境信息，prior 不会真正学习不同环境之间的差异。

### 7.5 MPD cost 与 baseline cost 不同

MPD inference cost 在 `mpd/inference/cost_guides.py`；传统规划器 cost 在 `mpd/motion_planning_baselines/.../cost_functions.py`。修改前先确认当前 planner 使用哪套实现。

## 8. 推荐的修改顺序

进行新实验时，建议按以下顺序操作：

1. 先在环境文件和数据生成 YAML 中确认几何、start/goal 采样范围。
2. 用少量任务运行 `generate_trajectories.py` 并可视化。
3. 合并 HDF5，检查数据集目录中的 `args.yaml`。
4. 在 `scripts/train/train.py` 中用小 `n_task_samples` 和较少 step 做 smoke training。
5. 用匹配的推理 YAML 加载新训练目录。
6. 先测试 `diffusion_prior`，再测试 `mpd`，区分 prior 质量和 cost guidance 效果。
7. 最后再扩大数据量、训练步数和批量实验规模。

这样出现问题时，可以判断错误来自环境、数据、模型 prior，还是 inference guidance，而不必同时排查整条链路。
