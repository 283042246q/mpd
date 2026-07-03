# New Project Run Guide

本文档说明迁移到 IsaacLab 后，本项目各类程序应该如何启动。命令默认从仓库根目录执行：

```bash
cd /home/eric/MotionPlanningDiffusion/mpd-splines-public
```

## 环境约定

MPD 主环境：

```bash
source set_env_variables.sh
conda activate mpd-splines-public-cu128
```

IsaacLab 环境：

```bash
export ISAACLAB_ROOT=/home/eric/IsaacLab_ori
export ISAACLAB_CONDA_ENV=env_isaaclab_ori
```

推荐保持两个环境分离：

- `mpd-splines-public`：训练、数据、采样、优化、metrics、调用 IsaacLab 子进程。
- `env_isaaclab_ori`：Isaac Sim / IsaacLab 仿真验证。

如果只做 import、文档检查或 CPU 路径验证，在 RTX 5060 + 旧 PyTorch 环境上建议隐藏 GPU：

```bash
CUDA_VISIBLE_DEVICES='' python ...
```

不要在需要 `--sim_backend isaaclab` 且 `--isaaclab_device cuda:0` 的命令前设置 `CUDA_VISIBLE_DEVICES=''`，因为 IsaacLab evaluator 是子进程，会继承这个环境变量。需要 IsaacLab 用 GPU 时，MPD 主进程用 `--device cpu` 即可。

## 标准完整流程

新 project 建议按下面顺序跑。已有公开数据和预训练模型时，优先用下载数据；只有需要重新生成数据集时才跑完整数据生成。

1. 准备仓库、环境、数据和模型。

```bash
cd /home/eric/MotionPlanningDiffusion/mpd-splines-public
source set_env_variables.sh
conda activate mpd-splines-public-cu128
export ISAACLAB_ROOT=/home/eric/IsaacLab_ori
export ISAACLAB_CONDA_ENV=env_isaaclab_ori
```

如果使用 README 中的公开包：

```bash
tar -xvf data_public.tar.gz
ln -s data_public/data_trajectories data_trajectories
ln -s data_public/data_trained_models data_trained_models
```

2. 可选：生成新数据集。

```bash
cd scripts/generate_data
python launch_generate_trajectories.py
```

`launch_generate_trajectories.py` 会先询问是否继续，因为它可能覆盖 `data_trajectories/` 下的数据。生成后合并数据，并按需要翻转轨迹做数据增强：

```bash
python post_process_generated_dataset.py --data_dir ../../data_trajectories/<DATASET_DIR>
python flip_solution_paths.py
```

`flip_solution_paths.py` 仍需要先在脚本内确认 `PATH_TO_DATASETS` 指向目标数据集。

3. 训练模型。

```bash
cd scripts/train
python train.py --device cuda:0 --debug False ----results_dir /home/eric/MotionPlanningDiffusion/mpd-splines-public/scripts/train/logs/
```

批量训练使用 launcher：

```bash
python launch_train_diffusion_models-v00.py
python launch_train_cvae_models-v00.py
```

如果当前 MPD 环境的 PyTorch 不支持本机显卡架构，先用 CPU 验证流程：

```bash
CUDA_VISIBLE_DEVICES='' python train.py --device cpu
```

4. 跑推理，不启动物理仿真。

```bash
cd ../inference
python inference.py \
  --cfg_inference_path ./cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml \
  --sim_backend none \
  --device cuda:0 \
  --results_dir /home/eric/MotionPlanningDiffusion/mpd-splines-public/scripts/inference/logs/
```

在 RTX 5060 + 旧 PyTorch 环境上，MPD 推理也建议先走 CPU：

```bash
CUDA_VISIBLE_DEVICES='' python inference.py \
  --cfg_inference_path ./cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml \
  --sim_backend none \
  --device cpu
```

5. 对 Panda 轨迹跑 IsaacLab 物理验证。

MPD 主进程可以继续用 CPU，IsaacLab 子进程用 `env_isaaclab_ori` 和 `cuda:0`：

```bash
python inference.py \
  --cfg_inference_path ./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml \
  --sim_backend isaaclab \
  --device cuda:0 \
  --isaaclab_root /home/eric/IsaacLab_ori \
  --isaaclab_conda_env env_isaaclab_ori \
  --isaaclab_device cuda:0 \
  --isaaclab_timeout_s 900 \
  --results_dir /home/eric/MotionPlanningDiffusion/mpd-splines-public/scripts/inference/logs/ \
  --isaaclab_headless False
```

6. 检查结果。

常规推理、训练输出在 `logs/`；数据生成输出在 `data_trajectories/`；训练模型输出在 `data_trained_models/`。IsaacLab 验证会在 `logs/` 额外生成 `isaaclab-trajectories-XXX.pt`、`isaaclab-statistics-XXX.json` 和 `isaaclab-evaluator-XXX.log`。

7. 可视化或 replay。

```bash
cd ../generate_data
python visualize_trajectories.py ../../data_trajectories/<DATASET_DIR> --sim-backend none

cd ../..
python mpd/motion_planning_baselines/examples/panda_isaac_replay.py \
  --sim_backend isaaclab \
  --output_dir logs/panda_replay \
  --isaaclab_root /home/eric/IsaacLab_ori \
  --isaaclab_conda_env env_isaaclab_ori \
  --isaaclab_device cuda:0
```

## 流程输入输出和数据流

这一节按数据链路说明每一步读什么、写什么，以及下一步如何使用上一步结果。公开数据包已经包含 `data_trajectories/` 和 `data_trained_models/` 时，可以跳过数据生成和训练，直接从“推理”开始。

### 总体链路

```text
OMPL/RRTConnect 先验轨迹
  -> data_trajectories/<DATASET_DIR>/**/dataset.hdf5
  -> data_trajectories/<DATASET_DIR>/dataset_merged.hdf5
  -> data_trajectories/<DATASET_DIR>/dataset_merged_doubled.hdf5
  -> 训练输出 <MODEL_DIR>/args.yaml + <MODEL_DIR>/checkpoints/*.pth
  -> inference cfg 中的 model_dir_*
  -> logs/<seed>/results_single_plan-XXX.pt
  -> IsaacLab 读取有效轨迹并写 isaaclab-*.json / *.log
```

### 数据生成：先验/专家轨迹

入口：

```bash
cd scripts/generate_data
python generate_trajectories.py
python launch_generate_trajectories.py
```

输入：

- `generate_trajectories.py` 或 `launch_generate_trajectories.py` 中的 `env_id`、`robot_id`、`planner`、`num_tasks`、`num_trajectories_per_task` 等参数。
- 可选环境配置文件：`data_generation_cfgs/*.yaml`，例如 `EnvWarehouse-RobotPanda_v01.yaml`。
- PyBullet + OMPL，本项目默认常用 `RRTConnect` 生成可行解；这些轨迹就是后续训练用的先验/专家轨迹数据。

输出：

- 单个任务分片写到 `results_dir/dataset.hdf5`。
- 同目录还会有 `args.yaml` 和 `timing_stats.pkl`。
- 批量脚本默认把结果组织到 `../../data_trajectories/<DATASET_DIR>/...` 下。

下一步如何使用：

- `post_process_generated_dataset.py` 会递归读取 `data_trajectories/<DATASET_DIR>/**/dataset.hdf5`，把分片合成一个训练文件。

### 数据后处理和翻转

合并分片：

```bash
cd scripts/generate_data
python post_process_generated_dataset.py --data_dir ../../data_trajectories/<DATASET_DIR>
```

输入：

- `../../data_trajectories/<DATASET_DIR>/**/dataset.hdf5`
- 任意一个分片里的 `args.yaml`

输出：

- `../../data_trajectories/<DATASET_DIR>/dataset_merged.hdf5`
- `../../data_trajectories/<DATASET_DIR>/args.yaml`

翻转轨迹：

```bash
python flip_solution_paths.py
```

输入：

- 脚本内 `PATH_TO_DATASETS` 匹配到的 `dataset_merged.hdf5`

输出：

- 同目录生成 `dataset_merged_doubled.hdf5`
- `sol_path` 会追加一份反向轨迹，`task_id` 会为反向样本生成新的 id。

下一步如何使用：

- 训练默认读取 `data_trajectories/<dataset_subdir>/args.yaml` 和 `dataset_file_merged`。
- `train.py` 默认 `dataset_file_merged="dataset_merged_doubled.hdf5"`，所以常规训练应先确保该文件存在。

### 训练：数据集到模型

入口：

```bash
cd scripts/train
python train.py --device cuda:0 --debug False --results_dir /home/eric/MotionPlanningDiffusion/mpd-splines-public/scripts/inference/logs/
```

输入：

- `dataset_subdir`：对应 `data_trajectories/<dataset_subdir>/`
- `dataset_file_merged`：默认 `dataset_merged_doubled.hdf5`
- `data_trajectories/<dataset_subdir>/args.yaml`：提供环境、机器人、planner、任务数量等数据集元信息。

输出：

- 直接运行 `train.py` 时，默认输出到 `logs/<seed>/`。
- 用 `launch_train_*.py` 时，输出到 launcher 生成的 `logs/<launch_name...>/<参数目录>/<seed>/`；如果脚本设置了 `base_dir`，则写到对应 base dir。
- 训练目录中关键文件：
  - `args.yaml`：训练参数，推理阶段会读取。
  - `train_subset_indices.pt`、`val_subset_indices.pt`：训练/验证 split，推理阶段会复用，保证 start-goal 选择和训练 split 对齐。
  - `checkpoints/model_current.pth`
  - `checkpoints/ema_model_current.pth`
  - `checkpoints/*_iter_XXXXXX.pth`
  - `checkpoints/train_losses.npy`、`checkpoints/val_losses.npy`

下一步如何使用：

- 推理 cfg 中的 `model_dir_ddpm_bspline`、`model_dir_cvae_bspline`、`model_dir_ddpm_waypoints` 要指向训练输出目录，也就是包含 `args.yaml` 和 `checkpoints/` 的那一层。
- 如果训练时 `use_ema=True`，推理会加载 `checkpoints/ema_model_current.pth`；否则加载 `checkpoints/model_current.pth`。

示例：

```yaml
model_dir_ddpm_bspline: '/home/eric/MotionPlanningDiffusion/mpd-splines-public/scripts/train/logs/<TRAIN_RUN>/0'
```

公开模型包已经通过 `data_trained_models -> data_public/data_trained_models` 提供预训练模型；使用公开模型时只需要让 `scripts/inference/cfgs/*.yaml` 的 `model_dir_*` 指向对应公开模型目录。

### 推理：模型到候选轨迹和最优轨迹

入口：

```bash
cd scripts/inference
python inference.py \
  --cfg_inference_path ./cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml \
  --sim_backend none \
  --device cuda:0 \
  --results_dir /home/eric/MotionPlanningDiffusion/mpd-splines-public/scripts/inference/logs/
```

输入：

- `cfg_inference_path`：选择环境、模型目录、planner 算法、采样数量、trajectory duration、guide 参数等。
- `model_dir_*`：根据 `planner_alg` 和 `model_selection` 选出最终 `model_dir`。
- `<MODEL_DIR>/args.yaml`：恢复训练时的数据集、模型结构、trajectory 表示。
- `<MODEL_DIR>/checkpoints/ema_model_current.pth` 或 `model_current.pth`：生成轨迹的模型。
- `<MODEL_DIR>/train_subset_indices.pt`、`val_subset_indices.pt`：加载训练时保存的 split。
- `data_trajectories/<dataset_subdir>/<dataset_file_merged>`：推理仍会加载数据集，用于构造 planning task、环境、机器人、start-goal 样本和归一化参数。

输出：

- 默认写到 `logs/<seed>/`，也可以用 `--results_dir <DIR>` 指定根目录。
- `args.yaml`：本次 CLI 展开后的运行参数。
- `args_inference.yaml`：最终推理配置，包含展开后的 `model_dir`。
- `results_single_plan-000.pt`、`results_single_plan-001.pt` ...：每个 start-goal 一个结果文件。

`results_single_plan-XXX.pt` 中常用字段：

- `control_points_iters`：采样/优化过程中每轮控制点。
- `q_trajs_pos_iters`、`q_trajs_vel_iters`、`q_trajs_acc_iters`：每轮转换后的轨迹。
- `q_trajs_pos_valid`：通过 MPD 几何/约束检查的有效轨迹 batch。
- `q_trajs_pos_best`：按 `best_trajectory_selection` 选出的最优轨迹。
- `metrics`：路径长度、平滑度、碰撞/成功等规划指标。
- `sim_statistics`：如果启用了 IsaacLab/IsaacGym，会记录物理验证统计。

如果开启渲染选项，还会额外输出：

- `inference-joint_space-time-opt-iters-XXX.mp4`
- `inference-joint_space-env-opt-iters-XXX.mp4`
- `inference-robot-env-opt-iters-XXX.mp4`
- `inference-robot-env-XXX.mp4`

下一步如何使用：

- 只看 MPD 规划结果时，读取 `results_single_plan-XXX.pt` 中的 `q_trajs_pos_best`、`metrics`。
- 需要物理验证时，`inference.py --sim_backend isaaclab` 会自动把 `q_trajs_pos_valid` 转成 `[H, B, D]` 并传给 IsaacLab evaluator。

### IsaacLab 物理验证：有效轨迹到接触统计

入口有两种。推荐从推理入口自动调用：

```bash
python inference.py \
  --cfg_inference_path ./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml \
  --sim_backend isaaclab \
  --device cuda:0 \
  --isaaclab_root /home/eric/IsaacLab_ori \
  --isaaclab_conda_env env_isaaclab_ori \
  --isaaclab_device cuda:0 \
  --results_dir /home/eric/MotionPlanningDiffusion/mpd-splines-public/scripts/inference/logs/ \
  --isaaclab_headless False
```

输入：

- `results_single_plan.q_trajs_pos_valid`
- Panda joint-space trajectory，传入 evaluator 前转换为 `[H, B, D]`
- IsaacLab 环境 `env_isaaclab_ori` 和 `ISAACLAB_ROOT`

输出：

- `logs/<seed>/isaaclab-trajectories-XXX.pt`：给 IsaacLab 子进程的轨迹 payload，包含 `q_trajs_pos`、`q_pos_starts`、`q_pos_goal`、`robot_name`、`env_name`。
- `logs/<seed>/isaaclab-statistics-XXX.json`：接触统计、无碰撞比例、first collision step、action repeat 等。
- `logs/<seed>/isaaclab-evaluator-XXX.log`：IsaacLab 子进程 stdout/stderr。
- 同一轮的 `results_single_plan-XXX.pt` 中会同步保存 `sim_statistics`。

直接运行 evaluator 时，输入必须是 `torch.save` 文件，至少包含：

```python
{"q_trajs_pos": tensor}  # shape: [H, B, D]
```

下一步如何使用：

- 批量统计成功率时优先读取 `isaaclab-statistics-XXX.json`。
- 对单次规划做复盘时读取 `results_single_plan-XXX.pt`，同时查看其中的 `metrics` 和 `sim_statistics`。

## 本地 GPU 和云端 GPU 提交

### 本地单机 GPU

本地工作站不需要 Slurm 提交，直接在对应目录运行脚本即可。MPD 环境能正常识别显卡时使用 `--device cuda:0`：

```bash
cd /home/eric/MotionPlanningDiffusion/mpd-splines-public
source set_env_variables.sh
conda activate mpd-splines-public-cu128

cd scripts/train
python train.py --device cuda:0

cd ../inference
python inference.py \
  --cfg_inference_path ./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml \
  --sim_backend none \
  --device cuda:0
```

当前 RTX 5060 如果仍使用旧 MPD/PyTorch 环境，训练和推理主进程先用 CPU；需要仿真时只让 IsaacLab 子进程使用 GPU：

```bash
cd scripts/inference
python inference.py \
  --cfg_inference_path ./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml \
  --sim_backend isaaclab \
  --device cuda:0 \
  --isaaclab_root /home/eric/IsaacLab_ori \
  --isaaclab_conda_env env_isaaclab_ori \
  --isaaclab_device cuda:0
```

本地也可以用 `launch_*` 批量脚本。`experiment_launcher.utils.is_local()` 返回 true 时，`launcher.run(LOCAL)` 会在本机运行，不会提交 Slurm。运行前按机器资源检查这些常量：

```python
USE_CUDA = True
N_EXPS_IN_PARALLEL = 1
CONDA_ENV = "mpd-splines-public"
```

仓库里的 launcher 旧默认值多为 `CONDA_ENV = "mpd-splines"`；在这台机器上应改成实际环境 `mpd-splines-public`，或者确保系统里存在同名环境。

### 云端单机 GPU

如果云端机器只有一张或几张 GPU、没有 Slurm，按本地单机方式运行。建议用 `tmux` 或云平台的任务入口保持进程：

```bash
cd /path/to/mpd-splines-public
source set_env_variables.sh
conda activate mpd-splines-public-cu128
export ISAACLAB_ROOT=/path/to/IsaacLab
export ISAACLAB_CONDA_ENV=env_isaaclab_ori

cd scripts/train
python train.py --device cuda:0
```

云端路径通常和本机不同，因此需要同步修改：

- `ISAACLAB_ROOT` 和 `ISAACLAB_CONDA_ENV`
- `data_trajectories`、`data_trained_models` 的真实路径或软链接
- inference cfg 中的 `model_dir_ddpm_bspline`、`model_dir_cvae_bspline`、`model_dir_ddpm_waypoints`
- launcher 脚本中的 `CONDA_ENV`

### Slurm 集群 GPU

有 Slurm 的云端或集群使用 `launch_*` 脚本提交。`is_local()` 返回 false 时，`Launcher` 会按脚本内的资源配置提交作业。

提交前重点检查每个 launcher 顶部的配置：

```python
LOCAL = is_local()
USE_CUDA = True
N_EXPS_IN_PARALLEL = 1
PARTITION = "gpu"
GRES = "gpu:1"
CONSTRAINT = "rtx3090|a5000"
CONDA_ENV = "mpd-splines-public-cu128"
```

`PARTITION`、`GRES`、`CONSTRAINT` 必须和集群实际队列名称一致；如果集群不支持 `CONSTRAINT`，设为 `None`。数据生成脚本默认 `USE_CUDA = False`，会走 CPU 分区；只有确认生成器需要 GPU 时才改成 `USE_CUDA = True`。

常用提交入口：

```bash
cd scripts/train
python launch_train_diffusion_models-v00.py
python launch_train_cvae_models-v00.py

cd ../inference
python launch_inference-experiments.py

cd ../generate_data
python launch_generate_trajectories.py
```

批量推理如果要启用 IsaacLab，在 `scripts/inference/launch_inference-experiments.py` 的 `default_options` 中设置：

```python
sim_backend="isaaclab"
isaaclab_root="/path/to/IsaacLab"
isaaclab_conda_env="env_isaaclab_ori"
isaaclab_device="cuda:0"
isaaclab_headless=True
```

集群上跑 IsaacLab 还需要确认计算节点能访问 IsaacLab 安装目录、Omniverse/S3 资产缓存和输出目录；第一次加载 Panda USD 资产会比较慢。

### 提交前检查

只提交本次文档改动时不要使用 `git add -A`，避免把无关文件带进去：

```bash
git status --short
conda run -n mpd-splines-public pre-commit run --files NEW_PROJECT_RUN_GUIDE.md
git add NEW_PROJECT_RUN_GUIDE.md
git commit -m "docs: add new project run workflow"
git push
```

## 推理

默认不做物理仿真验证：

```bash
cd scripts/inference
python inference.py \
  --cfg_inference_path ./cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml \
  --sim_backend none \
  --device cpu
```

使用 IsaacLab 做 Panda 轨迹物理验证：

```bash
cd scripts/inference
python inference.py \
  --cfg_inference_path ./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml \
  --sim_backend isaaclab \
  --device cuda:0 \
  --isaaclab_root /home/eric/IsaacLab_ori \
  --isaaclab_conda_env env_isaaclab_ori \
  --isaaclab_device cuda:0 \
  --isaaclab_timeout_s 900
```

兼容旧 IsaacGym 后端，只在旧 IsaacGym 可用环境中使用：

```bash
cd scripts/inference
python inference.py \
  --cfg_inference_path ./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml \
  --sim_backend isaacgym
```

推理输出默认写到 `logs/`。IsaacLab 后端会额外写：

- `isaaclab-trajectories-XXX.pt`
- `isaaclab-statistics-XXX.json`
- `isaaclab-evaluator-XXX.log`
- `results_single_plan-XXX.pt` 中的 `sim_statistics`

## 直接运行 IsaacLab Evaluator

输入必须是 `torch.save` 文件，包含 `q_trajs_pos`，shape 为 `[H, B, D]`。Panda arm 轨迹通常是 `D=7`。

```bash
conda run --no-capture-output -n env_isaaclab_ori \
  /home/eric/IsaacLab_ori/isaaclab.sh -p scripts/isaaclab/evaluate_mpd_trajectories.py \
  --input logs/trajectories.pt \
  --output logs/isaaclab_statistics.json \
  --headless \
  --device cuda:0
```

第一次运行可能会从 Omniverse S3 加载 IsaacLab Panda USD 资产，耗时较长。后续缓存完成后会快一些。

## 批量推理实验

批量实验入口：

```bash
cd scripts/inference
python launch_inference-experiments.py
```

默认配置中 `sim_backend="none"`。如果要批量启用 IsaacLab，在 `scripts/inference/launch_inference-experiments.py` 的 `default_options` 中设置：

```python
sim_backend="isaaclab"
isaaclab_root="/home/eric/IsaacLab_ori"
isaaclab_conda_env="env_isaaclab_ori"
isaaclab_device="cuda:0"
```

## 训练

单模型训练：

```bash
cd scripts/train
python train.py --device cuda:0
```

批量训练：

```bash
cd scripts/train
python launch_train_diffusion_models-v00.py
python launch_train_cvae_models-v00.py
python launch_train_diffusion_models-EnvEmpty2D-v00.py
```

训练入口已不再顶层依赖 IsaacGym。

## 数据生成和后处理

生成单个数据集：

```bash
cd scripts/generate_data
python generate_trajectories.py
```

批量生成：

```bash
cd scripts/generate_data
python launch_generate_trajectories.py
```

合并生成的数据：

```bash
cd scripts/generate_data
python post_process_generated_dataset.py --data_dir ../../data_trajectories/<DATASET_DIR>
```

翻转轨迹数据集：

```bash
cd scripts/generate_data
python flip_solution_paths.py
```

`flip_solution_paths.py` 仍需要在脚本内确认目标数据路径。

## 数据可视化

默认只做 MPD/PyBullet 可视化，不启动物理仿真后端：

```bash
cd scripts/generate_data
python visualize_trajectories.py ../../data_trajectories/<DATASET_DIR> --sim-backend none
```

使用 IsaacLab 验证可视化轨迹，当前主要支持 Panda 轨迹：

```bash
cd scripts/generate_data
python visualize_trajectories.py ../../data_trajectories/<PANDA_DATASET_DIR> \
  --sim-backend isaaclab \
  --isaaclab-root /home/eric/IsaacLab_ori \
  --isaaclab-conda-env env_isaaclab_ori \
  --isaaclab-device cuda:0
```

旧 IsaacGym 可视化只在旧环境中显式启用：

```bash
cd scripts/generate_data
python visualize_trajectories.py ../../data_trajectories/<PANDA_DATASET_DIR> --sim-backend isaacgym
```

## Panda Replay 示例

默认使用 IsaacLab evaluator：

```bash
python mpd/motion_planning_baselines/examples/panda_isaac_replay.py \
  --sim_backend isaaclab \
  --output_dir logs/panda_replay \
  --isaaclab_root /home/eric/IsaacLab_ori \
  --isaaclab_conda_env env_isaaclab_ori \
  --isaaclab_device cuda:0
```

只构建轨迹、不跑仿真：

```bash
python mpd/motion_planning_baselines/examples/panda_isaac_replay.py --sim_backend none
```

旧 IsaacGym replay：

```bash
python mpd/motion_planning_baselines/examples/panda_isaac_replay.py --sim_backend isaacgym
```

IsaacLab 视觉 replay 和视频导出：

```bash
/home/eric/IsaacLab_ori/isaaclab.sh -p scripts/isaaclab/replay_mpd_trajectory.py \
  --input scripts/inference/logs/<RUN_ID>/isaaclab-trajectories-000.pt \
  --trajectory_index 0 \
  --output_video scripts/inference/logs/<RUN_ID>/isaaclab-replay-000.mp4 \
  --screenshot_path scripts/inference/logs/<RUN_ID>/isaaclab-replay-000.png \
  --output_json scripts/inference/logs/<RUN_ID>/isaaclab-replay-000.json \
  --device cuda:0 \
  --headless \
  --enable_cameras
```

`inference.py` 的 IsaacLab batch evaluator 只负责碰撞统计，不再提供 `--render_isaaclab_movie`。需要视觉确认或 mp4 时，先用 inference 生成 `isaaclab-trajectories-XXX.pt`，再运行上面的 replay 脚本。

replay 脚本默认在视频、截图和 JSON 全部写完后快速退出进程，避免 Isaac Sim 5.1 在 `SimulationApp.close()` 阶段卡住。调试 Isaac Sim 资源清理时可以额外加 `--graceful_shutdown`，但正常导出视频不需要。

## Baseline 示例

示例脚本位于：

```bash
mpd/motion_planning_baselines/examples/
```

常用入口：

```bash
python mpd/motion_planning_baselines/examples/pointmass_2d_CHOMP.py
python mpd/motion_planning_baselines/examples/panda_spheres_CHOMP.py
python mpd/motion_planning_baselines/examples/planar_2_link_RRT.py
```

这些入口已清理未使用的顶层 IsaacGym import。旧 IsaacGym backend 文件仍保留在：

```text
mpd/torch_robotics/torch_robotics/isaac_gym_envs/motion_planning_envs.py
```

只有显式选择 IsaacGym backend 时才应进入该路径。

## 当前限制

- IsaacLab evaluator 第一版只稳定支持 Panda joint-space trajectory replay。
- MPD 障碍物到 IsaacLab scene 目前支持 sphere 和 box primitive；更复杂 mesh/非 primitive 障碍物仍未迁移。
- IsaacLab evaluator 负责批量 headless 统计；IsaacLab 视频和截图由 `scripts/isaaclab/replay_mpd_trajectory.py` 负责。
- 旧 `mpd-splines-public` 环境直接暴露 RTX 5060 时，可能在 Theseus/PyTorch CUDA import 阶段报 `sm_120` 不兼容；需要 CPU 路径验证时使用 `CUDA_VISIBLE_DEVICES=''`。
