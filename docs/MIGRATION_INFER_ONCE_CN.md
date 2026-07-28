# MPD 仓库迁移、Conda 环境与 `infer_once.py` 依赖说明

本文针对当前仓库的实际代码状态，而不是只照抄根目录旧版 `README.md`。目标是在另一台 Ubuntu/NVIDIA 电脑上克隆仓库，并运行：

```text
scripts/runtime/infer_once.py
```

检查日期：2026-07-27。

## 1. 结论先行

1. `infer_once.py` 本身没有写死 `/home/eric/...`。它通过 `__file__` 自动计算仓库根目录，因此仓库可以克隆到任意位置。
2. 单次 Warehouse 推理真正必须修改的是
   `scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-runtime.yaml`
   中选中模型的 `model_dir_ddpm_bspline`。当前值指向旧目录
   `${HOME}/Projects/MotionPlanningDiffusion/mpd-splines/...`，并不指向本仓库。
3. `data_public/`、`data_trained_models`、`data_trajectories` 和 `deps/isaacgym` 都不会随 Git 克隆。模型和数据必须单独复制/下载，再重建两个相对符号链接。
4. `address_token.txt` 指定当前实际 MPD 环境名为 `mpd-splines-public-cu128`。实机检查得到：

   | 项目 | 当前值 |
   |---|---|
   | Python | 3.10.20 |
   | PyTorch | 2.7.1+cu128 |
   | torchvision | 0.22.1+cu128 |
   | torchaudio | 2.7.1+cu128 |
   | NumPy | 1.26.4 |
   | SciPy | 1.15.2 |
   | CUDA runtime used by PyTorch | 12.8 |

5. 根目录旧 `environment.yml` 是另一套老环境：Python 3.8.19、Torch 2.0.0+cu118、Isaac Gym Preview 4。它与当前已验证的 `mpd-splines-public-cu128` 不一致。
6. 只运行当前未重构的 `infer_once.py`：

   - **需要 PyBullet、OMPL 和仓库内的 `pb_ompl`**，但不会启动 PyBullet GUI 或用 PyBullet 做本次碰撞检测；
   - **不需要 Isaac Lab**；
   - **不需要 Isaac Gym**；
   - 不需要 ROS 2。

## 2. 克隆与非 Git 资产

### 2.1 克隆仓库和子模块

```bash
git clone --recurse-submodules <此仓库的 Git URL> mpd-splines-public
cd mpd-splines-public
git submodule update --init --recursive
```

当前仓库有三个子模块：

```text
deps/experiment_launcher
deps/pybullet_ompl
deps/theseus
```

如果子模块服务器需要权限，应单独配置 SSH key 或凭据管理器。不要把访问令牌写入 clone URL、脚本、README 或提交记录。

### 2.2 复制/下载数据和模型

根目录旧 README 给出的公共数据包下载地址是：

```text
https://drive.google.com/file/d/1KG5ejn0g0KkDuUK6tPUqfmRYCNoKzK4K/view?usp=drive_link
```

解压后的目标结构应类似：

```text
mpd-splines-public/
  data_public/
    data_trained_models/
    data_trajectories/
```

然后在仓库根目录重建相对链接：

```bash
ln -s data_public/data_trajectories data_trajectories
ln -s data_public/data_trained_models data_trained_models
```

当前电脑上的这两个链接本来就是相对链接，因此链接文字本身可迁移；问题是其目标 `data_public/` 被 `.gitignore` 忽略，不会被 clone。

检查：

```bash
test -d data_trained_models
test -d data_trajectories
find -L data_trained_models -type f \
  \( -name model_current.pth -o -name ema_model_current.pth \) | head
```

## 3. `infer_once.py` 必须修改的路径

### 3.1 模型目录

文件：

```text
scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-runtime.yaml
```

当前固定配置为：

```yaml
model_selection: bspline
planner_alg: mpd
```

所以当前实际读取的是 `model_dir_ddpm_bspline`。该路径必须指向一个包含以下文件的目录：

```text
<model_dir>/
  args.yaml
  checkpoints/
    ema_model_current.pth   # args.yaml 中 use_ema=true 时
    # 或 model_current.pth  # use_ema=false 时
```

推荐不要再依赖特定用户的 `${HOME}/Projects/...` 布局。可在 shell 中定义仓库根目录：

```bash
export MPD_REPO_ROOT="$(pwd -P)"
```

然后把 YAML 第 4 行的开头改成：

```yaml
model_dir_ddpm_bspline: '${MPD_REPO_ROOT}/data_trained_models/launch_train_diffusion_models-v04_2024-09-18_08-12-54/.../0'
```

其中 `...` 必须保留为模型实际完整子目录，不能原样写三个点。最稳妥的方法是在新电脑上找到目标 checkpoint，再取其上两级目录：

```bash
find -L "$MPD_REPO_ROOT/data_trained_models" \
  -path '*EnvWarehouse-RobotPanda*ParametricTrajectoryBspline/0/checkpoints/ema_model_current.pth'
```

`model_dir_cvae_bspline` 和 `model_dir_ddpm_waypoints` 在当前 runtime 参数下不会被选中；只有以后切换算法/轨迹表示时才需修改。

### 3.2 请求、输出和配置参数

以下路径不在代码中写死，而是调用方传入：

```bash
python scripts/runtime/infer_once.py \
  --request /absolute/path/to/request.json \
  --output-dir /absolute/path/to/output \
  --config "$MPD_REPO_ROOT/scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-runtime.yaml" \
  --device cuda:0
```

`--request` 和 `--output-dir` 可以位于仓库外，只需当前用户有读写权限。

## 4. 其他写死路径：按使用场景修改

这些项不影响上述纯 `infer_once.py` 路径，但在运行相应功能前要处理。

| 文件 | 当前写死内容 | 何时需要修改 |
|---|---|---|
| `setup.sh:5` | `/home/eric/anaconda3/bin/conda` | 使用旧安装脚本时 |
| `setup.sh:47` | `/home/eric/anaconda3/envs/${CONDA_ENV_NAME}/bin/python` | 从源码编译 OMPL 时；应改为 `$CONDA_PREFIX/bin/python` |
| `setup.sh:7` | 环境名 `mpd-splines-public` | 当前实际环境名是 `mpd-splines-public-cu128` |
| `set_env_variables.sh:3-4` | `/home/eric/IsaacLab_ori`、`env_isaaclab_ori` | 仅 Isaac Lab 评估/回放 |
| `scripts/inference/inference.py:548-549` | Isaac Lab 根目录和环境名的默认值 | 从普通 inference 流程调用 Isaac Lab 评估时 |
| `scripts/isaaclab/subprocess_utils.py` | 同上 | Isaac Lab 子进程评估/回放 |
| `scripts/generate_data/visualize_trajectories.py` | 同上 | 数据可视化调用 Isaac Lab 时 |
| `scripts/inference/launch_inference-experiments.py` | 环境名和 Isaac Lab 路径 | 批量 inference/评估时 |
| `scripts/train/launch_train_*.py` | `mpd-splines-public-cu128` | 新环境使用其他名字时 |
| `scripts/generate_data/launch_generate_trajectories.py` | `mpd-splines-public-cu128` | 新环境使用其他名字时 |
| `scripts/generate_data/flip_solution_paths.py:54` | `/home/carvalho/Projects/...` | 使用该数据后处理脚本时 |
| `scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-config_file_v01_00.yaml:20` | `/home/eric/.../start-goal-states.yaml` | 使用此旧配置且 `start_goal_source=states_file` 时 |
| `mpd/torch_robotics/.../isaac_gym_envs/motion_planning_envs.py:238` | `${HOME}/Projects/.../deps/isaacgym/assets` | 使用 Isaac Gym 环境时 |
| `scripts/runtime/README.md` | `/home/eric/...` 示例命令 | 只影响文档示例，不影响程序 |
| 根目录 `README.md` | `~/Projects/...`、`~/Downloads/...` | 只影响旧版安装示例 |

另外，多份 `scripts/inference/cfgs/*.yaml` 使用
`${HOME}/Projects/MotionPlanningDiffusion/mpd-splines/data_trained_models/...`。
只有实际选择的配置需要改；如果希望整个仓库可迁移，建议统一替换为
`${MPD_REPO_ROOT}/data_trained_models/...`，并在运行前导出 `MPD_REPO_ROOT`。

## 5. Conda 环境安装

### 5.1 不建议直接运行旧 `setup.sh`

旧脚本会：

- 使用固定的 `/home/eric/anaconda3`；
- 创建旧 `environment.yml` 中的 Python 3.8/CUDA 11.8 环境；
- 强制要求并安装 Isaac Gym；
- 删除同名已有环境；
- 在固定 Conda 路径下编译 OMPL；
- 最后强制删除 `ncurses`。

这与当前 CUDA 12.8 环境不一致，也包含对单次推理不需要的组件。因此迁移
`infer_once.py` 时不要直接执行它。

### 5.2 推荐：从当前工作电脑导出，再清理本地 editable 包

在当前工作电脑执行：

```bash
conda env export -n mpd-splines-public-cu128 --no-builds \
  | sed '/^prefix:/d' \
  > environment-cu128-export.yml
```

编辑 `environment-cu128-export.yml`：

1. 保留 `name: mpd-splines-public-cu128`，或改成新名字；
2. 在 `pip:` 列表中加入：

   ```yaml
   - --extra-index-url https://download.pytorch.org/whl/cu128
   ```

3. 从 `pip:` 列表删除以下六个本地 editable 项，避免 pip 从 PyPI 安装同名但错误的项目：

   ```text
   experiment-launcher
   mp-baselines
   mpd
   pb-ompl
   torch-robotics
   torchkin
   ```

把清理后的环境文件带到新电脑，在仓库根目录执行：

```bash
conda env create -f environment-cu128-export.yml
conda activate mpd-splines-public-cu128

python -m pip install -e deps/experiment_launcher
python -m pip install -e deps/theseus/torchkin
python -m pip install -e mpd/torch_robotics
python -m pip install -e mpd/motion_planning_baselines
python -m pip install -e deps/pybullet_ompl
python -m pip install -e .
```

editable 安装会记录仓库绝对路径，所以即使直接复制过旧 Conda 环境，换电脑或移动仓库后也必须重新执行上面的 `pip install -e`。

### 5.3 没有导出文件时的最小推理环境基线

下面是按当前实机版本和实际导入链整理的基线。它比旧
`environment.yml` 小，但仍建议优先使用上一节的完整导出。

```bash
conda create -n mpd-splines-public-cu128 -c conda-forge -y \
  python=3.10.20 pip numpy=1.26.4 scipy=1.15.2 \
  pinocchio=2.7.0

conda activate mpd-splines-public-cu128

python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  dotmap==1.3.30 einops==0.8.2 h5py==3.16.0 \
  matplotlib==3.10.9 scikit-learn==1.7.2 tqdm==4.68.3 \
  PyYAML==6.0.3 GitPython==3.1.50 \
  torchlie==0.1.0 cholespy==2.2.0 \
  urdf-parser-py==0.0.4 trimesh==4.12.2 \
  sobol-seq==0.2.0 ghalton==0.6.1 \
  pybullet==3.2.7 ompl==2.0.1

python -m pip install -e deps/experiment_launcher
python -m pip install -e deps/theseus/torchkin
python -m pip install -e mpd/torch_robotics
python -m pip install -e mpd/motion_planning_baselines
python -m pip install -e deps/pybullet_ompl
python -m pip install -e .
```

注意：

- PyTorch CUDA 12.8 wheel 仍要求新电脑的 NVIDIA 驱动支持相应 CUDA runtime；
- 无 NVIDIA GPU 时可以改装 CPU 版 PyTorch，并用 `--device cpu` 尝试，但当前主要验证路径是 CUDA；
- 如果 `ompl==2.0.1` 在目标平台没有可用 wheel，需要按
  `deps/pybullet_ompl/README.md` 从源码构建；构建时 Python 路径使用
  `"$CONDA_PREFIX/bin/python"`；
- 不要再执行旧脚本中的无版本 `pip install numpy --upgrade`，否则可能升级到 NumPy 2.x，破坏当前已验证的 NumPy 1.26.4 组合。

### 5.4 环境变量

只运行 `infer_once.py` 时：

```bash
conda activate mpd-splines-public-cu128
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export MPD_REPO_ROOT="$(pwd -P)"
```

不需要设置 `ISAACLAB_ROOT` 或 `ISAACLAB_CONDA_ENV`。

## 6. `infer_once.py` 实际使用的库

### 6.1 直接导入

标准库包括：

```text
argparse, hashlib, json, math, os, pathlib, subprocess,
sys, tempfile, time, typing
```

直接第三方/项目包包括：

```text
numpy
torch
dotmap
mpd
torch_robotics
```

`scripts.isaaclab.scene_payload` 虽然路径名包含 `isaaclab`，但该文件只是把 MPD
障碍物序列化为字典；它只导入 NumPy 和 `torch_robotics`，不导入 Isaac Lab。

### 6.2 运行时导入链中的主要第三方库

当前 Warehouse DDPM/B-spline 路径还会加载：

```text
scipy
scikit-learn
matplotlib
einops
h5py
PyYAML
tqdm
torchlie
torchkin
pybullet
ompl
pb_ompl（仓库子模块）
```

另有仓库内 editable 包：

```text
mpd
torch_robotics
mp_baselines
experiment_launcher
torchkin
pb_ompl
```

其中部分库是由于模块顶层导入而被要求，不代表本次推理会调用其所有功能。

### 6.3 PyBullet 到底要不要

**当前代码要安装。**

导入链为：

```text
infer_once.py
  -> mpd.inference.inference / mpd.utils.loaders
  -> mpd.datasets.trajectories_dataset_bspline
  -> pb_ompl.pb_ompl
  -> pb_ompl.utils
  -> pybullet
```

此外 `mpd.inference.inference` 也直接从 `pb_ompl.pb_ompl` 导入函数。因此卸载
PyBullet 后，程序会在加载 planner 之前就报 `ModuleNotFoundError`/`ImportError`。

但本次单次推理的碰撞检查实际使用 MPD/`torch_robotics` 的可微几何模型，不会连接
PyBullet server，也不会打开 GUI。若以后把 `pb_ompl` 相关导入改成延迟导入，并将
纯推理所需 B-spline 工具拆开，才可能从最小环境中移除 PyBullet/OMPL。

### 6.4 Isaac Lab 和 Isaac Gym 到底要不要

**只运行 `infer_once.py` 时都不需要。**

已在当前 Conda 环境中用导入拦截器强制禁止 `isaaclab`、`isaacgym` 和 `omni`
模块，然后导入 `infer_once.py` 的完整 planner/loader/scene-payload 入口；导入成功，
且没有这些模块进入 `sys.modules`。

它们只用于其他路径：

- Isaac Lab：`scripts/isaaclab/evaluate_mpd_trajectories.py`、
  `replay_mpd_trajectory.py` 等评估/回放；
- Isaac Gym：`torch_robotics/isaac_gym_envs` 和旧版可视化/仿真路径。

因此无需复制 `address_token.txt` 中记录的独立 Isaac Lab 目录和环境，也无需下载旧
README 所说的 Isaac Gym Preview 4，除非确实要运行上述仿真功能。

## 7. 安装后的检查

### 7.1 版本和 CUDA

```bash
python -c "import sys, torch, numpy; \
print(sys.version); \
print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
print(numpy.__version__)"
```

预期核心版本：

```text
Python 3.10.20
torch 2.7.1+cu128
CUDA 12.8
NumPy 1.26.4
```

### 7.2 核心导入

```bash
python -c "import pybullet, ompl, pb_ompl, torchkin, torchlie; \
from mpd.inference.inference import GenerativeOptimizationPlanner; \
from mpd.utils.loaders import get_planning_task_and_dataset; \
print('imports ok')"
```

### 7.3 模型路径

```bash
python - <<'PY'
import os
from pathlib import Path
import yaml

cfg = Path(os.environ["MPD_REPO_ROOT"]) / \
    "scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-runtime.yaml"
data = yaml.safe_load(cfg.read_text())
model_dir = Path(os.path.expandvars(os.path.expanduser(data["model_dir_ddpm_bspline"]))).resolve()
print("model_dir:", model_dir)
print("args.yaml:", (model_dir / "args.yaml").is_file())
print("checkpoints:", list((model_dir / "checkpoints").glob("*model_current.pth")))
PY
```

三个检查都通过后，再按照 `scripts/runtime/README.md` 准备请求 JSON 并运行单次推理。

## 8. 凭据安全

`address_token.txt` 含明文 GitHub personal access token。该 token 与 Conda 安装和
`infer_once.py` 无关，不应复制到新电脑的仓库、不应提交 Git，也不应写进任何 Markdown。

当前检查显示它没有被 Git 跟踪；应继续保持未跟踪状态，并建议：

1. 立即在 GitHub 撤销并重新生成其中现有 token；
2. 使用 SSH key、Git credential manager 或环境变量管理新凭据；
3. 把 `address_token.txt` 加入本地全局 ignore，或在确认团队约定后加入仓库
   `.gitignore`；
4. 如果该 token 曾出现在远端提交、日志、聊天或终端共享输出中，按已泄露处理。

## 9. 最短迁移清单

- [ ] `git clone --recurse-submodules`
- [ ] 单独恢复 `data_public/`
- [ ] 重建 `data_trained_models`、`data_trajectories` 相对链接
- [ ] 建立 Python 3.10 / Torch 2.7.1+cu128 环境
- [ ] 重新 editable 安装六个仓库内包
- [ ] 安装 PyBullet、OMPL；不为纯单次推理安装 Isaac Lab/Isaac Gym
- [ ] 设置 `MPD_REPO_ROOT`
- [ ] 修改 runtime YAML 的 `model_dir_ddpm_bspline`
- [ ] 运行版本、导入和模型路径检查
- [ ] 用 `--request`、`--output-dir` 和 `--device cuda:0` 执行 `infer_once.py`
