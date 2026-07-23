# MPD Warehouse 单次推理接口

本文说明 `scripts/runtime/infer_once.py` 的用途、输入输出位置、数据格式、调用方式和内部运行流程。

## 1. 接口定位

`infer_once.py` 是 MPD 仓库内的无 ROS 单次推理入口。它负责：

- 读取一个不可变的 JSON 规划请求；
- 在独立 MPD Python/Conda 环境中加载 Warehouse Panda 模型；
- 使用当前 Panda 模型从 `q_pos_goal` 计算 `ee_pose_goal`；
- 调用一次 `GenerativeOptimizationPlanner.plan_trajectory()`；
- 验证最佳轨迹的形状、有限值、时间轴、起点连续性、关节限位和 MPD 场景碰撞；
- 导出不依赖 pickle、PyTorch 类或 ROS 消息的 `NPZ + JSON` 中性结果。

它不会：

- 发布 ROS 2 topic、service 或 action；
- 向机器人或控制器发送命令；
- 启动 IsaacLab、Isaac Gym、PyBullet GUI、RViz 或轨迹回放；
- 自动生成 `results_single_plan-000.pt`、`inference-report-000.txt`、MP4 或图片；
- 在推理完成后重新读取真实机器人状态或真实场景。

因此，`infer_once.py` 的输出只是“可供上层验收的规划结果”，不是可直接执行的真机命令。

## 2. 相关文件位置

仓库根目录假设为：

```text
/home/eric/MotionPlanningDiffusion/mpd-splines-public
```

运行入口：

```text
scripts/runtime/infer_once.py
```

默认推理配置：

```text
scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-runtime.yaml
```

输入文件位置由 `--request` 指定，输出目录由 `--output-dir` 指定。输入和输出可以放在仓库外，但应使用绝对路径，且运行 MPD 的用户必须拥有相应读写权限。

## 3. 调用方式

使用当前验证过的 MPD 环境：

```bash
conda run --no-capture-output \
  -n mpd-splines-public-cu128 \
  python /home/eric/MotionPlanningDiffusion/mpd-splines-public/scripts/runtime/infer_once.py \
  --request /absolute/path/to/request.json \
  --output-dir /absolute/path/to/request_output \
  --device cuda:0
```

显式指定配置：

```bash
conda run --no-capture-output \
  -n mpd-splines-public-cu128 \
  python /home/eric/MotionPlanningDiffusion/mpd-splines-public/scripts/runtime/infer_once.py \
  --request /absolute/path/to/request.json \
  --output-dir /absolute/path/to/request_output \
  --config /home/eric/MotionPlanningDiffusion/mpd-splines-public/scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-runtime.yaml \
  --device cuda:0
```

脚本使用自身位置定位仓库根目录，因此也可以从其他工作目录通过绝对脚本路径调用。

命令行参数：

| 参数 | 必需 | 默认值 | 含义 |
|---|---|---|---|
| `--request` | 是 | 无 | 输入 JSON 文件路径 |
| `--output-dir` | 是 | 无 | 本次请求的输出目录 |
| `--config` | 否 | Warehouse runtime YAML | 推理配置路径 |
| `--device` | 否 | `cuda:0` | PyTorch 设备 |

查看帮助：

```bash
python scripts/runtime/infer_once.py --help
```

## 4. 输入位置和内容

### 4.1 输入位置

输入是 `--request` 指向的单个 UTF-8 JSON 文件。例如：

```text
/tmp/mpd_requests/550e8400-e29b-41d4-a716-446655440000/request.json
```

脚本不会修改或复制该输入文件，但会计算原始文件内容的 SHA-256 并写入 `result.json`。

### 4.2 最小请求示例

```json
{
  "schema_version": 1,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "joint_names": [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7"
  ],
  "q_pos_start": [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0],
  "q_pos_goal": [0.2, -0.3, 0.1, -1.8, 0.2, 1.6, 0.1],
  "scene_id": "EnvWarehouseExtraObjectsV00",
  "seed": 12345
}
```

这里不传 `ee_pose_goal`。脚本加载 MPD Panda 后，通过同一套正运动学从 `q_pos_goal` 计算末端目标，避免上层运行时与 MPD 使用不同 FK 实现。

### 4.3 完整请求示例

```json
{
  "schema_version": 1,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "robot_model": "franka_panda",
  "planning_frame": "panda_link0",
  "joint_names": [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7"
  ],
  "q_pos_start": [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0],
  "q_pos_goal": [0.2, -0.3, 0.1, -1.8, 0.2, 1.6, 0.1],
  "q_vel_start": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "q_vel_goal": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "q_acc_start": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "q_acc_goal": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "joint_state_stamp": "ROS timestamp or adapter-defined timestamp",
  "scene_id": "EnvWarehouseExtraObjectsV00",
  "scene_hash": "adapter-scene-snapshot-sha256",
  "seed": 12345
}
```

### 4.4 输入字段

| 字段 | 必需 | 格式/单位 | 规则 |
|---|---|---|---|
| `schema_version` | 是 | 整数 | 当前必须为 `1` |
| `request_id` | 是 | 非空字符串 | 原样写回结果，用于请求结果配对 |
| `joint_names` | 是 | 7 个字符串 | 顺序必须严格为 `panda_joint1..7` |
| `q_pos_start` | 是 | `[7]`，rad | 必须有限、在 Panda 位置限位内且在当前 MPD 场景中无碰撞 |
| `q_pos_goal` | 是 | `[7]`，rad | 规则同 `q_pos_start`，同时用于计算 `ee_pose_goal` |
| `scene_id` | 是 | 非空字符串 | 当前必须为 `EnvWarehouseExtraObjectsV00` |
| `seed` | 是 | 整数 | 范围为 `[0, 2**32)` |
| `q_vel_start` | 否 | `[7]`，rad/s | 默认全零，若传入则检查 Panda 速度限位 |
| `q_vel_goal` | 否 | `[7]`，rad/s | 默认全零，若传入则检查 Panda 速度限位 |
| `q_acc_start` | 否 | `[7]`，rad/s² | 默认全零，若传入则检查 Panda 加速度限位 |
| `q_acc_goal` | 否 | `[7]`，rad/s² | 默认全零，若传入则检查 Panda 加速度限位 |
| `robot_model` | 否 | 字符串 | 缺省为 `franka_panda`；传入时也必须为此值 |
| `planning_frame` | 否 | 字符串 | 缺省为 `panda_link0`；传入时也必须为此值 |
| `joint_state_stamp` | 否 | 任意 JSON 值 | 仅记录和回传；脚本本身不判断状态是否过期 |
| `scene_hash` | 否 | 非空字符串 | 上层场景快照标识，仅记录和回传 |

所有关节向量必须正好是 7 个数，不能包含 `NaN` 或 `Inf`。

## 5. 输出位置和内容

### 5.1 输出目录

输出写入 `--output-dir`：

```text
/absolute/path/to/request_output/
  result.json
  trajectory.npz
```

成功时只有上述两个接口文件。脚本不会在该目录创建 `.pt`、文本 inference report、视频或仿真文件。

启动一次请求时，脚本会先删除输出目录中旧的 `trajectory.npz`。发生任何失败时也会再次删除它，防止上层误读上一次请求遗留的轨迹。

文件先写入同目录临时文件，刷新到磁盘后再使用原子重命名替换正式文件。成功路径先完成 `trajectory.npz`，最后写 `result.json`；上层应把成功状态的 `result.json` 视为本次输出完成标志。

### 5.2 `trajectory.npz`

`trajectory.npz` 只在 `result.json.status == "success"` 时存在。

| 数组 | 类型 | 形状 | 单位/含义 |
|---|---|---|---|
| `positions` | `float64` | `[128, 7]` | Panda 关节位置，rad |
| `velocities` | `float64` | `[128, 7]` | Panda 关节速度，rad/s |
| `accelerations` | `float64` | `[128, 7]` | Panda 关节加速度，rad/s² |
| `time_from_start` | `float64` | `[128]` | 从 `0.0` 到 `10.0` 秒，严格递增 |
| `joint_names` | Unicode 字符串 | `[7]` | 与输入完全相同的固定 Panda 关节顺序 |

读取示例：

```python
import numpy as np

trajectory = np.load("/absolute/path/to/request_output/trajectory.npz")
positions = trajectory["positions"]
velocities = trajectory["velocities"]
accelerations = trajectory["accelerations"]
time_from_start = trajectory["time_from_start"]
joint_names = trajectory["joint_names"].tolist()
```

不要让 ROS 侧加载 MPD 的 `.pt`/pickle 文件；ROS adapter 只需读取这个 NPZ 和对应 JSON。

### 5.3 成功的 `result.json`

成功结果的主要结构为：

```text
schema_version
status = "success"
request_id
joint_names
trajectory_file = "trajectory.npz"
request
scene
model
config
mpd_source
trajectory
candidates
best_trajectory_diagnostics
timing
created_unix_time
```

各部分含义：

| 部分 | 内容 |
|---|---|
| `request` | 请求 schema、原始 JSON SHA-256、seed、机器人/坐标系和状态时间戳 |
| `scene` | 实际加载的场景、上层传入的 scene hash、MPD 场景几何 SHA-256 |
| `model` | 模型目录、训练参数路径及哈希、checkpoint 路径及完整 SHA-256 |
| `config` | 本次 runtime YAML 的绝对路径和 SHA-256 |
| `mpd_source` | MPD Git commit 和工作区是否 dirty |
| `trajectory` | 轨迹点数、持续时间、单位、起点误差、速度/加速度限位利用率 |
| `candidates` | 生成、有效、碰撞和各类关节限位违规候选数量 |
| `best_trajectory_diagnostics` | 选优方法、各指标归一化/权重、末端误差、路径长度和平滑度 |
| `timing` | MPD planner 内部的总推理、生成器和 cost guide 时间 |

`timing.inference_total_sec` 只表示 `plan_trajectory()` 内部推理计时，不包含数据集/模型初始化、checkpoint SHA-256 计算和文件写入时间。若上层需要完整请求延迟，应在启动子进程前后自行计时。

候选统计中的碰撞或限位违规数可能大于零，因为 MPD 会生成一批候选再过滤。可执行性关注的是：

- `status == "success"`；
- `candidates.valid > 0`；
- 最佳轨迹已经通过脚本的二次限位和碰撞检查。

### 5.4 失败的 `result.json`

失败时不生成 `trajectory.npz`，只写：

```json
{
  "schema_version": 1,
  "status": "invalid_request",
  "request_id": "如果能够读取则回传",
  "trajectory_file": null,
  "request_sha256": "如果能够读取则记录",
  "error": {
    "type": "RequestValidationError",
    "message": "具体失败原因"
  },
  "created_unix_time": 0.0
}
```

状态和进程退出码：

| `status` | 退出码 | 含义 |
|---|---:|---|
| `success` | `0` | 推理和输出验收成功 |
| `inference_error` | `1` | 未分类的加载、推理或系统异常 |
| `invalid_request` | `2` | JSON、字段、关节、场景或输入状态不符合契约 |
| `invalid_configuration` | `2` | runtime YAML、模型路径、设备或固定配置不符合契约 |
| `no_valid_trajectory` | `3` | MPD 没有产生有效轨迹 |
| `invalid_result` | `4` | 最佳轨迹存在，但输出形状、时间、有限值、起点、限位或碰撞复核失败 |

上层必须同时检查子进程退出码和 `result.json.status`，并确认 `request_id` 与当前请求一致。

## 6. 程序内部运行流程

```text
CLI 参数
  │
  ├─ 解析 --request / --output-dir / --config / --device
  ├─ 创建输出目录并删除旧 trajectory.npz
  │
  ▼
读取 request.json 原始字节
  ├─ JSON 解码
  └─ 计算 request SHA-256
  │
  ▼
请求静态校验
  ├─ schema_version / request_id / seed
  ├─ Panda joint_names 精确顺序
  ├─ q_pos / q_vel / q_acc 形状和有限值
  └─ robot_model / planning_frame / scene_id
  │
  ▼
加载 runtime YAML
  ├─ 固定 states_file、BSpline、MPD、DDIM
  ├─ 固定 EnvWarehouseExtraObjectsV00
  └─ 固定 100 候选、128 点、10 秒
  │
  ▼
加载训练 args、数据归一化信息、Warehouse、RobotPanda
  └─ 确认 checkpoint 路径存在，但此时还不加载模型权重
  │
  ▼
输入动态校验
  ├─ 起点/终点关节位置限位
  ├─ 起点/终点 MPD 碰撞检测
  └─ 可选边界速度/加速度限位
  │
  ▼
RobotPanda FK(q_pos_goal) → ee_pose_goal
  │
  ▼
构造 GenerativeOptimizationPlanner 并加载 checkpoint
  │
  ▼
plan_trajectory(...)
  ├─ 生成 100 条候选
  ├─ cost guidance
  ├─ 碰撞和位置/速度/加速度限位过滤
  └─ weighted_metrics 选择 q_trajs_pos_best
  │
  ▼
最佳轨迹二次验收
  ├─ 必须存在且形状为 [128, 7]
  ├─ 全部位置/速度/加速度/时间必须有限
  ├─ 时间必须从 0 开始、严格递增并在 10 秒结束
  ├─ 第一点必须与 q_pos_start 在 1e-5 rad 内一致
  ├─ 不得超过 Panda 位置/速度/加速度限制
  └─ 不得在当前 MPD Warehouse 场景中碰撞
  │
  ▼
计算 metrics 和请求/场景/配置/模型/Git 元数据
  │
  ▼
原子写 trajectory.npz
  │
  ▼
原子写 status=success 的 result.json
```

任何一步抛出已分类错误时，程序都会删除 `trajectory.npz`、写失败 `result.json` 并返回非零退出码。

## 7. Runtime YAML 的固定配置

当前专用配置固定：

| 参数 | 值 |
|---|---|
| `start_goal_source` | `states_file`，表示只接受显式状态，不随机采样区域 |
| `env_id_replace` | `EnvWarehouseExtraObjectsV00` |
| `model_selection` | `bspline` |
| `planner_alg` | `mpd` |
| `diffusion_sampling_method` | `ddim` |
| `n_trajectory_samples` | `100` |
| `num_T_pts` | `128` |
| `trajectory_duration` | `10.0 s` |
| `ddim_sampling_timesteps` | `15` |
| `best_trajectory_selection` | `weighted_metrics` |

`infer_once.py` 会主动检查这些固定值。不能通过临时 YAML 静默换成另一机器人、另一场景、另一轨迹长度或另一规划方法而继续使用相同输出契约。

## 8. 与 ROS 2 adapter 的责任边界

ROS 2/physical runtime adapter 在调用前仍需负责：

1. 按名称从最新 `/joint_states` 提取 `panda_joint1..7`，不能依赖消息数组偶然顺序；
2. 检查状态时间戳新鲜度；
3. 冻结目标、TF、场景和抓取状态快照；
4. 为每个请求创建唯一 `request_id` 和独立输出目录；
5. 在 MPD 独立环境中启动本脚本并检查退出码；
6. 验证 `result.json.request_id` 与当前请求匹配；
7. 推理结束后再次读取机器人和场景，若起点漂移或场景变化则废弃轨迹；
8. 将 NPZ 转换为 `TrajectoryPlanResult` 或 `JointTrajectory` 后，交给 Execution Manager/JTC；
9. 在假硬件、仿真和现场安全门全部通过前，不得发送到真机。

`scene_hash` 和 `joint_state_stamp` 当前只由脚本记录，不会自动连接 ROS 2 或判断实时状态是否变化。这些检查必须由 adapter 在 MPD 子进程前后完成。


但 conda run 不会自动执行仓库的 set_env_variables.sh。因此 adapter 应显式补充需要的变量：

env = os.environ.copy()
env.update(
  {
      "ISAACLAB_ROOT": "/home/eric/IsaacLab_ori",
      "ISAACLAB_CONDA_ENV": "env_isaaclab_ori",
  }
)

command = [
  "/home/eric/anaconda3/bin/conda",
  "run",
  "--no-capture-output",
  "-n",
  "mpd-splines-public-cu128",
  "python",
  "/home/eric/MotionPlanningDiffusion/mpd-splines-public/scripts/runtime/
  infer_once.py",
    后续参数
]

subprocess.run(
  command,
  cwd="/home/eric/MotionPlanningDiffusion/mpd-splines-public",
  env=env,
  timeout=900,
  check=True,
)

## Adapter 推荐结构

ROS adapter：Pixi/ROS 2 Jazzy
      |
      | subprocess + conda run
      v
MPD infer_once：mpd-splines-public-cu128
      |
      | NPZ + JSON
      v
ROS adapter 校验并生成 JointTrajectory

不要：

在 ROS adapter 进程内 import MPD
在 ROS adapter 所在 shell 中永久 activate MPD
只换 Python 路径但继承 Pixi 动态库环境
