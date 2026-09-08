# Marvin 双臂 MPD 与 ROS 2 实施方案

> 状态：设计稿，面向本仓库当前代码（2026-09-03）。本文描述新增代码、接口、数据和验收顺序，不表示这些功能已经实现。
>
> 参考论文：[Motion Planning Diffusion: Learning and Adapting Robot Motion Planning with Diffusion Models](../Motion_Planning_Diffusion_Learning_and_Adapting_Robot_Motion_Planning_With_Diffusion_Models.pdf)。论文的核心做法是：以 B-spline 控制点为扩散模型输出，在推理阶段通过可微代价引导进行适配；论文实验使用固定轨迹时长。本文沿用该空间轨迹先验，同时把时间处理限定在推理/重定时阶段，不把时间 `t` 加入 diffusion 训练变量。

## 1. 结论先行

本项目不应把 Franka 的 7-DoF 类和接口直接改成 14-DoF，也不应让两个互不知情的 7-DoF MPD 实例同时规划。建议新增一条完全隔离的 Marvin 双臂链路：

1. MPD 中新增 Marvin 14-DoF 机器人、双末端 FK/Jacobian、双臂/载荷碰撞和协同闭链代价。
2. 独立任务和协同搬运分别训练两个 14-DoF checkpoint，第一版都采用 `q_start + q_goal` 条件，不改原网络输入协议。
3. 静态环境新增 Marvin 专用的离线 inference 和 one-shot runtime 入口。
4. 动态环境依次实现三档：冻结快照且无时间代价、固定时间表的动态碰撞代价、推理期优化 timing spline。第三档仍只对空间控制点做 diffusion 去噪，时间不参与训练。
5. ROS 2 新增双臂 action、planner adapter、bringup 和一个 14 关节 `JointTrajectoryController`。协同轨迹只发送一个原子化的 14 关节 goal。
6. 原 Franka 的脚本、配置、checkpoint、socket 和 ROS 2 package 保持原样；新功能使用新文件、新包名、新入口和默认关闭的 launch profile。

推荐的全链路为：

```text
ROS 2 双臂 Action Goal + DynamicWorld(version)
                  │
                  ▼
mpd_bimanual_planner_adapter ── latest-only / cancel / deadline
                  │ IPC schema: marvin_bimanual/v1
                  ▼
驻留 GPU worker ── 14D diffusion(P) ── 双臂代价引导
                  │                       ├─ robot/world/self/inter-arm
                  │                       ├─ joint/velocity/acceleration
                  │                       ├─ closure/object goal/payload
                  │                       └─ dynamic O(t), optional timing c
                  ▼
Top-K 稠密复验 + 最新 world_version 复验 + 重定时
                  │
                  ▼
单个 14-joint FollowJointTrajectory goal
                  │
                  ▼
bimanual_arm_jtc ── Marvin 左右臂同步执行/取消/制动
```

## 2. 范围与明确不做的内容

### 2.1 第一版支持的任务模式

统一关节顺序固定为：

```text
[Joint1_L, Joint2_L, Joint3_L, Joint4_L, Joint5_L, Joint6_L, Joint7_L,
 Joint1_R, Joint2_R, Joint3_R, Joint4_R, Joint5_R, Joint6_R, Joint7_R]
```

所有边界、IPC、HDF5、模型、ROS trajectory 和日志都必须使用这一顺序，禁止运行时按字典顺序推断。

| `task_mode` | 自由变量 | 目标 | 必须满足的约束 |
|---|---:|---|---|
| `left_only` | 左 7 维 | 左臂关节或末端目标 | 右臂全程等于起点；仍检查全机器人碰撞 |
| `right_only` | 右 7 维 | 右臂关节或末端目标 | 左臂全程等于起点；仍检查全机器人碰撞 |
| `dual_independent` | 14 维 | 左右臂各自独立目标 | 两臂可同时运动，但统一规划以检查臂间碰撞 |
| `cooperative_rigid` | 14 维 | 被共同抓取物体的目标位姿 | 两个固定抓取变换、闭链、载荷碰撞、统一时间轴 |

这里的“独立”表示目标和任务互不耦合，不表示规划器彼此独立。两个单独的 7-DoF 规划器无法可靠处理臂间碰撞、共享动态障碍和同时取消，因此不能作为 `dual_independent` 的正式执行路径。

### 2.2 第一版不包含

- 不把 `t`、持续时间或 timing spline 系数放入 diffusion 训练张量。
- 不训练一个同时覆盖所有模式的统一条件模型；先使用独立运动与协同搬运两个 checkpoint。
- 不做抓取搜索。`cooperative_rigid` 假定物体已被两臂稳定抓住，`T_object_left_grasp` 与 `T_object_right_grasp` 已标定。
- 不以几何规划代替双臂力控、阻抗控制、负载分配或滑移检测。真机搬运必须由控制层额外保证内力和接触安全。
- MVP 的协同动态重规划采用“停稳/保持后换轨”；连续运动中的闭链无缝桥接放到后续阶段。

## 3. 当前代码基线与可复用边界

### 3.1 MPD 侧

当前训练和数据主链已经大部分维度无关：

- `mpd/datasets/trajectories_dataset_bspline.py` 从 `sol_path` 拟合 B-spline，并产生 `control_points`、`q_start`、`q_goal` 等字段；除 Panda 的兼容分支外可承载 14 维。
- `scripts/train/train.py` 从 dataset 自动读取 `state_dim` 和 learnable control-point 数；`ContextModelQs` 可自然把双臂 `q_start + q_goal` 变成 28 维条件。
- `mpd/models/diffusion_models/models.py::TemporalUnet` 支持任意 `state_dim`，因此第一版不需要改主干网络实现。
- `mpd/inference/inference.py::GenerativeOptimizationPlanner`、`CostGuideManagerParametricTrajectory` 和 `PlanningTask` 的结构可复用，但现有末端接口只有一个 EE，不能直接表达双末端或被抓物体闭链。
- `mpd/inference/dynamic_collision.py` 已有固定容量的 sphere/box/capsule 动态世界表示。
- `mpd/inference/space_time_guidance.py` 与 `mpd/parametric_trajectory/timing_spline.py` 已有“空间 diffusion + 推理期 timing spline”的 Phase 5 能力，应扩展而不是重新设计。
- `scripts/runtime/` 已有驻留 worker、latest-only 重规划、Top-K、重定时、schema v3 timing 输出等基础设施，但 FR3 runtime contract 仍是 7 维硬编码。

### 3.2 ROS 2 侧

- `physical_ai_runtime/src/apps/marvin_motion_planning_bringup` 已有左右臂各自的 JSPC/JTC 和 7+7 关节定义，但 `groups.bimanual.enabled` 当前明确 fail-fast。
- `mpd_planner_adapter` 与 `mpd_dynamic_planner_adapter` 硬编码 `fr3_joint1..7`、`franka_fr3` 和 Franka topic，不能直接复用为 Marvin 入口。
- `manipulation_execution_manager` 能转发任意 `JointTrajectory`，可新增一个 14 关节实例而无需修改其核心。
- 当前左右两个 JTC 是两个 action server；它们不能保证协同轨迹的原子开始、原子取消和同一时间基准。

## 4. 机器人模型：先建立唯一的 14-DoF 几何事实源

### 4.1 新增文件

在 MPD 仓库新增：

```text
mpd/torch_robotics/torch_robotics/robots/robot_marvin_bimanual.py
mpd/torch_robotics/torch_robotics/robots/__init__.py              # 仅新增 export
mpd/torch_robotics/torch_robotics/data/urdf/robots/marvin/
  marvin_bimanual_mpd.urdf
  meshes/                                                         # 只放已解析的运行时 mesh
mpd/torch_robotics/torch_robotics/data/configs/robots/marvin/
  joint_limits.yaml
  collision_spheres.yaml
  collision_parent_bounds.yaml
  self_collision_pairs.yaml
  grasp_profiles.yaml
scripts/robots/export_marvin_mpd_model.py
tests/test_marvin_model_contract.py
```

`export_marvin_mpd_model.py` 从 ROS 的规范模型
`physical_ai_runtime/src/embodiments/robots/marvin/marvin_description/urdf/marvin.urdf.xacro`
导出扁平 URDF：关闭 `ros2_control`、解析 `package://`、保留碰撞几何，并写入源 xacro、mesh 和挂载参数的 SHA-256。不要手工维护第二套运动学参数。

`RobotMarvinBimanual` 至少提供：

```python
JOINT_NAMES = (...14 个固定名称...)
LEFT_SLICE = slice(0, 7)
RIGHT_SLICE = slice(7, 14)
link_name_ee_left = "flange_L"
link_name_ee_right = "flange_R"

fk_left(q14) -> T_world_left
fk_right(q14) -> T_world_right
jfk_left(q14) -> (J_left, T_world_left)
jfk_right(q14) -> (J_right, T_world_right)
jfk_bimanual(q14) -> ((J_left, T_left), (J_right, T_right))
split_q(q14) / merge_q(q_left, q_right)
```

基类要求的单一 `link_name_ee` 可指向左末端以保持通用工具兼容，但双臂规划代码不得调用这个含糊接口。

碰撞模型必须包括：

- 左臂、右臂和底座的 collision spheres；
- 同臂非相邻 link 的自碰撞 pair；
- 所有可能相交的左-右 link pair；
- 协同模式中随物体位姿运动的 payload spheres/boxes；
- 抓取接触 link 与 payload 之间的 allow-list，仅排除真实接触对，不能整体关闭 payload-robot 碰撞。

关节位置限制从 Marvin URDF 导出。速度/加速度/jerk 采用控制器允许值与硬件规格中的较小值；当前 bringup 中的保守基线为 `2.0 rad/s`、`8.0 rad/s²`、`20.0 rad/s³`，最终必须由真机负责人确认。

### 4.2 模型一致性门禁

`test_marvin_model_contract.py` 应在 100 组合法随机 `q14` 上对比 ROS/KDL 或 PyKDL 与 torchkin：

- 两个末端平移误差建议起始门限 `< 1 mm`；
- 两个末端旋转误差建议起始门限 `< 0.1°`；
- 14 个 joint 名称、顺序和 limits 完全一致；
- 零位姿、左右基座挂载变换、mesh checksum 一致。

这些是建议的初始验收值，需要根据模型精度校准，但不能跳过一致性测试。

## 5. 训练数据

### 5.1 公共 HDF5 协议

新增专用生成器：

```text
scripts/generate_data/generate_marvin_bimanual_trajectories.py
data_generation_cfgs/EnvMarvinTable-RobotMarvinBimanual-independent.yaml
data_generation_cfgs/EnvMarvinTable-RobotMarvinBimanual-cooperative.yaml
mpd/torch_robotics/torch_robotics/environments/env_marvin_table.py
```

每条样本保留现有 loader 所需字段，并增加审计元数据：

| 字段 | 形状/类型 | 说明 |
|---|---|---|
| `sol_path` | `[H_raw, 14]` | 经验证的原始关节路径 |
| `q_start`, `q_goal` | `[14]` | diffusion 条件与边界 |
| `task_id` | scalar/string | 场景/任务分组标识 |
| `task_mode` | enum/string | 四种模式之一 |
| `scene_id`, `scene_hash` | string | 防止数据/场景漂移 |
| `active_joint_mask` | `[14] bool` | 单臂模式冻结维度 |
| `object_start_pose`, `object_goal_pose` | `[7]` | 协同模式，位置 + quaternion |
| `T_object_left_grasp`, `T_object_right_grasp` | `[4,4]` | 固定抓取标定 |
| `min_clearance`, `closure_error` | scalar/vector | 离线质量审计 |
| `generator_version`, `robot_model_hash` | string | 数据可追溯性 |

现有 `TrajectoryDatasetBspline` 第一版只读取 `sol_path/q_start/q_goal/task_id`；新增元数据由生成审计、评测和 runtime goal-IK 使用。这样不需要修改原单臂 dataset。

数据集目录和 checkpoint 必须物理隔离：

```text
data_trajectories_bimanual/EnvMarvinTable-RobotMarvinBimanual-independent-v1/
data_trajectories_bimanual/EnvMarvinTable-RobotMarvinBimanual-cooperative-v1/
logs/marvin_bimanual/independent_v1/
logs/marvin_bimanual/cooperative_v1/
```

### 5.2 独立运动数据生成

使用一个 Marvin 复合 URDF 上的 14D RRTConnect，而不是两个 7D RRTConnect：

1. 采样合法 `q_start` 与左右目标。
2. `left_only` 时令右侧状态空间上下界临时收缩到 `q_start[7:14]`；`right_only` 对称处理。仅把 `q_goal` 设成起点不够，因为 OMPL 中间状态仍可能移动冻结臂。
3. `dual_independent` 允许 14 维同时变化，检查环境、自碰撞和臂间碰撞。
4. RRTConnect 后执行 shortcut/smoothing，再拟合 degree-5 rest-to-rest B-spline。
5. 对稠密采样点和相邻点扫掠都复验 joint limits、碰撞、速度和加速度。
6. 按 `task_id/scene_id` 切分 train/val/test，不能逐轨迹随机切分造成同一任务泄漏。

建议先均衡三类独立任务，例如 `left_only:right_only:dual_independent = 1:1:2`，再根据线上任务分布重采样。路径反转只用于静态环境且保持 rest-to-rest 边界；左右镜像增强只有在关节符号、基座镜像和末端 frame 映射经过单元测试后才能开启。

### 5.3 协同搬运数据生成

普通 14D RRT 很难命中低维闭链流形，不能只在路径末端检查抓取关系。推荐流程：

1. 从合法的双臂抓取姿态得到固定变换 `T_O_L`、`T_O_R`，其中 `T_O_L` 表示 object 到 left grasp 的变换。
2. 在可行物体位姿空间 SE(3) 中规划/采样物体路径。
3. 对每个物体 waypoint 进行双臂 IK，以上一步 `q14` warm-start，并同时最小化双末端残差、关节变化和限位余量。
4. 对失败区段细分 object waypoint 或重采样；成功后联合平滑 `q14`。
5. 全路径验证闭链、payload-world、payload-robot、robot-world、自碰撞、臂间碰撞和动力学限制。
6. 保存 `sol_path`，并把最终物体目标对应的 IK 解作为 `q_goal`。

若后续引入支持 manifold constraint 的 OMPL constrained planning，可替代第 2～4 步，但输出协议不变。

协同数据的硬性有效性条件为：每个稠密 waypoint 均满足闭链，而不是只满足起点和终点。建议初始数据门限为平移闭链误差 `< 2 mm`、旋转误差 `< 1°`，再按真机柔顺性标定。

## 6. 网络训练

### 6.1 第一版不改网络结构代码

新增两个训练配置/薄入口：

```text
scripts/train/cfgs/marvin_bimanual_independent.yaml
scripts/train/cfgs/marvin_bimanual_cooperative.yaml
scripts/train/train_marvin_bimanual.py
```

`train_marvin_bimanual.py` 只负责检查 `RobotMarvinBimanual`、14D 数据协议、模式和输出目录，然后调用现有训练主流程。不要复制 `TemporalUnet`。

两套模型均采用：

```text
x             = learnable B-spline control points, [B, N_cp_learnable, 14]
condition     = concat(q_start, q_goal), [B, 28]
diffusion t   = 原有 diffusion noise step
physical time = 不进入训练
```

协同请求中的 object goal 先经双臂 goal-IK 得到 `q_goal`；闭链与物体目标在推理代价和最终验证中再次强制。这样第一版不用改现有单 EE context model。

训练配置建议：

- B-spline degree 继续使用 5，边界控制点按现有 rest-to-rest 规则固定。
- `N_cp_learnable` 必须兼容 U-Net 下采样深度；启动时打印并 assert，不能依赖 padding 偶然通过。
- 先以现有 base dimension 32 做可比基线；若 14D 欠拟合，再单独把 Marvin 配置提高到 64，不改 Panda 配置。
- batch size 从 128 或 256 起步，根据 GPU 显存实测，而不是照搬论文 Panda 的 512。
- 每个关节独立统计 normalization；训练 manifest 固化 mean/std、joint order、robot hash 和 dataset hash。
- checkpoint 加载时验证 `state_dim=14`、`context_q_dim=28`、控制点数、robot hash 和 task family。

### 6.2 后续统一模型（非 MVP）

只有在两套模型稳定后，再考虑新增 `ContextModelBimanual`，输入 `q_start`、左右/物体目标、抓取变换、`task_mode` 和 active mask。该方案会改变训练协议、数据分布和 checkpoint 兼容性，不应阻塞第一版。

## 7. 双臂规划任务与代价

新增模块，避免修改原 `CostGuideManagerParametricTrajectory` 的单 EE 快路径：

```text
mpd/bimanual/__init__.py
mpd/bimanual/planning_task.py
mpd/bimanual/goal_ik.py
mpd/bimanual/costs.py
mpd/bimanual/cost_guide.py
mpd/bimanual/dynamic_cost_guide.py
mpd/bimanual/space_time_cost_guide.py
mpd/bimanual/trajectory_validator.py
mpd/bimanual/runtime_contract.py
tests/bimanual/
```

`BimanualPlanningTask` 持有两个末端目标、任务模式、active mask、payload geometry 和抓取变换。`BimanualCostGuideManagerParametricTrajectory` 可复用 B-spline 映射、normalization、碰撞缓存和梯度积分工具，但必须显式调用左右 FK/Jacobian。

公共代价包括：

```text
C = w_env C_robot_world
  + w_self C_self_and_interarm
  + w_payload C_payload_world_and_robot
  + w_limit C_joint_limit
  + w_vel C_velocity
  + w_acc C_acceleration
  + w_goal C_terminal_goal
  + w_hold C_inactive_arm_hold
  + w_close C_closed_chain
```

其中单臂冻结代价可写为：

```text
C_hold = Σ_s ||q_inactive(s) - q_inactive,start||²
```

并在采样后做硬投影，使冻结臂控制点直接等于起点。代价只是改善梯度，硬投影才是约束。

协同模式下，从两侧推断物体位姿：

```text
T_WO^L(q) = T_WL(q) · inv(T_OL)
T_WO^R(q) = T_WR(q) · inv(T_OR)
C_close    = Σ_s || Log(inv(T_WO^L) · T_WO^R) ||²_W
C_obj_goal = || Log(inv(T_WO(H)) · T_WO,goal) ||²_W
```

公式中的变换方向必须在 `grasp_profiles.yaml` 和 action 文档中固定，并用 round-trip 单测验证。payload 的碰撞几何由融合后的 `T_WO` 变换；闭链未达门限的 candidate 直接判 invalid，不能仅靠加权排序。

最终 validator 必须输出至少：

- `valid` 与结构化 failure code；
- 最小 robot-world、inter-arm、payload-world clearance；
- 最大 joint/velocity/acceleration violation；
- 最大/终点闭链平移和旋转误差；
- 左右末端终点误差、物体终点误差；
- 使用的 `world_version`、robot/model/config hash。

## 8. 静态环境单次推理

新增入口，不改变 `scripts/inference/inference.py` 和现有 FR3 runtime：

```text
scripts/inference/inference_marvin_bimanual.py
scripts/inference/cfgs/config_EnvMarvinTable-RobotMarvinBimanual-independent.yaml
scripts/inference/cfgs/config_EnvMarvinTable-RobotMarvinBimanual-cooperative.yaml

scripts/runtime/infer_once_marvin_bimanual.py
scripts/runtime/runtime_engine_marvin_bimanual.py
scripts/runtime/infer_server_marvin_bimanual.py
scripts/runtime/infer_client_marvin_bimanual.py
```

`inference_marvin_bimanual.py` 仿照现有离线评测入口，额外支持 `task_mode`、双末端/物体目标、payload 与 grasp profile，并渲染左右臂、物体和 clearance。`infer_once_marvin_bimanual.py` 用于 ROS 无关的单请求 contract 验证。

建议的请求 schema `marvin_bimanual_request/v1`：

```json
{
  "request_id": "uuid",
  "task_mode": "dual_independent",
  "joint_names": ["Joint1_L", "...", "Joint7_R"],
  "q_start": [0.0, "... 14 values"],
  "q_goal": [0.0, "... optional 14 values"],
  "left_goal_pose": null,
  "right_goal_pose": null,
  "object_goal_pose": null,
  "grasp_profile": null,
  "scene": {},
  "world_version": 42,
  "deadline_monotonic_ns": 0
}
```

响应 schema `marvin_bimanual_result/v1`：

```json
{
  "request_id": "uuid",
  "status": "ok",
  "joint_names": ["Joint1_L", "...", "Joint7_R"],
  "positions": [[0.0, "... 14 values"]],
  "velocities": [[0.0, "... 14 values"]],
  "accelerations": [[0.0, "... 14 values"]],
  "time_from_start": [0.0, 0.01],
  "world_version": 42,
  "validation": {}
}
```

所有请求必须严格检查数组长度、有限值、joint 顺序、四元数归一化、frame、schema version、deadline 和 mode 所需字段；禁止静默补齐或重排。

静态推理顺序：

1. 解析目标；若给 pose/object goal，求一个或多个合法 `q_goal`。
2. 根据 task family 选择 checkpoint。
3. 生成多个 14D B-spline candidate，并施加静态双臂代价引导。
4. 对 Top-K 做更高密度验证。
5. 选择最优有效路径；无有效解返回结构化失败，不返回“最不坏”的碰撞路径。
6. 用 TOPPRA/Ruckig 做统一 14D 重定时，生成同一 `time_from_start`。

## 9. 动态世界与多次 replan

### 9.1 三种时间处理必须分开命名

| runtime mode | diffusion 训练变量 | 推理优化变量 | 动态障碍查询 | 能否主动等待 |
|---|---|---|---|---|
| `snapshot_no_time` | 空间 `P` | 空间 `P` | 把请求时障碍冻结为静态快照 | 否 |
| `fixed_time_dynamic` | 空间 `P` | 空间 `P` | 按固定/预重定时时间表查询 `O(t)` | 有限，不能优化等待时机 |
| `inference_time_optimized` | 空间 `P` | 空间 `P` + timing `c` | 按优化后的 `t(s;c)` 查询 `O(t)` | 是 |

前两档对应“无时间 cost”和“有动态时间查询但不优化时间”；第三档有显式 timing cost，但仍不训练时间 diffusion。

### 9.2 `snapshot_no_time`

新增：

```text
scripts/runtime/dynamic_runtime_engine_marvin_bimanual.py
scripts/runtime/infer_dynamic_server_marvin_bimanual.py
```

每次收到较新的 `world_version` 或轨迹安全性下降时，将障碍当前位姿冻结，重新规划空间路径，再做普通重定时。它实现简单且应作为第一个动态里程碑，但无法利用障碍未来会移开的事实；collision guard 必须沿当前执行前缀持续预测并触发 replan/stop。

### 9.3 `fixed_time_dynamic`

复用现有 `dynamic_collision.py` 的 sphere/box/capsule 固定容量 world buffer：

1. 由路径长度或上一次重定时得到固定 `time_from_start[H]`。
2. 根据恒速/KF 预测得到 `O(t_h)`。
3. 对空间控制点 `P` 求动态碰撞梯度，时间表在本次引导中不变。
4. 输出后用同一时间表复验；若重定时改变时间表，必须重新动态复验，不能复用旧结论。

### 9.4 `inference_time_optimized`

新增 Marvin 封装入口：

```text
scripts/runtime/space_time_runtime_engine_marvin_bimanual.py
scripts/runtime/infer_space_time_server_marvin_bimanual.py
```

扩展现有 `InferenceOnlySpaceTimeGuide`，保持：

```text
diffusion model: p(P | q_start, q_goal)
timing spline:   t(s; c), c 仅在 inference 初始化并优化
```

优化目标建议为：

```text
C(P,c) = C_dynamic(P, t(s;c))
       + C_robot_and_payload(P)
       + C_closed_chain(P)
       + C_velocity(P,c)
       + C_acceleration(P,c)
       + λ_T · duration(c)
       + λ_s · timing_smoothness(c)
```

每个 candidate 可以有自己的 `c` 和 `time_from_start`，沿用现有 candidate-specific timing schema。协同模式的两臂和 payload 必须共享同一个 timing spline/总时长，禁止左右臂分别优化时间。

### 9.5 通用 replan 状态机

双臂 adapter/worker 应沿用现有 Phase 1～5 的安全结构，并扩大到 14D：

```text
IDLE → PLAN(request_version)
     → TOP_K_DENSE_VALIDATE
     → REVALIDATE(latest_world_version)
     → HANDOFF_OR_REJECT
     → EXECUTING
     ├─ new goal / unsafe prefix / newer world → cancel old compute, latest-only REPLAN
     ├─ guard warning → controlled brake
     └─ goal reached → IDLE
```

必须实现：

- 驻留 GPU worker；一个进程内只允许一个推理线程操作模型/CUDA buffer。
- latest-only 队列：新请求覆盖未开始的旧请求；在运行中的旧结果以 generation id 丢弃。
- request deadline、`world_version` 和 `valid_until`；过期结果永不下发。
- Top-K 稠密验证后，再针对 ROS 最新动态世界做一次独立复验。
- 14D joint state freshness、tracking error 和 collision guard。
- 任意一臂失败均取消整个 14D goal，并对两臂统一制动。

独立模式可以使用现有思想的 14D quintic bridge。协同模式下逐关节 quintic bridge 通常会破坏闭链，因此：

- MVP：只在两臂速度接近 0、闭链满足门限时切换新轨迹，即 stop/hold/replan/resume。
- 后续：新增 `constrained_bimanual_bridge.py`，在桥接区间对闭链流形做投影/优化，并稠密验证 payload 与动态障碍后才允许运动中换轨。

## 10. ROS 2 新增代码

为确保现有 Marvin per-arm bringup 和 Franka MPD 完全不变，新增独立 package：

```text
physical_ai_runtime/src/interfaces/manipulation_planning_interfaces/
  CMakeLists.txt
  package.xml
  action/PlanBimanualTrajectory.action
  msg/DynamicObject.msg
  msg/DynamicWorld.msg

physical_ai_runtime/src/motion_planning/motion_planners/mpd_bimanual_planner_adapter/
  package.xml
  setup.py
  setup.cfg
  resource/mpd_bimanual_planner_adapter
  mpd_bimanual_planner_adapter/
    __init__.py
    contract.py
    ipc_client.py
    backend.py
    goal_resolver.py
    replan_coordinator.py
    trajectory_adapter.py
    latest_world_buffer.py
    collision_guard.py
    constrained_handoff.py
    node.py
  config/
    static.yaml
    dynamic_snapshot.yaml
    dynamic_fixed_time.yaml
    dynamic_space_time.yaml
  launch/
    planner.launch.py
    planner_fake_hardware.launch.py
  test/

physical_ai_runtime/src/apps/marvin_mpd_bimanual_bringup/
  package.xml
  setup.py
  config/
    controllers_bimanual.yaml
    execution_bimanual.yaml
    planning_static.yaml
    planning_dynamic_snapshot.yaml
    planning_dynamic_fixed_time.yaml
    planning_dynamic_space_time.yaml
  launch/
    marvin_mpd_bimanual.launch.py
    marvin_mpd_bimanual_fake_hardware.launch.py
  test/
```

不要解除当前 `marvin_motion_planning_bringup/planning_config.py` 对 `groups.bimanual` 的 fail-fast；新 package 成熟前它是有效的防误用保护。新 bringup 默认 `execute:=false`，必须显式选择 bimanual profile。

### 10.1 Action 接口

`PlanBimanualTrajectory.action` 建议：

```text
uint8 LEFT_ONLY=0
uint8 RIGHT_ONLY=1
uint8 DUAL_INDEPENDENT=2
uint8 COOPERATIVE_RIGID=3

std_msgs/Header header
uint8 task_mode
string request_id
string[] joint_names
float64[] q_goal
bool has_left_goal_pose
geometry_msgs/PoseStamped left_goal_pose
bool has_right_goal_pose
geometry_msgs/PoseStamped right_goal_pose
bool has_object_goal_pose
geometry_msgs/PoseStamped object_goal_pose
string grasp_profile
bool execute
float64 planning_budget_s
---
bool success
string request_id
string error_code
string message
uint64 world_version
trajectory_msgs/JointTrajectory trajectory
float64 min_clearance_m
float64 max_closure_translation_error_m
float64 max_closure_rotation_error_rad
---
uint8 stage
float32 progress
uint64 world_version
string detail
```

使用 action 而不是 service，是因为规划可超时、取消且需要 feedback。目标 frame 必须统一转换到 `world`，转换时间戳失败时拒绝请求，禁止使用最新 TF 悄悄替代指定时间。

`DynamicObject.msg` 至少包含稳定 `object_id`、shape type/size、pose、twist、6x6 covariance、有效期和可选 inflation；`DynamicWorld.msg` 包含 `Header`、单调递增 `world_version`、`valid_until` 和对象数组。adapter 可以先把 typed message 转成现有 worker JSON/buffer 结构。

### 10.2 Adapter 职责

`mpd_bimanual_planner_adapter` 必须完成而不是交给 worker 猜测：

- 从 `/joint_states` 按固定名称提取 14D 状态，并拒绝缺失、重复或过期 joint。
- action goal 的 mode/字段/frame/单位校验。
- ROS 时间与 worker monotonic deadline 的显式转换。
- dynamic world 版本管理、prediction horizon 和 covariance inflation。
- 发送/取消 IPC 请求，丢弃旧 generation 的迟到响应。
- 把 worker 结果转换为一个 14-joint `JointTrajectory`。
- 下发前做 joint order、首点连续性、单调时间、limits 和最新世界复验。
- 管理 JTC controller ownership、execution manager goal、guard、replan 和统一 stop。

### 10.3 14 关节控制器与原子执行

`controllers_bimanual.yaml` 在保留 joint state broadcaster 的同时新增：

```yaml
controller_manager:
  ros__parameters:
    bimanual_arm_jtc:
      type: joint_trajectory_controller/JointTrajectoryController

bimanual_arm_jtc:
  ros__parameters:
    joints:
      [Joint1_L, Joint2_L, Joint3_L, Joint4_L, Joint5_L, Joint6_L, Joint7_L,
       Joint1_R, Joint2_R, Joint3_R, Joint4_R, Joint5_R, Joint6_R, Joint7_R]
    command_interfaces: [position]
    state_interfaces: [position, velocity]
    allow_partial_joints_goal: false
```

`execution_bimanual.yaml` 新增一个 `em_bimanual`，action 指向：

```text
/bimanual_arm_jtc/follow_joint_trajectory
```

控制器资源互斥规则：

- bimanual 模式启动/执行前，以 STRICT 模式停用 `left_arm_jspc/right_arm_jspc/left_arm_jtc/right_arm_jtc`，再激活 `bimanual_arm_jtc`。
- 切回 per-arm 模式时反向切换。
- 任意切换失败都不执行，不允许多个 controller 同时 claim 同一 joint command interface。
- 即使是 `dual_independent`，只要两臂同时运动，也发送一个 14-joint goal。

这需要先在 fake hardware 验证当前 controller manager 和 Marvin hardware interface 支持一个 controller 同时 claim 两组关节。

## 11. 配置键建议

新 MPD/ROS 配置都应显式包含：

```yaml
robot:
  model: marvin_bimanual
  joint_order: [Joint1_L, Joint2_L, Joint3_L, Joint4_L, Joint5_L, Joint6_L, Joint7_L,
                Joint1_R, Joint2_R, Joint3_R, Joint4_R, Joint5_R, Joint6_R, Joint7_R]
  planning_frame: world

planner:
  task_family: independent       # independent | cooperative
  runtime_mode: snapshot_no_time # snapshot_no_time | fixed_time_dynamic | inference_time_optimized
  n_trajectory_samples: 64
  top_k_dense_validation: 8
  deadline_s: 1.0

cooperative:
  grasp_profile: default_box
  max_closure_translation_error_m: 0.002
  max_closure_rotation_error_rad: 0.01745
  allow_moving_handoff: false

replan:
  latest_only: true
  min_period_s: 0.1
  state_max_age_s: 0.05
  world_max_age_s: 0.1
  stop_on_any_arm_fault: true
```

样本数、deadline、闭链门限和周期都是初始值，最终按 GPU 和真机基准决定。配置加载必须拒绝未知键、非法组合和 checkpoint family 不匹配。

## 12. 分阶段实施顺序

### Phase 0：模型与 contract（必须先完成）

- 导出 `marvin_bimanual_mpd.urdf`，实现双 FK/Jacobian 和碰撞模型。
- 固化 14 joint 顺序、frame、limits、grasp transform 方向和 schema。
- 完成 ROS/MPD FK 一致性测试和 IPC round-trip 测试。

退出条件：随机状态 FK 门禁通过；同一个 `q14` 在 MPD、ROS 和 trajectory 中含义一致。

### Phase 1：独立运动数据与静态 MPD

- 生成小规模 smoke 数据，再生成正式 independent 数据。
- 训练 14D independent checkpoint。
- 实现 MPD-only `inference_marvin_bimanual.py` 和 one-shot runtime。

退出条件：四类静态场景中的前三种任务可产生无碰撞 14D 路径；冻结臂最大位移在数值门限内；held-out scene 指标可复现。

### Phase 2：协同数据与静态 MPD

- 完成 dual IK/object path 数据生成。
- 训练 cooperative checkpoint。
- 实现 closure、payload、object goal cost 和硬验证。

退出条件：全路径闭链及 payload clearance 达标，不只是终点达标；故意错误的 grasp profile 必须 fail closed。

### Phase 3：ROS 2 plan-only

- 增加 interfaces、adapter 和新 bringup；`execute:=false`。
- RViz/fake world 中验证 action cancel、超时、旧结果丢弃和 trajectory 可视化。

退出条件：不启动任何硬件 command controller 也能完成端到端规划；原 Marvin/Franka launch 测试保持通过。

### Phase 4：14-joint 原子执行

- 增加 combined JTC 与 `em_bimanual`。
- fake hardware 验证 controller STRICT 切换、一个 goal、统一 cancel/stop。
- 再在低速真机验证 left/right/dual independent。

退出条件：左右轨迹使用同一 `time_from_start`；任一侧 fault 会取消并制动两侧；不存在资源双重 claim。

### Phase 5：动态快照，无时间代价

- 接入 typed `DynamicWorld`、latest-only replan、guard 和 14D bridge。
- 协同模式仅支持停稳换轨。

退出条件：连续 world update 下不会执行过期结果；障碍进入保护区会 replan 或统一安全停机。

### Phase 6：固定时间动态代价

- 对 `O(t)` 做固定时间表查询。
- 校验 retiming 前后时间表一致性和最新世界复验。

退出条件：回放相同动态场景得到可复现 collision-time 指标；超 prediction horizon 时 fail closed 或降级到配置允许的保守停止。

### Phase 7：推理期 timing 优化

- 把现有 timing spline 和 candidate-specific time schema 接到 Marvin 双臂 guide。
- joint optimization 同时计算 `∂C/∂P` 与 `∂C/∂c`，模型 checkpoint 仍是空间模型。

退出条件：能在测试场景中通过减速/等待避让，且 velocity、acceleration、duration、闭链和最新动态世界验证全部通过。

### Phase 8：连续协同换轨（可选）

- 实现 constrained bridge 或闭链投影的滚动轨迹拼接。
- 在仿真、硬件在环和低速真机逐级开放。

## 13. 测试与验收矩阵

### 13.1 MPD 单元/集成测试

建议新增：

```text
tests/test_marvin_model_contract.py
tests/bimanual/test_joint_order_contract.py
tests/bimanual/test_dual_fk_jacobian_finite_difference.py
tests/bimanual/test_interarm_collision.py
tests/bimanual/test_payload_collision.py
tests/bimanual/test_closed_chain_cost_gradient.py
tests/bimanual/test_inactive_arm_projection.py
tests/bimanual/test_goal_ik.py
tests/bimanual/test_bimanual_dense_validator.py
tests/bimanual/test_runtime_contract_v1.py
tests/bimanual/test_dynamic_world_versioning.py
tests/bimanual/test_fixed_time_dynamic_cost.py
tests/bimanual/test_space_time_gradient_fd.py
```

关键反例必须覆盖：joint 顺序交换、缺 joint、NaN、过期 world、payload 穿过桌面、两臂互撞、闭链终点正确但中途断开、retiming 后撞上动态物体、协同运动中收到旧 replan 结果。

### 13.2 ROS 2 测试

新 package 至少包含：

- action goal 校验、cancel、deadline 和 feedback；
- TF 时间戳/目标 frame 失败；
- joint state freshness 与固定顺序提取；
- IPC worker 断开、schema 不匹配、迟到 response；
- combined JTC controller ownership 与 STRICT switch；
- 一个 14-joint goal 的首点连续性和时间单调性；
- latest-world revalidation、guard、brake 和双臂统一 stop；
- launch test：`execute:=false` 默认不会激活 command controller；
- regression：原 `marvin_motion_planning_bringup` 与 Franka MPD adapter 测试仍通过。

### 13.3 指标

每个 task mode 和 runtime mode 分开记录：

- planning success / validated success / execution success；
- robot-world、inter-arm、payload-world 最小 clearance；
- closure translation/rotation 的 max、p95、terminal；
- goal error、path length、duration、最大速度/加速度/jerk；
- replan p50/p95/p99 latency、deadline miss、stale result discard；
- collision guard trigger、brake count、false stop；
- GPU 显存和 warm/cold start 时间。

14D 与更多 collision spheres 会显著增加成本，不能只报告平均推理时间；以 p95/p99 和 deadline miss 决定 candidate 数、Top-K、稠密点数和梯度步数。

## 14. 预期命令（完成上述文件后）

以下只是建议的最终 CLI contract，当前文件尚未实现：

```bash
# 生成独立运动数据
python scripts/generate_data/generate_marvin_bimanual_trajectories.py \
  --config data_generation_cfgs/EnvMarvinTable-RobotMarvinBimanual-independent.yaml

# 训练两个空间 diffusion checkpoint
python scripts/train/train_marvin_bimanual.py \
  --config scripts/train/cfgs/marvin_bimanual_independent.yaml
python scripts/train/train_marvin_bimanual.py \
  --config scripts/train/cfgs/marvin_bimanual_cooperative.yaml

# MPD-only 静态单次推理
python scripts/inference/inference_marvin_bimanual.py \
  --config scripts/inference/cfgs/config_EnvMarvinTable-RobotMarvinBimanual-independent.yaml

# 驻留动态/space-time worker
python scripts/runtime/infer_dynamic_server_marvin_bimanual.py --config <dynamic.yaml>
python scripts/runtime/infer_space_time_server_marvin_bimanual.py --config <space_time.yaml>

# ROS 2，先 fake hardware 且 plan-only
ros2 launch marvin_mpd_bimanual_bringup marvin_mpd_bimanual_fake_hardware.launch.py \
  runtime_mode:=snapshot_no_time execute:=false
```

具体参数解析风格应跟随仓库已有 `experiment_launcher` 和 ROS launch 约定；实现时不要为了匹配上面示意命令而另造第三套配置系统。

## 15. 兼容性、发布与回滚

### 15.1 明确保持不变

以下现有入口和包不得改名、改默认配置或改 schema：

```text
scripts/inference/inference.py
scripts/runtime/infer_once.py
scripts/runtime/infer_server.py
scripts/runtime/infer_dynamic_server.py
scripts/runtime/infer_space_time_server.py
physical_ai_runtime/.../mpd_planner_adapter
physical_ai_runtime/.../mpd_dynamic_planner_adapter
physical_ai_runtime/.../marvin_motion_planning_bringup
```

原 Panda/Franka dataset、normalization、checkpoint、socket 路径和 launch entry 保持原样。允许对通用 `__init__.py`/打包文件做“只增加新 export/entry point”的变更，但需要 regression test 证明旧 import 和 CLI 不变。

### 15.2 命名隔离

- socket：例如 `/tmp/mpd_marvin_bimanual_static.sock`、`dynamic.sock`、`space_time.sock`；
- ROS namespace：`/marvin/mpd_bimanual`；
- action：`/marvin/mpd_bimanual/plan`；
- controller：`bimanual_arm_jtc`；
- checkpoint：按 `independent/cooperative` 分目录；
- schema：`marvin_bimanual_request/v1`，不复用 FR3 schema version。

### 15.3 发布顺序

每个 phase 都以 feature flag/profile 隔离；默认继续走现有单臂/per-arm 入口。回滚只需停止新 worker/launch 并切回原 controller profile，不删除原数据或覆盖原 checkpoint。真机开放顺序必须是 plan-only → fake hardware → 仿真 → hardware-in-the-loop → 低速无载荷 → 低速载荷。

## 16. 最容易踩坑的五点

1. **双臂独立目标不等于两个独立规划器**：必须用 14D 统一检查互撞和共享障碍。
2. **闭链不能只做终点代价**：协同搬运必须在整个 B-spline 稠密轨迹上验证。
3. **两个 JTC goal 不等于同步执行**：协同任务必须使用一个 14-joint action goal。
4. **普通 quintic handoff 会破坏刚性抓取**：协同动态重规划第一版只能停稳换轨。
5. **retiming 会改变动态碰撞结论**：任何时间表变化后都必须重新按 `O(t)` 验证。

按这一顺序实施，最小可用版本是 Phase 0～4：静态环境下支持单臂、双臂独立和刚性协同的 14D 规划与原子执行。Phase 5～7 再逐级增加动态世界能力，其中 Phase 7 已满足“有时间 cost，但暂不把 `t` 放进 diffusion 训练”的要求。
