# MPD 与 ROS 2 机械臂平台集成规划方案

> 状态：Draft v1  
> 编制日期：2026-08-05  
> 首期对象：Franka FR3/Panda、单臂、关节空间全局轨迹  
> 依据：本仓库当前实现、论文 *Motion Planning Diffusion: Learning and Adapting Robot Motion Planning With Diffusion Models*、[“近两年 MPD 改进”聊天记录](https://chatgpt.com/share/6a72e02f-41c0-83ee-b087-c1b637f16f3c)

## 1. 目标与范围

本项目的首要目标不是立刻重写 MPD 算法，而是建立一条可复现、可测试、可取消、可审计的完整链路：

```text
ROS 2 当前关节状态 + 目标 + 场景快照
                  ↓
          MPD 全局轨迹规划
                  ↓
       结果校验与安全时间参数化
                  ↓
     Execution Manager 来源仲裁
                  ↓
 FollowJointTrajectory / ros2_control
                  ↓
      fake hardware → 低速实机
```

首期范围：

- FR3/Panda 7 自由度单臂；
- 点到点全局规划，一次请求返回一条完整的定时关节轨迹；
- 关节目标和末端位姿目标；
- 固定 Warehouse 场景加少量新增障碍物；
- 通过 Execution Manager 下发 `joint_trajectory_goal`；
- 离线、fake hardware 和低速实机的分级验证；
- 对 MPD 推理耗时、有效率、碰撞率和执行跟踪误差进行统一记录。

首期不做：

- 不把 MPD、PyTorch、CUDA 或 Python 推理放入 `ros2_control` 实时循环；
- 不直接向 libfranka 发送 1 kHz 位置/力矩命令；
- 不在首期支持双臂、夹爪协同、动态障碍物闭环重规划；
- 不把训练、数据清洗或 IsaacLab 依赖并入 ROS 2 Pixi/colcon 环境；
- 不将“MPD 输出了一条轨迹”等同于“该轨迹可安全上实机”。

## 2. 论文结论与工程含义

论文中的 MPD 在低维 B-spline 控制点空间学习多模态轨迹先验，并在反向扩散期间加入目标、碰撞、关节限制、速度和加速度等代价梯度。B-spline 使轨迹天然平滑、可高频插值，并比 128 个独立 waypoint 使用更低维的扩散变量。

对本项目最重要的论文结论是：

- MPD 的价值是用学习到的多模态先验初始化并约束优化，而不是替代所有几何验证和控制器安全机制；
- 新增障碍物下，纯 diffusion prior 可能失效，推理时的 cost guidance 是适应新场景的关键；
- Panda 复杂场景中，论文报告 100 条候选、15 个 DDIM 步、12 次 cost gradient 更新合计约 `0.56 s`，其中 diffusion 网络约 `0.057 s`，主要瓶颈是代价梯度；
- 论文固定轨迹时长，未解决在线时长优化；
- 论文只关注结构变化有限的环境，跨布局泛化仍需要环境编码或更大数据覆盖；
- 软代价不构成实机安全保证，最终仍需独立的限位、连续碰撞和控制器可执行性验证。

因此工程优先级应为：先打通稳定接口和验证门禁，再减少无效 Cost Guide，最后才考虑更换扩散采样器或重新训练模型。

## 3. 当前仓库基线

### 3.1 仓库责任边界

| 目录 | 当前责任 | 本方案中的定位 |
|---|---|---|
| `MotionPlanningDiffusion/mpd` | 数据生成、训练、MPD 推理、IsaacLab 验证、梯度剪枝实验 | 离线算法仓库和独立 GPU 推理环境 |
| `physical_ai_runtime` | ROS 2 通信、planner adapter、Execution Manager、控制器、机器人描述、录制 | 在线运行时与安全集成边界 |
| `runtime_resources` | FR3/Marvin/Piper 应用 bringup 和演示 | 具体机器人的 launch/config 组合 |

训练和在线运行时保持进程、依赖和所有权隔离。两侧只通过版本化规划请求与轨迹结果契约耦合。

### 3.2 已有能力

MPD 侧已有：

- `scripts/runtime/infer_once.py`：单请求 JSON 输入，`result.json + trajectory.npz` 输出；
- schema v1 的 `request_id` 回传、退出码、错误状态和原子结果写入；
- `[128, 7]` 的位置、速度、加速度和严格递增时间戳；
- 起点、关节限位、速度/加速度、有限值和 dense trajectory 的结果校验；
- `scripts/runtime/README.md` 中较完整的 adapter 责任说明；
- IsaacLab 统计、replay 和实机迁移分级文档；
- A2P-fast 梯度剪枝、dense validator 和多场景基准。

ROS 2 侧已有：

- `GlobalTrajectoryBackend.plan()` 和 `TrajectoryPlanResult` 的后端中立契约；
- `EMCommandSink` 将完整轨迹发布到 `/action_sources/<source>/joint_trajectory_goal`；
- Execution Manager 将该契约转发至 `FollowJointTrajectory`；
- `diffusion_planner_example.py` 已能通过独立 Conda 子进程调用 `infer_once.py`；
- 结果 joint order、shape、有限值和时间严格递增检查；
- FR3 fake hardware、JTC 和 execution manager 的现有 bringup 基础。

### 3.3 当前缺口

| 缺口 | 影响 | 首期处理 |
|---|---|---|
| 示例脚本写死 Conda、环境名和仓库绝对路径 | 无法迁移或部署 | 全部改为参数/YAML，启动时验证 |
| 在 ROS timer callback 中同步阻塞子进程，超时可达 900 s | 节点无响应，无法可靠取消 | 工作线程/异步请求；单请求互斥；超时后终止子进程组 |
| 每个请求重新加载模型和 CUDA | 冷启动大、资源抖动 | Gate 1 允许；Gate 3 引入常驻 worker |
| MPD 当前 runtime 配置只接受固定 Warehouse scene ID | 真实场景快照不能直接输入 | 首期固定场景；后续引入受限的 box/sphere scene payload |
| ROS 示例绕过 `GlobalTrajectoryBackend`，直接发布 EM topic | 架构重复、测试面分裂 | 实现正式 `mpd_planner_adapter` 后端并复用 runtime/sink |
| 规划请求没有正式的取消/反馈入口 | 长任务难管理 | 首期使用节点内部异步状态；需要跨节点请求时再引入 action |
| 输出虽有动力学字段，但缺少 controller 侧独立安全门 | 不能直接执行 | adapter 与 execution 前各保留一层校验 |
| 当前 constrained scene 有效率很低 | 算法尚未达到实机质量门槛 | 与接口集成分轨推进，不用低质量场景证明可用性 |
| `docs/MOTION_PLANNER_SOURCE_INTERFACE.md` 被代码引用但当前缺失 | 权威接口说明断链 | Gate 0 补齐或修正引用 |

### 3.4 当前性能基线

2026-08-04 的 100 候选、15 DDIM 步、3 个 active guidance steps、每步 6 次 guide、128 点 dense validation 结果显示：

| 场景 | `Inference + Dense` p50 | Valid | 结论 |
|---|---:|---:|---|
| Warehouse-clear | 约 `0.69 s` | `98%` | 可作为首期成功场景 |
| Warehouse-single | 约 `0.77 s` | `94%` | 可作为新增障碍基线 |
| Warehouse-narrow-0.20 | 约 `0.76 s` | `87%` | 可进入仿真压力测试 |
| Warehouse-narrow-0.14 | 约 `0.76 s` | `84%` | 仅作压力测试 |
| ThreePillars | 约 `0.71 s` | `5%` | 尚不能作为质量通过证据 |
| DrawerToShelf | 约 `0.71 s` | `1%` | 需训练/算法改进 |
| ToDrawer | 约 `0.72 s` | `1%` | 需训练/算法改进 |

`fused` B-spline 映射相对 `materialized` 没有稳定收益；当前应优化 FK/Jacobian、碰撞距离和空间梯度，而不是继续优化末端映射形式。

## 4. 目标架构

### 4.1 组件关系

```text
                    physical_ai_runtime (ROS 2 / Pixi)

 /joint_states ──> StateCache
                       │
 PoseStamped / JointTarget + World snapshot
                       │
                       v
              mpd_planner_source_node
              ├─ 请求生命周期、取消、诊断
              ├─ GlobalTrajectoryPlannerRuntime
              └─ MPDGlobalTrajectoryBackend
                            │
               versioned JSON/NPZ contract
                            │ process boundary
                            v
                 MPD worker (Conda + CUDA)
                 ├─ model warmup/cache
                 ├─ MPD inference
                 ├─ dense validation
                 └─ structured diagnostics
                            │
                            v
       /action_sources/mpd/joint_trajectory_goal
                            │
                            v
                 Execution Manager
                            │
                            v
             FollowJointTrajectory action
                            │
                            v
            fake/real ros2_control hardware
```

### 4.2 包与文件建议

```text
physical_ai_runtime/src/motion_planning/motion_planners/
  mpd_planner_adapter/
    mpd_planner_adapter/
      backend.py              # GlobalTrajectoryBackend 实现，不导入 rclpy
      worker_client.py        # subprocess 或常驻 worker 传输
      contract.py             # JSON/NPZ 编解码和严格校验
      config.py               # 路径、超时、场景、设备参数
      planner_node.py         # ROS 请求生命周期与 EM sink
    config/
      fr3_warehouse.yaml
    launch/
      mpd_planner.launch.py
    test/

runtime_resources/
  franka_motion_demos/
    launch/
      mpd_global_trajectory_demo.launch.py
    config/
      mpd_execution_manager.yaml
```

不建议把 adapter 放在 `policy_inference/examples`：MPD 在本项目中是请求式全局 motion planner，应实现 `GlobalTrajectoryBackend`，并沿用已有 planner source 与 EM 契约。

### 4.3 通信选择

首期复用标准 ROS 类型：

- 当前状态：`sensor_msgs/msg/JointState`；
- 笛卡尔目标：`geometry_msgs/msg/PoseStamped`；
- 结果/执行：`trajectory_msgs/msg/JointTrajectory` 与 `control_msgs/action/FollowJointTrajectory`；
- 诊断：先使用结构化日志和 `diagnostic_msgs`，不在命令 topic 填 JSON。

规划耗时超过 1 秒时天然适合 action，因为需要反馈、取消和结果。第一阶段若只有一个内部消费者，可先由 application 节点管理一次规划任务，不立即创建自定义接口。出现两个以上独立请求方，或需要跨进程返回失败原因/候选统计时，再在 `src/interfaces` 增加最小的 `PlanJointTrajectory.action`，不要提前设计通用 Physical AI 接口层。

## 5. 数据契约

### 5.1 请求 schema v1

沿用现有 `infer_once.py` 字段，并冻结为 adapter 与 MPD worker 的边界：

```json
{
  "schema_version": 1,
  "request_id": "uuid",
  "robot_model": "franka_fr3",
  "planning_frame": "fr3_link0",
  "joint_names": ["fr3_joint1", "...", "fr3_joint7"],
  "goal_type": "joint|cartesian",
  "q_pos_start": [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0],
  "q_pos_goal": [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0],
  "ee_pose_goal": {
    "position_xyz": [0.4, 0.0, 0.5],
    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]
  },
  "q_vel_start": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "q_vel_goal": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "q_acc_start": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "q_acc_goal": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "scene_id": "EnvWarehouseExtraObjectsV00",
  "seed": 12345
}
```

约束：

- `joint_names` 必须与 URDF、JointState 和控制器顺序严格一致；
- adapter 从最新且未过期的 JointState 生成 `q_pos_start`，不信任调用方提供的旧起点；
- `PoseStamped` 必须在请求时刻转换到 `planning_frame`，TF 失败或过期直接拒绝；
- 首期 `scene_id` 只能从白名单选择，不能用字符串静默切换任意 checkpoint；
- 同一请求使用唯一工作目录和 `request_id`，结果必须回显并匹配。

### 5.2 结果 schema v1

`result.json` 负责状态和诊断，`trajectory.npz` 负责数值数组：

```text
positions        float64 [T, 7] rad
velocities       float64 [T, 7] rad/s
accelerations    float64 [T, 7] rad/s²
time_from_start  float64 [T]    s
joint_names              [7]
```

成功必须同时满足：

- worker 退出码为 0；
- `status == "success"`；
- `request_id` 匹配；
- `T >= 2`，shape 完全一致，无 NaN/Inf；
- 时间从 0 开始或按 adapter 规则归一化，之后严格递增；
- 首点与请求起点误差在阈值内；
- joint order 完全匹配，不做隐式位置映射；
- 全轨迹满足软件限位、安全 margin、速度和加速度约束；
- dense continuous collision check 通过；
- 诊断中记录模型、配置、Git commit/dirty、seed、候选数和各阶段耗时。

失败必须返回明确类别：`invalid_request`、`invalid_configuration`、`timeout`、`no_valid_trajectory`、`invalid_result`、`canceled` 或 `inference_error`。失败时禁止发布旧轨迹或部分轨迹。

### 5.3 场景契约演进

场景按以下顺序演进：

1. `scene_id` 固定 Warehouse 场景；
2. schema v2 增加有限数量的 `WorldBox`/`WorldSphere`，带 frame、stamp、name 和尺寸；
3. 支持 mesh URI，但不在 ROS 消息内嵌顶点；
4. 最后才评估 voxel/point cloud，因为它会显著增加坐标、传输和碰撞后端复杂度。

任何场景快照都必须在 adapter 中转换到 MPD planning frame，并记录时间戳。动态障碍物闭环重规划不属于首期。

## 6. 分阶段实施与验证门禁

以下各 Gate 必须按顺序通过。每一级只增加一种新风险。

### Gate 0：冻结契约与可复现基线

目标：不改算法、不接 ROS 运动，证明固定请求可重复生成可验证轨迹。

任务：

- 冻结 request/result schema v1；
- 将绝对路径移入配置或环境变量；
- 为 `infer_once.py` 增加/保留契约测试；
- 固定 checkpoint、runtime YAML、seed、设备和输出元数据；
- 补齐缺失的 planner source 权威接口文档或修正代码引用；
- 记录 Warehouse-clear 和 Warehouse-single 的基线结果。

建议命令：

```bash
conda run --no-capture-output -n mpd-splines-public \
  python MotionPlanningDiffusion/mpd/scripts/runtime/infer_once.py \
  --request <request.json> \
  --output-dir <output-dir>
```

预期证据：

- `result.json`、`trajectory.npz` 和结构化耗时；
- 同 seed 重复运行轨迹一致或在已声明的确定性边界内一致；
- 无效 shape、joint order、scene ID 和起点会返回非零退出码；
- 没有任何 ROS 命令 topic 发布。

通过条件：契约测试全通过；Warehouse-clear 至少 100 个独立 seed 的 request success rate 达到 `>= 99%`，不是只统计 batch 内 valid fraction。

### Gate 1：ROS 2 adapter 离线集成

目标：实现正式 `MPDGlobalTrajectoryBackend`，结果只进入 `RecordingCommandSink`，不连接控制器。

任务：

- 新建 `mpd_planner_adapter`；
- 将当前 example 的 subprocess 逻辑下沉到 ROS-free `worker_client.py`；
- 实现 `GlobalTrajectoryBackend` 与 `TrajectoryPlanResult` 转换；
- 所有路径、超时、环境、设备、场景和 source name 参数化；
- 后台线程执行规划，主 executor 保持可响应；
- 支持一条活动请求、显式 busy/reject、timeout 和 cancel；
- 保存每次请求与诊断，不保存用户未授权的大型中间 tensor。

建议命令：

```bash
cd physical_ai_runtime
pixi run test-motion-planners
```

预期证据：

- backend 单元测试可用 fake worker 返回成功、无有效轨迹、超时、错误 schema；
- planner node 在规划期间仍能处理诊断与取消；
- `RecordingCommandSink` 得到一条完整轨迹；
- worker 失败时 sink 不得到任何轨迹。

通过条件：adapter 与 MPD 环境隔离；ROS/Pixi 的 `PYTHONPATH` 不污染 Conda；所有失败路径可观测且不发布动作。

### Gate 2：Execution Manager 契约验证

目标：只验证 ROS topic、来源仲裁和 JTC action 转发，不启动机械臂运动。

任务：

- 配置 source `mpd`，仅启用 `joint_trajectory_goal`；
- 验证 EM 优先级、active source、过期请求和并发来源行为；
- 检查 QoS：命令为 RELIABLE、KEEP_LAST 1、VOLATILE；
- 使用 mock `FollowJointTrajectory` server 检查 goal 内容与取消传播。

建议命令：

```bash
cd physical_ai_runtime
pixi run test-execution-manager
```

预期证据：

- `/action_sources/mpd/joint_trajectory_goal` 类型为 `JointTrajectory`；
- mock action server 只收到校验后的目标；
- 新 goal 可按既定策略替换/拒绝旧 goal；
- planner/EM 任一侧取消后，action goal 不再执行。

通过条件：launch test 全通过；错误轨迹、空轨迹和错误 joint order 均不能到达 JTC。

### Gate 3：模型常驻与性能稳定

目标：消除每请求 Conda 启动、模型加载和 CUDA 初始化开销，同时保持进程隔离。

任务：

- 将一次性 CLI 演进为常驻单 worker；
- 启动时 warmup，并上报 `ready/model/config/device`；
- 使用 Unix domain socket 或受限本机 IPC，协议仍复用 schema v1；
- 单 worker 串行执行，队列深度首期为 1；
- worker crash 后 adapter 进入 not-ready，不自动重放旧 motion request；
- 保留 `infer_once.py` 作为确定性回归与故障降级工具。

预期证据：

- 冷启动时间与稳态 request latency 分开记录；
- 100 次请求 GPU 内存不持续增长；
- worker kill/restart 后节点不会发布旧结果；
- p50/p95/p99 和各阶段耗时可查询。

通过条件：Warehouse-clear 稳态端到端规划 p95 `<= 1.0 s`；100 次运行无资源泄漏、无 request/result 串线。

### Gate 4：fake hardware 轨迹执行

目标：在 FR3 fake hardware 上验证关节顺序、时间、控制器接收和轨迹方向。

安全边界：允许 fake hardware 运动，不连接真实机器人。

任务：

- 启动 FR3 fake hardware、JTC、EM 和 MPD adapter；
- 先执行无障碍、短距离、保守速度的关节目标；
- RViz 对照 MPD/IsaacLab replay；
- 记录 planned、commanded、desired、actual、controller result；
- 验证规划过程中起点变化的处理：发布前重新检查实际状态与轨迹首点。

预期证据：

- action 返回成功；
- joint order 无镜像或错位；
- 轨迹时间严格递增，控制器无 tolerance violation；
- actual 与 desired 的最大误差在 fake hardware 预期范围内；
- rosbag/MCAP 可复现本次请求、结果和执行状态。

通过条件：至少 100 次离线目标连续通过，0 次 controller reject、0 次错误 source 抢占、0 次无记录动作。

### Gate 5：仿真场景闭环验证

目标：验证 MPD scene、ROS planning frame 和仿真障碍物三者一致。

任务：

- 只支持 box/sphere 场景；
- 为每个 obstacle 记录尺寸、pose、frame 和 stamp；
- 在 IsaacLab 与 ROS 仿真中使用同一份场景来源；
- 对轨迹做 MPD dense check 和独立仿真碰撞统计；
- 对 Warehouse-clear、single、narrow-0.20 分别跑独立 seed。

通过条件：

- clear/single 的 request success rate `>= 99%/95%`；
- 所有准备执行的轨迹独立仿真碰撞率为 0；
- MPD validator 与仿真碰撞判定不一致率低于 `1%`，且每个不一致案例均有保存和解释；
- narrow 场景只作为压力测试，不阻塞首期实机空场闭环。

### Gate 6：实机只读联调

目标：连接真实 FR3，只读取状态、TF、控制器与急停状态，不发布运动。

安全边界：planner sink 强制为 `recording`；EM 的 MPD source 禁用。

检查：

- robot model、URDF、joint names、单位和 planning frame 一致；
- JointState 新鲜度和顺序稳定；
- 当前状态与候选轨迹首点误差 `< 0.02 rad`；
- 急停、示教器、protective stop 和控制器状态可观测；
- 现场场景尺寸/pose 已测量并保留额外 collision margin。

通过条件：完整 operator checklist、状态快照、TF 检查和离线计划归档；全程没有运动命令。

### Gate 7：低速空场实机执行

目标：第一次真实动作只验证方向、平滑性、停止和跟踪，不验证复杂避障任务。

前置条件：Gate 0–6 全部通过，操作员明确批准本次轨迹。

限制：

- 单臂、无夹爪、空工作区；
- 速度/加速度缩放 `<= 0.1`；
- 小关节位移和短轨迹；
- 操作员在急停旁；
- 支持 action cancel、超时和 protective stop；
- 发布前再次校验 current state 与轨迹起点。

通过条件：

- 10 次逐步扩大的低速执行均无 protective stop、超限、抖动或方向错误；
- planned/desired/actual 全部记录；
- 最大跟踪误差、速度和加速度不超过预先设定阈值；
- 任一次异常都回退到 Gate 4 或 Gate 5，不在现场临时放宽阈值。

### Gate 8：真实障碍场景

目标：在已标定、静态、保守膨胀的真实 Warehouse 场景验证规划适应能力。

通过条件：

- 每次场景变化都重新生成 scene snapshot、MPD 结果、IsaacLab/仿真统计和 operator approval；
- 实物障碍采用比视觉模型更大的安全膨胀；
- 先单障碍、远离机械臂，再逐步缩小 clearance；
- 未建立动态场景更新与停止机制前，不引入运动障碍物。

## 7. 算法优化路线

算法路线与 ROS 集成 Gate 并行开发，但只有稳定分支和完整验证结果才能进入 Gate 5 以后。

### A0：建立可比较基线

必须统一：独立 seed 数、候选数、DDIM 步数、guide 调用次数、dense 点数、场景、设备和 warmup。报告至少包含：

- request success rate：一次请求能否找到至少一条最终有效轨迹；
- candidate valid fraction；
- 独立 dense collision rate；
- goal position/orientation error；
- path length、加速度、jerk；
- generator、guide、dense validation、selection、总 wall time；
- GPU 峰值内存。

禁止用同 seed 的 timing repeat 充当质量样本。

### A1：动态、稀疏 Cost Guidance（最高优先级）

聊天记录中最适合当前仓库的近期思路是把 guidance 从固定调度改为“时间门控 × 预测稳定性 × 当前风险”，同时保留 GPU 批处理：

```text
完整 batch 做 diffusion U-Net
             ↓
在预测干净轨迹 x_recon 上做低成本 coarse risk check
             ↓
构造 active mask / active sub-batch
             ↓
仅对风险轨迹做完整 FK + Jacobian + collision gradient
             ↓
scatter 回完整 batch，继续统一 DDIM timestep
```

实施顺序：

1. 固定 DDIM 和 guide 参数，只记录风险、稳定性和 active ratio；
2. 使用 mask 关闭明显安全轨迹的完整梯度；
3. 比较固定 batch mask 与压缩 active sub-batch；
4. 加入 hysteresis，避免 guidance 频繁开关；
5. 最后才动态改变 guide scale、prior scale、trust radius 或 guide steps。

风险概率第一版用 clearance/综合约束代价构造并校准。不要机械地将 `q/(1-q)` 直接乘原 cost gradient；概率一致版本优先使用安全后验对应的梯度，并设置概率截断、梯度归一化和最大更新限制。赔率版本只作为单独消融。

验收：相同质量门槛下 `guide_sec` p50 降低至少 `30%`，Warehouse request success rate 下降不超过 1 个百分点，最小 clearance 不变差。

### A2：重写 Cost Guide 热点

当前基准证明主要耗时不在 B-spline 映射。优化顺序：

1. 一次 FK/Jacobian 供环境碰撞、自碰撞和目标 cost 共用；
2. 速度、加速度、joint limit 等解析梯度移出 autograd；
3. 复用静态场景 SDF 与机器人几何缓存；
4. profile 后再选择 `torch.compile`、CUDA Graph、Triton、Warp 或 cuRobo 碰撞后端；
5. 每次替换都运行数值等价测试和端到端质量回归。

验收：梯度方向/数值误差在预定容差内；无碰撞判定漂移；`guide_sec` 在 A1 基础上再下降至少 `25%`。

### A3：减少扩散步骤

依次测试 DDIM `15 → 12 → 10 → 8`，保持其他因素不变。由于 diffusion 当前只占总时延小部分，不应把 DPM-Solver++、Consistency Model 或 Flow Matching 作为首期主线。

验收：选取 Pareto 点；request success、clearance 和多样性达到 A0 门槛，且端到端 p95 有可测收益。若收益小于 `10%`，停止继续复杂化采样器。

### A4：复用后期去噪候选与 trajectory stitching

借鉴 GPD：保留最后 `K=5` 个 `x_recon`，将 `batch × K` 展开为相关候选池；逐段连续碰撞检查后，从低代价轨迹开始，必要时切换到另一候选的安全前向窗口。

必须满足：

- 检查 segment，不只检查 waypoint；
- stitch 目标只能严格向前，防止循环；
- 连接先试直线，再用短时限 RRT-Connect；
- 每次连接和 B-spline 重拟合后重新做 dense validation；
- 明确记录局部规划器耗时，不能把它隐藏在 diffusion 指标内。

该方案可提高困难拓扑下的成功率，但不是首期 ROS 集成依赖。ThreePillars/Drawer 有效率仍很低时，不应直接把 stitching 结果用于实机。

### A5：Top-K GPU 后端修复

长期推荐结构：少量多模态 MPD 候选 → Top-K → cuRobo/Warp 轨迹优化 → 独立 dense validation。第一步仅把 cuRobo 作为最终优化器，不立即替换 MPD 反向扩散中的碰撞后端。

验收：与纯 MPD 相比提高 request success 或 clearance，同时端到端延迟满足部署预算；所有修复后轨迹仍经过相同的独立 validator。

### A6：研究项

以下属于后续研究，不进入首期交付承诺：

- 学习校准的 `P(unsafe | x_recon, t, scene)` 风险模型；
- 轨迹时长或 phase-time B-spline 联合优化；
- 环境编码与跨布局泛化；
- Flow Matching/Consistency 少步先验；
- 动态障碍物的 warm-start/receding-horizon 重规划；
- 安全投影、控制 barrier function 或形式化安全层。

## 8. 测试矩阵

| 层级 | 测试 | 关键断言 |
|---|---|---|
| MPD 单元 | request/result schema | shape、类型、joint order、request ID、退出码 |
| MPD 数值 | validator | 起点、限位、速度、加速度、continuous collision |
| Adapter 单元 | fake worker | success/timeout/cancel/crash/invalid result |
| Backend 契约 | `GlobalTrajectoryBackend` | ROS-free、never partial、diagnostics 完整 |
| ROS node | state/target freshness | stale/missing state 和 TF 失败时拒绝 |
| EM 集成 | source contract | 仅有效完整轨迹到达 mock JTC |
| launch test | startup/shutdown | worker/adapter/EM 可控退出，无孤儿进程 |
| 性能 | repeated independent requests | p50/p95/p99、GPU memory、无泄漏 |
| fake hardware | controller execution | joint order、time、tracking、cancel |
| 仿真 | scene consistency | MPD 与独立碰撞判定一致 |
| 实机只读 | state/TF/controller | 零动作、状态新鲜、急停可观测 |
| 实机低速 | conservative motion | 跟踪误差、超限、停止、完整记录 |

每个回归结果保存：代码版本、dirty 状态、checkpoint hash、配置 hash、请求、结果、硬件/GPU、ROS domain、时间和判定。大型模型和数据不提交 Git，但必须有可恢复的版本标识。

## 9. 可观测性与数据记录

每次规划生成唯一 `request_id`，贯穿以下记录：

- 请求接收、state/TF/scene snapshot 时间；
- queue、warmup、inference、validation、publish 和 execution 延迟；
- 模型/checkpoint、配置、seed、候选数和算法开关；
- success/failure category 与详细 reason；
- 候选有效率、碰撞率、goal error、clearance、平滑度；
- 发布的 `JointTrajectory`；
- EM source selection 和 JTC action status；
- desired/actual JointState 与 controller error；
- 实机阶段的相机、机器人状态、操作员审批和异常事件。

ROS 运行时优先使用 rosbag2 MCAP 保存原始 stamped streams；训练集转换和时间对齐留在离线侧。

## 10. 安全策略

- 默认 sink 为 `recording` 或 fake hardware；真实运动必须显式启动专用 launch。
- 任何配置缺失、TF 失败、状态过期、scene 过期、worker not-ready、超时、validator 失败均 fail closed。
- 不缓存并重放上一次成功轨迹作为错误降级。
- 规划期间机器人状态变化超过起点容差时，结果作废并重新规划。
- Execution Manager 是唯一正常硬件命令入口；direct sink 只用于无硬件 bringup/debug。
- 实机轨迹必须有速度/加速度安全缩放、joint margin、continuous collision check 和 operator approval。
- 首期一次只允许一个 planning request 和一个 execution goal。
- 真实运动前清理跨机器遗留 `CYCLONEDDS_URI/ROS_DOMAIN_ID`，确认只连接目标机器人。
- 所有 GPU/ML/规划计算位于非实时进程，控制器继续由 ros2_control 管理。

## 11. 交付物与建议排期

| 迭代 | 建议周期 | 交付物 | 退出条件 |
|---|---:|---|---|
| I0 | 3–5 天 | schema v1、基线报告、权威接口文档 | Gate 0 |
| I1 | 1 周 | `mpd_planner_adapter`、fake worker 测试 | Gate 1 |
| I2 | 1 周 | EM/JTC mock 集成与 launch test | Gate 2 |
| I3 | 1 周 | 常驻 worker、warmup、性能/故障测试 | Gate 3 |
| I4 | 1 周 | FR3 fake hardware demo、MCAP 记录 | Gate 4 |
| I5 | 1–2 周 | 统一 box/sphere 场景与 IsaacLab/ROS 仿真回归 | Gate 5 |
| I6 | 现场窗口 | 实机只读检查、低速空场执行 | Gate 6–7 |
| I7 | 现场窗口 | 单静态障碍物验证 | Gate 8 |

算法 A1/A2 可从 I1 后并行研究，但在稳定分支上必须通过 A0 全量回归后才能进入 I5。

## 12. 首个可执行 Sprint

第一周只做以下事项：

1. 冻结并测试 schema v1，建立 100 个独立 seed 的 Warehouse-clear/single 基线；
2. 新建 `mpd_planner_adapter`，实现 ROS-free backend、contract 和 fake worker；
3. 把 Conda、脚本、配置、checkpoint、device 和 timeout 全部参数化；
4. 将同步 timer callback 改为可取消的后台规划任务；
5. 用 `RecordingCommandSink` 完成成功、失败、超时、取消四条测试；
6. 补齐 planner source 权威接口文档，明确 MPD 属于 global trajectory family；
7. 暂不启动 JTC，不改 MPD guidance，不接实机。

Sprint 完成的判据是：一条 ROS 侧请求能够在不阻塞 executor 的情况下调用隔离的 MPD 环境，返回严格校验的 `TrajectoryPlanResult`；任何异常都不发布轨迹，并能用 `request_id` 找到完整诊断。

## 13. 决策记录

| 决策 | 理由 |
|---|---|
| MPD 定位为 `GlobalTrajectoryBackend` | 一次请求产生完整定时关节轨迹，与现有架构一致 |
| ROS 与 MPD 使用进程边界 | Python/CUDA/Conda 依赖与 ROS Jazzy/Pixi 隔离 |
| 首期继续使用 JSON/NPZ schema | 已实现、可审计、便于回归；先稳定语义再优化传输 |
| 先 subprocess，后常驻 worker | 先缩小集成风险，再解决冷启动性能 |
| 只通过 EM 进入控制器 | 保留来源仲裁、取消、状态和统一执行入口 |
| 首期固定 Warehouse + box/sphere | 当前 checkpoint 和 runtime 已有基线，避免同时引入感知复杂度 |
| 算法先优化 Cost Guide | 论文和当前基准都表明其为主耗时 |
| 不把低 candidate valid fraction 当作请求失败率 | 二者统计含义不同，必须分别报告 |
| 不以 fused mapping 作为近期优化方向 | 当前 42 次基准未显示稳定收益 |
| 实机按只读、空场低速、静态障碍逐级推进 | 每一级只增加一种风险，失败可明确回退 |

