# Marvin 双臂 MPD Inference 实施计划

> 状态：可执行设计稿，基于 2026-09-09 当前代码。
>
> 范围：Warehouse 环境、Marvin 14-DoF、以 `dual_independent` 为首个任务模式；从 MPD-only 单次推理，经 Isaac Lab 检测/回放与 ROS 2 单次规划，演进到动态世界重规划和推理期时空 cost guide。
>
> 关系：本文是 [MARVIN_BIMANUAL_MPD_ROS2_IMPLEMENTATION_PLAN.md](MARVIN_BIMANUAL_MPD_ROS2_IMPLEMENTATION_PLAN.md) 的 inference 专项落地计划。总体方案中仍以 `q_start + q_goal` 为首版网络条件的描述已经落后于当前实现；Warehouse 独立任务应以本文所述 scheme-3 双 EE context 为准。

## 1. 结论先行

当前网络侧已经适配双 EE goal context，但推理链路尚未适配完成：

- 当前训练配置使用 `q_start(14) + left_ee_goal(12) + right_ee_goal(12) + active_ee_mask(2)`，原始 context 恰为 40 维。
- A/B/C/D 四个网络变体都能消费该 context，输出仍是 `[B, H, 14]`；22 个原始 B-spline 控制点对应 17 个可学习点，不需要为了 inference 重新拟合 spline。
- `TrajectoryDatasetBspline` 已能构造双 EE context，但通用 `GenerativeOptimizationPlanner.plan_trajectory()` 仍按单 EE 接口调用，没有传 `active_ee_mask`。
- `scripts/inference/inference_marvin_bimanual.py` 目前只是 14D 线性插值 fallback；Marvin static/dynamic/space-time runtime 也主要是占位封装。
- ROS 2 侧已有双臂 action、IPC client、world buffer、14-joint JTC 和 bringup 骨架，但 planner node 当前固定返回 `WORKER_NOT_CONFIGURED`；配置指向 Unix socket，现有 Marvin worker 却仍是 stdin 行式 JSON，二者尚未连通。
- Isaac Lab 的检测与回放脚本目前只支持 Panda，不能直接加载 Marvin 14D 轨迹。

因此推荐顺序是：

```text
双 EE 张量/请求契约冻结
    → MPD-only 静态单次推理
    → Isaac Lab 独立检测与回放
    → ROS 2 one-shot plan-only / fake hardware
    → 常驻静态 worker
    → 动态快照 latest-only replan
    → 固定时间动态碰撞
    → inference-only timing spline / 时空 cost guide
```

两个问题的直接答案：

1. Franka 的 EE goal cost 数学形式可以复用，但现有类不能原样复用。需要把“一个 goal、一个 `jfk_s_ee()`”扩展为左右两个固定槽位、mask、每臂 FK/Jacobian，并把两侧 7D Jacobian 的梯度分别 scatter 到 14D。
2. Franka 默认 B3 剪枝策略可以作为双臂第一版生产基线复用，但不能只复制 YAML 就认为完成。endpoint-only、parent-link kinematics、dense parent fast path 和稀疏 `J^Tg` 的思想可复用；双 EE endpoint 和臂间 self-collision 必须补等价性测试，而且 Marvin/Pika 碰撞球数量远多于 Panda，必须重新测显存与延迟。

## 2. 首版范围与非目标

首个闭环只支持：

- `task_mode=dual_independent`；左右臂同时运动、分别到达自己的 EE pose goal。
- 一个统一的 14D MPD 规划问题，而不是两个互不知情的 7D planner。
- Warehouse 静态场景，以及后续同一场景上的动态障碍物。
- 左右臂共享一个 B-spline 参数和一个时间轴。
- ROS 侧先 `plan_only=true`，再 fake hardware，最后才允许真实硬件执行。

首个闭环不包含：

- `left_only`、`right_only` 的执行支持；contract 可以保留，但等 `dual_independent` 稳定后再启用。
- `cooperative_rigid`、闭链、payload、抓取搜索或双臂力控。
- 把物理时间加入 diffusion 训练输入或输出。
- 两条独立的 7-joint `FollowJointTrajectory` goal；正式执行必须发送一条原子化 14-joint goal。
- 在 Python/MPD replan 节点中实现硬实时避障。该层只能提供有截止时间的异步重规划，硬停止仍由控制器、安全 PLC/硬件和独立 collision guard 负责。

## 3. 当前基线与缺口

| 部分 | 当前可用能力 | inference 前必须补齐 |
|---|---|---|
| 网络 | A/B/C/D 均支持 40D 双 EE context；输出 14D 轨迹 | checkpoint/config/normalizer/variant 一致性校验 |
| Dataset | 支持 `[2,3,4]` EE pose 与 `[2]` mask | runtime 必须显式传 mask，禁止从单 EE API 猜测 |
| Marvin robot | 14D 固定顺序、双 FK、双 6x7 Jacobian、碰撞 FK/Jacobian 已存在 | 双 EE cost 使用的 6x14 Jacobian view/scatter |
| Warehouse | `EnvWarehouseMarvinBimanual` 与 primitive scene 已存在 | 生成、训练、runtime 的 scene id/version/hash 对齐 |
| Cost guide | Franka 完整，Marvin `cost_guide.py` 只是 facade | 双 EE、双臂 self/inter-arm、14D joint costs 与梯度 |
| Dense validator | 通用 dense validator 可复用；Marvin validator 是轻量骨架 | 区分 environment/intra-arm/inter-arm，输出双 EE 误差 |
| MPD-only | Marvin 脚本只做线性插值 | 加载真实 checkpoint、采样、guide、Top-K、保存 artifact |
| Isaac Lab | Panda 单臂检测/回放成熟 | Marvin USD、14D joint map、双 TCP、接触分类 |
| one-shot runtime | Franka `infer_once.py` 成熟 | 建立严格的 Marvin request/result/artifact 契约 |
| resident runtime | Franka static/dynamic/space-time 成熟 | Marvin engine/server/client 仍为占位或缺失 |
| ROS adapter | action、IPC client、world buffer、trajectory adapter 已有骨架 | node 未调用 worker、未执行/取消、未完成 replan 状态机 |
| bringup | 14-joint JTC 与 fake hardware launch 已有 | plan-only demo、start drift/结果复验、EM/动作链联调 |

需要特别修正两个配置漂移：

- 现有 Marvin inference YAML 名为 `EnvMarvinTable`，checkpoint 仍指向 `independent_v1`；当前训练实际是 Warehouse v3 双 EE 模型。
- runtime 加载时必须比较 checkpoint `args.yaml` 中的 `state_dim=14`、`context_q_dim=14`、`raw_context_dim=40`、`context_ee_goal_pose_bimanual=true`、网络 variant、17 个 learnable control points、dataset/robot/scene hash；不允许仅凭目录名加载。

## 4. 冻结的双 EE inference 契约

### 4.1 唯一关节顺序

所有 MPD、NPZ、Isaac Lab、IPC、ROS action 和 JTC 均固定为：

```text
[Joint1_L, Joint2_L, Joint3_L, Joint4_L, Joint5_L, Joint6_L, Joint7_L,
 Joint1_R, Joint2_R, Joint3_R, Joint4_R, Joint5_R, Joint6_R, Joint7_R]
```

任何输入都按名称重排到该顺序；缺失、重复或未知关节直接失败。结果中也必须原样携带该列表，不能只靠维度判断。

### 4.2 网络输入

首版运行时统一构造：

| 字段 | 形状 | 说明 |
|---|---:|---|
| `q_start` | `[14]` | ROS 最新关节状态，左 7 后右 7 |
| `ee_goal_pose` | `[2,3,4]` | 固定槽位顺序 `[left,right]`，旋转矩阵和平移 |
| `active_ee_mask` | `[2]` | `dual_independent` 固定为 `[1,1]` |
| raw context | `[40]` | `q_start14 + rotations18 + positions6 + mask2` |
| model output | `[N,17,14]` | learnable B-spline control points |

外部 ROS/JSON pose 使用 `[x,y,z,qx,qy,qz,qw]`；进入模型前统一转成 `[3,4]`，四元数先归一化并拒绝零范数。

`q_goal` 不是该 checkpoint 的网络条件。若请求提供 `q_goal`，只允许用于以下用途之一：

- 从 FK 生成两个 EE goal；
- 作为 goal IK/posture ranking 的参考解；
- 记录可重复性元数据。

EE-context B-spline 的终点控制点是模型变量，不能再把 IK 的 `q_goal` 当作硬位置边界覆盖它。

### 4.3 请求与结果

在真正实现前将当前 provisional `marvin_bimanual_request/v1` 固化为严格版本；至少包含：

```json
{
  "schema": "marvin_bimanual_request/v2",
  "request_id": "uuid",
  "task_mode": "dual_independent",
  "runtime_mode": "snapshot_no_time",
  "robot_model": "marvin_bimanual",
  "planning_frame": "world",
  "joint_names": ["Joint1_L", "...", "Joint7_R"],
  "q_start": [0.0],
  "q_velocity_start": [0.0],
  "left_goal_pose": {"frame_id": "world", "pose_xyzw": [0.0]},
  "right_goal_pose": {"frame_id": "world", "pose_xyzw": [0.0]},
  "scene_id": "EnvWarehouseMarvinBimanual",
  "scene_version": "marvin_warehouse_v2",
  "scene_hash": "sha256",
  "world_version": 0,
  "deadline_unix_ns": 0,
  "seed": 12345
}
```

数组示例中的 `[0.0]` 只表示字段位置；schema validator 必须要求准确的 14 或 7 个值。`dual_independent` 必须同时提供左右 goal，禁止当前 v1 中“只要任意一个 goal 即通过”的宽松判断。

结果采用 `result.json + trajectory.npz`：

- JSON 保存状态、版本、hash、双 EE 误差、clearance、候选统计和各阶段耗时。
- NPZ 保存 best 与 Top-K 的 `positions/velocities/accelerations/time_from_start`。
- 每条轨迹形状为 `[T,14]`；时间从 0 开始、严格递增。
- `status` 统一为 `success/no_valid_trajectory/invalid_request/stale/deadline_exceeded/fault`，不要同时维护 `ok`、`OK` 和 `success` 三套业务语义。
- 动态结果必须记录 `request_seq`、`world_version`、snapshot stamp、`valid_until` 与 trajectory start time。

### 4.4 对通用 planner 的最小兼容改动

不复制一份完整 `GenerativeOptimizationPlanner`。建议做一个向后兼容的 goal bundle 扩展：

```python
plan_trajectory(
    q_pos_start,
    q_pos_goal,
    ee_pose_goal,          # 单臂 [3,4]，双臂 [2,3,4]
    active_ee_mask=None,   # 单臂保持 None，双臂必须 [2]
    ...,
)
```

需要连通三处：

1. `dataset.create_data_sample_normalized(..., active_ee_mask=...)`；
2. planning task 的双 goal setter；
3. candidate ranking/metrics 的双 EE 分支。

原 Panda 输入与输出必须逐位保持不变，并用现有 Franka inference tests 做回归门禁。

## 5. `dual_independent` cost 设计

### 5.1 总目标

首版不加入闭链与 payload：

```text
C_static = w_env   C_robot_world
         + w_intra C_intra_arm_self
         + w_inter C_inter_arm
         + w_lim   C_joint_limit
         + w_vel   C_velocity
         + w_acc   C_acceleration
         + w_len   C_path_length
         + w_goal  C_dual_ee_terminal
```

其中 cost guide 负责把样本推向可行域，dense validator 才负责最终安全判定。没有有效 candidate 时必须失败，不能返回总 cost 最低但发生碰撞的轨迹。

### 5.2 双 EE goal cost

Franka 的 SE(3) 误差与 Jacobian transpose 计算可以复用。对手臂 `a∈{L,R}`：

```text
e_a = Log(T_goal,a · inv(T_ee,a(q_H))) = [rho_a, phi_a]

C_pos  = (1 / max(sum(mask),1)) Σ_a mask_a · 0.5 ||rho_a / sigma_pos||²
C_ori  = (1 / max(sum(mask),1)) Σ_a mask_a · 0.5 ||phi_a / sigma_ori||²
C_goal = w_pos C_pos + w_ori C_ori
```

这样从单臂 mask 扩展到双臂时不会仅因 active goal 数翻倍而改变 guidance 量级。首版 `dual_independent` 的 mask 恒为 `[1,1]`。

实现要求：

- 左右 goal 固定槽位，不按字典或收到顺序拼接。
- 只在末端轨迹点计算两次 EE FK/Jacobian，保留 Franka 的 endpoint-only 策略。
- `RobotMarvinBimanual.jfk_left/right()` 当前各返回 6x7；cost 层应明确 scatter 为左 `[..., :7]`、右 `[..., 7:]` 的 6x14 梯度，或新增经过单测的 full-14 dual EE API。
- 位置与旋转分别配置权重和尺度；不能用同一数值直接相加米与弧度。
- guide 使用 active-arm 平均，最终 validator 分别检查左右臂，任意一侧超限即 invalid。
- ranking 不能只对两臂误差求和，否则可能牺牲一臂。建议用“归一化平均值 + 最差臂项”：

```text
S_goal = lambda_mean · mean(E_left,E_right)
       + lambda_max  · max(E_left,E_right)
```

其中 `E_a` 是位置和姿态的归一化误差。初期令 `lambda_max > 0`，具体数值通过验证集标定。

### 5.3 碰撞、自碰撞与臂间碰撞

双臂不能把 `inter-arm` 当成附属诊断：

- robot-world：复用静态 SDF 与所有 Marvin/Pika collision spheres。
- intra-arm self：左臂内部和右臂内部合法 pair。
- inter-arm：左右 link pair，单独统计 clearance 和失败码。
- 底座/固定结构：按自碰撞 pair 配置纳入，不允许由于“双臂独立任务”而关闭。

现有 self-collision field 可以继续做统一梯度计算，但 pair metadata 必须能把 dense 结果拆成 `minimum_left_self_clearance`、`minimum_right_self_clearance` 和 `minimum_interarm_clearance`。生产阈值应由碰撞模型误差和 Isaac/真实机器人标定得到，不直接把 `interarm_clearance_cost(..., margin=0.08)` 的占位默认值当作安全规范。

### 5.4 限位、动力学与冗余姿态

- joint limit、velocity、acceleration cost 可按 14D 广播复用，但要使用 Marvin 的逐关节 limits。
- 所有 cost 先按 active waypoint、sphere/pair 数做归一化，避免碰撞球数量增加导致权重失真。
- 可增加弱 `terminal_posture` 或 manipulability cost，解决同一双 EE goal 的冗余解选择；默认关闭，先证明纯双 EE context 能稳定工作。
- 输出轨迹统一做 14D 重定时，两臂共享 `time_from_start`。

### 5.5 初始权重策略

Franka 的数值只能作为 ablation 起点，不能直接作为 Marvin 最终配置。推荐调参顺序：

1. 关闭 guidance，测网络 prior 的双 EE 误差和有效率。
2. 只开 `C_goal`，调到左右终点误差满足门限且不出现梯度爆炸。
3. 加 robot-world。
4. 加 intra/inter-arm；单独观察 inter-arm recall。
5. 加 joint/velocity/acceleration。
6. 最后调 path length/smoothness 与候选 ranking。

每一步使用相同 seed、相同请求集、相同候选数，报告 success rate、左右 EE P50/P95、各类碰撞 false-negative、延迟和峰值显存。

## 6. Franka 默认 B3 剪枝的复用边界

建议先实现一个小 batch、全量计算的 oracle，再启用 B3；B3 通过等价性测试后才成为默认路径。

| B3 项 | 双臂处理 | 结论 |
|---|---|---|
| `endpoint.ee_only_last_point=true` | 同一末端点计算左右两个 EE | 直接复用策略，需双 EE 单测 |
| `parent_link_kinematics=true` | Marvin 已有 canonical parent-link FK/Jacobian | 可复用 |
| `dense_parent_fast_path=true` | 保持 `[candidate,T,14]` dense layout | 可复用，重测显存 |
| `active_link_pruning=true` | 只跳过零碰撞梯度的 `J^Tg` | 可复用，覆盖 intra/inter pair |
| `compute_costs_with_xrecon=false` | 与 Franka 一致 | 可复用 |
| candidate pruning | 继续关闭 | 首版不启用 |
| temporal/conditional pruning | 继续关闭 | 首版不启用 |
| parent bounds/span certificate | 继续关闭 | 首版不启用 |
| link broad phase/cache | 继续关闭 | 首版不启用 |
| fused/sparse B-spline mapping | 继续关闭 | 首版不启用 |
| ranked dense early exit | 静态可在全量 oracle 后启用；动态必须全候选检查 | 分阶段启用 |

必须增加以下等价门禁：

- legacy/full 与 B3 的双 EE cost、14D gradient、更新后 control points 对齐。
- 双 EE endpoint-only 与全时间计算在终点梯度上对齐。
- parent-link 与 fine-sphere collision gradient 对齐。
- inter-arm pair 在 active-link pruning 前后 collision decision 完全相同。
- 同一 seed 下 valid mask 和 best candidate 一致；浮点差异使用预先固定的容差。

性能上不能假设 B3 对 Marvin 足够。Marvin/Pika fine collision spheres 超过 Panda，B3 仍会对全部候选和时间点执行 SDF；初期从较小 `n_trajectory_samples` 开始，测得显存后再提高。任何进一步的 candidate/temporal/span pruning 都单独做消融，不与首个可运行版本绑在一起。

## 7. 分阶段实施

### Phase 0：契约和可重复性门禁

目标：在运行 checkpoint 前消除模型、数据、场景与接口漂移。

工作项：

- 新增 Warehouse 双 EE runtime YAML，指向实际训练目录；不复用旧 `EnvMarvinTable` 文件名。
- 固化 request/result v2、joint order、pose format、frame、scene version/hash。
- checkpoint loader 校验网络 A/B/C/D variant 和 40D context 元数据。
- 为通用 planner 增加 `active_ee_mask` 的向后兼容传递。
- 将 `BimanualPlanningTask` 接到 loader，或提供功能等价的生产 task adapter；不能继续只使用当前轻量 dataclass facade。
- 建立 10～20 个固定 Warehouse 请求的 golden suite，覆盖左右目标交换、接近臂间碰撞和冗余 IK。

验收：

- context shape 精确为 `[N,40]`，左右槽位交换测试能够检测错误接线。
- 单臂 Franka 的既有 inference/contract tests 不回归。
- 同一 checkpoint、seed、请求产生可重复的候选初值和一致的 metadata。

### Phase 1：MPD-only Warehouse 单次静态推理

主要改造：

- `scripts/inference/inference_marvin_bimanual.py`：移除线性插值作为默认结果，接入真实 loader、checkpoint、diffusion sampling、双臂 guide、ranking 和 dense validation。线性插值仅可保留为显式 `--backend contract_stub` 测试选项。
- `mpd/bimanual/cost_guide.py`：实现真正的 14D cost manager，并复用通用 B-spline/collision gradient 工具。
- `mpd/bimanual/costs.py`：新增 dual EE position/orientation cost 与 pair-aware collision helpers。
- `mpd/bimanual/trajectory_validator.py`：接通 environment/self/inter-arm/limits/双 EE dense 检查。
- 新增 `config_EnvWarehouse-RobotMarvinBimanual-independent-runtime.yaml`。

单次流程：

```text
request v2
  → 严格验证/坐标变换
  → q_start + dual EE + [1,1]
  → diffusion candidates
  → static bimanual guide
  → 14D spline reconstruction
  → 全量/Top-K dense validation
  → balanced dual-EE ranking
  → result.json + trajectory.npz + scene.json
```

验收：

- best trajectory 起点与请求一致，速度/加速度边界满足配置。
- 左右 EE 均达到门限；报告分别给出 left/right，不只给平均值。
- Warehouse、intra-arm、inter-arm 均无 dense collision。
- 无有效轨迹返回结构化失败。
- full oracle 与 B3 的正确性等价测试通过后，B3 才可设为 runtime 默认。

### Phase 2：Isaac Lab 检测与回放

当前 `evaluate_mpd_trajectories.py` 和 `replay_mpd_trajectory.py` 是 Panda-only。为避免破坏 Franka，优先新增 Marvin 专用薄入口和共享 loader，而不是在脚本中堆积机器人分支：

```text
scripts/isaaclab/marvin_bimanual_asset.py
scripts/isaaclab/evaluate_marvin_bimanual_trajectories.py
scripts/isaaclab/replay_marvin_bimanual_trajectory.py
```

先决条件：

- 提供由规范 Marvin/Pika 描述转换并固化 hash 的 USD；若尚无 USD，该资产转换是本阶段的硬门禁。
- 明确 Isaac joint names 到 MPD canonical 14D 的映射。
- 明确 `left_pika_gripper_tcp`、`right_pika_gripper_tcp` 对应的 Isaac body。
- scene payload 继续复用 `export_isaaclab_scene_payload(..., include_boxes=True)`，并验证 Warehouse box 的 pose/size/frame。

检测器读取 Phase 1 的 artifact，至少输出：

- 每条 Top-K trajectory 的 contact/no-contact；
- left-world、right-world、inter-arm 接触分类；
- joint limit 和 tracking error；
- Isaac 双 TCP 终点误差；
- MPD dense 判定与 Isaac 判定的 confusion matrix；
- robot USD、scene payload 和 trajectory 的 hash。

回放器同时绘制左右 TCP 轨迹、目标 frame、发生碰撞的首个 waypoint，并可导出 mp4 与最终截图。

验收：

- 14D 轨迹不会误写到 gripper/mimic joints。
- 起点 pose 和 MPD FK 与 Isaac FK 在约定门限内。
- golden suite 中 MPD 与 Isaac 的安全判定无 false-negative；若几何近似造成差异，先扩大 MPD margin，不通过忽略接触解决。

#### Phase 1/2 直接入口的起终点来源

`scripts/inference/inference_marvin_bimanual.py` 是 Phase 1/2 的直接入口，
`--request` 可选。不提供现成 request 时，入口根据配置中的
`start_goal_source`（也可用同名 CLI 参数覆盖）生成严格 v2 request，并将实际使用的
request 保存到输出目录的 `request.json`。三个生产来源为：

- `dataset`：读取 `dataset_subdir/dataset_file_merged`，只在
  `task_mode=dual_independent` 的行中按 `--sample-index` 选择，并直接读取该行的
  `q_start`、`q_goal` 和双 EE goal pose。
- `states_file`：读取 14D `q_start/q_goal` 列表。每个向量严格采用左臂 7 维在前、
  右臂 7 维在后的 canonical 顺序；未提供 EE pose 时由 `q_goal` 做双臂 FK。
- `regions`：读取左右臂的 Warehouse 区域约束，复用生产数据生成器的
  Pinocchio IK、PyBullet/torch 几何和 endpoint collision gate，生成有效的 14D 起终点。

仓库内样板：

```text
scripts/inference/cfgs/start_goal_states/EnvWarehouse-RobotMarvinBimanual-states.yaml
scripts/inference/cfgs/start_goal_regions/EnvWarehouse-RobotMarvinBimanual-regions.yaml
```

Phase 1，使用配置默认的 `states_file`：

```bash
conda run --no-capture-output -n mpd-splines-public \
  python scripts/inference/inference_marvin_bimanual.py \
  --config scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-independent-runtime.yaml \
  --sample-index 0 \
  --output-dir /tmp/marvin-phase1-states \
  --device cuda:0
```

Phase 1，分别使用训练数据或区域采样：

```bash
conda run --no-capture-output -n mpd-splines-public \
  python scripts/inference/inference_marvin_bimanual.py \
  --config scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-independent-runtime.yaml \
  --start-goal-source dataset --sample-index 0 \
  --output-dir /tmp/marvin-phase1-dataset --device cuda:0

conda run --no-capture-output -n mpd-splines-public \
  python scripts/inference/inference_marvin_bimanual.py \
  --config scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-independent-runtime.yaml \
  --start-goal-source regions \
  --start-goal-file scripts/inference/cfgs/start_goal_regions/EnvWarehouse-RobotMarvinBimanual-regions.yaml \
  --seed 12345 --sample-index 0 \
  --output-dir /tmp/marvin-phase1-regions --device cuda:0
```

Phase 2 在同一条直接命令上增加 Isaac Lab 检测与回放：

```bash
conda run --no-capture-output -n mpd-splines-public \
  python scripts/inference/inference_marvin_bimanual.py \
  --config scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-independent-runtime.yaml \
  --start-goal-source states_file \
  --start-goal-file scripts/inference/cfgs/start_goal_states/EnvWarehouse-RobotMarvinBimanual-states.yaml \
  --sample-index 0 --output-dir /tmp/marvin-phase2-states \
  --device cuda:0 --sim-backend isaaclab \
  --isaaclab-conda-env env_isaaclab --isaaclab-device cuda:0
```

`--sample-index -1` 表示让 `dataset/states_file` 按 `--seed` 做确定性选择；
`regions` 用 `seed + sample-index` 形成可复现的采样流。旧的
`--request /path/request.json` 调用仍受支持。Phase 3 的
`scripts/runtime/infer_once_marvin_bimanual.py` 以及 Phase 4/5 的 socket/ROS
请求契约保持不变，仍要求调用方显式提供 request。

### Phase 3：ROS 2 单次规划连接

先复刻 Franka `send_mpd_trajectory.py` 的隔离进程方式，不立即引入动态 worker：

```text
ROS JointState + 双 Pose goal
  → one-shot demo node
  → 文件 request.json
  → 独立 MPD Conda 进程 infer_once_marvin_bimanual.py
  → result.json / trajectory.npz
  → ROS 侧再次校验
  → plan-only 输出或一个 14-joint FollowJointTrajectory goal
```

工作项：

- 将 `scripts/runtime/infer_once_marvin_bimanual.py` 改造成与 Franka `infer_once.py` 同等级的严格 one-shot 入口，而不是转调线性插值脚本。
- 在 `physical_ai_runtime/src/apps/marvin_mpd_bimanual_bringup` 增加 one-shot demo；默认 `plan_only=true`。
- 从最新 `JointState` 按 canonical names 构造 `q_start`，检查消息 age、有限值和 14 个关节完整性。
- 目标 pose 必须变换到 `world` planning frame，并记录 TF timestamp。
- 子进程移除 ROS `PYTHONPATH/PYTHONHOME`，沿用 Franka Conda 隔离方式。
- ROS 侧重新检查 trajectory shape、有限值、起点误差、时间单调、result/checkpoint/scene hash。
- 执行时只向 `/bimanual_arm_jtc/follow_joint_trajectory` 发送一条 14-joint goal；任意一臂 fault 取消整条 goal。

三步放行：

1. `plan_only=true`：只生成与检查 artifact。
2. fake hardware：执行小幅、宽 clearance 的 golden trajectory。
3. real hardware：必须由现场人员明确启用，保留急停和低速限制；不作为自动默认。

验收：

- ROS 请求、MPD artifact、Isaac 回放使用同一个 `request_id`。
- 规划期间 joint state 漂移超过门限时结果作废，不下发旧起点轨迹。
- fake hardware 两臂使用同一时间基准，action 成功/取消对 14 关节是原子的。

### Phase 4：常驻静态 worker

one-shot 闭环稳定后再消除模型重复加载：

- 把 `runtime_engine_marvin_bimanual.py` 从 planner callback wrapper 改成真正持有 planning task、dataset、checkpoint、GPU planner 的 resident engine。
- 新增 Marvin static Unix-socket server/client，复用 `ipc_protocol.py` 的长度前缀 JSON、`health/plan/shutdown`、`request_seq`、deadline 与原子 artifact 写入。
- 删除当前“配置指向 socket、worker 却读取 stdin”的协议不一致。
- ROS adapter 进程不 import torch/CUDA；GPU 只由 worker 的一个非重入规划区拥有。
- health 返回 model/config/scene/robot hash、warmup 时间、GPU device 和 dense/B3 开关。

验收：

- worker 启动只加载一次模型，连续请求不增长 GPU 内存。
- 同时请求时只允许一个 planner 操作 CUDA；其他请求明确返回 BUSY/STALE，不能并发踩 buffer。
- one-shot 与 resident 对相同 seed/request 的输出和 validator 结果一致。

### Phase 5：动态世界的 latest-only 实时重规划

第一版动态模式为 `snapshot_no_time`：障碍物在每次 plan 时冻结为当前静态快照，不做未来位置预测。这里的“实时”定义为有截止时间的在线 replan，而不是硬实时线程。

ROS 状态机：

```text
IDLE
  → PLAN(seq, world_version, deadline)
  → TOP_K_VALIDATE
  → REVALIDATE_LATEST_WORLD
  → HANDOFF_OR_REJECT
  → EXECUTING
      ├─ newer world / unsafe prefix → invalidate old generation → REPLAN
      ├─ warning threshold           → controlled brake / hold
      └─ goal reached                → IDLE
```

工作项：

- 将 `DynamicWorld.msg` 转为 worker 的 fixed-capacity world schema，严格处理 frame、stamp、`valid_until`、shape、twist、covariance 和 inflation。
- 完成 `LatestOnlyReplanCoordinator`：新 generation 使旧结果失效；Python 无法中断的 GPU kernel 可运行完，但结果不得下发。
- adapter 在 worker 返回 Top-K 后使用最新 world 再验证；`world_version` 不一致或 snapshot 过期则丢弃。
- execution prefix 持续做短时预测；warning 触发 replan，brake threshold 触发整条 14D goal 的受控制动。
- 独立任务允许在低速状态做统一 14D bridge；首版若 bridge 的安全性无法证明，则采用 cancel → hold → replan → resume。
- 记录 world timeline、plan generation、handoff、取消原因和实际 joint states，供 Isaac 动态回放。

建议从 Franka `mpd_dynamic_planner_adapter` 移植经过测试的 candidate selector、execution prefix、handoff/braking、replay recorder 和 revalidation diagnostics；不要只扩展当前几十行的 Marvin skeleton。

验收：

- 乱序/重复 `world_version` 被拒绝。
- deadline 后返回的 trajectory 永不执行。
- 新目标或新世界到达后，旧 generation 即使随后成功也被丢弃。
- 动态障碍进入警戒区时两臂一起 replan 或一起停止，不出现单臂继续执行。
- 动态场景日志可在 Isaac Lab 中确定性回放。

### Phase 6：固定时间动态碰撞

在 snapshot replan 稳定后，引入 `fixed_time_dynamic`，作为时空优化前的独立基线：

- 复用 Franka `FixedCapacityDynamicWorld`、`StaticDynamicCollisionField` 和恒速预测。
- 两臂所有 collision spheres 使用同一固定 `time_from_start[T]` 查询 `O(t)`。
- covariance/inflation 进入障碍安全包络。
- guide 后使用完全相同的时间表做全候选 dense check；动态模式关闭 ranked early-exit。
- 如果后续 TOPPRA/Ruckig 改变时间表，必须按新时间重新验证，否则结果无效。

该阶段仍只优化空间控制点 `P`，不优化时间。它用于隔离回答“失败来自动态预测，还是来自 timing 优化”。

验收：

- 静止动态物体退化为静态结果。
- 恒速物体的解析位置与查询时间逐点一致。
- 所有候选都完成 candidate-specific dynamic dense check。
- snapshot、fixed-time 两种模式使用不同 socket/config，能独立回滚。

### Phase 7：推理期时间维度 cost guide

最后实现 `inference_time_optimized`。空间 diffusion checkpoint 保持不变，每个空间 candidate 新增自己的单调 timing spline `t(s;c)`：

```text
C(P,c) = C_static(P)
       + w_dyn C_dynamic(P, t(s;c))
       + w_vel C_velocity(P,c)
       + w_acc C_acceleration(P,c)
       + w_T   duration(c)
       + w_ts  timing_smoothness(c)
```

约束：

- `dt/ds > 0`，duration 在配置的 `[T_min,T_max]` 内。
- 左右臂共享一个 timing spline，禁止各自等待或各自重定时。
- 动态碰撞、速度、加速度都使用 candidate-specific time；不能用固定时间的缓存冒充。
- 首版关闭 candidate-specific dynamic pruning，保留静态 B3；等价与性能证据充分后再优化。
- full dense check 的时间网格必须与 timing evaluation 网格一致。
- ranking 同时考虑 balanced dual-EE、静态空间质量、dynamic risk、duration 和 timing smoothness。

改造 `BimanualSpaceTimeCostGuide` 和 `MarvinBimanualSpaceTimeRuntimeEngine` 时复用 Franka 的 `InferenceOnlySpaceTimeGuide`、timing contract 和 schema v3 artifact；只扩展 14D/双臂 cost，不重新设计时间优化器。

验收：

- 关闭动态物体时结果退化到静态空间规划加合法 timing。
- 与 fixed-time 基线相比，等待/加速绕行场景的成功率提高，且不存在 collision false-negative。
- 每个 Top-K candidate 都保存自己的 timestamps、duration 和 timing control points。
- 执行前使用最新 world 和候选自己的时间表再次复验。

## 8. 文件级实施清单

### MPD 仓库

| 文件 | 动作 |
|---|---|
| `mpd/bimanual/runtime_contract.py` | 固化 v2 schema、严格双 goal/frame/hash/deadline/result 验证 |
| `mpd/bimanual/planning_task.py` | 从轻量容器补成可接通现有 PlanningTask/collision field 的双 goal task |
| `mpd/bimanual/costs.py` | 双 EE、pair-aware intra/inter-arm cost；保留后续 cooperative cost |
| `mpd/bimanual/cost_guide.py` | 真正的 static 14D guide，复用 B-spline 和 B3 工具 |
| `mpd/bimanual/trajectory_validator.py` | full dense validation 与结构化双臂诊断 |
| `mpd/inference/inference.py` | 最小向后兼容扩展：传 mask、双 EE ranking hook |
| `mpd/utils/loaders.py` | 双 EE 模式选择 production bimanual task |
| `scripts/inference/inference_marvin_bimanual.py` | 真实 MPD-only 单次推理 |
| `scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-independent-runtime.yaml` | 新增当前 Warehouse v3 runtime 配置 |
| `scripts/runtime/infer_once_marvin_bimanual.py` | 严格 one-shot + artifact |
| `scripts/runtime/runtime_engine_marvin_bimanual.py` | 常驻静态 engine |
| `scripts/runtime/infer_server_marvin_bimanual.py` | 新增长度前缀 Unix socket service |
| `scripts/runtime/infer_client_marvin_bimanual.py` | 新增 health/plan/shutdown CLI |
| `scripts/runtime/dynamic_runtime_engine_marvin_bimanual.py` | snapshot/fixed-time 动态 engine |
| `scripts/runtime/infer_dynamic_server_marvin_bimanual.py` | 用 resident service 替换 stdin placeholder |
| `scripts/runtime/space_time_runtime_engine_marvin_bimanual.py` | 14D candidate-specific timing |
| `scripts/runtime/infer_space_time_server_marvin_bimanual.py` | 独立时空 socket service |
| `scripts/isaaclab/*marvin_bimanual*` | Marvin asset、检测与回放入口 |

### physical_ai_runtime 仓库

| 文件/包 | 动作 |
|---|---|
| `manipulation_planning_interfaces` | action/msg 版本与严格 contract 对齐；保持字段向后兼容或显式升版 |
| `mpd_bimanual_planner_adapter/backend.py` | 完成 health/update-world/plan/result artifact 读取 |
| `ipc_client.py` | 复用统一 `ipc_protocol` 语义、大小/超时/状态校验 |
| `node.py` | 完成 action、JointState、TF、worker、plan-only、执行与取消 |
| `latest_world_buffer.py` | frame/shape/covariance/valid_until 验证和 worker payload 转换 |
| `replan_coordinator.py` | latest-only generation、deadline、旧结果抛弃 |
| `trajectory_adapter.py` | 14D positions/velocities/accelerations/timestamps 全字段转换 |
| `collision_guard.py` | 前缀预测、warning/brake 两级策略、任一臂 fault 联动 |
| `config/*.yaml` | static/snapshot/fixed-time/space-time 分离 socket 与参数 |
| `marvin_mpd_bimanual_bringup` | one-shot demo、fake-hardware 联调、14-joint 原子 action |

已有骨架应原地完善；Franka package、socket、launch 和 checkpoint 路径保持不变。

## 9. 测试与验收矩阵

| 层级 | 必测内容 | 通过条件 |
|---|---|---|
| Context | 左右 pose/mask/normalization/variant A-D | 40D 契约一致，左右不串槽 |
| EE gradient | 双臂解析梯度 vs finite difference | 在固定容差内一致，非本臂 7 列为零 |
| B3 | full vs B3 cost/gradient/valid/best | decision 一致，无 inter-arm 漏检 |
| Static MPD | golden Warehouse 请求集 | 双 EE 与安全门限通过，无效时 fail closed |
| Isaac | MPD vs Isaac FK/contact | 无安全 false-negative，hash 可追踪 |
| Contract | malformed JSON/shape/frame/hash/deadline | 全部结构化拒绝 |
| IPC | partial frame/oversize/BUSY/STALE/restart | 不崩溃、不返回错请求结果 |
| ROS plan-only | JointState/TF/start drift/result | 旧状态或漂移结果不下发 |
| Fake execution | 14-joint goal/cancel/fault | 原子开始、原子取消、两臂同时间轴 |
| Dynamic | out-of-order world/latest-only/deadline | 只执行最新、有效快照的结果 |
| Space-time | monotonic timing/full dense check | 每 candidate 时间合法并安全 |

建议报告的核心指标：

- planning success rate；
- left/right EE position/orientation P50、P95 与 worst-arm error；
- environment、left self、right self、inter-arm 最小 clearance；
- joint/velocity/acceleration 最大利用率；
- generator、guide、dense check、ranking、IPC、ROS total latency；
- warm/cold latency、P50/P95/P99、峰值 GPU 显存；
- replan trigger-to-new-goal latency、stale/busy/deadline rate；
- MPD 与 Isaac collision confusion matrix。

## 10. 建议的提交拆分

为便于回归和回滚，按以下顺序独立提交：

1. contract v2、goal bundle、双 EE tensor tests；
2. dual EE cost/full-gradient oracle；
3. B3 双臂适配与等价测试；
4. MPD-only Warehouse inference 与 artifact；
5. Isaac Lab Marvin asset/evaluate/replay；
6. one-shot ROS plan-only；
7. fake hardware 14D 原子执行；
8. resident static worker 与 ROS action adapter；
9. snapshot latest-only replan；
10. fixed-time dynamic guide；
11. inference-only timing spline。

每个提交都应保持 Franka tests 通过；动态和 space-time 使用独立入口与 socket，任何阶段都可以退回上一个已验证模式。

## 11. 第一轮实现的推荐切片

下一步只做 Phase 0 + Phase 1，不同时改 ROS 和动态世界。最小交付物为：

```text
Warehouse 双 EE runtime config
+ strict request/result v2
+ plan_trajectory active_ee_mask 接线
+ dual EE terminal cost
+ 14D static collision/limits guide
+ full dense validator
+ real inference_marvin_bimanual.py
+ 10～20 个 golden requests
+ full-vs-B3 equivalence tests
```

该切片通过后，Phase 2 的 Isaac Lab 才能作为独立物理引擎验证器；Isaac 结果稳定后再让 ROS 读取同一 artifact。这个顺序能把模型/cost 错误、仿真资产错误和 ROS 执行错误分开定位。
