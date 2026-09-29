# 将 ST-RRT* 接入 ToDrawer 随机动态世界 benchmark

本文记录已完成的接入及复现实验步骤。目标是在同一批 ToDrawer 场景中比较 MPD 与 OMPL ST-RRT*，复用 ROS fake hardware 执行、动态世界观测、轨迹拼接与安全检查、manifest 记录和 IsaacLab 回放。

## 当前实现与提交

接入位于两个独立的 `codex/strrt-benchmark` 分支，未改动原有 `mpd_dynamic_planner_adapter` ROS 包及其接口。

| 步骤 | 仓库 | 提交 | 内容 |
| --- | --- | --- | --- |
| 规划 worker | `MotionPlanningDiffusion/mpd` | `10d35b4` | ST-RRT*、碰撞模型、动态 IPC |
| ROS 新包 | `physical_ai_runtime` | `534b5e8` | `strrt_planner_adapter`、独立 artifact 解码、launch/config |
| 内核验证优化 | `MotionPlanningDiffusion/mpd` | `aff6cb9` | 批量边检查、目标/终端保持复核 |
| ROS 退出处理 | `physical_ai_runtime` | `a368c5e` | 中断时清理节点 |
| 计时拆分 | `MotionPlanningDiffusion/mpd` | `44984b3` | 分开记录 IK、预检、OMPL 求解、后处理 |
| 同框架 benchmark | `MotionPlanningDiffusion/mpd` | `c2f51e2` | pipeline 分支、配对场景、指标与报告 |

ST-RRT* 仅支持 `absolute_world_time`；现有默认 modes 未加入 `strrt`。同一 benchmark run 为 worker 固定一次 OMPL 随机种子（worker seed + 1），请求内的 seed 仍记录在产物中，但 OMPL 不允许在同一个进程中途重新设全局种子。`cost_mpd_weight` 在 ST-RRT* ROS config 为 0；复用节点的部分日志和 manifest 时序字段仍含 `MPD` 名称，表示共享执行管线，并非调用 MPD 模型。

## 1. 先固定比较边界

现有流程是 `benchmark_todrawer_random.py` → `run_dynamic_demo_pipeline.sh` → 常驻规划 worker + ROS fake hardware → `replay-manifest.json` → 可选 IsaacLab 回放。IsaacLab 回放发生在 ROS 执行之后；它不是闭环规划器的运行环境。ST-RRT* 应接在 worker/ROS 规划器位置，而不是只接在回放脚本里。

首轮使用 `--timing-protocol absolute_world_time`。这个协议让同一场景的障碍物在相同世界时刻穿越；`motion_aligned` 会按各 mode 的首轮规划时间平移穿越时刻，加入新 mode 还需要新的时间画像与校准。当前 `_mode_timing_shift()` **即使在 absolute 分支也会先查时间画像**，接入时须按第 6 节调整。比较时固定场景文件、目标位姿、起点、动态世界观测、ROS 规划预算、执行控制器及安全阈值。记录算法自己的求解耗时，同时单列 worker 往返和轨迹后处理耗时。

推荐最小实现：**一个独立 ST-RRT* worker + 一个 ROS backend + 一个 launch 分支 + 一个 benchmark mode**。保留现有 `MpdDynamicReplanNode` 的调度、桥接、最新世界复核、guard、刹车、JTC 和 recorder；不要改 MPD 模型或原有 mode 的结果。

## 2. 已核对的代码入口

| 环节 | 现有文件 | 接入动作 |
| --- | --- | --- |
| 成对场景、运行与报告 | `scripts/isaaclab/benchmark_todrawer_random.py` | 增加 `strrt` mode、启动参数与通用指标 |
| 单次运行编排 | `scripts/isaaclab/run_dynamic_demo_pipeline.sh` | 增加 `--phase strrt` 的 worker/launch 分支 |
| Worker 协议 | `scripts/runtime/infer_server.py`、`infer_dynamic_server.py`、`ipc_protocol.py` | 复用长度前缀 JSON、`health`/`update_world`/`plan`/`shutdown` 和结果落盘方式 |
| ROS 规划入口 | `physical_ai_runtime/.../mpd_dynamic_planner_adapter/mpd_dynamic_planner_adapter/replan_node.py` | 复用调度、handoff、guard、JTC、recorder |
| ROS backend 示例 | 同目录 `backend.py`、`space_time_replan_node.py` | 新增 ST-RRT* 解码器及 node 入口，不伪装成 MPD v3 artifact |
| 动态几何复核 | 同目录 `collision_guard.py`、`dynamic_world.py` | 复用障碍物位置预测、膨胀和碰撞球定义 |
| 回放记录 | 同目录 `replay_recorder.py` | 接收统一的 `TrajectoryPlanResult`，无需知道规划算法 |

文中 `physical_ai_runtime/...` 均指 `/home/eric/Projects/physical_ai_runtime/src/motion_planning/motion_planners`。

## 3. OMPL 环境验证

`mpd-splines-public` 默认搜索路径中的 `ompl` 只是仓库子目录的 namespace；直接 `import ompl.geometric` 会失败。实际编译好的绑定在 `deps/pybullet_ompl/ompl/py-bindings`，其中含 `STRRTstar`、`SpaceTimeStateSpace` 和 `MinimizeArrivalTime`。已用该绑定运行仓库自带的 `SpaceTimePlanning.py`，成功得到时空路径。

```bash
cd /home/eric/Projects/MotionPlanningDiffusion/mpd
OMPL_BINDINGS="$PWD/deps/pybullet_ompl/ompl/py-bindings"
PYTHONPATH="$OMPL_BINDINGS${PYTHONPATH:+:$PYTHONPATH}" \
LD_LIBRARY_PATH="/home/eric/anaconda3/envs/mpd-splines-public/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
/home/eric/anaconda3/envs/mpd-splines-public/bin/python -c \
  'from ompl import base as ob, geometric as og; assert ob.SpaceTimeStateSpace and og.STRRTstar; print("OMPL ST-RRT* ready")'
```

在 pipeline 的 **ST-RRT* worker 分支**设置上述 `PYTHONPATH` 和 `LD_LIBRARY_PATH`；现有 MPD worker 分支仍保持原有环境。不要只在交互 shell 设置，否则 `conda run` 启动的 worker 可能找不到绑定。

## 4. Worker 的规划问题

建议新增 `scripts/runtime/strrt_server.py` 和 `scripts/runtime/strrt_engine.py`。`strrt_engine.py` 只依赖 OMPL、机器人模型、静态场景、动态碰撞和必要的 IK，不加载 diffusion checkpoint。worker 对外保持现有 IPC 的 `schema_version=1`。

### 4.1 输入与时钟

从现有 `DynamicMpdWorkerClient` 请求读取。若复用 `DynamicResidentPlannerService`，它会将外层 `world_version`、`trajectory_start_unix_ns` 复制进 engine 请求的 `_dynamic_world_version`、`_trajectory_start_unix_ns`：

- `request.q_pos_start`、`q_vel_start`、`q_acc_start`、`joint_names`、`goal_type`、`ee_pose_goal` 或 `q_pos_goal`、`seed`；
- `world_version`、`trajectory_start_unix_ns`、`deadline_unix_ns`；
- 最近一次 `update_world` 的完整 `DynamicWorldSnapshot`，包含观测时间、有效期、几何、速度、协方差和膨胀参数。

定义 OMPL 状态为 `(q[7], τ)`，其中 `τ=0` 是**计划 handoff 时刻**，障碍物查询使用 `t_abs = trajectory_start_unix_ns × 1e-9 + τ`。不得把求解器启动时刻、世界观测时刻或场景起始时刻误当成轨迹零点。每个请求固定一份世界快照及版本；求解后由 ROS 继续按最新版本复核。

### 4.2 空间、目标与有效性

1. 7 维关节上下界使用与当前 FR3 任务一致的模型；创建 `RealVectorStateSpace(7)` 和 `SpaceTimeStateSpace`。时间上界首先取现有 14 s 轨迹范围，并裁到 `world.valid_until_unix_ns - trajectory_start_unix_ns`；上界非正时返回明确失败。`vMax` 只是关节空间的标量速度界，仍要逐关节核对真实速度限值。
2. 对 joint goal 直接使用给定关节状态；对本 benchmark 的 pose goal，用与 MPD 一致的目标坐标系和位姿容差求一组或多组 IK 解。预先记录 IK 耗时和失败原因；不能从 MPD 的结果轨迹反取 goal。目标到达时间应可变，勿把 `τ` 固定为零。可参考 `deps/pybullet_ompl/ompl/demos/SpaceTimePlanning.py` 的 ST-RRT* 建模方式。
3. `isValid(q, τ)` 同时检查关节界、机器人自碰撞、静态环境、以及动态障碍物在 `t_abs` 的碰撞。机器人碰撞球中心和半径应来自当前 MPD 使用的 FR3 碰撞模型，不能只检测末端；静态场景使用相同 ToDrawer 几何。动态障碍物使用快照中的位置、速度、姿态和膨胀规则，禁止读取场景文件中未来的真值轨迹作为规划 oracle。
4. 自定义 `MotionValidator`：拒绝 `τ₂ ≤ τ₁`、超速度的边；沿边以不大于现有 guard 的 `0.02 s` 时间间隔并结合关节空间分辨率采样，逐点检查 `isValid`。仓库自带 demo 的 validator 只查终点和平均速度，**不足以检测本任务中途穿越的障碍物**。
5. 先用求解器预算找一条精确解。ST-RRT* 是几何时空规划器，`vMax` 不会自动保证关节加速度、jerk 或移动中起点的导数连续。求解时间、IK、插值、平滑、碰撞球生成和产物写盘都计入实际 deadline；超时返回失败，不延长 ROS 的 2.5 s 规划预算。

### 4.3 从时空路径到可执行轨迹

输出 `q(t)`、`dq(t)`、`ddq(t)` 的严格递增时间采样，首点为请求 handoff 状态。把原始时空路径转为连续轨迹时，必须匹配请求的 `q_vel_start`/`q_acc_start`，检查逐关节速度、加速度、jerk、目标容差及终端保持；随后**对最终连续轨迹重新做静态、自碰撞和时空动态碰撞检测**。普通空间 shortcut、B-spline 拟合或事后重新定时都会改变到达障碍物的时间，不能沿用原路径的碰撞结论。

如平滑后的轨迹不可行，本次规划应返回 `PLAN_FAILED` 并记录 `postprocess_invalid`；不要把原始折线路径直接送给 JTC。首轮可把后处理失败率作为单独指标，用它定位几何规划与执行接口间的差距。

## 5. Worker 与 ROS 的最小契约

复用 `DynamicResidentPlannerService` 的请求锁、版本检查、deadline、`request-N/trajectory.npz`、`result.json`、`response.json` 和 health 生命周期。若复用该类，ST-RRT* engine 需实现它实际调用的 `health()`、`update_world()`、`plan()`、`instance_id`、`dynamic_world.world_version`，`plan()` 返回 `PlanArtifacts(result_payload, trajectory_arrays)`。health 应报告 `planner: "strrtstar"`，不要伪造 MPD 的 `dense_validation.fully_warmed`。

无解、后处理不合格、目标 IK 失败属于单次请求失败：使用现有 `RuntimeContractError` 派生异常或等价的 service 分支写出 `PLAN_FAILED`/`result.json`，并让 worker 回到 READY。不能把预期中的无解抛成普通异常，否则 `ResidentPlannerService` 会进入 FAULT，后续场景无法继续。

建议 ST-RRT* artifact 用独立、简单的 schema，并新增 ROS `StrrtGlobalTrajectoryBackend` 解码它；**无需填充虚假的 MPD top-K、score 或 timing schema v3**。最小成功产物：

| 字段 | 形状/要求 | 用途 |
| --- | --- | --- |
| `joint_names` | `(7,)`，顺序为 `fr3_joint1`…`fr3_joint7` | ROS/JTC 关节对应 |
| `positions`、`velocities`、`accelerations` | `(H,7)`，有限数值，`H≥2` | 拼接与执行 |
| `time_from_start` | `(H,)`，首点 `0`，严格递增 | 绝对 handoff 时间换算 |
| `collision_sphere_positions` | `(H,S,3)` | ROS 最新世界复核和 guard |
| `collision_sphere_radii` | `(S,)` | 与现有碰撞球模型一致 |

ROS backend 把它转换成 `TrajectoryPlanResult`，将碰撞球数组放入 `diagnostics`，并保留 `request_seq`、`world_version`、`handoff_unix_ns`、worker 耗时和产物路径。现有 `replan_node.py` 在没有 `top_k_candidates` 时会按单候选处理，缺失 `mpd_selection_score` 时默认 0；`collision_plan_from_result()` 必须能从 diagnostics 获得碰撞球。`warmup()` 对 ST-RRT* 只核验 worker READY、算法身份和几何模型版本。

新增 ROS node 入口和 launch，例如 `strrt_replan_node.py`、`replan_strrt.launch.py`、`replan_strrt_fake_hardware.launch.py`，沿用 `replan_space_time.yaml` 中的规划时限、桥接、guard 和刹车参数，并在 `setup.py` 注册入口。节点名称及 YAML 顶层 key 要匹配。唯一应改变的决策部分是后端返回的候选轨迹；若复用 `MpdDynamicReplanNode`，需在报告中如实标注共用的 MPD 命名评分项，并确保单候选的 MPD score 权重实际为零。

## 6. 接入 pipeline 与 benchmark

### Pipeline

在 `run_dynamic_demo_pipeline.sh` 中增加 `--phase strrt`：选择新 worker 脚本、新 socket 名和新 ROS fake-hardware launch；保留 `--world-scenario-file`、`--planner-seed`、`--plan-rate-hz`、`--duration-sec`、`--allow-brake`、`--skip-render`、static-scene export、scenario preflight、manifest/timing 验证。不要把 `--config` 的 diffusion 模型路径或 `--device cuda:0` 无条件传给 ST-RRT* worker；其模型、静态场景和规划参数应有自己的明确 CLI。`--skip-render` 仍应完成 ROS 评测及 manifest 验证。

### Benchmark

在 `benchmark_todrawer_random.py`：

1. `MODE_SPECS` 增加 `"strrt": ("strrt", None)`；`DEFAULT_MODES` 暂不加入，避免改变现有默认实验。
2. 调整 `_mode_timing_shift()`：`absolute_world_time` 在查当前 mode 的 `MODE_TIMING_PROFILES` **之前**返回零平移，并对所有 mode 使用同一个 parked-Franka 初始保护截止时间，例如 `MODE_TIMING_PROFILES["joint"].expected_goal_s - GOAL_CROSSING_RESERVE_S`；随后保留现有几何和启动安全检查。`generate_benchmark_suite()` 的 `reference_mode` 在 absolute 协议下也应固定为已有的 `joint`，而不是依赖 `--modes` 的首项。这样无需为绝对时间协议编造 ST-RRT* 的首轮规划画像，且交换 mode 顺序不会改变采样场景。
3. 仅对 ST-RRT* 增加必要的 CLI/`run-spec.json` 配置，例如 OMPL range、边检查间隔、求解预算和几何版本；其余场景、种子和运行次序沿用现有逻辑。
4. 让 `extract_run_metrics()` 接受 ST-RRT* `result.json` 的通用 `status`、`planner_name`、`solve_s`、`postprocess_s`、`total_s`。MPD 专属的 NFE、Corridor、top-K 等字段填 `null`，不要填 0。当前 `_parse_ros_log()` 用固定的 `dynamic MPD replanner started` 文本提取起始时间；新节点应改用共同的记录字段，或使解析器同时识别 ST-RRT* 日志。
5. 保持 `valid_dynamic_success` 的既有定义（到达、穿越前运动、无 brake），另外记录最终轨迹的碰撞/clearance 检查结果。此指标本身不代表“无碰撞”。
6. 增加新输出目录；`materialize_suite()` 会核对 `suite.json` 的 mode 列表，不允许向旧实验目录直接追加新 mode。

首轮只比较 `joint` 与 `strrt`，使用 `absolute_world_time`。这样同一 `scenario_id`/repeat 才能按相同障碍物时间表成对比较。若以后需要 `motion_aligned`，先为 `strrt` 加 `MODE_TIMING_PROFILES` 并用独立先导实验校准，随后重新生成整批 suite。

## 7. 分阶段验收

1. **依赖**：上面的 import 检查通过；运行 `deps/pybullet_ompl/ompl/demos/SpaceTimePlanning.py` 得到路径。此项已在本机验证。
2. **规划内核**：固定起点/终点且无障碍时得到精确解；移动障碍横穿时允许在安全时刻等待或绕行；插入边中点碰撞时 motion validator 必须拒绝；过期世界或不可达目标明确失败。
3. **产物契约**：用一条已知轨迹检查首点导数、时间严格递增、关节与碰撞球数组形状；将它交给 ROS `collision_plan_from_result()` 和 recorder，二者均通过。
4. **ROS 小试**：先 `plan_only:=true` 核对版本、deadline 和 latest-world guard，再用 `safe_control` 场景执行。确认 manifest、JTC handoff、guard 与刹车均有记录。
5. **配对 benchmark**：新输出目录、一个环境/类别和一个 repeat 做 smoke test；检查两 mode 的场景 JSON 内容与 planner seed，运行后检查 `runs.csv` 和 `report.md`。最后再扩大样本，并任选一组加 `--render` 检查 IsaacLab 回放。

接入完成后的示例命令：

```bash
cd /home/eric/Projects/MotionPlanningDiffusion/mpd
/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  scripts/isaaclab/benchmark_todrawer_random.py \
  --output-dir scripts/isaaclab/logs/todrawer-strrt-smoke \
  --modes joint strrt \
  --timing-protocol absolute_world_time \
  --categories safe_control \
  --environment-count-per-category 1 \
  --planner-repeats 1 \
  --duration-sec 35
```

实际比较报告至少列出：完成/manifest 数、目标到达率、`valid_dynamic_success`、brake、guard 拒绝、最终轨迹最小 clearance、首轮规划完成时间、求解及总延迟、关节路径长度与执行时长；同时保留原始失败原因和场景/seed，以便复查。

## 本机验收记录与限制

已验证编译版 OMPL 能导入 `STRRTstar`；固定关节目标得到严格递增、首点导数连续的轨迹。构造移动球体时，边两端有效而中途碰撞，批量边检测返回无效。worker 的一次成功请求返回 `OK`，不可达目标返回 `PLAN_FAILED`，健康状态仍为 `READY`。新 ROS 包通过 `pixi run build --packages-up-to strrt_planner_adapter` 构建并能导入。

使用 `safe_control` 的 `scenario-005`、35 秒、同一 seed、`--skip-render` 做了 1 对 1 smoke test。两个 mode 使用的场景文件 SHA256 相同；两次 pipeline 都完成并产生 manifest、`runs.csv` 与 `report.md`，均到达目标且无刹车、无命令间隙。ST-RRT* 有求解和后处理失败请求，按现有 deadline 规则拒绝；`valid_dynamic_success` 在此单例中两个 mode 都是 false。这个单例只证明接线、记录与评测可用，不代表两算法统计性能。IsaacLab 视频回放及更多场景尚未验收。
