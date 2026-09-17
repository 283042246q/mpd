# ToDrawer 随机动态 Benchmark 重设计与验收方案

> 适用仓库：
>
> - `283042246q/mpd`
> - `283042246q/physical_ai_runtime`
>
> 核对基线：2026-09-17 远程 `main`
>
> 目标：把当前 ToDrawer 随机动态 benchmark 从“元数据宣称 execution-synced、实际 runtime 是 world-node-start clock”的混合状态，改为**明确的 world-clock 动态场景**：场景加载且机器人状态可用后动态物体立即运动；Kalman 在机器人首次规划前已有多帧观测并能够估计速度；动态障碍在机器人真实执行窗口内穿越 robot-aware anchors；不同类别的 simultaneous / staggered / mixed 时序随机性受约束且可解释；生成阶段即可排除明显的静态环境/机器人基座不可行场景。

---

## 0. 先区分“当前代码事实”和“本方案建议”

### 当前远程代码事实

1. `benchmark_todrawer_random.py` 当前主 anchors 为：

   ```python
   BASE_CROSSINGS = (
       ((-0.68831415, -1.22503140, 0.65347225), (0.61993897, 0.78465003, 0.0)),
       ((-0.02146948, 0.40876295, 0.46108561), (0.99858189, -0.05323737, 0.0)),
       ((-0.50, -0.30, 0.5), (0.0, 0.0, 1.0)),
   )
   ```

   但 scenario JSON 同时写着：

   ```text
   motion_clock = first_execution_accepted
   ARM_OPERATION_ARRIVAL_WINDOW_S = (0.9, 3.0)
   ```

2. 实际 `dynamic_world_demo` 并没有等待 `/execution_started`。节点初始化时直接：

   ```python
   self._started = time.time()
   ```

   之后以默认 10 Hz：

   ```python
   elapsed = time.time() - self._started
   ```

   更新并发布动态物体。

3. 当前 `replan_node.py` 中没有 `/execution_started` 发布逻辑。因此 benchmark JSON 的 `first_execution_accepted` 语义与 runtime 实际行为不一致。

4. 动态世界状态估计使用 `ConstantVelocityKalmanFilter`：

   ```text
   state = [px, py, pz, vx, vy, vz]
   ```

   新 track 第一帧速度初始化为 0，后续位置观测通过 Kalman update 更新速度；给 worker 的 snapshot 已包含 `linear_velocity` 与 `covariance_6x6`。

5. ToDrawer benchmark 默认通过 pipeline 覆盖 `plan_rate_hz=1.0`。

6. Phase-5 当前：

   ```text
   planning_budget_s = 2.5
   commit_margin_s   = 0.15
   command_lead_s    = 0.05
   bridge_minimum_duration_s = 0.20
   ```

   `replan_node.py` 中：

   ```python
   planning_deadline = now + planning_budget_s
   bridge_start = planning_deadline + max(command_lead_s, commit_margin_s)
   ```

   因此：

   ```text
   bridge_start - planning_submitted = 2.65 s
   ```

   benchmark 1 Hz 下首次 `_schedule()` 通常约 world t≈1 s，所以首次 bridge nominal 约 world t≈3.65 s；handoff 至少再晚 0.20 s，即 nominal ≥3.85 s。

7. Phase-4 当前：

   ```text
   planning_budget_s = 1.4
   commit_margin_s   = 0.15
   ```

   所以：

   ```text
   bridge_start - planning_submitted = 1.55 s
   ```

8. ToDrawer EE 目标位置为：

   ```text
   (-0.2301621, 0.5245667, 0.3053491) m
   ```

   frame 为 Panda/FR3 base frame (`fr3_link0`)。

9. 静态环境由 `EnvOpenDrawerShelf` 的固定 boxes 构成；随机 benchmark 当前不随机静态家具。

### 本方案建议

下面出现的新 anchors、时间窗口、warm-up 参数、静态环境预检和 category scheduling 规则均属于**建议修改**，需要通过本文件后半部分的量化与可视化验收后再作为最终 benchmark baseline。

---

# 1. 总体目标语义

修改后统一采用：

```text
controller / joint-state available
        +
scenario file 已加载
        ↓
world clock t = 0
        ↓
动态物体开始运动
        ↓
10 Hz position observations
        ↓
Kalman velocity estimation warm-up
        ↓
第一次 MPD planning
        ↓
规划期间 world 继续运动
        ↓
first bridge / MPD suffix 执行
        ↓
动态物体在 robot-aware anchor 时窗穿越
```

必须明确：

- world clock **不依赖机械臂是否已有合法轨迹**；
- world clock **不依赖 controller 是否接受第一条 trajectory**；
- 第一条规划失败时，动态世界仍继续运动；
- 第一次规划必须使用已经积累多帧观测后的 Kalman velocity，而不是仅第一帧的 `v=0`；
- Phase-4 因计算预算短而更早执行、Phase-5 因计算预算长而更晚执行，这是 end-to-end benchmark 应保留的真实差异，不允许为了“等慢方法”暂停世界。

---

# 2. 新主 anchors：空间坐标与随机范围

修改文件：

```text
mpd/scripts/isaaclab/benchmark_todrawer_random.py
```

## 2.1 建议替换 `BASE_CROSSINGS`

建议第一版：

```python
BASE_CROSSINGS = (
    # A0: early arm / forearm region
    ((0.280, 0.060, 0.620), (1.0, 0.0, 0.0)),

    # A1: middle forearm / wrist region
    ((0.050, 0.120, 0.580), (1.0, 0.0, 0.0)),

    # A2: late drawer-approach region
    ((-0.180, 0.220, 0.560), (1.0, 0.0, 0.0)),
)
```

建议增加显式名称，而不是仅靠 tuple index：

```python
BASE_CROSSING_NAMES = ("early", "middle", "drawer_approach")
```

或者更彻底改成 dataclass / dict：

```python
BASE_CROSSINGS = (
    {
        "id": "A0",
        "name": "early",
        "anchor": (0.280, 0.060, 0.620),
        "direction": (1.0, 0.0, 0.0),
        "nominal_time_s": 4.35,
    },
    ...
)
```

推荐后者，因为后续 schedule 不应再依赖“随机 object index 恰好对应某个 anchor”。

## 2.2 空间随机范围

保留当前机制：

```text
普通 category：
anchor xyz 每轴独立 ±0.015 m

inflated_dense：
anchor xyz 每轴独立 ±0.020 m
```

因此：

### A0 normal

```text
x: 0.265 ~ 0.295
y: 0.045 ~ 0.075
z: 0.605 ~ 0.635
```

### A1 normal

```text
x: 0.035 ~ 0.065
y: 0.105 ~ 0.135
z: 0.565 ~ 0.595
```

### A2 normal

```text
x: -0.195 ~ -0.165
y: 0.205 ~ 0.235
z: 0.545 ~ 0.575
```

### `inflated_dense`

```text
A0:
x 0.260~0.300
y 0.040~0.080
z 0.600~0.640

A1:
x 0.030~0.070
y 0.100~0.140
z 0.560~0.600

A2:
x -0.200~-0.160
y 0.200~0.240
z 0.540~0.580
```

## 2.3 与目标位置的距离

ToDrawer EE goal：

```text
(-0.2301621, 0.5245667, 0.3053491)
```

三个建议 anchor 到 goal 的 3D 距离约：

```text
A0: 0.7583 m
A1: 0.5636 m
A2: 0.4002 m
```

它们构成一个由 early → middle → drawer approach 逐步接近任务目标的空间序列。

## 2.4 最大 dense obstacle 在 crossing 时的静态 AABB 余量

`inflated_dense` 最大 box：

```text
size_xyz max = (0.22, 0.20, 0.30) m
base inflation max = 0.05 m
anchor jitter max = 0.02 m
```

将 jitter 也保守地算入包络后，有效半尺寸：

```text
(0.18, 0.17, 0.22) m
```

与当前 `EnvOpenDrawerShelf` 固定 boxes 做 AABB 计算，三个 anchor 在**crossing 中心位置**的最小静态余量约：

```text
A0: 0.1005 m
A1: 0.0600 m
A2: 0.0400 m
```

注意：

> 这只证明 nominal crossing 处有余量，不证明整段 curved / speed-varying trajectory 都不会碰静态环境。

因此必须增加第 6 节的**全时间动态物体 vs 静态环境预检**。

---

# 3. 新时间基线

仍修改：

```text
mpd/scripts/isaaclab/benchmark_todrawer_random.py
```

删除旧语义：

```python
ARM_OPERATION_ARRIVAL_WINDOW_S = (0.9, 3.0)
GENERATION_REVISION = "execution-synced-workspace-arrival-v4"
```

建议新增：

```python
BASE_CROSSING_NOMINAL_TIME_S = {
    0: 4.35,  # A0
    1: 5.25,  # A1
    2: 6.15,  # A2
}

SCENARIO_TIME_JITTER_S = 0.15
OBJECT_TIME_JITTER_S = 0.10

PRIMARY_CROSSING_WINDOW_S = (4.0, 6.5)
```

基础规则：

```text
t(anchor_i)
= nominal_time(anchor_i)
+ scenario_global_jitter
+ object_local_jitter
```

其中：

```text
scenario_global_jitter ~ U(-0.15, +0.15)
object_local_jitter    ~ U(-0.10, +0.10)
```

因此普通基础范围：

```text
A0: 4.10 ~ 4.60 s
A1: 5.00 ~ 5.50 s
A2: 5.90 ~ 6.40 s
```

相邻 anchor 的自然时间差：

```text
0.70 ~ 1.10 s
```

---

# 4. 各 category 的最终建议 schedule

核心原则：

1. **anchor 表示 robot motion phase**，不能完全随机打乱 early/middle/late 的时间关系；
2. **物体身份 / motion model 可以随机**，避免 motion model 与“先到/后到”形成固定混杂；
3. simultaneous 类允许两个相邻 anchor 向中间时间靠拢；
4. staggered 类保留 early→late 顺序，但具体物体随机分配；
5. A0+A2 不作为常规 simultaneous pair。

---

## 4.1 `single_crossing`

- anchor：A0/A1/A2 等概率随机一个；
- crossing time：

```text
A0: 4.10–4.60
A1: 5.00–5.50
A2: 5.90–6.40
```

- motion 保持当前 `constant_velocity`；
- object identity 无额外规则。

---

## 4.2 `staggered_multi`

### 2 objects

推荐只从：

```text
(A0, A1)
(A1, A2)
```

中随机选择一对。

时间保持各自基础窗口：

```text
A0→A1: gap 0.70–1.10 s
A1→A2: gap 0.70–1.10 s
```

如果未来想增加 hard 变体，可少量加入 `(A0,A2)`，但不要作为普通样本主体。

### 3 objects

固定占用：

```text
A0 → A1 → A2
```

时间：

```text
A0 4.10–4.60
A1 5.00–5.50
A2 5.90–6.40
```

motion type 仍从 `constant_velocity / constant_acceleration` 中随机，但**先后顺序不绑定 motion type**。

---

## 4.3 `simultaneous_multi`

只使用相邻 pair：

```text
A0 + A1
A1 + A2
```

### A0 + A1

```text
shared center = 4.80 + U(-0.15, +0.15)
each object   = center + U(-0.10, +0.10)

最终单体范围约：
4.55–5.05 s
```

两个物体最大 crossing-time 差：

```text
<= 0.20 s
```

### A1 + A2

```text
shared center = 5.70 + U(-0.15, +0.15)
each object   = center + U(-0.10, +0.10)

最终单体范围约：
5.45–5.95 s
```

要求：

- 哪个物体先到随机；
- `constant_acceleration` 与 `sinusoidal_curve` 不固定先后。

不建议把 A0+A2 放进常规 simultaneous，因为它们 nominal phase 相差约 1.8 s。若保留少量 extreme case，应单独标记，例如：

```text
simultaneous_nonlocal_extreme
```

而不是混在普通统计里。

---

## 4.4 `fast_crossing`

### 1 object

同 `single_crossing` 的 anchor/time 规则。

### 2 objects

随机相邻 pair：

```text
A0+A1
A1+A2
```

建议相邻 crossing gap：

```text
0.60–0.90 s
```

目的：fast object 更快，因此可把连续威胁压得更紧。

---

## 4.5 `inflated_dense`

固定 A0+A1+A2 三条通道，时间不要过于随机：

```text
A0: 4.17–4.53
A1: 5.07–5.43
A2: 5.97–6.33
```

推荐相邻 gap：

```text
0.74–1.06 s
```

当前 motion types：

```text
constant_velocity
constant_acceleration
curved_speed_variation
```

必须改成：

```python
motion_types = [
    "constant_velocity",
    "constant_acceleration",
    "curved_speed_variation",
]
rng.shuffle(motion_types)
```

不能再让某种 motion model 永远 earliest / middle / latest。

由于 dense object 最大且有 curved motion：

> 必须通过全轨迹静态环境预检；若失败应 bounded-resample，而不是生成一个物理上不可行的 benchmark。

---

## 4.6 `safe_control`

空间 anchors 保持当前代码：

```python
SAFE_CONTROL_CROSSINGS = (
    ((0.55, -0.40, 0.55), (0.0, 1.0, 0.0)),
    ((0.45, -0.55, 0.70), (1.0, 0.0, 0.0)),
)
```

普通 ±0.015 m jitter。

时间从旧 `1.5–2.4 s` 改为机器人真实运动期：

```text
shared center ~ U(4.6, 5.8)
individual jitter ±0.20 s

最终整体约：
4.40–6.00 s
```

目的：

- 在和主 benchmark 相近的 world time 内运动；
- 但空间上保留 low-conflict control；
- 作为“运动本身存在，但与 robot path 低冲突”的对照。

---

## 4.7 `accelerating_crossing`

随机相邻 pair：

```text
A0+A1
A1+A2
```

建议 gap：

```text
0.70–1.10 s
```

两个物体同为 `constant_acceleration`，anchor phase 保持 early→late。

---

## 4.8 `curved_crossing`

采用 `simultaneous_multi` 同样的 shared-center 规则：

```text
A0+A1: 4.55–5.05
A1+A2: 5.45–5.95
```

两个物体先后完全随机，最大时间差约 0.20 s。

由于 lateral amplitude 当前可到 `0.12 m`：

> 必须执行全轨迹静态环境预检，尤其 A2 附近。

---

## 4.9 `uncertain_motion`

相邻 pair：

```text
A0+A1
A1+A2
```

建议 gap：

```text
0.65–1.00 s
```

对象身份随机。

---

## 4.10 `mixed_motion_multi`

当前 motion models：

```text
constant_acceleration
sinusoidal_curve
curved_speed_variation
```

当前代码把第三个固定为 delayed object，这是不必要的 confound。

建议：

1. 先 `rng.shuffle(motion_types)`；
2. 前两个占 near-simultaneous group；
3. 第三个作为 delayed object。

推荐：

```text
A0 + A1 near-simultaneous:
shared center = 4.80 ± 0.15
each ±0.10
=> 4.55–5.05 s

A2 delayed:
相对 shared center + U(0.90, 1.20)
=> 大致 5.55–6.15 s
```

这样 CA / sinusoidal / curved-speed 都有机会成为：

```text
near-simultaneous object 1
near-simultaneous object 2
delayed object
```

---

# 5. 重写 schedule API：不要再“先生成时间、再随机 corridor”

当前代码：

```python
crossing_times = _arrival_schedule(rng, category, object_count)
corridor_indices = rng.sample(range(len(BASE_CROSSINGS)), object_count)
```

导致：

> time 与 anchor 的对应关系是隐式随机的。

建议改为显式 schedule object，例如：

```python
@dataclass(frozen=True)
class CrossingAssignment:
    corridor_index: int
    crossing_time_s: float
    role: str  # early / middle / late / simultaneous_a / simultaneous_b / delayed
```

然后：

```python
assignments = _arrival_schedule(
    rng,
    category=category,
    object_count=object_count,
)
```

由 `_arrival_schedule()` **同时决定**：

- corridor；
- crossing time；
- schedule role；
- 是否 simultaneous/staggered。

随后再随机 motion type → assignment 的映射。

这样可以直接测试：

```text
anchor 与时间是否符合 category contract
motion model 是否被固定绑定到某一 temporal role
```

---

# 6. 增加动态物体 vs 静态环境的全轨迹 feasibility check

当前 generator 只明确检查：

```text
dynamic object vs robot-base exclusion AABB
```

这是不够的。

新增 helper，推荐仍放：

```text
mpd/scripts/isaaclab/benchmark_todrawer_random.py
```

或者拆成：

```text
mpd/scripts/isaaclab/todrawer_scenario_validation.py
```

推荐后者，便于测试复用。

## 6.1 几何来源

不要复制静态 box 数值。

直接从：

```text
mpd/torch_robotics/torch_robotics/environments/env_open_drawer_shelf.py
```

读取：

```text
DRAWER_CABINET_BOXES
OPEN_BOTTOM_DRAWER_BOXES
ADJACENT_SHELF_BOXES
```

或者从 pipeline 导出的 `static-scene.json` 做第二层验证。

## 6.2 检查方式

使用与 world publisher 相同的 `object_position_at()` 方程。

建议 sample：

```text
dt = 0.02–0.05 s
t = 0 ... DESIGN_EPISODE_DURATION_S
```

对每一时刻计算：

```text
dynamic inflated AABB
vs
所有 static box AABB
```

### hard condition

```text
minimum_static_clearance_m > 0.0
```

建议 generator 更保守：

```text
minimum_static_clearance_m >= 0.01 m
```

### warning threshold

```text
minimum_static_clearance_m < 0.02 m
```

记录 warning，但是否 reject 可根据实际生成成功率调整。

## 6.3 curved motion 必须检查整个轨迹

不能只检查 `t=crossing_time`，因为：

```text
lateral_amplitude_m <= 0.12
```

A2 周围曲线运动最可能在 crossing 前后向固定抽屉/上层 drawer front 偏移。

## 6.4 bounded resampling

推荐：

```text
MAX_OBJECT_RESAMPLE_ATTEMPTS = 50
```

流程：

```text
采样 object
→ base clearance
→ static clearance
→ 不满足则重新采样 motion/anchor jitter/direction
→ 50 次仍失败则明确 raise
```

不要无限循环。

每个 object 记录：

```json
{
  "minimum_robot_base_clearance_m": ...,
  "minimum_static_environment_clearance_m": ...
}
```

---

# 7. 修正 scenario clock 元数据

修改：

```text
mpd/scripts/isaaclab/benchmark_todrawer_random.py
```

删除：

```json
{
  "mode": "first_execution_accepted",
  "trigger_topics": [...]
}
```

建议改：

```json
{
  "mode": "first_robot_state_after_scenario_load",
  "independent_of_execution": true
}
```

或者，如果不实现 joint-state gate，则最少写成：

```json
{
  "mode": "world_publisher_started",
  "independent_of_execution": true
}
```

同时更新：

```python
GENERATION_REVISION = "world-clock-robot-aware-crossings-v1"
```

建议 schema bump：

```text
scenario schema_version: 2 → 3
suite schema_version:    3 → 4
report schema_version:   4 → 5
```

原因：clock semantics 和 schedule contract 都发生了实质变化，不应与旧 suite 混用。

删除/替换字段：

```text
arm_operation_arrival_window_s
vertical_crossings_retained
```

建议变为：

```json
{
  "primary_crossing_window_s": [4.0, 6.5],
  "anchor_schedule": {
    "A0": {"role": "early", "nominal_time_s": 4.35},
    "A1": {"role": "middle", "nominal_time_s": 5.25},
    "A2": {"role": "drawer_approach", "nominal_time_s": 6.15}
  }
}
```

---

# 8. world clock 应改为“场景已加载 + robot state ready”，而不是固定 wall-clock init

修改：

```text
physical_ai_runtime/
src/motion_planning/motion_planners/mpd_dynamic_planner_adapter/
mpd_dynamic_planner_adapter/world_demo_node.py
```

## 8.1 当前问题

当前：

```python
self._started = time.time()
```

这意味着 node constructor 完成瞬间就是 world t=0。

更严格的目标：

```text
scenario payload 已成功 parse
+
已经收到第一条有效 /franka/joint_states
        ↓
world t = 0
```

这样“ROS/controller ready”不再依赖猜测性的 5 秒 timer。

## 8.2 建议参数

增加：

```text
clock_start_mode: first_joint_state
joint_state_topic: /franka/joint_states
clock_start_timeout_s: 10.0
```

允许 fallback：

```text
clock_start_mode: node_start
```

用于简单单元测试。

## 8.3 推荐使用 monotonic clock

elapsed 不应使用可被系统时钟校正影响的 `time.time()`：

```python
self._started_monotonic = time.monotonic()
elapsed = time.monotonic() - self._started_monotonic
```

但 observation stamp 仍保持：

```python
stamp_unix_ns = time.time_ns()
```

即：

```text
scenario elapsed → monotonic
跨节点时间戳      → unix time
```

## 8.4 第一次 joint state 触发

建议：

```python
def _on_joint_state(...):
    if self._started_monotonic is None:
        self._started_monotonic = time.monotonic()
        self._scenario_start_unix_ns = time.time_ns()
```

world publisher 在 start 前不发布 moving world，或发布一个明确的 `not_started` 状态；推荐最简单地：

```text
未 start → 不发布 scenario observations
start 后  → 10 Hz 正常发布
```

## 8.5 增加日志

必须日志打印：

```text
scenario loaded
first robot state received
world clock started
first world observation published
```

并包含 unix timestamp，方便自动计算。

---

# 9. Kalman 初始速度 warm-up：把“通常有多帧”变成代码契约

修改：

```text
physical_ai_runtime/
src/motion_planning/motion_planners/mpd_dynamic_planner_adapter/
mpd_dynamic_planner_adapter/dynamic_world.py
```

以及：

```text
.../mpd_dynamic_planner_adapter/replan_node.py
```

## 9.1 `dynamic_world.py`

当前第一帧：

```python
state = [position, 0, 0, 0]
```

所以不能把“已经看到 1 帧 object”等同于“速度估计 ready”。

建议 `_Track` 增加：

```python
observation_count: int
first_stamp_unix_ns: int
last_stamp_unix_ns: int
```

新 track：

```text
observation_count = 1
```

每次 `track.filter.update(...)`：

```text
observation_count += 1
last_stamp_unix_ns = stamp
```

增加 manager API：

```python
def velocity_estimates_ready(
    self,
    *,
    min_observations: int,
    min_track_age_s: float,
) -> bool:
    ...
```

条件：

- 当前 snapshot 非空；
- 所有当前 active tracks：
  - `observation_count >= min_observations`
  - `last_stamp - first_stamp >= min_track_age_s`

## 9.2 `replan_node.py`

新增参数：

```text
initial_world_warmup_enabled: true
initial_world_min_observations: 5
initial_world_min_track_age_s: 0.40
```

推荐第一版：

```text
5 observations
0.40 s
```

原因：

- publisher 10 Hz；
- 5 帧意味着至少约 0.4 s 的位置变化；
- benchmark 1 Hz 首次 schedule 正常仍会等到约 1 s，因此通常会得到更多帧；
- 参数只是把“速度可估计”变成 hard contract，不强行额外拖慢正常路径。

只在**第一次 planning submission 前** gate：

```text
if no planning has ever been submitted:
    if world velocity estimates not ready:
        skip this schedule tick
```

首次 planning 已提交后：

- 不再做这个 initial gate；
- 继续使用现有 `max_world_age_s`、guard 和 prediction freshness 逻辑。

## 9.3 新增 diagnostics/counter

建议：

```text
initial_world_warmup_skips
initial_world_ready_unix_ns
initial_world_ready_observation_count_min
initial_world_ready_track_age_min_s
```

写入 ROS log，最好也写入 replay manifest event。

---

# 10. 配置文件修改

至少修改：

```text
physical_ai_runtime/
src/motion_planning/motion_planners/mpd_dynamic_planner_adapter/
config/replan_dynamic.yaml

physical_ai_runtime/
src/motion_planning/motion_planners/mpd_dynamic_planner_adapter/
config/replan_space_time.yaml
```

若 `replan_dynamic_aligned.yaml` 有独立参数，也同步。

增加：

```yaml
initial_world_warmup_enabled: true
initial_world_min_observations: 5
initial_world_min_track_age_s: 0.40
```

**不要因为本次修改顺手改变：**

```text
Phase-4 planning_budget_s = 1.4
Phase-5 planning_budget_s = 2.5
commit_margin_s = 0.15
guard_rate_hz = 20
```

这些属于 planner/runtime benchmark 条件，应该保持，避免同时改变过多变量。

---

# 11. fake-hardware launch 修改

修改：

```text
physical_ai_runtime/.../launch/replan_space_time_fake_hardware.launch.py
physical_ai_runtime/.../launch/replan_dynamic_fake_hardware.launch.py
```

当前 `TimerAction(period=5.0, ...)` 可以暂时保留作为 controller bringup 的粗粒度延时，但 world clock **不要再由这个 5 s timer 直接定义**。

给 `dynamic_world_demo` 增加：

```python
parameters=[
    {
        "scenario": ...,
        "scenario_file": ...,
        "clock_start_mode": "first_joint_state",
        "joint_state_topic": "/franka/joint_states",
        "clock_start_timeout_s": 10.0,
    }
]
```

这样：

```text
launch +5s
→ world node/replanner process 启动
→ scenario parse
→ 等第一条真实 joint state
→ world clock t=0
```

注意：

> world clock 仍完全不依赖 planner 是否成功、机械臂是否开始运动。

---

# 12. `run_dynamic_demo_pipeline.sh` 修改

文件：

```text
mpd/scripts/isaaclab/run_dynamic_demo_pipeline.sh
```

当前已有：

- static scene export；
- MPD worker health check；
- ROS fake hardware + world + replanner；
- replay manifest validation；
- timing continuity validation；
- IsaacLab deterministic replay video。

这些都保留。

建议增加两步。

## 12.1 ROS 前：scenario preflight

在启动 ROS 前，如果传入 `--world-scenario-file`：

```bash
python scripts/isaaclab/validate_todrawer_random_suite.py \
  --scenario-file "$WORLD_SCENARIO_FILE" \
  --static-scene "$STATIC_SCENE"
```

单 scenario validator 和 suite validator 可共用模块。

硬失败条件：

```text
schema/clock 不一致
crossing schedule 不符合 category contract
base collision
static environment collision
NaN/Inf
direction 非单位/零向量
crossing_time 不在合法范围
```

## 12.2 ROS 后：输出 world-relative timing metrics

从 manifest 中计算：

```text
world_start
first_planning_submitted
first_command_start
first_bridge_start
first_handoff
first_actual_joint_motion（如果可从 trajectory/robot state得到）
goal
```

统一转换为：

```text
time_since_world_start_s
```

追加到：

```text
to_drawer-replan-timing.json
```

---

# 13. benchmark report 增加时序字段

修改：

```text
mpd/scripts/isaaclab/benchmark_todrawer_random.py
```

建议 `REPORT_FIELDS` 新增：

```text
world_start_unix_s
first_planning_submit_from_world_s
first_command_start_from_world_s
first_bridge_start_from_world_s
first_handoff_from_world_s

initial_world_warmup_observations
initial_world_warmup_age_s

scheduled_crossing_time_min_s
scheduled_crossing_time_max_s

minimum_static_environment_clearance_m
```

如果 manifest 当前没有 world start event，可先将“第一条 world snapshot stamp”作为近似 world start，但正式版本最好由 world node/replay event 明确记录。

---

# 14. 新增离线 validator

建议新文件：

```text
mpd/scripts/isaaclab/validate_todrawer_random_suite.py
```

职责：

## 14.1 单场景结构验证

检查：

```text
schema
frame_id == fr3_link0
category
objects 数量
anchor/corridor 合法
motion model 合法
所有数值 finite
direction norm ≈ 1
```

## 14.2 crossing 数学一致性

对每个 object：

```python
position = object_position_at(object, object["crossing_time_s"])
```

要求：

```text
||position - anchor_position|| < 1e-9 ~ 1e-7 m
```

因为现有 motion equations 在 relative_time=0 时应严格回到 anchor。

## 14.3 category timing contract

例如：

### simultaneous

```text
max(t_i) - min(t_i) <= 0.20 s
```

### staggered adjacent

```text
0.70 <= gap <= 1.10 s
```

### inflated_dense

```text
0.74 <= adjacent gap <= 1.06 s
```

### mixed

```text
first pair difference <= 0.20 s
delayed - simultaneous_center in [0.90, 1.20] s
```

### safe_control

```text
all crossing times in [4.40, 6.00]
```

## 14.4 空间 contract

普通：

```text
anchor offset from base anchor <= 0.015 m per axis
```

dense：

```text
<= 0.020 m per axis
```

## 14.5 base/static clearance

hard:

```text
robot-base exclusion clearance > 0
static environment clearance >= 0.01 m
```

warning:

```text
static clearance < 0.02 m
```

## 14.6 Monte Carlo distribution check

新增 CLI：

```bash
python scripts/isaaclab/validate_todrawer_random_suite.py \
  --monte-carlo 10000 \
  --seed 20260829
```

输出：

```text
各 category 样本数
各 anchor 使用频率
各 motion model × temporal role 频率
simultaneous 谁先到的比例
crossing time mean/std/min/max
static clearance min/p1/p5/median
base clearance min/p1/p5
resample attempts 分布
```

不要用很窄的随机频率 hard-fail；主要作为统计报告。

可设 broad warning：

```text
三个 anchor 在大样本中任一使用比例 <25% 或 >42%
→ warning

mixed/inflated_dense 中某 motion model 某 temporal role 长期为 0
→ hard fail（说明仍存在绑定）
```

---

# 15. 新增自动化单元测试

推荐新文件：

```text
mpd/scripts/isaaclab/tests/test_todrawer_random_scenarios.py
```

如果仓库现有 pytest 结构不同，则 Codex 应放入现有 tests 目录，但不要新造第二套测试体系。

至少覆盖：

1. `object_position_at(crossing_time) == anchor`
2. 固定 seed suite deterministic
3. 新 revision/schema 正确
4. 不再出现 `first_execution_accepted`
5. `single_crossing` 时间合法
6. 2/3-object staggered gap 合法
7. simultaneous 最大差 <=0.20 s
8. mixed motion role permutation 非固定
9. inflated_dense motion role permutation 非固定
10. anchor jitter 上界
11. base exclusion 无交叉
12. static environment 全轨迹无交叉
13. safe_control 只使用 S0/S1
14. A0/A1/A2 assignment 不出现非法 pair
15. bounded resample 达到上限时明确失败

runtime 侧建议新增/扩展测试：

```text
physical_ai_runtime/.../test/
```

覆盖：

1. Kalman 第一帧 velocity=0；
2. 多帧匀速观测后 velocity 向真值收敛；
3. `observation_count` 正确；
4. warm-up 5 帧之前 `velocity_estimates_ready=False`；
5. 5 帧且 age>=0.4 后为 True；
6. replanner 第一次 schedule 在 not-ready world 下不 submit；
7. ready 后能够 submit；
8. gate 只影响第一次 planning；
9. stale world 仍由现有 freshness check 拦截；
10. world clock 不需要 execution event。

---

# 16. 量化验收标准

以下数值建议作为修改完成后的第一版 acceptance criteria。

## 16.1 suite generation

用：

```bash
python scripts/isaaclab/benchmark_todrawer_random.py \
  --scenario-count 1000 \
  --repeats 1 \
  --dry-run \
  --output-dir /tmp/todrawer-suite-check
```

要求：

```text
1000/1000 scenarios 成功 materialize
0 个 NaN/Inf
0 个 robot-base intersection
0 个 static-environment intersection
0 个 schedule contract violation
```

如果 bounded resampling 失败率 >0.1%，说明 anchor/motion 范围过激，应调参数，而不是提高 retry 到非常大。

## 16.2 crossing center

所有 object：

```text
position(crossing_time) 与 anchor_position 误差
max < 1e-7 m
```

## 16.3 static clearance

硬要求：

```text
minimum_static_environment_clearance_m >= 0.01 m
```

建议报告：

```text
minimum
1st percentile
5th percentile
median
```

如果大量样本集中在 1–2 cm：

> 即使没碰撞，也说明设计过于贴近静态家具，应重新移 anchor / 缩 lateral amplitude。

## 16.4 Kalman warm-up

在 fake hardware benchmark 中：

```text
first planning submit 前：
所有 active object observation_count >= 5
所有 active track age >= 0.40 s
```

100% runs 满足。

## 16.5 Kalman velocity accuracy

因为 benchmark 的 ground-truth motion model 已知，可新增离线/日志对比。

对于 constant velocity 场景，在第一次 planning 时：

```text
velocity vector error:
||v_est - v_true||
```

建议第一版目标：

```text
median < 0.03 m/s
p95    < 0.08 m/s
```

方向误差（速度足够大时）建议：

```text
median angular error < 10°
p95 < 25°
```

这些阈值需要跑数据后再微调；若明显达不到，优先检查 filter/process noise/measurement covariance，而不是直接放宽 benchmark。

对于 acceleration/curved motion：

- CV Kalman 不应被要求精确拟合真实加速度；
- 主要报告短时 velocity error 和 covariance coverage；
- 不用把 model mismatch 当程序 bug。

## 16.6 world → first planning

benchmark 1 Hz：

理想：

```text
first planning submit from world start:
约 0.8–1.3 s
```

不建议做过窄 hard assert。

硬要求只需：

```text
>= 0.40 s
并满足 observation_count/track_age warm-up
```

## 16.7 Phase-5 planning timing

代码确定：

```text
bridge_start - planning_submitted
≈ 2.65 s
```

建议自动检查：

```text
2.64 <= value <= 2.70 s
```

若实现中有明确 scheduler rounding，可根据真实日志放宽到 ±0.05 s。

handoff：

```text
handoff - bridge_start >= 0.20 s
```

## 16.8 Phase-4 planning timing

代码确定：

```text
bridge_start - planning_submitted
≈ 1.55 s
```

检查：

```text
1.54 <= value <= 1.60 s
```

## 16.9 首次真实运动

不要 hardcode “一定 3.65 s”。

应统计：

```text
first actual motion from world start
```

Phase-5 正常第一轮成功时预期约：

```text
3.5–4.1 s
```

若第一轮 planning 失败，应该更晚，且**world 不能暂停**。

自动验证的重点是：

```text
首轮失败情况下：
world snapshots 仍持续更新
object positions 仍持续变化
```

## 16.10 category 时序

### simultaneous

```text
pair crossing difference <= 0.20 s
```

### normal staggered

```text
0.70–1.10 s
```

### fast two-object

```text
0.60–0.90 s
```

### uncertain

```text
0.65–1.00 s
```

### inflated_dense

```text
0.74–1.06 s
```

### mixed

```text
near-simultaneous pair <= 0.20 s
delayed gap = 0.90–1.20 s
```

---

# 17. 可视化验收

建议新增：

```text
mpd/scripts/isaaclab/visualize_todrawer_random_suite.py
```

输入：

```text
suite.json
static-scene.json
可选 replay-manifest.json
```

输出至少三张图。

## 17.1 `anchors_xy.png`

Top-down XY：

显示：

- robot base exclusion AABB；
- cabinet/drawer/shelf static AABBs；
- A0/A1/A2；
- S0/S1；
- anchor jitter rectangle；
- direction arrows；
- sampled dynamic object center trajectory；
- 最大 inflated envelope；
- EE goal。

人工检查：

- A0/A1/A2 是否确实由 robot-side → drawer-side；
- S0/S1 是否明显低冲突；
- 曲线轨迹是否扫进静态家具；
- A2 是否过于贴近 drawer front。

## 17.2 `anchors_yz.png`

侧视 Y-Z：

重点检查：

- drawer front/top；
- A0/A1/A2 高度；
- 最大 box + inflation；
- curved lateral 不影响 z 时，是否仍有足够静态 clearance。

## 17.3 `scenario_timeline.png`

时间轴：

```text
world t=0
Kalman observations
first plan
first bridge
first handoff
A0 crossing
A1 crossing
A2 crossing
goal
```

不同 mode 用不同 marker，但不要为了“对齐”去修改 world clock。

这张图是人工检查“0.9–3.0 s 旧问题是否真的消失”的最直观方式。

---

# 18. IsaacLab replay 人工验收

现有 pipeline 已经支持：

```text
replay_mpd_trajectory.py
--output_video
--screenshot_path
--prediction_horizon_s 3.0
--prediction_samples 10
```

因此不用新造整套 renderer。

建议每个 category 至少抽：

```text
3 个 seed × 关键 mode
```

先做小规模：

```text
single_crossing
staggered_multi
simultaneous_multi
inflated_dense
safe_control
mixed_motion_multi
```

优先检查 Phase-5 joint / factorized 中一个，以及 Phase-4 baseline。

人工看视频时必须回答以下问题：

1. world 是否在 robot 第一次运动之前已经明显运动？
2. robot 第一次运动时，动态障碍是否已具备合理速度趋势？
3. A0 是否主要干扰 early arm workspace？
4. A1 是否主要干扰中段 arm/wrist？
5. A2 是否真的在 drawer approach 阶段产生交互，而不是在机器人到达前就离开？
6. simultaneous 是否视觉上近同时？
7. staggered 是否明显接续但不是同时？
8. safe_control 是否“动态但低冲突”？
9. dense/curved object 是否出现穿家具、穿 robot base、从奇怪方向飞入场景等 artifact？
10. 不同 mode 的 world movement 是否完全独立于 planner inference speed？

任何一项不满足，都不能只看 success rate 判断 benchmark 已完成。

---

# 19. 首轮建议实验矩阵

修改后不要直接跑 50×5×所有 modes。

分 4 层。

## Stage A：纯生成测试

```text
10,000 synthetic scenarios
不启动 ROS/GPU
```

检查 distribution + geometry + timing。

## Stage B：world/Kalman only

启动：

```text
fake hardware
dynamic_world_demo
replanner（可 plan_only / stub）
```

重点记录：

```text
world start
10 Hz observation
Kalman velocity
warm-up release
```

至少：

```text
10 categories × 3 seeds
```

## Stage C：小规模真实 planner

```text
10 categories
每类 2 scenarios
1 repeat
Phase-4 + Phase-5 joint + 1 个 factorized
```

约 60 runs。

检查：

- timing；
- no static collision；
- no base collision；
- first planning velocity；
- replay video。

## Stage D：正式 benchmark

Stage A-C 全通过后再跑：

```text
50 scenarios
5 repeats
全部目标 modes
```

---

# 20. Codex 实施任务清单

可以直接把下面清单交给 Codex。

## A. `mpd` repo

### 修改

```text
scripts/isaaclab/benchmark_todrawer_random.py
```

完成：

- [ ] 替换 3 个 BASE anchors；
- [ ] 方向改为建议的 robot-aware crossing direction；
- [ ] 删除旧 `first_execution_accepted` clock 元数据；
- [ ] 新 world-clock schema/revision；
- [ ] nominal times 4.35/5.25/6.15；
- [ ] scene jitter ±0.15 s；
- [ ] object jitter ±0.10 s；
- [ ] 重写 `_arrival_schedule()`，使其显式返回 corridor+time assignment；
- [ ] simultaneous 只常规使用相邻 anchors；
- [ ] staggered 保持 anchor phase 顺序；
- [ ] mixed motion order shuffle；
- [ ] inflated_dense motion order shuffle；
- [ ] 保留 safe-control anchors，但修改时间；
- [ ] 增加 static environment full-trajectory clearance；
- [ ] bounded resampling；
- [ ] scenario JSON 记录 `anchor_id/schedule_role/nominal_crossing_time`；
- [ ] report 新增 world-relative timing / static clearance 字段。

### 新增

```text
scripts/isaaclab/todrawer_scenario_validation.py
scripts/isaaclab/validate_todrawer_random_suite.py
scripts/isaaclab/visualize_todrawer_random_suite.py
```

如仓库已有合适的 geometry/validation 模块，优先复用，不要重复代码。

### 修改

```text
scripts/isaaclab/run_dynamic_demo_pipeline.sh
```

完成：

- [ ] ROS 前 scenario/static preflight；
- [ ] ROS 后 world-relative timing summary；
- [ ] 保留现有 manifest/timing/replay render。

---

## B. `physical_ai_runtime` repo

### 修改

```text
.../mpd_dynamic_planner_adapter/world_demo_node.py
```

- [ ] scenario clock 与 execution 解耦；
- [ ] first joint state 后启动 clock；
- [ ] elapsed 使用 `time.monotonic()`；
- [ ] unix observation stamp 仍用 `time.time_ns()`；
- [ ] 明确日志记录 scenario loaded / robot state ready / world started / first publish；
- [ ] 10 Hz 默认发布率不变。

### 修改

```text
.../mpd_dynamic_planner_adapter/dynamic_world.py
```

- [ ] `_Track` 增加 observation_count；
- [ ] first/last observation stamp；
- [ ] readiness API；
- [ ] 不改变 worker snapshot 中现有 position/velocity/covariance 语义。

### 修改

```text
.../mpd_dynamic_planner_adapter/replan_node.py
```

- [ ] 第一次 planning 前 warm-up gate；
- [ ] 5 observations；
- [ ] min track age 0.40 s；
- [ ] 只 gate initial planning；
- [ ] 增加 diagnostics/counters；
- [ ] 不改变后续 stale-world safety checks；
- [ ] 不改变 planner selection/guard 算法。

### 修改配置

```text
config/replan_dynamic.yaml
config/replan_space_time.yaml
config/replan_dynamic_aligned.yaml（若独立存在）
```

新增：

```yaml
initial_world_warmup_enabled: true
initial_world_min_observations: 5
initial_world_min_track_age_s: 0.40
```

### 修改 fake hardware launch

```text
launch/replan_dynamic_fake_hardware.launch.py
launch/replan_space_time_fake_hardware.launch.py
```

给 world demo 传：

```text
clock_start_mode=first_joint_state
joint_state_topic=/franka/joint_states
clock_start_timeout_s=10
```

保留/移除 5 秒 `TimerAction` 可以单独决定；但无论是否保留：

> TimerAction 不能再定义 scenario time origin。

---

# 21. Codex 自动检验命令建议

Codex 修改后至少执行以下类型的命令；具体环境入口按仓库已有 pixi/conda 设置调整。

## Python syntax

```bash
python -m py_compile \
  scripts/isaaclab/benchmark_todrawer_random.py \
  scripts/isaaclab/todrawer_scenario_validation.py \
  scripts/isaaclab/validate_todrawer_random_suite.py \
  scripts/isaaclab/visualize_todrawer_random_suite.py
```

runtime：

```bash
python -m py_compile \
  src/motion_planning/motion_planners/mpd_dynamic_planner_adapter/mpd_dynamic_planner_adapter/world_demo_node.py \
  src/motion_planning/motion_planners/mpd_dynamic_planner_adapter/mpd_dynamic_planner_adapter/dynamic_world.py \
  src/motion_planning/motion_planners/mpd_dynamic_planner_adapter/mpd_dynamic_planner_adapter/replan_node.py
```

## Unit tests

优先跑仓库原有测试入口，再跑新增 tests。

例如：

```bash
pytest -q <new-test-file>
```

不要因为某些 Isaac/ROS heavy tests 无法运行就跳过纯 Python schedule/geometry tests。

## Dry run

```bash
python scripts/isaaclab/benchmark_todrawer_random.py \
  --scenario-count 1000 \
  --repeats 1 \
  --dry-run \
  --output-dir /tmp/todrawer-new-schedule
```

## Validator

```bash
python scripts/isaaclab/validate_todrawer_random_suite.py \
  --suite /tmp/todrawer-new-schedule/suite.json
```

## Visualization

```bash
python scripts/isaaclab/visualize_todrawer_random_suite.py \
  --suite /tmp/todrawer-new-schedule/suite.json \
  --output-dir /tmp/todrawer-new-schedule/plots
```

---

# 22. Codex 静态代码审查问题

要求 Codex 修改后逐条回答，并引用代码位置。

1. 还有任何地方写 `first_execution_accepted` 吗？
2. `world_demo_node` 是否还依赖 planner/execution topic 才推进 elapsed？
3. 第一次 planning 失败时动态 world 是否继续推进？
4. 第一次 planning 前能否保证至少 5 帧 observation？
5. warm-up gate 是否只限制第一次 planning？
6. Phase-4/5 原 planning budgets 是否保持不变？
7. simultaneous 是否可能出现 >0.20 s 的 pair difference？
8. mixed 的 delayed motion model 是否仍固定？
9. inflated_dense 的 earliest/middle/latest motion model 是否仍固定？
10. A0/A1/A2 是否能从 scenario JSON 明确识别？
11. full trajectory static clearance 是否考虑：
    - object size；
    - base inflation；
    - anchor jitter；
    - curved lateral motion；
    - acceleration/speed variation；
12. bounded resampling 是否有明确上限？
13. safe_control 是否仍只使用 S0/S1？
14. schema/revision 是否 bump，避免读取旧 suite？
15. report 是否能显示 world-relative first plan/bridge/handoff？
16. replay render 是否仍正常？

任何一项回答“不确定”都应继续追代码，而不是默认通过。

---

# 23. 人工 code review 清单

人工看 `git diff` 时重点防止 Codex 做以下不必要修改：

- [ ] 不要改 MPD checkpoint；
- [ ] 不要改 diffusion model architecture；
- [ ] 不要改训练数据；
- [ ] 不要顺手改 cost weights；
- [ ] 不要改 Phase-4/5 planning budget；
- [ ] 不要为了让测试过而降低 collision threshold；
- [ ] 不要把 dynamic world 又绑回 trajectory execution；
- [ ] 不要把 world 在 inference 期间暂停；
- [ ] 不要复制一份静态环境 box 数值造成双 source of truth；
- [ ] 不要把 10 Hz world publisher 降低；
- [ ] 不要为了避免 static collision 把 curved amplitude 全局砍掉，优先 bounded resample / corridor-aware constraint；
- [ ] 不要让所有 category 都退化成同一种时序模板。

---

# 24. 最终人工验收标准

在认为“修改完成”之前，人工至少确认：

### 场景语义

- [ ] world 先动，robot 后动；
- [ ] 第一次 planning 时物体已经有明显非零 velocity estimate；
- [ ] planner 失败不会冻结 world；
- [ ] A0/A1/A2 的实际 crossing 与 early/mid/late robot phase大致吻合；
- [ ] safe_control 确实低冲突；
- [ ] simultaneous / staggered / mixed 肉眼符合名字。

### 几何

- [ ] 动态物体不穿柜体；
- [ ] 不穿抽屉；
- [ ] 不穿货架；
- [ ] 不穿 robot base；
- [ ] A2 不因 curved lateral motion 扫入 drawer front；
- [ ] large inflated object 没有视觉 penetration。

### 时间

- [ ] Phase-5 正常首轮 bridge 约 world 3.5–4.1 s；
- [ ] Phase-4 明显更早；
- [ ] A0 crossing 主要在 ~4.1–4.6；
- [ ] A1 ~5.0–5.5；
- [ ] A2 ~5.9–6.4；
- [ ] first-plan failure run 中 crossing 仍按 world time 发生。

### 统计

- [ ] motion model 不再固定绑定 temporal role；
- [ ] anchors 在大样本中均有合理覆盖；
- [ ] static clearance distribution 没有大量堆在 0 附近；
- [ ] suite generation failure/resample 次数不过高。

---

# 25. 推荐实施顺序

不要一次把所有修改混在一个巨大 commit。

建议：

## Commit 1：clock semantics + metadata

```text
world clock 改为 first robot state after scenario load
scenario schema/revision bump
删除 first_execution_accepted
```

验收：

```text
world 不依赖 execution
```

## Commit 2：Kalman warm-up contract

```text
observation_count
track age
initial planning gate
diagnostics
```

验收：

```text
first planning >=5 observations
```

## Commit 3：new anchors + scheduling

```text
A0/A1/A2
4.35/5.25/6.15
category rules
motion permutation
```

验收：

```text
dry-run + schedule unit tests
```

## Commit 4：static feasibility + bounded resample

验收：

```text
1000/10000 scenario Monte Carlo
0 penetration
```

## Commit 5：visualization + benchmark/report metrics

验收：

```text
plots + replay videos + world-relative timing report
```

最后再跑正式 benchmark。

---

# 26. 远程代码核对来源

以下均为本方案撰写时核对的远程 `main`：

- `mpd/scripts/isaaclab/benchmark_todrawer_random.py`  
  https://raw.githubusercontent.com/283042246q/mpd/main/scripts/isaaclab/benchmark_todrawer_random.py

- `mpd/scripts/isaaclab/run_dynamic_demo_pipeline.sh`  
  https://raw.githubusercontent.com/283042246q/mpd/main/scripts/isaaclab/run_dynamic_demo_pipeline.sh

- `mpd/.../env_open_drawer_shelf.py`  
  https://raw.githubusercontent.com/283042246q/mpd/main/mpd/torch_robotics/torch_robotics/environments/env_open_drawer_shelf.py

- `physical_ai_runtime/.../world_demo_node.py`  
  https://raw.githubusercontent.com/283042246q/physical_ai_runtime/main/src/motion_planning/motion_planners/mpd_dynamic_planner_adapter/mpd_dynamic_planner_adapter/world_demo_node.py

- `physical_ai_runtime/.../dynamic_world.py`  
  https://raw.githubusercontent.com/283042246q/physical_ai_runtime/main/src/motion_planning/motion_planners/mpd_dynamic_planner_adapter/mpd_dynamic_planner_adapter/dynamic_world.py

- `physical_ai_runtime/.../replan_node.py`  
  https://raw.githubusercontent.com/283042246q/physical_ai_runtime/main/src/motion_planning/motion_planners/mpd_dynamic_planner_adapter/mpd_dynamic_planner_adapter/replan_node.py

- `physical_ai_runtime/.../config/replan_space_time.yaml`  
  https://raw.githubusercontent.com/283042246q/physical_ai_runtime/main/src/motion_planning/motion_planners/mpd_dynamic_planner_adapter/config/replan_space_time.yaml

- `physical_ai_runtime/.../config/replan_dynamic.yaml`  
  https://raw.githubusercontent.com/283042246q/physical_ai_runtime/main/src/motion_planning/motion_planners/mpd_dynamic_planner_adapter/config/replan_dynamic.yaml

- `physical_ai_runtime/.../launch/replan_space_time_fake_hardware.launch.py`  
  https://raw.githubusercontent.com/283042246q/physical_ai_runtime/main/src/motion_planning/motion_planners/mpd_dynamic_planner_adapter/launch/replan_space_time_fake_hardware.launch.py

---

# 27. 最重要的验收原则

这次修改的最终目标不是“让 benchmark 更容易成功”，而是让 benchmark 的物理和时间语义正确：

```text
动态环境真实持续运动
        ↓
观测器先看到运动
        ↓
Kalman 得到速度
        ↓
MPD 在 moving world 中做第一次 planning
        ↓
planner latency 会真实影响执行时机
        ↓
障碍在 robot 真实 motion phase 穿越关键空间
        ↓
不同方法在完全相同 world clock 下比较
```

因此，成功率下降本身不代表设计失败。

真正需要避免的是：

```text
物体在 robot 动之前已经全部穿完；
planner 失败时 world 被暂停；
第一帧 velocity=0 就直接开始规划；
anchor 不在 robot swept workspace；
动态物体穿静态家具；
motion type 与到达顺序固定绑定；
为了对齐慢方法而人为冻结环境。
```

只要上述问题被消除，新的 ToDrawer random benchmark 才真正适合比较 Phase-4、Phase-5、joint/factorized Space-Time MPD 的 end-to-end 动态规划能力。
