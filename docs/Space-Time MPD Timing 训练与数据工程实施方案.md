# Space-Time MPD Timing 训练与数据工程实施方案

> 本文以当前仓库源码为准。文中使用以下状态标记：
>
> - **已实现**：当前仓库已有代码和测试支撑。
> - **原型**：已有局部代码或 smoke test，但尚未形成训练/数据闭环。
> - **待实现**：设计方案，不应误读为当前功能。

## 1. 结论与推荐路线

当前仓库已经实现了 candidate-specific 的时空代价、单调 TimingSpline、动态障碍评价、runtime timing contract，以及独立的 Factorized TimingDiffusion 训练链路；尚未把 learned timing sampler 接入 F1 runtime，也没有任何 joint-trained space-time diffusion。当前 `phase5_joint` 的含义仍是：

```text
Spatial MPD 只扩散空间控制点 P
    +
InferenceOnlySpaceTimeGuide 内部维护 timing 控制点 c
    +
在 CostGuide 激活步骤中同时计算 ∂C/∂P 和 ∂C/∂c
```

因此，`phase5_joint` 是 **推理期联合梯度优化**，不是 JointDual 训练。

推荐按以下顺序实施：

```text
统一 robot/URDF/limits 与 HDF5 schema
    ↓
RRTConnect 空间路径 → 固化为同一套 spatial B-spline P
    ↓
TOPP-RA 可行锚点 + 多种 retime + TimingSpline c 拟合
    ↓
F1：分离 Factorized + 最终短联合优化
    ↓
F2：使用相同模型做低噪声部分重采样交替
    ↓
F3：加入 rollout/repaired 数据，做最后低噪声区细粒度交替
    ↓
JointDual：联合训练双分支，默认同步反向扩散
```

TimingDiffusion 当前同时支持两种六维目标：现有非冗余 `c[6]`，以及解耦的 `[tau,r1..r5]`。前者作为最直接的兼容基线，后者以 $T_{min}(P,r)$ 和任务 $T_{max}$ 约束总时长，作为主要研究表示。两者共用同一条件网络与训练数据来源，不因切换表示而改变空间路径分布。

---

## 2. 当前仓库真实状态

### 2.1 空间数据生成与训练

**已实现：**

- `scripts/generate_data/generate_trajectories.py` 使用 PyBullet/OMPL，当前默认规划器是 `RRTConnect`。
- 原始数据包含 `sol_path`、`task_id`，在开启 `fit_bspline` 时还包含根级字段 `bspline_params_tt`、`bspline_params_cc` 和 `bspline_params_k`。
- `TrajectoryDatasetBspline` 从 `sol_path` 拟合或直接读取 B-spline，然后移除由起终点及零速度/零加速度边界决定的控制点，只把 learnable spatial control points 送入模型。
- 当前训练模型是单状态张量：

  \[
  x_P\in\mathbb R^{B\times H_P\times D_q}.
  \]

- `scripts/train/train.py` 当前默认 `n_diffusion_steps=100`、cosine schedule、batch size 128、AdamW、学习率 `3e-4`；实际实验配置可以覆盖这些默认值。
- `GaussianDiffusionModel.conditional_sample()` 当前只采样一个 `[B,H,state_dim]` 张量，没有结构化的 `(P,c)` sampler。
- `scripts/train/train_timing_diffusion.py` 已提供独立于上述空间训练代码的 TimingDiffusion 入口，直接读取 canonical shard，支持 `c` 与 `tau_r`，不实例化 planning environment、旧 dataset 或旧 trainer。

**当前限制：**

- `TrajectoryDatasetBspline` 仍只认识空间字段；TimingDiffusion 使用新的 `SpaceTimeTimingDataset`，不修改旧 loader 行为。
- `post_process_generated_dataset.py` 会把所有 shard 读入内存后合并，扩大为每条空间路径多个 timing variant 后会成为内存瓶颈。
- 现有 HDF5 没有 robot fingerprint、joint-name order、limits 版本和 timing representation contract。

### 2.2 TimingSpline

**已实现：** `mpd/parametric_trajectory/timing_spline.py`。

当前表示为：

\[
g(s;c)=B_t(s)c,
\]

\[
u(s;c)=\frac{dt}{ds}=u_{\min}+\operatorname{softplus}(g(s;c)),
\]

\[
t(s;c)=\int_0^s u(\xi;c)d\xi.
\]

因此 `time_from_start` 严格递增。当前默认 timing spline 是：

```text
num_control_points = 8
degree = 3
num_phase_points = 128
```

端点满足：

```text
c[1]  = c[0]
c[-2] = c[-1]
```

以保证 $u_s(0)=u_s(1)=0$。当前 runtime optimizer 进一步冻结前两个和后两个 timing control points，只更新 `c[2:-2]`。数据文件仍应保存完整 `[K=8]` 向量，并在 loader/decoder 中显式执行端点投影。

`TimingSpline` 类本身的 duration 默认范围是 2–15 秒；当前 `SpaceTimeGuidanceSettings` 默认覆盖为 6–14 秒、nominal duration 为 10 秒。数据集 manifest、训练配置和 runtime 必须记录并使用同一组范围，不能依赖不同类的默认值。

空间和时间的链式关系已经实现：

\[
\dot q=\frac{q_s}{u},
\]

\[
\ddot q=\frac{q_{ss}}{u^2}-\frac{q_su_s}{u^3}.
\]

当前表示能表达局部显著减速和 near-wait，但因为要求 `diff(time_from_start)>0`，不能表示严格的零进度 dwell。F1–JointDual 第一版不加入 exact dwell token。

### 2.3 当前 Space-Time guide

**已实现：** `mpd/inference/space_time_guidance.py`。

`SpaceTimeCostEvaluator` 当前包含：

- candidate-specific 动态障碍 clearance/risk；
- joint velocity violation；
- joint acceleration violation；
- duration；
- timing smoothness；
- 与原空间 CostGuide 合并的静态碰撞、平滑、目标等空间项。

当前三种 mode 的准确含义如下。

| mode | 空间 Diffusion | timing 状态 | 时空代价是否更新空间 |
|---|---|---|---|
| `phase5_scalar_duration` | 只扩散 $P$ | guide 内部等值 $c$，近似标量 duration | 否；空间仍使用原 spatial guide |
| `phase5_timing_only` | 只扩散 $P$ | guide 内部更新 $c$ | 否，`_spatial_descent()` 返回零 |
| `phase5_joint` | 只扩散 $P$ | guide 内部更新 $c$ | 是，通过当前 $c$ 计算并加入 $-\nabla_P C$ |

三种 mode 都没有 TimingDiffusion。

当前 guide 只在 diffusion sampler 激活 CostGuide 时更新 timing。CostGuide 启用比例由：

```text
t_start_guide_steps_fraction
```

控制，而不是硬编码“最后 30%”。现有配置包含约 `0.2`、`0.3` 和 `0.333`。后文统一记为：

\[
\rho_g\in[0.2,0.33].
\]

实验必须记录实际 `rho_g` 和 DDIM sampling steps，不能只写“后 30%”。

### 2.4 当前 fixed-time 动态 guidance

**已实现：** `mpd/inference/fixed_time_aligned_guidance.py`。

它保持原 spatial guide，并以 `dynamic_world.trajectory_duration_s` 在 dense horizon 上构造均匀时间。若实验定义为 10 秒均匀 timing，必须让：

```text
dynamic_world.trajectory_duration_s = 10.0
```

而不能只在论文或配置说明里称为 10 秒。

### 2.5 当前 runtime contract

**已实现：** `scripts/runtime/timing_contract.py` 和 `space_time_runtime_engine.py`。

- 每条 candidate 有独立的 `time_from_start`；
- timing 必须从零开始、严格递增、finite，且 duration 在配置范围内；
- trajectory artifact schema v3 保存 `topk_time_from_start`；
- timing schema 当前为 v1；
- runtime 会做 candidate-specific dense (q,\dot q,\ddot q) validation。

### 2.6 当前 TOPP-RA 与 `(P,c)` 数据状态

**已实现：** `scripts/spacetime_data/generate_spacetime_dataset.py` 已形成独立于旧训练 loader 的生产数据闭环：

- 流式读取 warehouse 根级 `sol_path/task_id`；
- 拟合与当前 MPD 零速度、零加速度边界一致的 full spatial B-spline `P`；
- 由 canonical `P` 本身构造 TOPP-RA geometric path，不二次拟合 waypoint；
- 按 URDF joint order 对齐当前 `joint_limits.yaml` 中的 `dq_max/ddq_max`；
- 生成 TOPP-RA anchor、duration-scaled、limit-scaled、local-slowdown 和 near-wait 七类 reference；
- 拟合当前 runtime 的 `TimingSpline c[8]`，并在拟合后重新计算 `q,dq,ddq`；
- 对超限 timing 自动做全局安全放慢，仍不满足 duration/limits 的 variant 记录 reject reason；
- 流式写 canonical HDF5 shard、robot bundle、manifest 和按 `base_path_id` 固定划分的 split。

**已实现：** `scripts/spacetime_data/validate_spacetime_dataset.py` 可独立复算 robot hash、schema/dtype/shape、TimingSpline duration、joint limits 和 split leakage。

原 `scripts/spacetime_data/test_toppra.py` 仍只是 API smoke test，不作为生产生成入口。

---

## 3. F1、F2、F3 与 JointDual 的统一定义

所有 factorized 方案共享两个先验：

\[
P\sim p_\theta(P\mid x),
\]

\[
c\sim p_\phi(c\mid P,x),
\]

其中 (x) 至少包含 start/goal；第一版不把动态世界输入 denoiser，当前世界仍由 CostGuide 注入。

### F1：完全分离 + 最终短联合优化

```text
SpaceDiffusion 完整采样 P
    ↓
TimingDiffusion 在固定 P 上完整采样 c
    ↓
3–10 次短 joint cost refinement，同时更新 P、c
```

空间阶段可以在最后 `rho_g` 使用弱的 10 秒均匀动态 guidance。Timing 阶段在自己的最后 `rho_g` 使用完整 candidate-specific timing cost。

### F2：低噪声部分重采样交替

```text
先执行 F1 的 Space + Timing
    ↓
将 P 重新加噪到低噪声 tP = rho_P TP
    ↓
固定当前 c，重新执行 Space 最后 rho_P
    ↓
将 c 重新加噪到低噪声 tc = rho_c Tc
    ↓
固定更新后的 P，重新执行 Timing 最后 rho_c
```

F2 与 F1 使用相同训练模型和基础 clean 数据。区别主要在 sampler。当前 sampler 尚未提供 arbitrary (x_t) initialization，需要新增 partial-noise/partial-reverse API。

### F3：低噪声区细粒度交替

```text
高噪声区主要建立空间拓扑
    ↓
P、c 都进入可信低噪声区后
    ↓
1 个 Space step → 1–2 个 Timing step → 重复
```

Space denoiser 仍可不输入 $c$，此时 timing 只通过 $-\nabla_P C(P,c)$ 影响空间。Timing denoiser 则会看到不断变化的空间 clean estimate，因此需要 rollout/repaired 数据增强。

### JointDual：联合训练的双分支

JointDual 不是把 (P,c) 塞进一个同形张量，而是：

\[
\hat\epsilon_P=f_P(P_{t_P},c_{t_c},t_P,t_c,x),
\]

\[
\hat\epsilon_c=f_c(c_{t_c},P_{t_P},t_c,t_P,x).
\]

两个分支用同一批配对 $(P,c)$ 做联合 score matching。默认推理采用同步更新：

```text
(P_t, c_t)
    ├── Space branch  → P_{t-1}
    └── Timing branch → c_{t-1}
```

JointDual 描述的是学习目标和网络耦合方式；它以后也可以换成交替 sampler，但论文主基线应先用同步更新，避免把模型收益和数值求解器收益混在一起。

---

## 4. 统一 robot、URDF 与 limits contract

### 4.1 当前 Panda 真实来源

当前 `RobotPanda` 使用：

```text
kinematic/collision URDF:
mpd/torch_robotics/torch_robotics/data/urdf/robots/
  franka_description/robots/panda_arm_hand_no_gripper.urdf

joint velocity/acceleration limits:
mpd/torch_robotics/torch_robotics/data/configs/panda/joint_limits.yaml

collision spheres:
mpd/torch_robotics/torch_robotics/data/configs/panda/panda_sphere_config.yaml

parent broad-phase bounds:
mpd/torch_robotics/torch_robotics/data/configs/panda/
  panda_parent_collision_bounds.yaml
```

当前位置上下限来自 URDF 的非 fixed joints；`dq_max/ddq_max` 来自 `joint_limits.yaml`。标准 URDF 没有统一的 acceleration limit 字段，所以不能宣称“所有限制只来自 URDF”。

### 4.2 统一 robot bundle

每个训练数据集目录必须包含不可变 robot bundle：

```text
dataset_root/
├── manifest.yaml
├── robot/
│   ├── robot.urdf
│   ├── joint_limits.yaml
│   ├── collision_spheres.yaml
│   └── collision_parent_bounds.yaml        # 可选
├── shards/
│   ├── part-00000.hdf5
│   └── ...
└── splits/
    ├── train_base_path_ids.npy
    ├── val_base_path_ids.npy
    └── test_base_path_ids.npy
```

要求：

1. `robot.urdf` 是 canonical source，不使用运行时生成的 `*_tmp_<pid>.urdf` 作为数据身份。
2. `active_joint_names` 按 URDF 中非 fixed joint 的解析顺序保存。
3. `joint_limits.yaml` 的 key 顺序必须与 `active_joint_names` 一致；loader 必须按 joint name 显式对齐，不能依赖 YAML 偶然顺序。
4. TOPP-RA、TimingSpline validator、训练 loader 和 runtime 必须使用同一组 resolved arrays：

   \[
   q_{\min},q_{\max},\dot q_{\max},\ddot q_{\max}.
   \]

5. 每个 HDF5 shard 保存 robot bundle 的 SHA-256，而不是绝对文件路径。
6. base link、EE link、是否含 gripper、tool/grasped-object profile 必须进入 manifest。
7. 单位固定为：joint position `rad`、velocity `rad/s`、acceleration `rad/s^2`、time `s`、Cartesian length `m`。

推荐 `manifest.yaml`：

```yaml
schema_version: spacetime_mpd_v1
robot:
  name: RobotPanda
  urdf: robot/robot.urdf
  urdf_sha256: "..."
  joint_limits: robot/joint_limits.yaml
  joint_limits_sha256: "..."
  collision_spheres: robot/collision_spheres.yaml
  collision_spheres_sha256: "..."
  base_link: panda_link0
  ee_link: panda_hand
  active_joint_names:
    - panda_joint1
    - panda_joint2
    - panda_joint3
    - panda_joint4
    - panda_joint5
    - panda_joint6
    - panda_joint7
  dof: 7
units:
  joint_position: rad
  joint_velocity: rad/s
  joint_acceleration: rad/s^2
  time: s
spatial_spline:
  degree: 5
  num_control_points: 22
  num_phase_points: 128
  zero_velocity_at_endpoints: true
  zero_acceleration_at_endpoints: true
timing_spline:
  representation: dt_ds_softplus_v1
  degree: 3
  num_control_points: 8
  num_phase_points: 128
  u_min: 0.05
```

这里的 `num_control_points` 必须写运行时实际解析后的数量，不是 `bspline_num_control_points_desired` 的请求值；当前 loader 会为满足 UNet horizon 和边界条件调整控制点数。

### 4.3 Robot fingerprint 校验

数据生成、训练和推理启动时均应检查：

```text
URDF hash
joint limits hash
collision spheres hash
active joint names and order
DOF
spatial/timing spline contract
```

任一不一致应默认报错。允许显式 `--allow-robot-mismatch` 只用于迁移诊断，不允许用于正式训练或 benchmark。

---

## 5. 统一 Space-Time HDF5 v1

### 5.1 设计原则

- clean canonical record 始终是一对 $(P,c)$，即使 F1 的 Space 模型只读取 $P$。
- 保存 full spatial control points，不只保存 loader 删除端点后的 learnable slice。
- 保存完整 timing control points `[K=8]`，同时保存由它解码得到的 duration 作一致性检查。
- 高斯 noisy states 在训练时在线生成，不写入 HDF5。
- F3 rollout clean estimate 属于新的 clean/repaired record，不与普通 Gaussian noise 混淆。
- dynamic scene 使用 scene table 去重，不为每条 variant 重复存储同一轨迹。

### 5.2 Canonical fields

| HDF5 path | shape | dtype | 说明 |
|---|---:|---|---|
| `/spatial/control_points` | `[N,H_P,D]` | float32 | full spatial B-spline $P$，统一为 control-point-major |
| `/spatial/degree` | scalar/attr | int16 | 当前 Panda 默认 5 |
| `/timing/control_points` | `[N,K]` | float32 | 完整 TimingSpline $c$ |
| `/timing/duration` | `[N]` | float32 | `TimingSpline.evaluate(c).duration` |
| `/timing/reference_time` | `[N,H_T]` | float32 | 可选，teacher 的 (t_{ref}(s)) |
| `/condition/q_start` | `[N,D]` | float32 | 起点 |
| `/condition/q_goal` | `[N,D]` | float32 | 终点 |
| `/index/task_id` | `[N]` | int64 | start/goal task |
| `/index/base_path_id` | `[N]` | int64 | 同一空间路径的 timing variants 共用 |
| `/index/variant_id` | `[N]` | int16 | timing variant |
| `/index/scene_id` | `[N]` | int64 | `-1` 表示 static |
| `/index/mode_id` | `[N]` | int16 | 可选 temporal mode |
| `/source/spatial` | `[N]` | uint8 | RRTConnect 等枚举 |
| `/source/timing` | `[N]` | uint8 | TOPPRA/time-scale/template/dynamic-opt |
| `/source/parent_sample_id` | `[N]` | int64 | F3 repaired record 的来源；普通样本为 `-1` |
| `/quality/timing_fit_rmse` | `[N]` | float32 | (t_{ref}\rightarrow c) 拟合误差 |
| `/quality/v_ratio_max` | `[N]` | float32 | `max(abs(dq)/dq_max)` |
| `/quality/a_ratio_max` | `[N]` | float32 | `max(abs(ddq)/ddq_max)` |
| `/quality/static_clearance_min` | `[N]` | float32 | dense static clearance |
| `/quality/dynamic_clearance_min` | `[N]` | float32 | dynamic sample；static 为 NaN |
| `/quality/accepted` | `[N]` | bool | 正式训练只读取 true |

### 5.2.1 Duration/shape 解耦派生字段

规范 `(P,c)` 不需要重新运行 RRT 或 TOPP-RA即可派生 `(P,T,r)`。仓库脚本
`scripts/spacetime_data/augment_normalized_timing.py` 默认取任务上限
`T_max=14s`，从实际 `TimingSpline c` 拟合五维相对时间形状 `r`，并根据
当前 `(P,r)` 与 Panda 速度/加速度限制计算采样动力学下界 `T_min(P,r)`：

| HDF5 path | shape | dtype | 说明 |
|---|---:|---|---|
| `/timing/shape_control_points` | `[N,5]` | float32 | 去掉常数 gauge 后的归一化 timing shape `r` |
| `/timing/t_min` | `[N]` | float32 | 速度、加速度和可选 floor 的最大下界 |
| `/timing/t_min_velocity` | `[N]` | float32 | 速度约束给出的下界 |
| `/timing/t_min_acceleration` | `[N]` | float32 | 加速度约束给出的下界 |
| `/timing/t_max` | `[N]` | float32 | 当前任务时限，默认 14s |
| `/timing/duration_fraction` | `[N]` | float32 | 未裁剪的 `(T-T_min)/(T_max-T_min)`，可用于审计越界样本 |
| `/timing/tau` | `[N]` | float32 | 端点数值裁剪后的 duration logit |
| `/quality/normalized_timing_fit_rmse` | `[N]` | float32 | 归一化累计时间曲线拟合 RMSE |
| `/quality/normalized_timing_density_clip_fraction` | `[N]` | float32 | 拟合目标低于 density floor 的采样比例 |
| `/quality/duration_logit_clipped` | `[N]` | bool | fraction 是否因 logit 有限化而裁剪 |
| `/quality/duration_bounds_valid` | `[N]` | bool | `(tau,r)` timing 训练必须额外过滤为 true |

派生脚本保留越界行而不静默改变时长；修改 `T_max` 后可重复运行并原子替换
每个 shard。基础 schema 仍为 v1，manifest 的 `normalized_timing` 节记录派生参数。

scene table 至少包含：

```text
/scenes/id
/scenes/horizon_s
/scenes/object_count
/scenes/object_shape/type/size
/scenes/object_pose_t0
/scenes/object_motion_parameters 或 sampled trajectory
```

### 5.3 Legacy migration

现有根级数据必须由 migration reader 支持：

```text
sol_path
task_id
bspline_params_tt
bspline_params_cc
bspline_params_k
```

特别注意 legacy `bspline_params_cc` 在当前 loader 中被转置后使用。迁移脚本必须根据 DOF、控制点数和 spline metadata 验证方向，再写成统一的：

```text
[N, H_P, D]
```

不要通过“某一维等于 7”静默猜测；不满足唯一判定时直接报错。

### 5.4 Split 规则

train/val/test 的主分组键必须是：

```text
base_path_id
```

`task_id` 和 `scene_id` 作为审计字段保存，但不能让同一 `base_path_id` 因 scene 不同而跨 split。同一空间路径的不同 retime、dynamic scenes，以及 F3 的全部 rollout/repaired descendants 都必须继承同一个 base-path split，否则 TimingDiffusion 测试会发生路径泄漏。

---

## 6. 数据生成：RRT 空间路径与多种 retime

### 6.1 总流程

```text
Robot bundle + scene/task seed
    ↓
RRTConnect 生成 collision-free sol_path
    ↓
path simplify
    ↓
fit/materialize 唯一 spatial B-spline P
    ↓
在完全相同的 P 上运行 TOPP-RA
    ↓
生成多种 timing reference t_ref(s)
    ↓
拟合当前 TimingSpline c
    ↓
用当前 TimingSpline 公式重建 q,dq,ddq,t
    ↓
static/dynamic dense validation
    ↓
写 canonical HDF5 shard
```

关键约束：TOPP-RA、timing fit、训练和 runtime 必须使用同一组 $P$。不能对 RRT waypoints 做一次 TOPP-RA、再把另一条拟合后的 B-spline 当作训练空间路径，否则 `(P,c)` 不再严格配对。

### 6.2 TOPP-RA 的职责

TOPP-RA 是固定几何路径的主要 feasibility teacher，但不应被描述为所有 timing mode 的唯一生成器。

推荐每条 base path 生成以下 variants：

| variant | 生成方法 | 是否主要使用 TOPP-RA | 作用 |
|---|---|---:|---|
| time-optimal anchor | 使用真实 `dq_max/ddq_max` 的 TOPP-RA | 是 | 给出固定路径的最快可行锚点 |
| limit-scaled | 随机缩放 velocity/acceleration limits 后重新 TOPP-RA | 是 | 产生不同全局速度和部分 shape |
| duration-scaled | 对 time-optimal trajectory 做 $t'=\rho t,\rho>1$ | 以 TOPP-RA 为锚点 | 严格保持或放松 v/a 可行性 |
| local slowdown | 对 (u(s)=dt/ds) 加局部 bump | 否 | 学局部减速 timing shape |
| near-wait | 对窄 phase 区间加大 (u(s)) | 否 | 学近似等待 |
| dynamic before/after | timing-only cost optimization，多初始化 | 否 | 学动态障碍两侧的 temporal modes |

全局时间缩放满足：

\[
t'=\rho t,\qquad
\dot q'=\frac{\dot q}{\rho},\qquad
\ddot q'=\frac{\ddot q}{\rho^2},
\]

所以当 $\rho\ge1$ 时，不会破坏原 TOPP-RA 的 velocity/acceleration 上界。

建议第一版每条空间路径生成 6–8 个 timing variants：

```text
1 × TOPP-RA time-optimal
2 × duration scaling，例如 rho ∈ {1.2, 1.5}
1–2 × limit-scaled TOPP-RA
1–2 × local slowdown / near-wait
可选 1–2 × dynamic timing mode
```

### 6.3 TOPP-RA 实现要求

生产脚本不能沿用 smoke test 中的硬编码 limits。应通过 robot bundle 或实例化相同 `RobotPanda` 得到：

```python
joint_names = manifest.robot.active_joint_names
q_min = robot.q_pos_min
q_max = robot.q_pos_max
dq_max = robot.dq_max
ddq_max = robot.ddq_max
```

检查：

- 数组顺序与 `joint_names` 完全一致；
- TOPP-RA path 直接由 canonical B-spline (q(s;P)) 构造；
- rest-to-rest boundary 与空间 B-spline 的零速度/零加速度边界一致；
- TOPP-RA 输出 `s(t)` 后，在固定 phase grid 上单调反演为 `t_ref(s)`；
- 对失败、不可逆、非 finite 或 duration 超限样本记录 reject reason，不静默丢弃。

### 6.4 拟合当前 TimingSpline

对每个 teacher reference (t_{ref}(s_i)) 求：

\[
c^*=\arg\min_c
\frac1H\sum_i\left(t(s_i;c)-t_{ref}(s_i)\right)^2
+\lambda_s\int_0^1\left(\frac{u_s}{u}\right)^2ds.
\]

每次优化后执行：

```text
c[1]  = c[0]
c[-2] = c[-1]
```

第一版建议直接存完整 (c\in\mathbb R^8)。TimingDiffusion 内部可选使用无冗余的 6 维表示：

```text
z_c = [c0, c2, c3, c4, c5, c7]
decode(z_c) = [c0,c0,c2,c3,c4,c5,c7,c7]
```

如果为了完全匹配当前 runtime optimizer，只训练 `c[2:-2]`，则必须同时把 endpoint values 和 nominal-duration 初始化策略写入 checkpoint metadata；否则同一个 4 维 latent 无法独立还原完整 timing。

### 6.5 Local slowdown 与 dynamic labels

局部 slowdown 可以在 TOPP-RA anchor 的 (u_{base}(s)) 上构造：

\[
u'(s)=u_{base}(s)
\left[1+a\exp\left(-\frac{(s-\mu)^2}{2\sigma^2}\right)\right].
\]

任何 local variant 都必须重新计算真实：

\[
q,\dot q,\ddot q,t,
\]

因为 $u_s$ 会进入 acceleration。不能用“只变慢所以一定可行”替代 validator。

动态 labels 可以复用 `SpaceTimeCostEvaluator` 和 `phase5_timing_only` 的 optimizer 逻辑，但当前仓库还没有固定 $P$ 的批量 label generator。应提取成无 sampler 副作用的接口：

```text
optimize_timing(P, c_init, world) -> c_star, breakdown, status
```

对每个 (P,W) 用 fast/slow/early-slow/late-slow/near-wait/random 多初始化，按 crossing time 或 safe-window signature 去重，保留不同 feasible modes。

第一版模型若不输入 world，动态 labels 只是在学习对 world 边缘化的 timing prior。建议先以 static TOPP-RA/templates 为主，动态 variants 占比控制在 10%–20%，当前 scene 仍由推理 CostGuide 决定。

---

## 7. F1/F2/F3/JointDual 的数据要求

| 方法 | clean 基础数据 | 额外数据 | 是否能复用 factorized checkpoint |
|---|---|---|---|
| F1 | 配对 `(P,c)` | 无 | 是 |
| F2 | 与 F1 完全相同 | 可选 generated-P augmentation | 是，模型不变 |
| F3 | `(P,c)` | 强烈建议 low-noise rollout/repaired `(P_tilde,c_tilde)` | Space 可复用；Timing 建议微调 |
| JointDual | 同一 canonical `(P,c)` | 更高的多路径/多 timing 覆盖；联合 corruption | 可以用 factorized 分支初始化，不能直接当最终 joint checkpoint |

### 7.1 F1 数据

Space 模型读取：

```text
(x, P)
```

Timing 模型读取：

```text
(x, P, c)
```

最终 joint cost refinement 不需要额外训练 label。

### 7.2 F2 数据

F2 的 partial re-noise 是标准 forward diffusion：

\[
P_t=\alpha_tP_0+\sigma_t\epsilon_P,
\qquad
c_t=\alpha_tc_0+\sigma_t\epsilon_c.
\]

这类状态已经由 diffusion training 在线覆盖，不需要写入数据集。只有当第一次真实 timing guide 对 $P$ 改动较大、导致第二次 TimingDiffusion 的条件路径超出训练分布时，才需要加入 generated/repaired path augmentation。

### 7.3 F3 rollout/repaired 数据

F3 的 TimingDiffusion 会看到：

```text
SpaceDiffusion 低噪声 clean estimate
CostGuide 修改后的中间路径
尚未完成的 partial reverse 路径
```

这些结构化误差不等于普通 Gaussian noise。数据生成应：

```text
运行真实 Space sampler
    ↓
收集最后 rho_g 的 P0_hat 和 guided P0_hat
    ↓
对每个 P_tilde 重新 TOPP-RA/Timing optimization
    ↓
得到新的可行 c_tilde
    ↓
写入 parent_sample_id 与 rollout step
```

不能把原 $c_0$ 直接复制给明显改变后的 $\tilde P$，否则可能制造错误的 velocity、acceleration 或动态碰撞标签。

### 7.4 JointDual 数据

JointDual 可以复用完全相同的 clean HDF5，但训练时需要在线联合加噪：

\[
P_{t_P}=\alpha^P_{t_P}P_0+\sigma^P_{t_P}\epsilon_P,
\]

\[
c_{t_c}=\alpha^c_{t_c}c_0+\sigma^c_{t_c}\epsilon_c.
\]

JointDual 同步基线可先令 (t_P=t_c)。若需要支持 F3 式异步/交替 sampler，再独立采样 (t_P,t_c)，覆盖：

```text
clean P + noisy c
noisy P + clean c
noisy P + noisy c
不同 noise level 的 P、c
```

JointDual 对数据覆盖要求高于 F1/F2。每个 task 最好有多个空间路径，每条或相近路径有多个 timing，而不是由单一 optimizer 产生一一对应的 `(P,c)`。

---

## 8. 训练方案

### 8.1 公共预处理

**Timing view 已实现：** `mpd/datasets/spacetime_timing_dataset.py` 直接流式读取 canonical HDF5。它不导入 `TrajectoryDatasetBspline`，也不通过旧 planning/training loader。Spatial/Joint view 留待 JointDual 实现：

```text
SpatialDatasetView
TimingDatasetView
JointDatasetView
```

三者读取同一个 canonical HDF5。

当前 normalization 分组：

```text
P: 按 joint 对 full spatial control points 统计 mean/std
c[6]: 独立逐维 mean/std
[tau,r5]: 独立逐维 mean/std
```

统计只使用 train split。normalizer、manifest hash、robot/spline contract、环境标签和 representation 均已写入 checkpoint。正式训练必须读取 canonical split 文件；只有显式 `allow_hash_split_fallback` 才允许未完成数据使用稳定的 `base_path_id` hash split。

### 8.2 F1/F2 TimingDiffusion

**训练与独立 sampler 已实现；F1 runtime 接入待实现。** Space checkpoint 保持不变。timing denoiser 为：

\[
\hat\epsilon_c=f_\phi(c_t,t,\operatorname{Enc}_P(P),x).
\]

当前实现没有复制完整 spatial TemporalUnet：

```text
full P[H_P,D]
  → 固定 B-spline basis 在 64 phase points 求 q, q_s, q_ss
  → 拼接 phase 坐标
  → width=128 的保序 dilated residual 1D CNN
  → 三次 stride-2 downsample + flatten
  → path embedding[256]

noisy timing latent[6] + sinusoidal diffusion time[128] + path embedding
  → hidden=256、6 层 FiLM residual MLP
  → predicted epsilon[6]
```

默认约 356 万参数。`c` 表示由 full `c[8]` 编码为 `[c0,c2,c3,c4,c5,c7]`；`tau_r` 表示为 `[tau,r1..r5]`，并过滤 `quality/duration_bounds_valid=false`。两种表示分别训练 checkpoint，不在同一模型中混合 representation token。

训练损失：

\[
L_c=\mathbb E\|\epsilon_c-\hat\epsilon_c\|^2.
\]

F2 不增加训练 loss，只增加 partial sampler 和推理消融。

独立训练入口和默认配置：

```text
scripts/train/train_timing_diffusion.py
scripts/train/cfgs/timing_diffusion_warehouse.yaml
```

训练使用 cosine beta schedule、100 diffusion steps、epsilon MSE、AdamW、gradient clipping 和 EMA。checkpoint 保存 model/EMA/optimizer/scaler/RNG，以支持严格 resume；模型本身已有 conditional ancestral sampling API。

### 8.3 F3 微调

F3 保持 Space model 不变，Timing model 使用混合条件路径：

```text
60% clean teacher P
20% generated final P
20% low-noise rollout/repaired P_tilde
```

比例是初始建议，最终根据条件分布偏移和 validation 调整。所有 $\tilde P$ 必须有重新验证或重新优化的 $\tilde c$。

### 8.4 JointDual

推荐结构：

```text
P_t ── Spatial TemporalUnet ── εP
          ↑             ↓
      cross adapter / FiLM / attention
          ↓             ↑
c_t ── small Timing branch ─── εc
```

训练损失按分支分别取 mean，避免空间维数淹没 timing：

\[
L=L_P+\lambda_cL_c,
\]

\[
L_P=\operatorname{mean}\|\epsilon_P-\hat\epsilon_P\|^2,
\qquad
L_c=\operatorname{mean}\|\epsilon_c-\hat\epsilon_c\|^2.
\]

起始建议 `lambda_c=1.0`，再 sweep `{0.5,1,2}`。

Checkpoint warm start：

1. 加载现有 spatial MPD 到 Space branch；
2. cross adapter 零初始化，使初始空间行为等价于原模型；
3. 加载 F1/F3 TimingDiffusion 到 Timing branch；
4. 先冻结 spatial backbone，只训练 timing/cross adapter；
5. 再用较低 spatial LR 联合微调。

JointDual 第一版不把 dynamic world 输入网络，只学习 joint kinematic/retiming prior；动态障碍继续由 CostGuide 注入。之后才能做 world-conditioned ablation。

---

## 9. 推理方案与 CostGuide 激活

### 9.1 公共规则

定义每条 chain 的低噪声 guide gate：

\[
g_P=\mathbf1[t_P\le\rho_PT_P],
\]

\[
g_c=\mathbf1[t_c\le\rho_cT_c].
\]

默认从当前配置范围开始：

```text
rho_P, rho_c ∈ [0.2, 0.33]
```

任何需要同时使用 (P,c) 的 dynamic cross guidance，只有在两者 clean estimate 都可信时启用：

\[
g_{cross}=g_Pg_c.
\]

Cost 应在 (hat P_0,hat c_0) 上计算，而不是直接在 noisy (P_t,c_t) 上解释物理时间。

### 9.2 F1 推理

#### Space stage

```text
前 1-rho_P：只做 spatial denoising
最后 rho_P：原 static spatial guide + 弱 fixed-10s dynamic guide
```

空间阶段推荐 cost：

\[
C_P^{10s}=
C_{static}+C_{goal}+C_{smooth}
+\lambda_{dyn}^{weak}C_{dyn}(P,c_{10s}).
\]

`lambda_dyn_weak` 建议从完整 timing 动态权重的 10%–30% 起步。不要在均匀 10 秒假设下强制 velocity/acceleration cost，因为真实 timing 尚未确定。

#### Timing stage

固定 (P^*)：

```text
前 1-rho_c：TimingDiffusion denoising
最后 rho_c：candidate-specific Timing CostGuide
```

\[
C_c=
C_{dynamic}+C_{velocity}+C_{acceleration}
+C_{duration}+C_{timing\_smooth}.
\]

只应用：

\[
-\nabla_cC(P^*,c).
\]

#### Final short joint refinement

执行 3–10 个小步：

\[
P\leftarrow P-\eta_P\nabla_PC(P,c),
\]

\[
c\leftarrow c-\eta_c\nabla_cC(P,c).
\]

每步后投影 endpoint timing constraints，并执行 duration bounds/backtracking。最终结果必须经过现有 dense validator。

### 9.3 F2 推理

F2 先得到 F1 的 $(P^{(0)},c^{(0)})$，然后：

```text
P^(0) forward-noise 到 tP=rho_P TP
    ↓
固定 c^(0)，只跑 Space 最后 rho_P
使用真实 c^(0) 的 dynamic/velocity/acceleration 对 P 的梯度
    ↓
得到 P^(1)
    ↓
c^(0) forward-noise 到 tc=rho_c Tc
    ↓
固定 P^(1)，只跑 Timing 最后 rho_c
    ↓
得到 c^(1)
```

空间第二遍可使用完整时空 cost 对 $P$ 的有效项：

- static collision；
- dynamic collision；
- spatial smooth/goal；
- velocity、acceleration 对 $P$ 的梯度。

Timing 第二遍使用 dynamic、velocity、acceleration、duration 和 timing smoothness 对 $c$ 的梯度。

同一 cost 在两个 block 出现不是重复计费：它们分别沿 (partial C/\partial P) 与 (partial C/\partial c) 优化。

### 9.4 F3 推理

推荐只在双方均进入低噪声区后细粒度交替：

```python
while low_noise_steps_remain:
    P0_hat = space_denoise_step(P_t)
    P0_hat = guide_P(P0_hat, c0_hat) if cross_ready else P0_hat
    P_t = space_reverse_update(P_t, P0_hat)

    for _ in range(timing_steps_per_space_step):
        c0_hat = timing_denoise_step(c_t, condition=P0_hat.detach())
        c0_hat = guide_c(P0_hat.detach(), c0_hat)
        c_t = timing_reverse_update(c_t, c0_hat)
```

初始建议：

```text
1 Space step : 1 Timing step
```

稳定后再尝试 `1:2`。如果顺序偏差明显，使用对称 block：

```text
半个 Space update → 完整 Timing update → 半个 Space update
```

F3 不是严格 Gibbs sampler，应称为 alternating guided denoising。

### 9.5 JointDual 推理

默认同步：

```python
eps_P = space_branch(P_t, c_t, t)
eps_c = timing_branch(c_t, P_t, t)

P0_hat, c0_hat = predict_clean(eps_P, eps_c)

if low_noise:
    cost = spacetime_cost(P0_hat, c0_hat, world)
    P0_hat -= eta_P * grad(cost, P0_hat)
    c0_hat -= eta_c * grad(cost, c0_hat)

P_t, c_t = joint_reverse_update(P0_hat, c0_hat)
```

两个分支读取同一旧状态并共同产生下一状态，避免 F3 的先后顺序误差。Timing 不会在同一 step 看到“刚更新后的 P”，但会在下一 reverse step 接收新状态；这与标准联合状态积分一致。

---

## 10. Cost 分配与梯度边界

| cost | Space stage 固定10秒 | Timing stage | F2/F3 Space block | JointDual |
|---|---:|---:|---:|---:|
| static collision | 强 | 对 $c$ 梯度为 0 | 强 | 两分支联合评价 |
| spatial smooth/goal | 强 | 对 $c$ 梯度为 0 | 强 | 两分支联合评价 |
| dynamic collision | 弱、名义 timing | 强、真实 timing | 强、真实 timing | 强、真实 timing |
| velocity | 第一遍不建议 | 强 | 强，对 $P$ 求梯度 | 强 |
| acceleration | 第一遍不建议 | 强 | 强，对 $P$ 求梯度 | 强 |
| duration | 不使用 | 强 | 对 $P$ 通常无梯度 | 强 |
| timing smoothness | 不使用 | 强 | 对 $P$ 无梯度 | 强 |

当前 `InferenceOnlySpaceTimeGuide` 已经能从同一个图得到空间与 timing 梯度，但它通过内部副作用更新 `self.timing_control_points`。支持 F1–JointDual 前应重构为：

```text
SpaceTimeCostGuide.evaluate(P, c, world)
    -> total, breakdown, timing_eval

SpaceTimeCostGuide.gradients(P, c, active=(P,c))
    -> grad_P, grad_c
```

sampler 决定应用哪个梯度，CostGuide 不再拥有隐式 optimizer state。为兼容当前 Phase 5 runtime，可保留 `InferenceOnlySpaceTimeGuide` 作为 wrapper/fallback。

---

## 11. 实现 PR 顺序

| PR | 内容 | 状态/依赖 | 关键测试 |
|---|---|---|---|
| PR0 | robot bundle + manifest + fingerprint | 已实现 | joint order/hash/units mismatch |
| PR1 | canonical HDF5 schema + legacy migration + streaming writer | 已实现 | roundtrip、transpose、split leakage |
| PR2 | production TOPP-RA retimer | 已实现 | real limits、known path、failure reasons |
| PR3 | `fit_timing_reference()` + timing variants + dense validator | 已实现 | monotonic、fit RMSE、v/a reconstruction |
| PR4 | canonical `TimingDatasetView` + TimingNormalizer | 已实现；Spatial/Joint view 随 JointDual 增加 | batch shapes、normalizer roundtrip、split fallback |
| PR5a | P-conditioned TimingDiffusion + 独立训练/采样/checkpoint | 已实现 | c/tau_r loss、sampling shape、真实 shard smoke、resume |
| PR5b | TimingDiffusion 接入 F1 runtime sampler | 待实现 | candidate-specific timing、dense validation |
| PR6 | stateless SpaceTimeCostGuide + F1 short joint refinement | 从现有 guide 提取 | grad P/c、mode parity |
| PR7 | arbitrary partial-noise sampler + F2 | 待实现 | forward/reverse level、deterministic DDIM |
| PR8 | rollout collector + retime repair + F3 fine-tune | 依赖 F1/F2 | parent split、repaired feasibility |
| PR9 | F3 structured alternating sampler | 依赖 PR8 | gate、step ratio、noise-level contract |
| PR10 | JointDual model/loss/checkpoint transplant | 依赖统一数据 | branch loss、zero adapter、sync sampler |
| PR11 | runtime learned timing integration | 依赖 F1 或 JointDual | artifact v3、fallback、dense validation |

必须优先保留/补充的数学测试：

```text
test_timing_monotonicity
test_timing_endpoint_derivative_constraints
test_qdot_against_finite_difference
test_qddot_against_finite_difference
test_fit_timing_reference
test_toppra_uses_manifest_joint_order
test_toppra_uses_runtime_limits
test_spacetime_hdf5_roundtrip
test_no_base_path_split_leakage
test_partial_noise_level_roundtrip
test_factorized_f2_reuses_f1_checkpoint
test_jointdual_branch_shapes
test_candidate_specific_timing_contract
```

---

## 12. 评估矩阵

| ID | 方法 | 训练模型 | 推理特点 |
|---|---|---|---|
| B0 | 当前 spatial MPD + fixed timing | 现有 spatial | 当前基线 |
| B1 | spatial MPD + post-hoc TOPP-RA | 现有 spatial | classical fixed-path retime |
| F1 | Factorized separated | spatial + timing | 两完整 chain + 短 joint refinement |
| F2 | Factorized partial alternating | 与 F1 相同 | 低噪声 P/c 部分重采样 |
| F3 | Factorized step alternating | F1 + rollout timing fine-tune | 最后低噪声区逐 step 交替 |
| J1 | JointDual synchronous | 联合双分支 | 同步联合反向扩散 |
| J2 | JointDual + alternating sampler | 与 J1 相同 | 只用于区分模型与 sampler 收益 |

离线至少报告：

```text
timing fit RMSE
duration error/distribution
monotonic violation
velocity/acceleration violation
static/dynamic minimum clearance
unguided/guided feasible ratio
timing mode coverage and entropy
best-of-B cost
```

在线至少报告：

```text
success / collision / dynamic collision
time-to-first-feasible
planning latency breakdown
execution duration
trajectory switch / timing mode switch
peak velocity / acceleration / jerk proxy
emergency interruption count
```

所有方法使用同一 robot fingerprint、scene seeds、candidate budget、DDIM step budget和 guide fraction。F2/F3 计算量更高时，同时报告等 wall-clock 和等 denoiser-evaluation 两组结果。

---

## 13. 验收门槛与最终建议

### 数据闭环

- robot/limits/collision fingerprint 100% 可追踪；
- accepted timing `diff(t)>0` 100%；
- dense `v_ratio_max<=1+tol`、`a_ratio_max<=1+tol`；
- TOPP-RA reference 到 TimingSpline 的 median relative timing RMSE 先以 `<1%` 为目标；
- split 中没有 `base_path_id` 或 parent lineage 泄漏；
- rejected samples 有明确 reason histogram。

当前 warehouse 扩大 smoke（1000 条 base paths、默认七类 timing）的实测结果为：6812 个 accepted `(P,c)`，所有 1000 条路径至少保留一个 timing；独立 validator 对全部 6812 条复算无违规。该结果用于验证数据工程闭环，不替代全量数据集的最终统计。

### F1/F2

- F1 TimingDiffusion unguided timing valid ratio显著高于随机/单一 nominal init；
- F2 必须复用 F1 checkpoint，只将收益归因于 partial alternating sampler；
- F2 相比 F1 改善 dynamic feasibility 时，不能明显损害 static feasibility 和空间 mode diversity。

### F3

- 对 clean P 和 rollout P 分别报告 TimingDiffusion 指标；
- 证明 rollout/repaired augmentation 能降低条件分布偏移；
- 报告 `1:1`、`1:2` 和对称 block 的顺序/延迟消融。

### JointDual

- 先以同步 sampler 作为主结果；
- 分支 loss、score norm 和 CostGuide norm 分别记录；
- spatial static success 相比原 MPD 的下降必须受控；
- JointDual 与 F3 比较时，同时控制数据量，避免把 rollout data 增量误认为模型结构收益。

最终推荐：

```text
第一可交付版本：统一 schema + TOPP-RA/retime + F1
第二可交付版本：同模型 F2
第三可交付版本：rollout/repaired 数据 + F3
论文主联合模型：JointDual synchronous
可选推理增强：JointDual alternating sampler
```

这条路线能把四个问题分开验证：

1. learned timing prior 是否有效；
2. 在不重训模型时，交替 sampler 是否有效；
3. 中间路径数据增强是否使细粒度交替稳定；
4. 联合训练的 dual-branch 是否真正学习到超出 CostGuide 的空间—时间相关性。
