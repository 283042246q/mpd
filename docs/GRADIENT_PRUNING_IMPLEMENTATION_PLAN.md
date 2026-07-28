# MPD 梯度剪枝与自适应碰撞 Guidance 分阶段实施方案

本文档将 `docs/grad_pruning_step_guidance.txt` 中的梯度剪枝设想拆分为可独立实现、验证和回退的阶段。

目标是在不重新训练 diffusion model 的前提下，减少 Warehouse/Panda 推理期间的碰撞梯度计算量，同时保持输出轨迹的安全性、困难场景覆盖率和末端位姿精度。

本文所说的“梯度剪枝”不是在完整 Jacobian 计算后将小梯度置零，而是：

```text
B-spline control points
        ↓
粗采样轨迹点，只计算位置、FK 和距离
        ↓
识别危险时间区间、危险物理 link 和危险碰撞 pair
        ↓
只对 active 状态计算碰撞 Jacobian
        ↓
映射回 B-spline control-point gradient
        ↓
独立 dense checker 验证最终候选
```

## 1. 当前基线与改进目标

### 1.1 当前计算路径

当前 `CostComposite` 每次被 DDIM guidance 调用时会：

1. 将归一化控制点还原为 B-spline 控制点；
2. 展开成 128 个位置、速度和加速度采样点；
3. 对全部采样点计算所有碰撞球的 FK 和 Jacobian；
4. 对全部采样点计算末端 FK 和 Jacobian；
5. 依次计算环境碰撞、自碰撞、关节限制、末端目标、速度和加速度梯度；
6. 将各 cost 的梯度映射回控制点。

关键代码：

- `mpd/inference/cost_guides.py`
- `mpd/models/diffusion_models/diffusion_model_base.py`
- `mpd/torch_robotics/torch_robotics/robots/robot_base.py`
- `mpd/torch_robotics/torch_robotics/torch_planning_objectives/fields/distance_fields.py`

Warehouse/Panda 论文配置通常是：

```text
candidate trajectories: 100
trajectory samples:     128
collision spheres:       56
DDIM steps:               15
active guide steps:        3
gradients per guide step:  4 或 6
```

因此主要优化对象是碰撞 FK、Jacobian、SDF 查询和自碰撞 pair，而不是 diffusion U-Net。

### 1.2 当前仓库可复用的基线

`scripts/inference/log/test2/sweep-ranked.csv` 中推荐的完整 coverage 组合：

```text
w1to3-i3-m6-p0.5
contexts:              15/15
mean valid rate:       84.53%
position error:        1.58 cm
orientation error:     1.46 deg
mean inference time:   2.07 s
```

以上结果可作为第一轮 A/B benchmark，但正式实验必须固定：

- checkpoint；
- start/goal context 列表；
- random seed；
- GPU 型号和进程环境；
- 预热次数；
- guidance 参数；
- dense checker 分辨率。

### 1.3 总体验收目标

第一版实现的 go/no-go 目标：

| 指标 | 目标 |
|---|---:|
| Guidance p50 加速 | `>= 1.8x` |
| 总推理 p50 加速 | `>= 1.5x` |
| 平均 valid rate 下降 | `<= 1` 个百分点 |
| Context success/coverage | 不下降 |
| 相对 512 点 checker 的漏碰撞率 | `0` |
| EE 位置误差增加 | `<= 2 mm` |
| EE 姿态误差增加 | `<= 0.2 deg` |
| 最差 context 的有效候选数 | 不低于基线，或有明确 fallback |

这些目标是最终验收要求，不要求每个中间阶段都立即获得完整加速。

## 2. 总体阶段划分

按以下顺序实施：

```text
Stage 0  基线冻结、细粒度 profiler 和 feature flags
   ↓
Stage 1  独立 dense safety oracle 与梯度等价测试
   ↓
Stage 2  低风险冗余计算消除：EE 只算末端、cost 分流
   ↓
Stage 3  固定 coarse-to-fine 时间风险选择器
   ↓
Stage 4  Active-time Jacobian：获得第一轮实质加速
   ↓
Stage 5  Link/sphere/self-collision 层级剪枝
   ↓
Stage 6  GPU 分桶、候选级调度与安全 fallback
   ↓
Stage 7  全量 benchmark、消融、IsaacLab 与真机前验收
```

原则：

- 每个阶段都保留 `enabled: false` 的回退开关；
- 每个阶段必须先通过全量模式等价测试，再开启剪枝；
- joint limits、velocity、acceleration、smoothness 第一版不做时间剪枝；
- 最终安全判定必须独立于 guidance 使用的 active set；
- 某阶段未满足退出条件时，不进入下一阶段。

## 3. Stage 0：基线冻结、Profiler 与开关

### 3.1 目标

在改变算法前获得稳定、可复现的计算时间分解，并建立统一配置入口。

### 3.2 实现内容

在 inference 配置中增加：

```yaml
gradient_pruning:
  enabled: false
  profile: false
  profile_per_guide_call: false
  record_active_statistics: false
```

在 `CostComposite.__call__` 中分别计时：

```text
B-spline expansion
collision FK
collision Jacobian
EE FK/Jacobian
environment SDF query
self-collision distance
self-collision backward
joint-space costs
gradient-to-control-point mapping
gradient aggregation/projection
dense final validation
```

如果当前 TorchKin API 无法拆开 FK 与 Jacobian 时间，Stage 0 先分别调用已有：

```text
robot.fk_collision_spheres
robot.jfk_s_collision_spheres
```

做独立 microbenchmark，不立即改变正式推理路径。

新增每次 guidance 的统计：

```text
guide_call_index
diffusion_timestep
n_candidates
n_time_points_total
n_time_points_active
n_spheres_total
n_spheres_active
n_self_pairs_total
n_self_pairs_active
min_environment_clearance
min_self_clearance
```

### 3.3 建议代码位置

```text
mpd/inference/cost_guides.py
mpd/models/diffusion_models/diffusion_model_base.py
scripts/inference/inference.py
scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-runtime.yaml
scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-paper_sweep.yaml
```

可以新增：

```text
mpd/inference/guidance_profiler.py
scripts/inference/benchmark_gradient_pruning.py
```

### 3.4 测试与退出条件

- 同一配置连续运行至少 10 次；
- 丢弃前 2～3 次 warmup；
- 记录 p50、p95，而不只记录均值；
- profiler 关闭时，推理时间额外开销小于 2%；
- profiler 开启前后生成结果在固定随机种子下保持一致。

交付物：

```text
baseline-timing.csv
baseline-active-statistics.csv
baseline-summary.md
```

## 4. Stage 1：独立 Dense Safety Oracle

### 4.1 目标

在任何剪枝发生前，建立与 guidance 计算路径独立的最终安全验证器。后续所有剪枝结果都以该验证器为准。

### 4.2 实现内容

新增统一 dense checker：

```python
DenseTrajectoryValidator.validate(
    control_points,
    num_points=512,
    check_environment=True,
    check_self_collision=True,
    check_joint_position=True,
    check_joint_velocity=True,
    check_joint_acceleration=True,
)
```

返回：

```text
trajectory_valid_mask
environment_collision_mask
self_collision_mask
joint_limit_mask
minimum_environment_clearance
minimum_self_clearance
first_invalid_index
```

要求：

- checker 使用完整碰撞球和完整配置的 self-collision pairs；
- checker 不复用 guidance 的 active mask；
- checker 默认不计算 Jacobian；
- benchmark 使用 512 点；
- runtime 可使用 256 点，并保留 512 点离线复核；
- 抓持物体存在时必须覆盖 object spheres 和 object-link pairs。

### 4.3 梯度等价测试

在剪枝接口中支持强制全量模式：

```yaml
gradient_pruning:
  enabled: true
  force_all_active: true
```

当所有时间点、球和 pair 都 active 时，新路径必须与旧路径比较：

```text
cost_environment
cost_self_collision
grad_environment_wrt_q
grad_self_wrt_q
grad_total_wrt_control_points
final DDIM update
```

建议容差：

```text
float64 unit test: rtol=1e-6, atol=1e-8
float32 GPU test:  rtol=1e-4, atol=1e-5
```

如稀疏张量重排导致浮点求和顺序变化，可放宽 GPU 容差，但必须记录最大相对误差。

### 4.4 测试场景

至少构造：

1. 全程远离障碍物；
2. 单个时间点穿过障碍物；
3. 两个粗采样点之间发生短暂碰撞；
4. 靠近但未进入 safety margin；
5. 自碰撞；
6. 同时存在环境碰撞和自碰撞；
7. 抓持物体碰撞；
8. 关节、速度或加速度超限。

### 4.5 退出条件

- dense checker 对上述人工场景判断正确；
- `force_all_active` 与旧实现梯度等价；
- 512 点 checker 成为后续 benchmark 的唯一安全 oracle；
- checker 失败时轨迹不会被标记为可执行。

## 5. Stage 2：低风险冗余计算消除

### 5.1 目标

先处理不改变 collision guidance 数值定义的冗余计算，降低后续重构复杂度。

### 5.2 EE Goal 只计算末端

当前 EE goal cost 最终只使用轨迹末端，但 EE FK/Jacobian 对全部 128 点计算。

改为：

```text
q_ee = q_traj_pos[:, -1, :]
robot.jfk_s_ee(q_ee)
```

EE goal 的控制点梯度仍通过末端 B-spline basis row 映射。

### 5.3 Cost 分流

将 cost 分为三类：

```text
Dense joint-space costs
  joint position/velocity/acceleration
  path length/smoothness

Sparseable collision costs
  environment collision
  self collision

Endpoint costs
  EE position/orientation/pose
```

第一版始终让 joint-space costs 使用完整 128 点。原因是这些计算便宜，而且固定降采样可能漏掉速度、加速度和关节限制局部极值。

建议新增内部结构：

```python
GuideKinematics:
    q_dense
    dq_dense
    ddq_dense
    q_collision
    collision_indices
    collision_poses
    collision_jacobians
    ee_pose
    ee_jacobian
```

避免所有 cost 继续依赖一个包含完整 FK/Jacobian 的参数列表。

### 5.4 退出条件

- EE cost 和梯度满足 Stage 1 等价测试；
- 所有现有单元测试通过；
- 100 条候选的 EE FK/Jacobian 时间明显下降；
- valid rate、EE error 和轨迹选择与基线无统计显著差异；
- 此阶段不要求达到整体 1.5x 加速。

## 6. Stage 3：固定 Coarse-to-Fine 时间风险选择器

### 6.1 目标

先实现确定性、固定候选索引的风险选择器，验证召回率；这一阶段可以暂时不追求真正的 Jacobian 加速。

### 6.2 Coarse 采样

从预计算的 128 点 B-spline basis 中选择固定 32 点：

```text
始点和终点必须保留
其余点尽量均匀覆盖 phase
索引固定并提前缓存
```

计算：

```text
q coarse
collision-sphere FK
environment clearance
self-collision clearance
joint-space movement
B-spline velocity/curvature proxy
```

不计算 collision Jacobian。

### 6.3 风险特征

环境碰撞球余量：

\[
c_{m,k}=\operatorname{sdf}(x_{m,k})-r_m-\epsilon
\]

自碰撞余量：

\[
c_{ij,k}=\|x_{i,k}-x_{j,k}\|-r_i-r_j-\epsilon_{\mathrm{self}}
\]

区间风险由以下条件共同决定：

```text
任一环境 clearance < environment_refine_margin
任一 self clearance < self_refine_margin
区间端点 SDF 差异大
关节位移 ||q[k+1]-q[k]|| 大
B-spline 速度或曲率大
粗采样点已经碰撞
```

### 6.4 Active index 生成

第一版不做完全动态递归，使用固定层级：

```text
Level 0: 32 点
Level 1: 对危险区间加入中点，最多 64 点
Level 2: 对仍危险区间继续加入点，最多 128 点
```

对 active 点做邻域膨胀：

```text
active index 左右各加入 1～2 个点
相邻 active 区间合并
始点和终点保留
```

### 6.5 非均匀积分权重

碰撞 cost 原来使用固定 128 点梯形积分。非均匀采样后不能直接求平均，必须按选中 phase 索引生成梯形或 Voronoi 权重：

\[
C_{\mathrm{coll}}
\approx
\sum_{k\in A} \omega_k C_{\mathrm{coll}}(q(s_k))
\]

其中 `A` 是 active/coarse 点集合，`\omega_k` 反映相邻 phase 间隔。

joint-space cost 仍使用原始 128 点和原始积分方式。

### 6.6 配置建议

```yaml
gradient_pruning:
  enabled: true
  force_all_active: false

  temporal:
    coarse_points: 32
    buckets: [32, 64, 128]
    environment_refine_margin: 0.08
    self_refine_margin: 0.06
    q_delta_threshold: null
    curvature_threshold: null
    neighbor_dilation: 2
    always_keep_endpoints: true
```

阈值必须根据 Stage 0 的 clearance 分布标定，以上数值仅作为初始实验值。

### 6.7 召回率测试

以完整 128/512 点轨迹为真值，统计：

```text
risk interval recall
collision waypoint recall
minimum-clearance interval recall
false-negative trajectories
average selected time points
p95 selected time points
```

重点检查“两个 coarse 点之间发生碰撞”的轨迹。

SDF 的 1-Lipschitz 性质只能在拥有区间内碰撞球运动上界时形成保守保证。若使用：

\[
\Delta x_{\max}\approx\|J(q)\|\|q_{k+1}-q_k\|
\]

但没有区间 Jacobian 上界，则该判据只能作为启发式，不能代替 dense checker。

### 6.8 退出条件

- 对测试集中所有真实碰撞轨迹，风险选择器召回率达到预设目标，建议 `>= 99.9%`；
- 512 点 oracle 下没有未被最终 checker 捕获的输出碰撞；
- 安全轨迹平均选点数不超过 40；
- 一般轨迹平均选点数不超过 64；
- 困难轨迹允许退化到 128 点；
- 最差 context 的风险召回率单独达标。

## 7. Stage 4：Active-Time Jacobian

### 7.1 目标

将 Stage 3 的 active 时间索引真正用于 Jacobian 计算。这是第一阶段预期获得明显端到端加速的改动。

### 7.2 实现策略

对每条轨迹得到 active indices 后：

```text
q_dense:       [B, 128, dof]
active index:  [B, K]
q_active:      [B, K, dof]
```

按固定 `K` 分桶：

```text
Bucket 32
Bucket 64
Bucket 128
```

每个桶单独调用 TorchKin：

```text
jfk_s_collision_spheres(q_active)
```

得到 active 时间点的碰撞球 pose/Jacobian，再映射回控制点梯度。

不要先调用完整：

```text
jfk_s_collision_spheres(q_dense)
```

然后才 gather/mask；这种实现不会减少主要计算量。

### 7.3 控制点梯度映射

对选中 phase 索引取对应 B-spline basis：

```text
B_active = B[active_indices]
```

计算：

\[
\frac{\partial C}{\partial w}
=
\sum_{k\in A}
\omega_k
\frac{\partial C}{\partial q(s_k)}
\frac{\partial q(s_k)}{\partial w}
\]

需要验证：

- active indices 与 `B_active` 顺序一致；
- 不同 bucket 的结果 scatter 回原 candidate 顺序；
- endpoint hard condition 不被覆盖；
- EE context 下最后控制点的特殊梯度规则仍保留；
- gradient clipping 仍发生在完整控制点梯度聚合之后。

### 7.4 空 active set

如果轨迹在 collision activation margin 外：

```text
environment collision gradient = 0
self-collision gradient = 0
```

不调用 collision Jacobian。

但仍计算：

```text
joint-space costs
EE endpoint cost
diffusion prior
```

### 7.5 退出条件

- `force_all_active` 与旧梯度等价；
- guidance p50 至少加速 1.5x；
- 总推理 p50 至少加速 1.25x；
- 平均 valid rate 下降不超过 1 个百分点；
- context coverage 不下降；
- p95 没有因分桶和多次 kernel launch 明显恶化；
- 输出全部通过 512 点离线 checker。

如果 Stage 4 没有明显加速，优先检查：

1. 是否仍在某处构造完整 Jacobian；
2. bucket 是否过碎，导致 kernel launch 开销过高；
3. `K=128` 的候选比例是否过高；
4. self-collision autograd 是否成为新瓶颈；
5. Python cost 循环和 tensor scatter 是否抵消收益。

## 8. Stage 5：空间层级剪枝

Stage 5 在时间剪枝稳定后实施，不与 Stage 4 同时开发，避免无法定位质量退化来源。

### 8.1 Stage 5A：物理 Link FK 代替 56 个虚拟球 Link

当前碰撞球作为虚拟 fixed links 加入 URDF，并由 TorchKin逐球返回 FK/Jacobian。

改为只计算 Panda 物理 link：

```text
panda_link1 ... panda_link7, panda_hand
```

碰撞球位置：

\[
x_{\mathrm{sphere}}
=
R_{\mathrm{link}}p_{\mathrm{local}}+t_{\mathrm{link}}
\]

球心位置 Jacobian由父 link spatial Jacobian和局部偏移计算。

收益：

```text
运动学输出 link 数量约从 56 降到 8
同一父 link 的球共享 pose/Jacobian
便于下一步做 link-level broad phase
```

必须新增父 link 路径和原 56-sphere 路径的 FK/Jacobian 数值等价测试。

### 8.2 Stage 5B：环境碰撞 Link Broad Phase

每个物理 link 建立保守包围体：

```text
bounding sphere
或 capsule
或 OBB
```

第一版推荐 bounding sphere，计算简单且容易保证包含全部细球。

```text
link bounding volume 远离环境
    → 跳过该 link 的细碰撞球

link bounding volume 接近环境
    → 展开该 link 的全部细球
```

包围体必须覆盖该 link 的所有配置球，并包含 safety/refine margin。

### 8.3 Stage 5C：Self-Collision Pair Broad Phase

当前配置的物理 link pairs 会展开成所有 sphere pairs。

改为：

```text
物理 link pair bounding-volume distance
        ↓
仅 active link pair 展开细球 pair
        ↓
仅最危险或 margin 内的细球 pair 产生梯度
```

不得增加当前配置明确排除的相邻 link pairs，否则会把正常机械连接误判为自碰撞。

抓持物体的 object-link pairs 必须保留，并独立测试。

### 8.4 Self-Collision 显式梯度

当前 self-collision 对球位置使用 autograd 求梯度。Stage 5 可进一步实现显式球间距离梯度：

\[
\frac{\partial d_{ij}}{\partial x_i}
=
\frac{x_i-x_j}{\|x_i-x_j\|}
\]

\[
\frac{\partial d_{ij}}{\partial x_j}
=
-\frac{x_i-x_j}{\|x_i-x_j\|}
\]

需要处理球心重合时的数值稳定性。

这一项应作为独立 commit 和独立消融，不与 broad phase 混在一起。

### 8.5 退出条件

- 父 link 路径与原 sphere-link 路径数值等价；
- link bounding volume 对全部细球是保守包含；
- pair broad phase 相对完整 sphere-pair 检查无 false negative；
- guidance 相对 Stage 4 再加速至少 1.2x；
- 总推理相对原始基线达到至少 1.5x；
- valid rate、context coverage 和 EE error 达到总体验收目标。

## 9. Stage 6：GPU 分桶、候选调度与 Fallback

### 9.1 GPU 分桶

完全动态的每轨迹采样数量不利于 GPU batch。使用固定桶：

```text
K=0     无碰撞 Jacobian
K=32    安全或低风险
K=64    一般风险
K=128   高风险/窄通道
```

每个 bucket 内保持规则 tensor shape。记录每个 diffusion step 的 bucket 分布。

如果 bucket 数量太少，可把小 bucket 合并，避免大量小 kernel。

### 9.2 候选级 Guidance 调度

在时间和空间剪枝稳定后，才加入逐候选调度：

```text
已经安全且 EE error 满足阈值
    → 跳过下一次 collision guidance 或减少 guide iterations

仍碰撞或 clearance 很小
    → 保持/增加 active points

多次迭代 cost 无下降
    → 升级到更密 bucket 或触发 dense correction
```

输入可包括：

```text
minimum environment clearance
minimum self clearance
collision waypoint ratio
EE position/orientation error
cost decrease rate
gradient norm
current diffusion noise level
remaining time budget
```

第一版使用规则控制器，不训练额外网络。

### 9.3 安全 Fallback

最终 dense checker 失败时：

```text
若同一 context 还有通过 checker 的候选
    → 丢弃失败候选，从安全候选中选最优

若所有候选都失败且仍有时间预算
    → 对 top-K 候选执行一次 128 点 dense correction
    → 再次运行独立 dense checker

若仍失败
    → 返回规划失败，不输出可执行轨迹
```

禁止：

- 因为 coarse guidance 判断安全就跳过最终 checker；
- 将 dense checker 失败仅作为 warning；
- 在没有复核的情况下执行“最接近安全”的碰撞轨迹。

### 9.4 退出条件

- p95 时间相对 Stage 5 不恶化；
- 困难 context 能自动进入 128 点 bucket；
- 已安全候选能减少无效 guidance 调用；
- fallback 后 context coverage 不低于原始基线；
- 所有被标记为成功的轨迹通过独立 dense checker。

## 10. Stage 7：全量评估与发布

### 10.1 消融组合

至少评估：

```text
A0 原始实现
A1 + EE endpoint only
A2 + temporal risk selection
A3 + active-time Jacobian
A4 + parent-link kinematics
A5 + environment link broad phase
A6 + self-collision pair broad phase
A7 + candidate schedule/fallback
```

每一项必须报告相对上一项和相对 A0 的变化。

### 10.2 场景集合

至少覆盖：

1. Warehouse 训练环境；
2. Warehouse additional objects；
3. 随机旋转或位姿变化的环境；
4. 窄通道；
5. 易发生 self-collision 的回折动作；
6. 抓持物体；
7. 当前 sweep 中的最差 context；
8. 无碰撞、低风险场景，用于衡量最大加速。

### 10.3 必报指标

正确性和安全：

```text
context success rate
valid trajectory fraction
environment collision fraction
self-collision fraction
joint-limit violation fraction
512-point checker false-negative rate
IsaacLab collision/contact rate
minimum clearance p5/p50
```

任务质量：

```text
EE position error
EE orientation error
joint-space path length
velocity
acceleration
jerk/smoothness
diversity
```

性能：

```text
total latency p50/p95
generator latency p50/p95
guidance latency p50/p95
FK latency
Jacobian latency
SDF latency
self-collision latency
dense checker latency
peak GPU memory
average/p95 active time points
average/p95 active links/spheres/pairs
bucket occupancy
fallback frequency
```

### 10.4 统计要求

- 使用配对 context 和配对 seed；
- 至少重复 3 次计时；
- 比较均值时同时报告置信区间或 bootstrap interval；
- valid rate 同时报告 micro average 和 per-context macro average；
- 单独报告最差 5 个 context；
- 不允许只报告“至少一条成功”，同时必须报告 100 条候选中的有效比例。

### 10.5 IsaacLab 与真机前条件

离线 MPD checker 通过后：

1. 在 IsaacLab 重放选中轨迹；
2. 检查接触、跟踪误差、首个碰撞时刻；
3. 增加标定/感知误差对应的安全裕量；
4. 限制速度和加速度；
5. 真机首次执行采用低速比例；
6. 控制器保留位置、速度、加速度硬限制和急停。

梯度剪枝只优化规划器计算，不构成真机安全保证。

## 11. 建议的代码结构

避免把所有逻辑继续堆进 `cost_guides.py`，建议拆分：

```text
mpd/inference/
├── cost_guides.py
├── guidance_profiler.py
├── guidance_kinematics.py
├── collision_risk_selector.py
├── active_jacobian.py
├── dense_trajectory_validator.py
└── guidance_scheduler.py
```

职责：

| 模块 | 职责 |
|---|---|
| `guidance_profiler.py` | 计时和 active-set 统计 |
| `guidance_kinematics.py` | dense/coarse/active/endpoint 运动学数据 |
| `collision_risk_selector.py` | coarse-to-fine 时间、link、pair 风险筛选 |
| `active_jacobian.py` | 分桶、gather、TorchKin 调用、scatter |
| `dense_trajectory_validator.py` | 独立最终安全检查 |
| `guidance_scheduler.py` | 候选级跳过、升级和 fallback 规则 |

配置建议统一放在：

```yaml
gradient_pruning:
  enabled: false
  force_all_active: false
  profile: false
  record_active_statistics: false

  temporal:
    coarse_points: 32
    buckets: [32, 64, 128]
    environment_refine_margin: 0.08
    self_refine_margin: 0.06
    neighbor_dilation: 2

  spatial:
    parent_link_kinematics: false
    environment_link_broad_phase: false
    self_link_pair_broad_phase: false

  validation:
    runtime_dense_points: 256
    benchmark_dense_points: 512
    dense_correction_on_failure: true
    dense_correction_top_k: 10
    reject_if_still_invalid: true

  scheduling:
    enabled: false
    skip_safe_candidates: false
    promote_on_stalled_cost: true
```

## 12. 测试计划

建议新增：

```text
tests/test_dense_trajectory_validator.py
tests/test_collision_risk_selector.py
tests/test_active_jacobian_equivalence.py
tests/test_parent_link_sphere_kinematics.py
tests/test_collision_broad_phase.py
tests/test_gradient_pruning_fallback.py
```

测试层次：

```text
L0 纯张量和索引单元测试
L1 小型 CPU FK/几何测试
L2 单个 GPU guidance gradient 等价测试
L3 单请求 Warehouse inference
L4 固定 context benchmark
L5 IsaacLab replay/evaluation
```

每次提交至少运行与改动对应的层次，不应只运行最终 inference。

## 13. 推荐提交顺序

建议保持小提交，便于 bisect 和回退：

```text
1. add guidance profiling and pruning config
2. add independent dense trajectory validator
3. split endpoint, dense-joint and collision cost paths
4. compute EE FK/Jacobian only at trajectory endpoint
5. add deterministic temporal risk selector
6. add nonuniform collision integration weights
7. add active-time Jacobian buckets
8. add parent-link collision-sphere kinematics
9. add environment link broad phase
10. add self-collision link-pair broad phase
11. add explicit self-collision gradients
12. add candidate scheduling and dense fallback
13. add benchmark and ablation reports
```

不要在同一个提交中同时改变：

- 风险阈值；
- Jacobian 计算路径；
- cost 权重；
- DDIM guidance schedule；
- 最终 validator 分辨率。

否则出现质量变化时无法判断原因。

## 14. 最终判断标准

只有同时满足以下条件，才默认在 runtime 配置中开启：

```text
1. force_all_active 与旧实现梯度等价；
2. 512 点 checker 对成功输出零漏碰撞；
3. context coverage 不下降；
4. 平均 valid rate 下降不超过 1 个百分点；
5. EE 精度满足总体验收阈值；
6. guidance p50 至少加速 1.8x；
7. 总推理 p50 至少加速 1.5x；
8. p95 延迟没有明显长尾退化；
9. 抓持物体和最差 context 通过专项测试；
10. IsaacLab 接触率和跟踪结果不劣于基线。
```

如果只满足速度目标但困难 context coverage 下降，应保持默认关闭，并继续增大 refine margin、邻域膨胀范围或调整 fallback，而不是接受平均指标掩盖局部失败。
