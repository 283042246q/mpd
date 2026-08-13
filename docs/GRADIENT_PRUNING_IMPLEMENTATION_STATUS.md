# 梯度裁剪实施状态与使用说明

本文件记录以下两份计划在当前仓库中的落地状态：

- `GRADIENT_PRUNING_IMPLEMENTATION_PLAN.md`
- `GRADIENT_PRUNING_SWITCH_AND_AUTOTEST_PLAN.md`

正式逐步开启的 GPU 消融结果见
[`GRADIENT_PRUNING_ABLATION_RESULTS_20260729.md`](GRADIENT_PRUNING_ABLATION_RESULTS_20260729.md)。

## 1. 已落地的主路径

### 严格总开关与兼容性

- Python 默认配置为无缓存 B3；YAML 缺少整个 `gradient_pruning` 段时也进入
  ParentLinkFast/no-grad/原映射/sparse `J^Tg`，但不创建 temporal selector、包络体预扫、
  link broad phase 或 selection/pose/SDF cache。
- 只有显式设置 `gradient_pruning.enabled: false` 才走原始 legacy 路径。
- `dense_validation` 是独立开关；baseline 与 pruning 可以使用完全相同的安全 oracle。
- 主 inference、runtime 和 paper-sweep YAML 均显式写出同一套无缓存 B3，方便配置自解释；
  其行为与 Python 缺省值一致。

### Guidance 裁剪

- 完整 128 点 B-spline 位置、速度和加速度仍用于 joint-space costs。
- EE goal 仅在末端点计算 FK/Jacobian。
- 碰撞风险扫描只计算 FK、球心距离和 clearance，不预先构造完整碰撞 Jacobian。
- 支持 `K=0/32/64/128` 候选分桶；`K=0` 不调用碰撞 Jacobian。
- active phase 使用对应 B-spline basis，并按非均匀 phase 做梯形积分后 scatter 回原候选顺序。
- diffusion timestep、guide call、active 点数、球数、pair 数、clearance、bucket occupancy 和分段耗时均可输出 CSV。
- task-space cost 使用显式 FK/Jacobian/SDF 梯度，整个 guide 在 `torch.no_grad()` 下执行，不再为手工梯度链构建 Autograd graph。
- 零权重 cost 在调用 cost function 前跳过；Velocity、Acceleration 和 PathLength 使用解析梯度。
- B-spline 映射与 trapezoid 积分可直接融合为 `[B,H,D] -> [B,K,D]` contraction，避免物化 `[B,H,K,D]`；旧路径由 `mapping.fused_bspline_integration: false` 保留。

### 空间运动学与自碰撞

- collision sphere 可由物理 parent link 的 pose/Jacobian 与局部偏移重建。
- Panda 实测将 56 个球映射到 8 个 parent links。
- ParentLinkFast 在 Temporal 关闭时保持规则 `[batch, horizon, dof]` 布局，直接执行 dense parent-link J-FK 和积分，不创建 temporal selector，也不做 active gather/scatter。
- Temporal 建立在 ParentLinkFast 基础设施之后：完整时间轴候选继续走 dense fast path，只有 K0/K32/K64 候选进入稀疏 bucket。
- self-collision 球间距离梯度改为显式解析梯度，并处理数值稳定性。

### 独立稠密安全校验

- runtime、benchmark 和 validator 默认值统一为 128 点，不复用 guidance active set；非 128 配置会显式报错。
- 检查环境碰撞、自碰撞、关节位置、速度和加速度。
- 返回逐轨迹有效标记、逐点碰撞标记、最小 clearance 和首个失败索引。
- `reject_invalid: true` 时只允许 oracle 通过的候选进入最终选择；全部失败时返回规划失败，不输出碰撞轨迹。

### 场景与基准

- 提供 deterministic 2D 场景工厂，以及 simple/narrow/Warehouse-Panda manifest。
- 基准脚本生成同 checkpoint、seed、start/goal、几何和 dense checker 的 paired baseline/pruning 配置。
- 支持生成、执行、失败场景重放和 `--fail-on-regression`。
- 输出 `paired-results.csv`、`timing-breakdown.csv`、`active-statistics.csv`、`failed-contexts.yaml` 和 `summary.md`。

## 2. 配置示例

```yaml
gradient_pruning:
  enabled: true
  force_all_active: false
  profile: true
  record_active_statistics: true
  endpoint:
    ee_only_last_point: true
  candidate:
    enabled: false
  temporal:
    enabled: true
    coarse_points: 32
    probe_midpoints: true
    reuse_selection_within_ddim_step: true
    buckets: [32, 64, 128]
    environment_refine_margin: 0.08
    self_refine_margin: 0.06
    neighbor_dilation: 2
    always_keep_endpoints: true
  spatial:
    parent_link_kinematics: true
    dense_parent_fast_path: true
    active_link_pruning: false
    environment_link_broad_phase: false
    self_link_pair_broad_phase: false
  mapping:
    fused_bspline_integration: true
    sparse_bspline_support: false
  scheduling:
    enabled: false

dense_validation:
  enabled: true
  runtime_points: 128
  benchmark_points: 128
  check_environment: true
  check_self_collision: true
  check_joint_limits: true
  reject_invalid: true
```

## 3. 验证命令

快速回归：

```bash
/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  -m unittest discover -s tests -p 'test_*.py'
```

完整配对 benchmark：

```bash
LD_LIBRARY_PATH=/home/eric/anaconda3/envs/mpd-splines-public/lib \
/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  scripts/inference/benchmark_gradient_pruning.py \
  --suite warehouse_panda \
  --device cuda:0 \
  --contexts 15 \
  --candidates 100 \
  --repeats 3 \
  --dense-points 128 \
  --fail-on-regression
```

## 4. 当前验收记录

- unittest（2026-08-08）：61 项通过，2 项因当前测试进程无 CUDA 条件跳过。
- Panda parent-link pose 最大误差：`1.11e-16`。
- Panda parent-link Jacobian 最大误差：`0`。
- CUDA paired smoke：4 个 context，baseline/pruning valid rate 均为 `100%`，coverage regression 为 `0`。
- smoke 仅有 4 条候选，不能替代计划要求的 `15 contexts × 100 candidates × 3 repeats` 性能验收。

## 5. 本轮进一步优化（2026-07-31）

- temporal risk selector 默认只对 `coarse_points` 和相邻区间中点做 FK/SDF/自碰撞扫描；原始 128 点索引空间仍保留给 active set 和 dense oracle。
- active bucket 不再无条件加入全部粗采样点。风险点和膨胀邻域优先保留，其余位置用于填充目标 bucket，因此单个局部风险可以实际进入 `K=32`。
- `compute_costs_with_xrecon: true` 已可用于 DDIM guidance：每个 diffusion step 固定 denoiser 的 `x0` 预测，在 clean B-spline 控制点上计算 guidance，并以 `max_perturb_x` 作为 trust region 后重新构造 DDIM 状态。
- DDPM 的 `guide_gradient_steps` 也支持同一 clean-prediction 模式，并改为显式 detached 更新，避免多次 guide 后 optimizer 仍持有旧 tensor。
- clean guidance 默认仍关闭；建议先在固定 seed、相同 dense checker 下做 `x_t`/`x0` 和 `1/2` 次 guide 的配对消融。

## 6. ParentLinkFast 与新版顺序（2026-08-03）

- 生产基础链为 `A0 → A1 → A2 → A2P → A2P-fast`；其后分成 clean-x0 A1 和 Temporal 两个独立实验分支，旧的 `A2 → Temporal → Parent-link` 只保留为诊断对照。
- A2P 使用 parent-link 运动学和通用全量 bucket，用于功能等价与增量计时。
- A2P-fast 在 Temporal 关闭时绕过 selector、active index、gather 和 bucket scatter。
- A3R 在 A2P-fast 上开启 Temporal；完整 horizon bucket 自动走 dense fast path，稀疏 bucket 保持原有非均匀 phase 积分。
- 同一 DDIM step 内只在 `guide_iteration=0` 扫描风险；后续 iteration 复用相同 selection。缓存键包含 context、diffusion timestep、batch/horizon、device 和 dtype，下一 step 或形状变化会重扫。
- selector 的 risk count、K 分桶和 phase top-K 已改为 GPU 批量张量计算；Jacobian 路径直接消费 packed `[B, H]` 索引矩阵，不再逐候选执行 `bool/nonzero/isin/unique`。
- clean-x0 DDIM 更新改为 A1：直接修正 detached `x0`，下一状态保留本轮 denoiser 原始 epsilon，不再根据 guided `x0` 重算噪声。
- 消融中的 clean-x0 已直接继承 A2P-fast，并强制 `temporal.enabled: false`；分支为 `a2pf_clean_x0`（4 guide steps）和 `a2pf_clean_x0_2steps`。Temporal 分支 `a3r_temporal_parent` 也独立以 A2P-fast 为父阶段。
- 单元测试覆盖 dense/subset 布局、A2P-fast 与通用全量路径等价，以及同批 K32/完整 horizon 混合时的 cost/gradient 等价。
- Temporal/cache/A1 定向测试 25 项通过；全量 unittest discovery 47 项通过。
- 更新后 Warehouse CUDA 小规模 smoke 的 selection cache 命中 `9/12`（每个 active DDIM step 首次扫描、随后三次复用）；A3R guide 从先前匹配 smoke 的 `0.175s` 降到 `0.167s`。由于单次 A2P-fast 基线波动较大，Temporal 仍保持默认关闭。该数据仅为 1 context、4 candidates、1 repeat，不作为正式性能验收。
- clean-x0-on-A2P-fast smoke：4-step guide `0.117s`，与 A2P-fast `0.118s` 基本持平，但 valid 从 `4/4` 降到 `3/4`；2-step guide `0.067s`、inference `0.124s`，valid 为 `4/4`，但 EE 误差、路径长度和平滑性均明显回退。因此 clean-x0 继续保持实验状态。
- 未加入 autoset，Temporal 仍默认由显式配置决定。

## 7. 明确边界

- 按需求未加入 autoset、阈值自动搜索或在线自动调参。
- `environment_link_broad_phase`、`self_link_pair_broad_phase` 和候选级 `scheduling` 默认关闭；当前配置解析器会拒绝将这些尚未完成独立保守性验证的选项设为 `true`，避免静默退化或产生安全假设。
- dense correction 重试不是安全成功的必要条件；oracle 全部拒绝时直接返回规划失败。
- 梯度裁剪只优化规划计算，不替代 IsaacLab、控制器硬限制或真机安全流程。

## 8. 显式梯度与融合映射更新（2026-08-03）

- `CostGuideManagerParametricTrajectory` 的 Legacy/Pruned 调用均改为 `torch.no_grad()`；输入即使带 `requires_grad=True`，返回 cost/gradient 也不携带 Autograd graph。
- `CostJointSpaceVelocity` 的解析梯度为 `q_velocity`，`CostJointSpaceAcceleration` 为 `q_acceleration`。
- PathLength 对相邻点差的半范数使用解析 segment 梯度，并显式处理零长度 segment；单测与 Autograd reference 对齐。
- 融合开关默认开启，仅作用于 pruning 的 B-spline 路径；Waypoints 和 `mapping.fused_bspline_integration: false` 继续走物化后积分的原逻辑。
- CPU float64 等价测试覆盖 dense、Temporal sparse bucket、EE endpoint 和 joint-space costs。
- Warehouse open-clearance、32 candidates、2 repeats 的 CUDA smoke：旧映射 Guide p50 `0.638 s`，融合映射 `0.599 s`，约 `1.066x`；两者 Valid 均为 `100%`。128 点 dense validation p50 约 `0.219 s`。
- Warehouse narrow-0.20 的 5-repeat no-grad 原路径/融合路径及旧记录对比见 [`GRADIENT_PRUNING_NOGRAD_FUSION_NARROW020_RESULTS_20260803.md`](GRADIENT_PRUNING_NOGRAD_FUSION_NARROW020_RESULTS_20260803.md)；该场景融合 Guide 仅 `1.017x`，不构成明显加速。
- 100 candidates、统一 `15 DDIM / 20% active / 6 guide iterations` 的 7 场景分阶段测试见 [`GRADIENT_PRUNING_100_CANDIDATE_STAGE_TIMING_RESULTS_20260804.md`](GRADIENT_PRUNING_100_CANDIDATE_STAGE_TIMING_RESULTS_20260804.md)。42 次 CUDA inference 中，diffusion p50 为 `49.3–53.3 ms`、Guide 为 `420.8–493.1 ms`、128 点 dense check 为 `214.7–221.2 ms`；fused 的 Guide 几何平均加速仅 `0.9956x`。

## 9. 四轴独立稀疏与 B0--B7 正式测试（2026-08-07）

- 候选、时间、link、控制点稀疏分别由 `candidate.enabled`、
  `temporal.enabled`、`spatial.active_link_pruning`、
  `mapping.sparse_bspline_support` 控制，互不隐式联动。
- 候选/时间语义已拆开：候选开关只允许安全候选进入 K0；时间开关只负责把危险候选
  分配到 K32/K64/K128。两者都关闭时保持 A2P-fast 的规则全量布局。
- link 稀疏对非零碰撞 task-space gradient 做 packed `Adjoint + J^Tg` 投影；FK、
  Jacobian 和 SDF 仍保持原路径，尚未启用未经保守性验证的 link broad phase。
- 控制点稀疏利用 B-spline 局部支撑，每个相位只 scatter 到 `degree+1` 个相关控制点；
  dense 与 per-candidate temporal phase 都有 float64 等价测试。
- 7 场景 × 8 变体 × 3 重复的 168 次 CUDA 测试全部成功。B3 link 的 Guide/
  inference/含 dense 总加速分别为 `1.668x/1.561x/1.341x`，且所有 valid/collision
  mask 与 B0 一致；B7 分别只有 `1.306x/1.265x/1.173x`。
- 当前推荐只默认打开 B3 link。候选单开为负优化；时间稀疏只在 active time 足够低时
  明显受益；控制点稀疏接近中性。未加入 autoset。
- 完整协议、逐场景时间、准确度和边界见
  [`GRADIENT_PRUNING_B0_B7_MULTI_SCENE_RESULTS_20260807.md`](GRADIENT_PRUNING_B0_B7_MULTI_SCENE_RESULTS_20260807.md)。

## 10. B3 默认、Link broad phase 与 Conditional temporal（2026-08-08）

- Warehouse 主 inference、runtime 和 paper-sweep 已显式默认启用 B3：ParentLinkFast、
  no-grad、原映射、sparse `J^Tg`；Python 缺省值同样为 B3，只有显式 `enabled: false`
  才走 legacy。
- 新增 `spatial.link_broad_phase.enabled`。它用全 128 点 inflated-margin FK-only/SDF
  scan 建立候选 parent mask，精确 guide 只对入选 parent 执行 J-FK、sphere SDF 和
  self-pair 梯度；支持非连续 sphere/pair 局部索引重映射。
- 新增 `temporal.conditional_enabled` 和
  `conditional_active_ratio_threshold: 0.35`。拟议 active ratio 超过阈值时直接回退 B3
  dense-time 路径；同一 DDIM step 继续复用 selection。
- 两个开关完全独立，消融为 B3/C1 broad/C2 conditional/C3 both。
- 7 场景 × 4 变体 × 3 重复共 84/84 次 CUDA 推理成功。C1/C2/C3 的 Guide 几何
  平均加速分别为 `0.690x/0.892x/0.645x`，均为负优化；继续默认关闭。
- C1、C2 的所有 valid/collision mask 与 B3 一致；C3 在 narrow-0.20 有 1/100 条
  valid→collision。完整结果见
  [`GRADIENT_PRUNING_B3_BROAD_PHASE_CONDITIONAL_RESULTS_20260808.md`](GRADIENT_PRUNING_B3_BROAD_PHASE_CONDITIONAL_RESULTS_20260808.md)。

## 11. C2 Parent 多粗球预扫描与 C1 回退（2026-08-09）

- Panda 8 个物理 parent 共使用 18 个保守局部粗球，每个 parent 按细球中心的主轴
  空间拓扑连续分段，短 link 使用 2 球，link5/hand 使用 3 球；由 56 个原始碰撞球
  自动生成；
  Robot 初始化时逐球验证包含关系，抓持物体等未保存 parent 自动生成运行时保守球。
- C1 最终恢复 `scan_geometry: fine_spheres`：保持 2026-08-08 原始语义，以 56 个细球
  和 548 个配置 self-pairs 做全 128 点预扫描；精确 guide 再按 parent mask 裁剪。
- 新增独立 `preselection.parent_bounds_scan`，不依赖 link broad phase；C2 消融默认打开，
  因而可在 `link_broad_phase.enabled: false` 时用多粗球完成 32+midpoint 时间粗扫。
- 保留 `scan_geometry: fine_spheres` 作为 2026-08-08 旧 C1 对照路径。
- 90 个仓库测试通过（另 2 个按原条件跳过）；环境与 self-collision 下界保守性均有
  独立测试。Warehouse open、10 candidates CUDA
  smoke 中 B3/实验性多粗球C1/C2 的 Guide 为 `0.120/0.251/0.145 s`，valid rate 均为
  90%。实验性多粗球C1相对前一单大球 smoke 的 Guide `0.447→0.251 s`，细球保留率
  `92.0%→85.1%`，self-pair 保留率 `73.2%→52.4%`；但 C1/C2 对 B3 仍分别只有
  `0.480x/0.828x`；C1 已按本节上述决定恢复细球版本，该小样本只保留为研究记录。
- 生成脚本：`scripts/generate_parent_collision_bounds.py`；保存数据：
  `mpd/torch_robotics/torch_robotics/data/configs/panda/panda_parent_collision_bounds.yaml`。

## 12. C1 Pose/SDF 预扫缓存与 Jacobian-only（2026-08-09）

- TorchKin 新增 pose-cache/Jacobian-only 接口；C1 预扫保存 related parent pose、细球
  pose/中心和逐对象 SDF distance，首次精确 guide 只为激活 parent 构造 Jacobian，SDF
  只补 gradient，不再重复 FK/distance。
- 同一 DDIM step 后续 guide iteration 的 q 已改变，只复用 selection，不复用旧几何。
- 92 个项目测试通过（另 2 个跳过）；cached Jacobian 与 J-FK 在 `1e-9/1e-10` 下等价。
- 统一 100 candidates、15 DDIM、20%、6 iterations、dense=128、seed0、3 repeats 的
  7 场景测试中，C1/B3 Guide 算术平均为 `0.454804/0.285201 s`，加速仅 `0.627x`；
  valid 均值 `54.24%/54.19%`，准确度基本不变。
- C1 继续作为默认关闭的研究开关；完整结果见
  [`GRADIENT_PRUNING_C1_SCAN_CACHE_RESULTS_20260809.md`](GRADIENT_PRUNING_C1_SCAN_CACHE_RESULTS_20260809.md)。

## 13. B3 上的连续 B-spline span certificate（2026-08-11）

- 新增独立 `span_certificate.enabled`，在 B3 fine-sphere、无 parent 包络体、无轨迹
  pose/distance/Jacobian cache 的条件下，用 derivative control points、配置无关的
  Jacobian-column reach bound 和 1-Lipschitz SDF lower bound 认证连续 knot span。
- 环境默认查询 GridMapSDF 的原始 analytic primitives；nearest-grid 查询只保留带量化
  误差的实验回退，避免把离散 grid 误当成连续 1-Lipschitz SDF。
- 环境碰撞和 self-collision 都必须通过；未认证 span 分层二分，depth 3 后映射回原 B3
  的 32/64/128 active bucket，最终 dense=128 validator 不变。
- 7 场景、100 candidates、3 repeats 中 valid/collision rate 与 B3 全部一致，但 Guide
  加速比几何平均仅 `0.408x`。首层认证率平均 `0.095%`，depth 3 后仍剩 `57.84%`
  未认证时间覆盖；严格 primitive SDF 预扫是最大开销。
- 该开关继续默认关闭，不加入 autoset。实现、逐层认证率和 FK/SDF/J-FK 分解见
  [`BSPLINE_SPAN_CERTIFICATE_B3_RESULTS_20260811.md`](BSPLINE_SPAN_CERTIFICATE_B3_RESULTS_20260811.md)。
