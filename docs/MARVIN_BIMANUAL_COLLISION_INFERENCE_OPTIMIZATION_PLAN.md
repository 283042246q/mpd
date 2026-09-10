# Marvin 双臂 MPD 碰撞推理优化实施计划

> 状态：实施基线，2026-09-10。
>
> 范围：Warehouse、Marvin 14-DoF、Pika 双夹爪、`dual_independent` inference。
>
> 约束：不改变 Panda/Franka 入口；不改变现有 Marvin Phase 1--5 入口的默认行为；四项优化分别有独立开关、独立验证入口和阶段提交。

## 1. 目标和结论

当前 production collision model 有 1035 个 fine spheres 和 375482 个 self/inter-arm fine pairs。主要显存问题不是 FK，而是 self collision cost 和 dense validator 一次物化 `[candidate,time,pair,3]` 或 `[candidate,time,pair]` 张量。

按以下顺序实施：

1. **Exact pair streaming**：分块计算全部 fine pairs，只保留每个 candidate/time 的最大 penetration 和赢家 pair；cost 与显式梯度语义不变。
2. **Exact validator chunking**：按 candidate、time、pair 三层分块，完整检查 1035 球；不接受 guidance mask，也不使用 reduced geometry。
3. **Conservative parent-bound broad phase**：用 physical-parent bounds 筛出 guide 中需要展开的 fine spheres/pairs；困难候选退化为 exact pair streaming。
4. **Foam reduced guide geometry**：仅 guidance 使用单独的低球数 Marvin robot；两套 Pika 合计目标为 90--110 球，production validator 继续使用原始 642 个 Pika 球和总计 1035 球。

最终数据流：

```text
MPD candidate q
  -> optional parent-bound scan
  -> full or selected guide fine pairs
  -> exact pair streaming cost/gradient
  -> generated candidates
  -> full production robot + exact chunked dense validator
  -> Isaac Lab replay/check
```

## 2. 独立开关和入口

建议配置增加以下字段；缺省值全部关闭，以保持现有入口结果：

```yaml
collision_optimization:
  pair_streaming:
    enabled: false
    pair_chunk_size: 4096
  reduced_guide_geometry:
    enabled: false
    profile: foam_pika_100

dense_validation:
  chunking:
    enabled: false
    candidate_chunk_size: 4
    time_chunk_size: 16
    self_pair_chunk_size: 4096

gradient_pruning:
  spatial:
    link_broad_phase:
      enabled: false
      scan_geometry: parent_bounds
      full_scan: true
      environment_margin: 0.20
      self_margin: 0.10
```

`gradient_pruning.spatial.self_link_pair_broad_phase` 和 `environment_link_broad_phase` 保持 unsupported，不把新实现伪装成这两个旧占位开关。Parent-bound 路径继续使用已经存在的 `link_broad_phase` 开关。

独立命令入口：

| 入口 | 用途 |
|---|---|
| `scripts/inference/benchmark_marvin_pair_streaming.py` | full tensor 与 pair streaming 的 cost/gradient、耗时、峰值 CUDA 显存和 OOM 对比 |
| `scripts/inference/benchmark_marvin_validator_chunking.py` | full validator 与 candidate/time/pair chunking 的结果等价性、耗时和显存对比 |
| `scripts/inference/benchmark_marvin_parent_broad_phase.py` | full guide 与 parent-bound guide 的 active parent/pair 比例、cost/gradient 和性能对比 |
| `scripts/robots/build_marvin_foam_guide_geometry.py` | 从仓库内本地 Pika meshes 和本地 Foam 生成 reduced guide YAML、元数据及覆盖报告 |
| `scripts/inference/benchmark_marvin_collision_optimizations.py` | 统一的单项与组合消融，捕获 CUDA OOM 而不中止整个矩阵 |

所有 benchmark 默认只读取资产并写入用户指定的 output directory，不覆盖 checkpoint、dataset 或 production collision YAML。

## 3. 阶段 A：Exact pair streaming

### 3.1 实现位置

- `CollisionSelfField.compute_distance_field_cost_and_gradient()` 增加可选 `pair_chunk_size`。
- 增加 streaming minimum API，供 validator 共用。
- Bimanual 三类 cost 继续分别传入原始 fine-pair indices，权重和积分方式不变。

### 3.2 算法

pair 按原始顺序分块。每块计算 clearance/penetration，只保存该块最大 penetration 的 pair、方向和全局 pair index；全局更新使用严格 `>`，保持 `torch.max` 的 first-index tie 语义。循环结束后只对最终赢家执行两次 `scatter_add_`。

空 pair 返回零 cost/gradient；全部非碰撞时 gradient 为零；零距离 pair 使用当前 eps-clamp 稳定子梯度。该路径不依赖 autograd 构建 pair 图。

### 3.3 验收

- chunk size `1、257、2048、4096、>P` 与 full tensor 对比。
- 覆盖无碰撞、单碰撞、多碰撞、正值 tie、pair subset 和 local link subset。
- cost 数值等价；非 tie 情况 gradient 数值等价；tie 情况赢家 pair 与原始顺序一致。
- CPU float32/float64 测试；CUDA 可用时报告峰值显存、耗时和 OOM。

## 4. 阶段 B：Exact validator chunking

### 4.1 实现位置

- `DenseTrajectoryValidator.validate()` 抽出 block evaluator。
- `CollisionSelfField` streaming minimum 返回 `[candidate,time]`，不返回 `[candidate,time,pair]`。
- `BimanualDenseTrajectoryValidator` 在同一次 pair traversal 中累计 left/right/inter-arm/base 四类 minimum。
- 删除 `_annotate()` 中为了分类 clearance 而执行的第二次 collision-sphere FK。

### 4.2 分块顺序

```text
candidate block
  -> time block
     -> collision-sphere FK
     -> environment clearance
     -> pair blocks
        -> overall/category running minimum
     -> 写入预分配 [B,H] mask/clearance
```

最终结果仍保留完整 `q_position/q_velocity/q_acceleration` 和 `[B,H]` masks，便于 artifact 与 replay；只消除中间 poses 和 pair clearance 的全量驻留。

动态环境调用必须同步切分 `trajectory_times`。第一版不做 waypoint early-exit，因为 failure diagnostics 和 minimum clearance 需要完整轨迹。

### 4.3 验收

- full 与 chunked 的 valid mask、各 collision mask、first invalid index、minimum clearance、四类 bimanual minimum 和 failure code 一致。
- 覆盖 candidate/time/pair 不能整除 chunk size 的情况。
- `validate_ranked_batches()`、padding slot、unchecked candidate 和 dynamic `trajectory_times` 回归。

## 5. 阶段 C：Parent-bound broad phase

### 5.1 保守性条件

每个 parent bound 必须包含该 physical parent 的所有 guide fine spheres。环境 SDF 满足 1-Lipschitz 时，`sdf(center)-radius` 是 bound 内几何的 clearance 下界；两个 parent bounds 的球面距离是其全部 child sphere pairs 的 clearance 下界。

Broad-phase margin 不得小于对应 guide cost 的激活距离，并增加数值 guard。任何缺失/非法 bound、动态 payload 未覆盖、未知 distance field 或 mask 构造失败都回退到全量 pair streaming。

### 5.2 预计算映射

Robot 初始化时生成并缓存：

- `parent -> fine sphere indices`；
- `fine pair -> parent pair id`；
- `parent pair -> fine pair indices`；
- `fine pair -> bimanual category`。

运行时不得反复用 `torch.isin` 扫描 37.5 万个 pairs。

### 5.3 运行时

第一版对全部 guide 时间点做 parent scan，但 mask 按 candidate 聚合，避免逐时间点形成大量小 bucket。保留独立的：

- environment parent mask；
- active parent-pair mask；
- kinematics parent union。

fine self-pair mask直接由 `active_parent_pair_mask[fine_pair_parent_pair_id]` 产生，不能仅用“两个 parent 均 active”推导，否则会加入无关 parent pairs。

优先采用一次计算全部约 26 个 parent pose/Jacobian 的规则张量路径：pose 用于 bounds，Jacobian 按 mask gather；不要 parent FK 粗筛后再次运行相同 FK。只有 profiler 证明有收益时，才使用按 mask 构造的 subset TorchKin 函数。

### 5.4 验收

- 加载时继续验证 bounds 覆盖每个 fine sphere。
- 对随机和接触边界状态验证：任一激活 fine sphere/pair 对应 parent mask 必须 active，false negative 为 0。
- 对相同 q 比较 full guide 与 broad-phase guide；在保守 margin 内 cost/gradient 等价。
- 报告 active parents、environment spheres、left/right/inter-arm fine pairs 的比例。

## 6. 阶段 D：Foam reduced guide geometry

### 6.1 资产策略

使用本地 `/home/eric/Projects/foam`，其 SphereTree 可执行文件已经存在；输入只使用仓库内归档的 Marvin/Pika STL，不下载远程资产。

仅重新拟合八个 Pika/tool links：左右 adaptor、gripper base、left finger、right finger。Marvin 原始 arm/base 的 393 个 spheres 不变。两套 Pika 合计目标 90--110 个 spheres，目标分配先按左右对称：

```text
每侧 adaptor       3--5
每侧 gripper base  28--34
每侧两根 finger   合计 14--18
每侧总计          45--55
```

构建脚本从多个 Foam method/depth/branch 候选中选择满足预算、误差最小的层，并记录 Foam commit、参数、mesh hash、每 link 球数和覆盖指标。生成资产存放在新的 `configs/marvin/pika_foam_guide/`，不修改 `configs/marvin/pika/`。

Foam medial spheres 不自动声明为 production-conservative。允许其作为优化引导近似，但最终结果必须由原始 1035 球 validator 复核。

### 6.2 双模型隔离

运行时建立：

- `planning_task.robot`：production Marvin，供 context、limits、EE、artifact 和 dense validator 使用；
- `guide_collision_robot`：相同 URDF/joint order，但加载 reduced guide collision YAML，只供 robot-world/self/inter-arm guide 使用。

EE goal、joint limit、velocity、acceleration 和 path-length cost 仍使用 production task。只有 collision cost 的 field、collision FK/Jacobian、sphere/pair metadata 切换到 guide robot。

结果 JSON 必须同时记录 production 和 guide geometry profile/hash/count，防止把 reduced geometry 误当作最终证书。

### 6.3 覆盖与质量门禁

- 统计 reduced spheres 对 Pika mesh surface/voxel samples 的 uncovered ratio 和最大 signed excess。
- 随机/边界 q 对比 reduced guide、production spheres、PyBullet/Isaac Lab。
- reduced guide 可出现与 production 不同的 cost，但最终 validator 不能漏检；重点统计 reduced guide 导致的 candidate rejection 和最终 success rate 变化。
- 若约 100 球版本导致成功率显著下降，保留 150/200 球 profile，不通过修改 production validator 掩盖问题。

## 7. 消融矩阵

固定 requests、seed、checkpoint、候选数和 CUDA device，至少运行：

| ID | Pair streaming | Validator chunk | Parent bounds | Foam guide |
|---|---:|---:|---:|---:|
| A0 | off | off | off | off |
| A1 | on | off | off | off |
| A2 | on | on | off | off |
| A3 | on | on | on | off |
| A4 | on | on | off | on |
| A5 | on | on | on | on |

另外分别执行每个单项入口，以区分组合交互和单项收益。

每次记录：

- 状态：success、no-valid、CUDA OOM、其他异常；
- `torch.cuda.max_memory_allocated/reserved`、运行前后 free/total；
- warmup 后 guide、dense validation、total 的 P50/P95；
- left/right/inter-arm active pairs 和粗筛比例；
- 最终 valid rate、左右 EE error、minimum environment/intra/inter-arm clearance；
- 相对 production full validator 的 false-positive/false-negative；
- Foam profile/hash 和 sphere/pair counts。

CUDA OOM 必须被单次 case 捕获、写入 JSON，并在清理 CUDA cache 后继续后续 case。A0 若 OOM，仍保留其 OOM 结果，不能因此跳过 A1--A5。

## 8. 阶段提交

1. `docs: plan Marvin collision inference optimizations`
2. `feat: stream Marvin self-collision pair reductions`
3. `feat: chunk Marvin dense trajectory validation`
4. `feat: add conservative Marvin parent-bound broad phase`
5. `feat: add Foam reduced Pika guide geometry`
6. `test: benchmark Marvin collision inference ablations`

每个 commit 前运行该阶段单元测试；最后在 `mpd-splines-public` 环境运行完整相关测试和可执行的 CUDA 消融。若系统当时没有足够空闲显存，报告 device-wide 占用并保留可复现命令，不用 CPU 结果冒充 CUDA 结果。
