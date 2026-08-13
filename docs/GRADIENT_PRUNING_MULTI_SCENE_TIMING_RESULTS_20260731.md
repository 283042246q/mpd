# Gradient Pruning 多场景计时结果（2026-07-31，2026-08-03 更新）

> 新流程在 Warehouse 四档、ThreePillars 和两个 Drawer 上的正式成对测试，见
> [GRADIENT_PRUNING_NEW_FLOW_MULTI_SCENE_RESULTS_20260803.md](GRADIENT_PRUNING_NEW_FLOW_MULTI_SCENE_RESULTS_20260803.md)。本页后半部分保留旧顺序数据用于追溯负优化来源。

## 2026-08-03 实现更新

本页原有数字来自 2026-07-31 的旧顺序 `A2 → Temporal → Parent-link`，用于说明负优化来源，不能直接当作新实现的 A2P-fast 计时结果。当前基础链与实验分支改为：

| 阶段 | 当前定义 |
|---|---|
| A0 | Legacy guidance |
| A1 | 新 dispatcher，完整时间轴、sphere-link Jacobian |
| A2 | A1 + EE goal 只在末端计算 |
| A2P | A2 + parent-link kinematics，通用全量 bucket 路径 |
| A2P-fast | A2P + dense parent fast path；Temporal 关闭时不创建 selector、不 gather/scatter |
| A2PF-C | A2P-fast + clean-x0 A1，Temporal 关闭；分别测试 4/2 guide steps |
| A3R | A2P-fast + Temporal；K0/K32/K64 走稀疏路径，完整时间轴候选回退到 dense parent fast path |

旧 `A3`（A2 后直接 Temporal、没有 Parent-link）和 full-scan 现在只保留为诊断对照。clean-x0 和 Temporal 都直接以 A2P-fast 为父阶段，避免把两个实验优化的时间与质量影响混在一起。

对应配置为：

```yaml
gradient_pruning:
  enabled: true
  endpoint:
    ee_only_last_point: true
  spatial:
    parent_link_kinematics: true
    dense_parent_fast_path: true
  temporal:
    enabled: false  # A2P-fast；设为 true 才是 A3R
    reuse_selection_within_ddim_step: true
```

`dense_parent_fast_path` 只有在 `parent_link_kinematics: true` 时生效。未加入 autoset；Temporal 仍由显式开关控制。

### 2026-08-03 CUDA smoke

RTX 4090 D、Warehouse open-clearance、1 context、4 candidates、1 repeat、128 点 validator 的端到端 smoke 如下。该规模只用于确认路径可执行和初步方向，不替代多 context、多 repeat 正式统计。

| 阶段 | Guide | 相对前一步 | Inference | Active | Valid |
|---|---:|---:|---:|---:|---:|
| A2P 通用全量路径 | 0.136 s | — | 0.191 s | 100.0% | 100.0% |
| A2P-fast | 0.118 s | 1.158x | 0.170 s | 100.0% | 100.0% |
| A3R Temporal on A2P-fast | 0.175 s | 0.672x | 0.226 s | 28.6% | 100.0% |
| 旧 Temporal、无 Parent-link（诊断） | 0.250 s | — | 0.303 s | 28.6% | 100.0% |

这次 smoke 说明 dense fast path 已消除一部分通用 bucket 开销；Temporal 虽比旧顺序快，但在该小样本上仍不足以抵消扫描和稀疏调度成本，因此继续默认关闭。

### 2026-08-03 Temporal cache/GPU selector 与 clean-x0 A1 smoke

在完全相同的小规模条件下重跑新版实现：同一 active DDIM step 的四次 guide iteration 中，第 0 次扫描，后 3 次复用 selection，实际 cache hit 为 `9/12=75%`。selector 分桶和 phase 选择使用 packed GPU 张量；clean-x0 使用保留原始 denoiser epsilon 的 A1。

| 指标 | 修改前 smoke | 修改后 smoke |
|---|---:|---:|
| A3R guide | 0.175 s | 0.167 s |
| A3R inference | 0.226 s | 0.219 s |
| A3R valid | 100.0% | 100.0% |
| A6 clean-x0 EE position mean | 0.057 | 0.035 |
| A6 clean-x0 EE orientation mean | 5.206 | 4.147 |

单次运行中 A3R 绝对 guide 时间下降约 `4.4%`，A1 的 EE 指标也优于此前错误的一致噪声重建分支；但 A2P-fast 本次基线波动明显，因此不能据此宣称 Temporal 已形成稳定净加速，仍需正式多场景重复测试。

### 2026-08-03 clean-x0 A1 on A2P-fast smoke

clean-x0 分支关闭 Temporal，直接使用 A2P-fast dense parent-link 基础设施。测试条件仍为 Warehouse open-clearance、1 context、4 candidates、1 repeat、128 点 validator。

| 阶段 | Guide | Inference | 含 dense 总时间 | Valid | Collision | EE pos mean | EE ori mean | Path mean | Smoothness mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A2P-fast | 0.118 s | 0.171 s | 0.378 s | 100% | 0% | 0.0134 | 1.295 | 5.623 | 38.23 |
| A2P-fast + clean-x0 A1，4 steps | 0.117 s | 0.166 s | 0.376 s | 75% | 25% | 0.0352 | 3.997 | 6.102（valid） | 102.14（valid） |
| A2P-fast + clean-x0 A1，2 steps | 0.067 s | 0.124 s | 0.334 s | 100% | 0% | 0.0308 | 4.376 | 6.114 | 98.13 |

结论：连接方式本身可行，4-step 没有形成有意义的速度收益且发生碰撞回退；2-step 相对 A2P-fast 的 guide/inference/含 dense 总时间分别约 `1.76x/1.37x/1.13x`，但 EE 误差约增至 `2.3x/3.4x`，轨迹也更长、更不平滑。当前参数下不建议启用，需要针对 clean 控制点空间重新调低 `guide_lr`、`max_perturb_x` 或 cost weight 后再做正式质量门槛测试。

## 2026-07-31 测试条件

- GPU：NVIDIA GeForce RTX 4090 D
- 每个阶段：1 context，32 candidates，2 repeats，seed 0
- 轨迹：128 点；独立 dense validator：512 点
- 所有对照使用相同 checkpoint、起终点、候选数和验证器
- Warehouse 正式结果来自 `benchmark_results/gradient_pruning_20260731/warehouse_cuda/`
- `warehouse/` 是 CUDA 不可用时的 CPU 回退诊断数据，不参与本报告结论

## 历史阶段定义

| 阶段 | 2026-07-31 定义 |
|---|---|
| A0 | Legacy guidance |
| A1 | 新 dispatcher，完整 128 点 Jacobian |
| A2 | A1 + EE goal 只在末端计算 |
| A3 | A2 + temporal bucket + coarse scan |
| A4 | A3 + parent-link kinematics |
| A5 | A2 + temporal bucket + full scan，仅作 coarse-scan 对照 |
| A6 | A4 + clean-x0 guidance，原 guide steps |
| A7 | A6 + 2 guide steps |

## 场景结论（历史实现）

表中“总加速”包括 inference 和 512 点 dense validation；K 列依次为 K0/K32/K64/K128。

| 场景 | 难度 | 推荐阶段 | Guide p50 | Guide 加速 | Inference p50 | 总加速 | Active | K 分布 | Valid | Collision |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| Warehouse open | simple | A1 | 0.213 s | 1.016x | 0.268 s | 1.003x | 100.0% | 0/0/0/100 | 93.8% | 3.1% |
| Warehouse obstacle | medium | A2 | 0.241 s | 1.047x | 0.297 s | 1.093x | 100.0% | 0/0/0/100 | 93.8% | 0.0% |
| Warehouse narrow 0.14 | hard | A1 | 0.225 s | 1.026x | 0.280 s | 1.012x | 100.0% | 0/0/0/100 | 62.5% | 18.8% |
| ThreePillars | hard | A2 | 0.573 s | 1.035x | 0.628 s | 1.021x | 100.0% | 0/0/0/100 | 9.4% | 87.5% |
| Drawer-to-shelf | hard | A2（仅计时） | 5.222 s | 1.313x | 5.335 s | 1.272x | 100.0% | 0/0/0/100 | 0.0% | 100.0% |
| To-drawer | medium | A1 | 1.017 s | 1.377x | 1.080 s | 1.192x | 100.0% | 0/0/0/100 | 46.9% | 53.1% |

Drawer-to-shelf 的 Legacy valid rate 已为 0%，因此该场景只能比较执行时间，不能用于证明功能安全。

## 功能消融（历史实现）

| 场景 | A1/A0 dispatcher | A2/A1 EE末端 | A3/A5 coarse/full | A4/A3 parent-link | A6/A4 clean-x0 | A7/A6 2 steps |
|---|---:|---:|---:|---:|---:|---:|
| Warehouse open | 1.016x | 1.005x | 1.001x | 1.271x | 1.045x | 2.201x |
| Warehouse obstacle | 1.016x | 1.030x | 1.042x | 1.364x | 1.094x | 1.852x |
| Warehouse narrow 0.14 | 1.026x | 0.944x | 1.037x | 1.323x | 0.957x | 1.978x |
| ThreePillars | 0.961x | 1.077x | 1.023x | 1.353x | 0.407x | 2.733x |
| Drawer-to-shelf | 1.279x | 1.027x | 0.975x | 1.178x | 1.992x | 5.206x |
| To-drawer | 1.377x | 0.801x | 0.742x | 1.563x | 1.313x | 2.414x |

`A3/A5` 大于 1 表示 coarse scan 比 full scan 快。Parent-link 在所有场景中均降低 A3 guide 时间，但旧 A4 相对 Legacy 的端到端时间仍为：Warehouse `0.90x/0.99x/0.90x`、ThreePillars `0.85x`、Drawer-to-shelf `0.55x`、To-drawer `0.78x`。这正是将 Parent-link 前移并增加 dense fast path 的原因。

## Temporal 与质量（历史实现）

| 场景 | Legacy valid | A3 valid | A4 valid | A3 Active | A3 K0/K32/K64/K128 |
|---|---:|---:|---:|---:|---|
| Warehouse open | 93.8% | 93.8% | 93.8% | 30.0% | 3.9/77.3/16.1/2.6 |
| Warehouse obstacle | 93.8% | 93.8% | 93.8% | 34.2% | 0.0/71.1/25.0/3.9 |
| Warehouse narrow 0.14 | 62.5% | 62.5% | 59.4% | 61.5% | 0.0/2.6/73.2/24.2 |
| ThreePillars | 9.4% | 9.4% | 9.4% | 76.1% | 0.0/0.5/47.0/52.5 |
| Drawer-to-shelf | 0.0% | 3.1% | 3.1% | 58.3% | 0.0/14.6/61.5/23.9 |
| To-drawer | 46.9% | 40.6% | 43.8% | 85.0% | 0.0/2.5/26.2/71.2 |

Temporal 确实减少了 active 点，但旧实现的风险扫描、分桶、索引与不规则张量调度开销大于 Jacobian 节省。新 A3R 已消除完整时间轴候选的不必要 gather/scatter，但是否形成净加速仍必须重新实测。

## Clean-x0 质量门槛（历史实现）

| 场景 | Legacy valid | A6 valid | A7 valid | A7 Guide 加速 | A7 总加速 |
|---|---:|---:|---:|---:|---:|
| Warehouse open | 93.8% | 90.6% | 87.5% | 1.463x | 1.069x |
| Warehouse obstacle | 93.8% | 87.5% | 90.6% | 1.396x | 1.149x |
| Warehouse narrow 0.14 | 62.5% | 50.0% | 62.5% | 1.167x | 1.034x |
| ThreePillars | 9.4% | 6.2% | 0.0% | 0.765x | 0.868x |
| Drawer-to-shelf | 0.0% | 0.0% | 0.0% | 5.411x | 3.551x |
| To-drawer | 46.9% | 31.2% | 3.1% | 2.162x | 1.477x |

A6/A7 的速度收益伴随明显且跨场景不稳定的质量回退，继续保持实验状态和默认关闭。

## 最新启用建议

1. 基础顺序使用 `A1 → A2 → A2P → A2P-fast`；A2P-fast 可在不开 Temporal 的情况下独立开启。
2. A2P 是功能等价对照，A2P-fast 才是生产候选；应首先比较 `A2P-fast/A2P`，分离规则 dense 布局带来的收益。
3. Temporal 只能接在 A2P-fast 后面。仅当 `A3R/A2P-fast` 在同一硬件、候选数、场景和 dense validator 下大于 1 且 valid rate 不下降时才开启。
4. A3R 中完整时间轴候选直接复用 dense parent fast path；K0/K32/K64 才承担稀疏 gather/scatter 成本。
5. Coarse/full scan 与旧的无 Parent-link Temporal 只作诊断，不进入生产累计开启链。
6. `compute_costs_with_xrecon` 和两步 guidance 默认关闭；没有 autoset、阈值自动搜索或在线自动启用。

## 原始结果

- `benchmark_results/gradient_pruning_20260731/warehouse_cuda/reports/ablation-summary.csv`
- `benchmark_results/gradient_pruning_20260731/three_pillars/reports/ablation-summary.csv`
- `benchmark_results/gradient_pruning_20260731/drawer_to_shelf/reports/ablation-summary.csv`
- `benchmark_results/gradient_pruning_20260731/to_drawer/reports/ablation-summary.csv`
