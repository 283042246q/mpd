# Gradient Pruning 新流程多场景测试（2026-08-03）

## 1. 结论

本轮在 7 个场景上完成 5 个关键流程点的成对测试，共 70 次 CUDA 推理。

1. **默认使用 A2P-fast**。它保持了 Legacy 的轨迹质量，7 场景 Guide 几何平均加速 `1.121x`，包含 512 点 dense validation 后总耗时几何平均加速 `1.059x`。
2. **Clean-x0/4 不应开启**。其 Guide 几何平均仅 `1.011x`，总耗时 `1.007x`，但在 ThreePillars 和两个 Drawer 上有效率降为 `0%–3.1%`，末端误差明显恶化。
3. **Clean-x0/2 只能作为速度实验，不应进入默认流程**。Guide 几何平均 `2.658x`、含 dense 总耗时 `1.516x`，主要来源是 guide iteration 从 4 减为 2，而不是 Clean-x0 本身；末端误差和困难场景有效率不通过质量门槛。
4. **Temporal 默认关闭**。它的 Active 点比例平均降到 `61.4%`，但 Guide 和含 dense 总耗时几何平均分别为 `0.953x`、`0.950x`，即总体负优化。7 个场景中只有 `warehouse/open_clearance` 值得开启。
5. ThreePillars、DrawerToShelf 和 ToDrawer 上 Legacy 本身的有效率也较低。本轮可用于**同输入的性能成对比较和质量 smoke gate**，不能替代多 seed 的规划成功率评测；这三个场景还需要匹配场景的训练模型或继续调 guidance。

## 2. 使用的现有测试基础设施

仓库内以下脚本可直接复用：

- `scripts/inference/benchmark_gradient_pruning_ablation.py`：同 checkpoint、同 seed、同起终点的增量消融，生成 timing、active statistics 和 dense validation 报告。本轮增加了 `--variants`，允许只执行新流程关键点。
- `scripts/inference/validate_constrained_panda_regions.py`：对 ThreePillars 和两个 Drawer 做区域采样、IK、OMPL 规划及 dense PyBullet/Torch 碰撞交叉校验。
- `tests/test_gradient_pruning_ablation.py`：校验关键 variant 的配置关系和筛选执行数量。

约束区域预检使用 `samples=5`、`plans=1`、`planner-time=10 s`、`interpolate-num=256`：

| 场景 | 结果 | 备注 |
|---|---|---|
| ToDrawer | PASS | 5 组采样、1 条 dense-valid OMPL 路径 |
| DrawerToShelf | PASS（重试后） | 第一条 OMPL 路径被 dense PyBullet 拒绝，第二条通过，说明该场景对离散碰撞检测较敏感 |
| ThreePillars | PASS | 5 组采样、1 条 dense-valid OMPL 路径 |

## 3. 测试设置

- 候选轨迹：32
- 上下文：1
- timing 重复：2
- dense checker：512 点
- 设备：`cuda:0`
- 所有 variant：相同 checkpoint、seed、场景和候选数
- 重复次数用于降低计时波动；因为 seed 相同，质量数字是 smoke test，不是两个独立统计样本
- `总耗时` = `inference_total + dense_validation`

关键流程点：

| 简称 | 配置 |
|---|---|
| Legacy | 旧全量 guidance |
| A2P-fast | Endpoint + ParentLink + dense parent fast path，Temporal 关闭 |
| Clean-4 | A2P-fast 后接 Clean-x0 A1，4 次 guide iteration |
| Clean-2 | A2P-fast 后接 Clean-x0 A1，2 次 guide iteration |
| Temporal | A2P-fast 后接 Temporal，同 DDIM step 复用 selection，GPU 向量化 selector |

## 4. 完整结果

`Guide×` 和 `总×` 均相对同场景 Legacy；大于 1 表示加速。`Active` 是进入精细计算的时间点比例。末端位置误差单位为米、姿态误差单位为度。

| 场景 | 流程点 | Guide s | Guide× | 总耗时 s | 总× | Active | Valid | Collision | EE pos | EE ori |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Warehouse-clear / simple | Legacy | 0.664 | 1.00 | 1.656 | 1.00 | 100.0% | 100.0% | 0.0% | 0.020 | 1.04 |
|  | A2P-fast | 0.596 | 1.12 | 1.576 | 1.05 | 100.0% | 100.0% | 0.0% | 0.020 | 1.04 |
|  | Clean-4 | 0.664 | 1.00 | 1.652 | 1.00 | 100.0% | 96.9% | 3.1% | 0.047 | 5.88 |
|  | Clean-2 | 0.316 | 2.10 | 1.311 | 1.26 | 100.0% | 93.8% | 6.2% | 0.040 | 4.83 |
|  | Temporal | 0.482 | 1.38 | 1.455 | 1.14 | 32.8% | 100.0% | 0.0% | 0.020 | 1.04 |
| Warehouse-single / medium | Legacy | 0.684 | 1.00 | 1.676 | 1.00 | 100.0% | 100.0% | 0.0% | 0.032 | 1.50 |
|  | A2P-fast | 0.596 | 1.15 | 1.586 | 1.06 | 100.0% | 100.0% | 0.0% | 0.032 | 1.50 |
|  | Clean-4 | 0.712 | 0.96 | 1.739 | 0.96 | 100.0% | 100.0% | 0.0% | 0.045 | 5.36 |
|  | Clean-2 | 0.336 | 2.04 | 1.348 | 1.24 | 100.0% | 100.0% | 0.0% | 0.043 | 4.52 |
|  | Temporal | 0.886 | 0.77 | 1.961 | 0.85 | 44.0% | 100.0% | 0.0% | 0.032 | 1.49 |
| Warehouse-narrow-0.20 / hard | Legacy | 0.689 | 1.00 | 1.672 | 1.00 | 100.0% | 90.6% | 0.0% | 0.030 | 2.45 |
|  | A2P-fast | 0.652 | 1.06 | 1.653 | 1.01 | 100.0% | 90.6% | 0.0% | 0.030 | 2.45 |
|  | Clean-4 | 0.701 | 0.98 | 1.693 | 0.99 | 100.0% | 90.6% | 0.0% | 0.048 | 6.68 |
|  | Clean-2 | 0.341 | 2.02 | 1.343 | 1.24 | 100.0% | 93.8% | 3.1% | 0.045 | 7.82 |
|  | Temporal | 0.668 | 1.03 | 1.683 | 0.99 | 31.0% | 87.5% | 3.1% | 0.030 | 2.46 |
| Warehouse-narrow-0.14 / hard | Legacy | 0.684 | 1.00 | 1.674 | 1.00 | 100.0% | 71.9% | 18.8% | 0.024 | 1.70 |
|  | A2P-fast | 0.618 | 1.11 | 1.614 | 1.04 | 100.0% | 71.9% | 18.8% | 0.024 | 1.70 |
|  | Clean-4 | 0.713 | 0.96 | 1.746 | 0.96 | 100.0% | 75.0% | 15.6% | 0.039 | 6.30 |
|  | Clean-2 | 0.340 | 2.01 | 1.388 | 1.21 | 100.0% | 75.0% | 15.6% | 0.034 | 4.93 |
|  | Temporal | 0.665 | 1.03 | 1.642 | 1.02 | 63.0% | 71.9% | 18.8% | 0.024 | 1.69 |
| ThreePillars / hard | Legacy | 2.103 | 1.00 | 3.158 | 1.00 | 100.0% | 3.1% | 90.6% | 0.020 | 1.36 |
|  | A2P-fast | 1.650 | 1.28 | 2.683 | 1.18 | 100.0% | 3.1% | 90.6% | 0.020 | 1.37 |
|  | Clean-4 | 1.683 | 1.25 | 2.662 | 1.19 | 100.0% | 0.0% | 100.0% | 0.112 | 28.24 |
|  | Clean-2 | 0.584 | 3.60 | 1.640 | 1.93 | 100.0% | 0.0% | 96.9% | 0.074 | 27.14 |
|  | Temporal | 1.939 | 1.08 | 2.986 | 1.06 | 84.2% | 3.1% | 90.6% | 0.020 | 1.38 |
| DrawerToShelf / hard | Legacy | 2.957 | 1.00 | 3.943 | 1.00 | 100.0% | 3.1% | 96.9% | 0.013 | 0.86 |
|  | A2P-fast | 2.582 | 1.15 | 3.574 | 1.10 | 100.0% | 3.1% | 96.9% | 0.013 | 0.84 |
|  | Clean-4 | 2.975 | 0.99 | 3.974 | 0.99 | 100.0% | 0.0% | 100.0% | 0.195 | 24.37 |
|  | Clean-2 | 0.561 | 5.27 | 1.644 | 2.40 | 100.0% | 0.0% | 100.0% | 0.197 | 23.28 |
|  | Temporal | 3.604 | 0.82 | 4.633 | 0.85 | 91.2% | 3.1% | 96.9% | 0.013 | 0.82 |
| ToDrawer / medium | Legacy | 1.699 | 1.00 | 2.696 | 1.00 | 100.0% | 28.1% | 71.9% | 0.025 | 1.40 |
|  | A2P-fast | 1.671 | 1.02 | 2.727 | 0.99 | 100.0% | 31.2% | 68.8% | 0.025 | 1.40 |
|  | Clean-4 | 1.769 | 0.96 | 2.767 | 0.97 | 100.0% | 3.1% | 96.9% | 0.150 | 29.26 |
|  | Clean-2 | 0.600 | 2.83 | 1.599 | 1.69 | 100.0% | 9.4% | 90.6% | 0.131 | 26.73 |
|  | Temporal | 2.388 | 0.71 | 3.418 | 0.79 | 83.6% | 28.1% | 71.9% | 0.025 | 1.37 |

## 5. 跨场景汇总

速度使用 7 个场景加速比的几何平均；Valid/Collision 是场景均值，只用于观察本次固定输入的质量变化。

| 流程点 | Guide 几何平均加速 | 含 dense 总加速 | Valid 均值 | Collision 均值 | Active 均值 | 判断 |
|---|---:|---:|---:|---:|---:|---|
| Legacy | 1.000x | 1.000x | 56.7% | 39.7% | 100.0% | 对照 |
| A2P-fast | **1.121x** | **1.059x** | 57.1% | 39.3% | 100.0% | **默认开启** |
| Clean-4 | 1.011x | 1.007x | 52.2% | 45.1% | 100.0% | 关闭：无稳定收益且质量退化 |
| Clean-2 | 2.658x | 1.516x | 53.1% | 44.6% | 100.0% | 仅实验：guide 次数变化导致不可等价 |
| Temporal | 0.953x | 0.950x | 56.2% | 40.2% | 61.4% | 默认关闭，按场景静态开启 |

## 6. 按场景开关建议（不使用 autoset）

| 场景 | A2P-fast | Clean-x0 | Temporal | 原因 |
|---|---|---|---|---|
| Warehouse-clear | 开 | 关 | **开** | Temporal 相对 A2P-fast Guide `1.237x`，总耗时相对 Legacy `1.138x`，质量相同 |
| Warehouse-single | 开 | 关 | 关 | Temporal 相对 A2P-fast Guide `0.673x`，明显负优化 |
| Warehouse-narrow-0.20 | 开 | 关 | 关 | Active 虽仅 31.0%，但总耗时仍为 Legacy 的 `1.007x`，收益不足且碰撞率增加 3.1pp |
| Warehouse-narrow-0.14 | 开 | 关 | 关 | Temporal 相对 A2P-fast Guide `0.930x`，总收益仅约 2%，不值得引入分支开销 |
| ThreePillars | 开 | 关 | 关 | Temporal 相对 A2P-fast Guide `0.851x`；基础有效率也需另行解决 |
| DrawerToShelf | 开 | 关 | 关 | Temporal 相对 A2P-fast Guide `0.716x`，总耗时负优化约 29.6% |
| ToDrawer | 开 | 关 | 关 | Temporal 相对 A2P-fast Guide `0.700x`，总耗时负优化约 25.3% |

不要只用 `active_time_ratio` 决定 Temporal：Warehouse-narrow-0.20 的 Active 为 31.0%，低于 Warehouse-clear 的 32.8%，但前者没有端到端收益。selector、分桶、稀疏 gather/scatter、kernel launch，以及各碰撞项的工作量共同决定盈亏。autoset 尚未实现时，应使用上述场景静态配置。

## 7. 对负优化步骤的解释与下一步

### Temporal

- 当前 selection 复用和 GPU 向量化已经消除了同 DDIM step 多 guide iteration 的重复选择，但没有消除稀疏索引、张量整理和小 kernel 的固定成本。
- Active 点较高（ThreePillars/Drawer 为 84%–91%）时，节省的 FK/Jacobian 不足以覆盖固定开销。
- 即使 Active 较低，也可能因为样本分布碎片化、bucket 形状变化和 dense validation 占比而没有端到端收益。
- 下一步应先加入**只读 profiler 数据**：selector、gather/scatter、FK、Jacobian、cost、backward 分项；再决定是否做固定 shape/persistent buffer。不要仅继续调时间阈值。

### Clean-x0

- Clean-4 相对 A2P-fast 在 6/7 场景的 Guide 都是负优化或近似持平，说明“改成 clean x0”本身没有计算优势。
- Clean-2 的高加速来自 guide iteration 减半；它改变了优化预算，且 EE orientation 在困难场景从约 `0.8–1.4°` 恶化到 `23–27°`，不能作为等价优化。
- 如需继续研究，应该单独把 guide step 数作为算法质量/速度超参数，不再归入梯度剪枝等价加速链。

### Dense validation

Warehouse 的 512 点 dense validation 约 `0.86–0.95 s`，在简单场景已经超过推理部分。因此 Guide `2x` 不会转化为总流程 `2x`。若部署流程允许，可独立研究分层 dense checker；这与 guidance 内的 Temporal 分桶是两个不同问题。

## 8. 结果文件

- `benchmark_results/gradient_pruning_new_flow_20260803/warehouse/reports/ablation-summary.csv`
- `benchmark_results/gradient_pruning_new_flow_20260803/three_pillars/reports/ablation-summary.csv`
- `benchmark_results/gradient_pruning_new_flow_20260803/drawer_to_shelf/reports/ablation-summary.csv`
- `benchmark_results/gradient_pruning_new_flow_20260803/to_drawer/reports/ablation-summary.csv`

每个目录还包含 generated configs、逐次 inference report、active statistics 和 executed command，能够按原 seed 复现。
