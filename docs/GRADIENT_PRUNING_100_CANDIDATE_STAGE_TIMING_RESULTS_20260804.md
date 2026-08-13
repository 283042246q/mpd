# 100 候选多环境分阶段计时（2026-08-04）

## 1. 结论

统一 diffusion/guide 调度后，各环境的纯 diffusion 和最终 dense check 耗时非常接近，环境差异主要出现在 cost guidance：

- diffusion：全部场景 p50 为 `49.3–53.3 ms`，最大差异约 `8.1%`；
- grad/guide：p50 为 `420.8–493.1 ms`，同一映射路径内最大差异约 `16%–17%`；
- 128 点 dense check：p50 为 `214.7–221.2 ms`（排除 ToDrawer fused 的单次 `254.7 ms` 抖动后，中位数仍为 `220.9 ms`），场景间差异约 `3%`；
- fused 相对 materialized 的 Guide 几何平均加速为 `0.9956x`，即 fused 平均慢约 `0.4%`；
- fused 相对 materialized 的 `inference + dense` 几何平均加速为 `0.9969x`，即 fused 平均慢约 `0.3%`。

因此，在当前 A2P-fast、100 candidates、128 点轨迹下，融合 B-spline 映射仍未形成可测的稳定收益。Guide 的主要耗时仍在 FK/Jacobian、碰撞距离及空间梯度，而不是最后的 B-spline 映射/积分。

ThreePillars 和两个 Drawer 的有效率仍然很低。本轮可以比较相同输入下的执行时间，但不能把这些场景作为规划质量通过的证据。

## 2. 统一测试口径

| 项目 | 设置 |
|---|---:|
| candidates | 100 |
| DDIM sampling steps | 15 |
| guide start fraction | 0.2 |
| active DDIM guide steps | `ceil(15 * 0.2) = 3` |
| guide iterations / active step | 6 |
| CostGuide 调用总数 | 18 |
| trajectory points | 128 |
| dense validation points | 128 |
| pruning | A2P-fast（Endpoint + ParentLink + dense parent fast path） |
| Temporal | 关闭 |
| Clean-x0 | 关闭 |
| mapping | materialized / fused 成对测试 |
| seed | 全部为 2 |
| timing repeats | 3 |
| context / repeat | 1 |

每次 repeat 使用相同 seed，因此 3 次重复用于估计计时中位数，不是 3 个独立质量样本。fused/materialized 按 repeat 交替执行，以减小固定顺序和热机偏差。

## 3. 分阶段耗时

以下全部为 3 次重复的 p50，单位为毫秒。`映射加速` 为 `materialized / fused`，大于 1 才表示 fused 更快。

| 场景 | Diffusion M | Grad M | Dense M | Diffusion F | Grad F | Dense F | Grad 映射加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Warehouse-clear | 50.3 | 421.6 | 216.1 | 50.4 | 420.8 | 215.1 | 1.002x |
| Warehouse-single | 53.3 | 488.5 | 221.2 | 52.4 | 493.1 | 220.1 | 0.991x |
| Warehouse-narrow-0.20 | 51.9 | 486.1 | 218.8 | 49.3 | 486.4 | 217.6 | 0.999x |
| Warehouse-narrow-0.14 | 52.1 | 488.0 | 217.4 | 52.8 | 491.0 | 218.7 | 0.994x |
| ThreePillars | 52.3 | 438.7 | 215.5 | 52.5 | 443.9 | 215.0 | 0.988x |
| DrawerToShelf | 52.3 | 435.9 | 217.2 | 52.0 | 440.0 | 216.7 | 0.991x |
| ToDrawer | 51.4 | 445.1 | 214.7 | 52.5 | 443.3 | 220.9 | 1.004x |
| **7 场景几何平均映射加速** |  |  |  |  |  |  | **0.9956x** |

`Diffusion` 对应报告中的 `generator_sec`，`Grad` 对应 `guide_sec`，`Dense` 对应 `dense_validation_sec`。

## 4. 推理合计和质量

`Inference` 是 diffusion、guide 及采样循环开销；`Inference + Dense` 不包含最终最佳轨迹选择和部分报告后处理。

| 场景 | Inference M ms | Inference+Dense M ms | Inference F ms | Inference+Dense F ms | Valid | Collision |
|---|---:|---:|---:|---:|---:|---:|
| Warehouse-clear | 475.6 | 691.7 | 474.8 | 689.9 | 98% | 0% |
| Warehouse-single | 544.9 | 766.2 | 549.7 | 770.1 | 94% | 6% |
| Warehouse-narrow-0.20 | 541.7 | 761.8 | 538.7 | 756.3 | 87% | 0% |
| Warehouse-narrow-0.14 | 542.4 | 759.8 | 547.3 | 765.3 | 84% | 14% |
| ThreePillars | 495.7 | 711.5 | 500.0 | 714.1 | 5% | 95% |
| DrawerToShelf | 492.8 | 710.5 | 496.6 | 713.8 | 1% | 99% |
| ToDrawer | 501.0 | 715.1 | 498.7 | 722.8 | 1% | 97% |

两条映射路径在每个场景得到相同的 Valid/Collision 决策。ToDrawer 的 `1% valid + 97% collision` 还包含 2% 的非碰撞约束失效。

## 5. 跨环境差异解释

本轮已经消除了原配置最明显的计时差异：Warehouse 原来是 `3 active steps * 4 iterations = 12` 次 Guide，而 ThreePillars/Drawer 原来是 `5 * 6 = 30` 次；现在全部统一为 `3 * 6 = 18` 次。

仍保留了场景原本的 cost/规划参数：

- Warehouse 使用组合的 `CostTaskSpaceEEGoalPose`；ThreePillars 和 Drawer 使用独立的 Position/Orientation cost；
- Warehouse 的 `ddim_scale_grad_prior=0.25`，ThreePillars/ToDrawer 为 `0.5`，DrawerToShelf 为 `0.35`；
- Warehouse 的 Velocity/Acceleration 权重为 `0.01/0.001`，其余场景为 `0.02/0.005`；这些权重都非零，因此不会触发零权重跳过；
- DrawerToShelf 的 `max_perturb_x=0.25`，其他场景为 `0.15`；
- 最佳轨迹选择方法不同，但该阶段不属于这里记录的 diffusion/grad/dense 三段计时。

权重数值本身通常不改变算子数量，但会改变采样轨迹、碰撞状态和空间有效集合。EE cost 的组织方式及环境碰撞几何也会改变 Guide 工作量。因此，统一调用次数后剩余约 17% 的 Guide 差异不能只解释为“场景难度”，更准确地说是环境几何、cost 结构和采样状态的共同结果。

## 6. 结果位置

- 原始根目录：`benchmark_results/gradient_pruning_100cand_stage_timing_20260804/`
- fused：`benchmark_results/gradient_pruning_100cand_stage_timing_20260804/fused/`
- materialized：`benchmark_results/gradient_pruning_100cand_stage_timing_20260804/materialized/`
- 每个运行目录包含 `args_inference.yaml`、`inference-report-000.txt`、`active-statistics.csv` 和保存的 plan tensor。

总计 `7 scenes * 2 mappings * 3 repeats = 42` 次 CUDA inference，42 份报告完整，未发现异常日志。
