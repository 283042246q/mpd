# Warehouse narrow-0.20：No-grad 原路径与融合路径对比（2026-08-03）

## 1. 目的

在 `Warehouse-narrow-0.20 / hard` 上只比较当前 no-grad CostGuide 下的两条 B-spline 路径：

1. 原物化路径：`mapping.fused_bspline_integration: false`；
2. 融合路径：`mapping.fused_bspline_integration: true`。

同时保留 2026-08-03 多场景正式测试中的 A2P-fast 旧记录。旧记录来自本轮 no-grad/解析梯度/融合实现之前，且使用 512 点 dense validation；当前两条路径统一使用 128 点。因此 Guide/Inference 可以作方向性比较，dense 和总耗时不能当作相同检查密度的等价比较。

## 2. 固定条件

| 项目 | 值 |
|---|---|
| 场景 | `narrow_020` / hard / gap 0.20 m |
| 流程 | A2P-fast，Temporal 关闭 |
| checkpoint | Warehouse Panda 相同 checkpoint |
| seed | 2 |
| candidates | 32 |
| contexts | 1 |
| 当前重复数 | 5，按 materialized/fused 交替执行 |
| 当前有效性筛选 | 128 点 |
| 旧记录重复数 | 2 |
| 旧记录有效性筛选 | 512 点 |

两条当前路径共同启用：

- 整个 CostGuide 使用 `torch.no_grad()`；
- 零权重 cost 跳过；
- Velocity、Acceleration 和 PathLength 使用解析梯度；
- ParentLink dense fast path；
- 相同 start/goal、候选数和 guidance 参数。

唯一配置差异是：

```yaml
gradient_pruning:
  mapping:
    fused_bspline_integration: false  # 原物化路径
    # 或 true                         # 融合路径
```

## 3. 当前 5 次原始计时

单位为秒。

| 路径 | Repeat | Guide | Generator | Inference | Dense-128 | Inference + dense |
|---|---:|---:|---:|---:|---:|---:|
| No-grad 原路径 | 0 | 0.755477 | 0.115353 | 0.873721 | 0.220014 | 1.093734 |
| No-grad 原路径 | 1 | 0.688917 | 0.109035 | 0.800421 | 0.223532 | 1.023954 |
| No-grad 原路径 | 2 | 0.663790 | 0.109370 | 0.775602 | 0.222460 | 0.998062 |
| No-grad 原路径 | 3 | 0.642921 | 0.108967 | 0.754423 | 0.219945 | 0.974368 |
| No-grad 原路径 | 4 | 0.677396 | 0.108481 | 0.788350 | 0.219828 | 1.008178 |
| No-grad 融合路径 | 0 | 0.811489 | 0.192756 | 1.008258 | 0.220258 | 1.228516 |
| No-grad 融合路径 | 1 | 0.680499 | 0.109499 | 0.792502 | 0.223398 | 1.015900 |
| No-grad 融合路径 | 2 | 0.665865 | 0.109903 | 0.778242 | 0.226040 | 1.004282 |
| No-grad 融合路径 | 3 | 0.655718 | 0.108440 | 0.766656 | 0.230202 | 0.996857 |
| No-grad 融合路径 | 4 | 0.632759 | 0.110246 | 0.745508 | 0.215601 | 0.961109 |

融合 repeat-0 同时出现 Generator 和 Guide 抖高；以下结论使用 5 次 p50，不用单次异常值。

## 4. p50 与旧记录

| 版本 | CostGuide graph | B-spline 映射 | Dense 点数 | Guide p50 | Generator p50 | Inference p50 | Dense p50 | 总耗时 p50 | Valid | Collision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧 A2P-fast 正式记录 | 原实现 | 原物化 | 512 | **0.652278** | 0.113862 | **0.774840** | 0.878566 | 1.653406 | 90.625% | 0.0% |
| 当前 No-grad 原路径 | no-grad + 解析 joint costs | 原物化 | 128 | 0.677396 | **0.109035** | 0.788350 | **0.220014** | 1.008178 | 90.625% | 0.0% |
| 当前 No-grad 融合路径 | no-grad + 解析 joint costs | 融合 | 128 | 0.665865 | 0.109903 | 0.778242 | 0.223398 | **1.004282** | 90.625% | 0.0% |

### 同为 No-grad：融合相对原路径

| 指标 | 加速比 | 变化 |
|---|---:|---:|
| Guide | `1.0173x` | -1.70% 时间 |
| Inference | `1.0130x` | -1.28% 时间 |
| Dense validation | `0.9849x` | +1.54% 时间，属于独立 validator 抖动 |
| Inference + dense | `1.0039x` | -0.39% 时间 |

### 当前路径相对旧 Guide 记录

- 当前 No-grad 原路径：`0.652278 / 0.677396 = 0.9629x`，比旧记录慢约 3.85%。
- 当前 No-grad 融合路径：`0.652278 / 0.665865 = 0.9796x`，比旧记录慢约 2.08%。
- 当前融合 Inference 为 `0.778242 s`，与旧记录 `0.774840 s` 相差约 0.44%，没有可辨识的整体推理加速。

因此本场景不能证明 no-grad、零权重跳过和解析 joint costs 带来明显加速。融合映射相对当前原路径有约 1–2% 的 Guide/Inference 收益，但加入 128 点 validator 后只剩约 0.4%。

## 5. 质量一致性

| 指标 | 旧记录 | No-grad 原路径 | No-grad 融合路径 |
|---|---:|---:|---:|
| Valid | 90.625% | 90.625% | 90.625% |
| Collision | 0.0% | 0.0% | 0.0% |
| EE position mean | 0.029895 m | 0.029895 m | 0.029895 m |
| EE orientation mean | 2.449545° | 2.449545° | 2.449552° |

融合与原路径最终 control points 平均绝对差为 `7.08e-06`、最大绝对差为 `0.00421`；轨迹点平均绝对差为 `5.30e-06`、最大绝对差为 `0.00202`。这是 float32 求和顺序在 60 次 guide call 中累积的差异，没有改变本次有效性或碰撞判断。

## 6. 128 点与旧 512 点的影响

当前 dense p50 约 `0.22 s`，旧 512 点为 `0.879 s`，约为原来的四分之一。当前总耗时相对旧记录约有 `1.64x` 改善，但主要来自有效性筛选从 512 点改为 128 点，不能归因于 no-grad 或 B-spline 融合。

结论：

1. 融合开关可以保留为默认开启，质量一致且有小幅收益；
2. 不应把本次结果表述为明显的梯度加速；
3. no-grad/解析小项的收益低于本场景的进程和 GPU 抖动；
4. 如需继续获得明显 Guide 提升，应转向减少映射次数、跨 cost 合并 q-space gradient，或减少 guide call，而不是继续优化当前单次融合 contraction。

## 7. 结果位置

- 当前原路径配置与 5 次结果：`benchmark_results/gradient_pruning_nograd_fusion_narrow020_20260803/materialized/`
- 当前融合路径配置与 5 次结果：`benchmark_results/gradient_pruning_nograd_fusion_narrow020_20260803/fused/`
- 旧记录：`benchmark_results/gradient_pruning_new_flow_20260803/warehouse/reports/ablation-summary.csv`

