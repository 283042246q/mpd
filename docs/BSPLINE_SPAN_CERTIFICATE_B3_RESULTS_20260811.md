# B3 基础上的保守 B-spline span certificate 验证（2026-08-11）

## 结论

连续时间安全证明已经作为独立开关接到 B3 上，并完成 7 个 Panda 场景测试。当前版本数值
安全，但性能为明确的负优化，不建议默认开启：

- 7 个场景中 B3 与 span certificate 的 dense-valid rate、collision rate 完全一致；
- 所有场景 Guide 都变慢，Guide 加速比（B3/Span）的几何平均为 `0.408x`；
- 首层 span 认证率平均只有 `0.095%`；
- 未认证时间覆盖率在 depth 0/1/2/3 后平均仍为
  `99.91% / 92.70% / 76.11% / 57.84%`；
- 严格连续 SDF 预扫是主要成本，同步诊断中每个 guide call 的中点
  FK/SDF/bound-arithmetic 平均为 `27.88 / 105.23 / 5.47 ms`；导数上界另为
  `0.61 ms`。

因此问题不在 derivative-control-point 上界的计算，而在全局运动界过松导致必须不断细分，
以及为保持 1-Lipschitz 证明而执行的 analytic primitive SDF 查询过贵。

## 实现范围

开关：

```yaml
gradient_pruning:
  span_certificate:
    enabled: true
    max_subdivision_depth: 3
    environment_safe_margin: 0.08
    self_safe_margin: 0.06
    jacobian_bound_mode: componentwise
    exact_sdf_for_certificate: true
    grid_error_scale: 1.0
    profile_stages: false
```

实验路径满足以下约束：

1. 父路径为 B3：A2P-fast、no-grad、原 B-spline 映射、active `J^Tg`；
2. 不使用 parent 包络体；预扫仍是原始 56 个 fine collision spheres；
3. 不启用 C1 link broad phase，不缓存轨迹 pose、sphere center、distance 或 Jacobian；
4. 配置无关的每球 Jacobian-column reach bound 只按机器人结构生成一次；
5. 每次 guide call 根据当前控制点重新计算 derivative control points 和 span certificate；
6. 认证失败的叶区间回到原 B3 J-FK/SDF/梯度路径；最终 dense=128 validator 不变。

逐关节形式比单个 Frobenius 界更紧，但仍保守：

\[
|q'_j(s)|\le V_{k,j},\qquad
\|\dot x_c(s)\|\le\sum_j R_{j,c}V_{k,j}=L_{k,c}.
\]

环境碰撞球下界为：

\[
d^{LB}_{k,c}=d_c(s_m)-L_{k,c}
\max(|s_m-s_l|,|s_r-s_m|).
\]

自碰撞球对使用：

\[
d^{LB}_{k,a,b}=d_{a,b}(s_m)-(L_{k,a}+L_{k,b})
\max(|s_m-s_l|,|s_r-s_m|).
\]

当前 GridMapSDF 是 nearest-grid 查询，本身不满足连续 1-Lipschitz。正式 certificate 默认直接
查询 GridMapSDF 保存的原始 sphere/box primitive，避免把经验 grid 查询误称为严格证明。
仍保留 grid-error 回退实验开关，但它不是本轮正式数据路径。

## 测试协议

- 100 candidates；
- 15 DDIM steps；
- 最后 20%（3 steps）开启 guidance；
- 每个 guided step 6 iterations，共 18 guide calls；
- Temporal、Clean-x0、融合映射、parent envelope、C1 scan cache 均关闭；
- dense validation=128；
- seed=0；同场景 B3/Span 使用相同 start/goal；
- latency 运行 3 次，报告 p50；
- 分项时间另跑 1 次同步诊断，不与 latency p50 混合。

TorchKin 当前只暴露联合 J-FK，因此不能把该 kernel 内的 FK 与 Jacobian 单独计时；表中将它
如实记为 J-FK。

## 时间、认证率与准确度

加速比定义为 `B3 Guide / Span Guide`，小于 1 表示负优化。`d0` 即首层认证后剩余的
原 span 时间测度，d1--d3 为后续二分层。

| 场景 | B3/Span Guide p50 (s) | 加速 | B3/Span inference (s) | B3/Span valid | B3/Span collision | 首层认证 | 剩余 d0/d1/d2/d3 | Active 点 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Warehouse open | 1.4358 / 2.6001 | 0.552x | 1.6681 / 2.8430 | 91% / 91% | 0% / 0% | 0.10% | 99.9/82.3/55.2/32.7% | 47.5% |
| Warehouse single | 1.5642 / 4.3821 | 0.357x | 1.8072 / 4.6327 | 91% / 91% | 0% / 0% | 0.00% | 100.0/97.7/73.1/49.4% | 68.2% |
| Warehouse narrow-0.20 | 1.6802 / 3.3644 | 0.499x | 1.9200 / 3.6825 | 90% / 90% | 0% / 0% | 0.46% | 99.5/78.0/49.7/30.6% | 43.5% |
| Warehouse narrow-0.14 | 1.6262 / 3.8102 | 0.427x | 1.8785 / 4.0955 | 75% / 75% | 17% / 17% | 0.10% | 99.9/91.1/73.0/55.6% | 75.0% |
| ThreePillars | 1.6283 / 3.4625 | 0.470x | 1.9187 / 3.7147 | 6% / 6% | 94% / 94% | 0.00% | 100.0/99.9/85.6/69.3% | 92.5% |
| Drawer-to-shelf | 1.6912 / 5.8381 | 0.290x | 1.9642 / 6.0779 | 0% / 0% | 100% / 100% | 0.00% | 100.0/100.0/99.6/87.4% | 99.5% |
| To-drawer | 1.7205 / 5.2601 | 0.327x | 1.9627 / 5.5053 | 28% / 28% | 67% / 67% | 0.00% | 100.0/99.9/96.5/79.9% | 91.2% |

## Bound 开销与省下的计算

以下是同步诊断的每 guide call p50。`筛选开销` 包含 derivative bound、中点 FK、连续
primitive SDF 和 lower-bound arithmetic；`省下` 为 B3 对应阶段减去 Span 活跃阶段，负数
表示分桶/dispatch 后反而更慢。

| 场景 | derivative/FK/SDF/bound (ms) | 筛选开销 (ms) | J-FK 省下 (ms) | active SDF 省下 (ms) | 筛选开销-省下 (ms) |
|---|---:|---:|---:|---:|---:|
| Warehouse open | 0.91 / 29.36 / 54.40 / 3.91 | 88.58 | -5.51 | 22.83 | +71.25 |
| Warehouse single | 0.91 / 32.53 / 108.92 / 4.96 | 147.31 | -12.82 | 10.98 | +149.15 |
| Warehouse narrow-0.20 | 0.50 / 25.14 / 69.10 / 3.45 | 98.18 | 6.63 | 28.89 | +62.67 |
| Warehouse narrow-0.14 | 0.42 / 19.33 / 98.17 / 4.70 | 122.62 | 9.15 | 18.14 | +95.33 |
| ThreePillars | 0.41 / 25.09 / 49.47 / 7.00 | 81.98 | -4.34 | -1.89 | +88.22 |
| Drawer-to-shelf | 0.42 / 26.64 / 181.91 / 7.11 | 216.08 | -2.99 | -1.00 | +220.07 |
| To-drawer | 0.71 / 37.07 / 174.64 / 7.18 | 219.60 | -9.08 | -1.44 | +230.12 |

即使在收益最好的 narrow-0.20 中，约 `35.5 ms` 的 J-FK/SDF 节省仍覆盖不了
`98.2 ms` 的 certificate 筛选。

## 为什么首层几乎无法认证

1. 初始 knot span 较宽，轨迹的 derivative-control-point 上界在扩散早期尤其大；
2. 配置无关的 kinematic reach bound 必须覆盖所有 Panda 姿态，比当前姿态 Jacobian 松；
3. 自碰撞使用 `L_a + L_b`，没有利用相邻 link 的共同运动抵消；
4. 每个 span 必须让全部 56 个环境球和全部 self-pair 同时通过；任一球不确定就细分；
5. Drawer/ThreePillars 中大部分轨迹本来就靠近或进入障碍，certificate 理应难以通过；
6. 到 depth 3 后仍需把叶区间映射到固定 32/64/128 bucket，active ratio 高时不能形成规则
   dense B3 那样高效的大 kernel。

## 后续优化判断

当前形式不应继续通过增加 subdivision depth 来优化：这会增加中点 FK/SDF 数，且高难场景
最终仍接近 full dense。

如果继续研究，建议按以下顺序：

1. **先做低成本 gate**：用控制点导数和障碍 AABB 距离判断理论下界是否有机会通过；预计
   certificate 覆盖低于阈值时直接执行 B3，不启动任何中点 FK/SDF。
2. **缩紧 self-collision 相对运动界**：对球对直接预计算每关节的相对 Jacobian-column
   reach bound，而不是简单相加两个绝对 reach；这是首层认证率的主要改进空间之一。
3. **混合 grid/analytic SDF**：地图内使用带严格量化余量的 grid lower bound，只对越界点
   或误差带内点查询 primitive；必须先补齐 grid projection 的连续误差证明。
4. **层级批量而非重复 dispatch**：一次 FK 得到多个 subdivision midpoint 层，或将固定
   depth 的所有中点放入单个规则 GPU tensor，再用 mask 做 bound；当前逐层 FK launch 很贵。
5. **更紧的局部 Jacobian 界**：需要 Hessian/interval bound 才能保持证明；仅采样当前
   Jacobian 最大值不能称为 conservative certificate。

在这些问题解决前，B3 仍是推荐默认路径，`span_certificate.enabled` 保持 `false`，也不加入
autoset。

## 代码、测试和结果

- 实现：`mpd/inference/collision_risk_selector.py`；
- derivative control points：`mpd/parametric_trajectory/trajectory_bspline.py`；
- 配置：`mpd/inference/guidance_config.py`；
- benchmark：`scripts/inference/benchmark_span_certificate.py`；
- 单元测试覆盖 derivative convex-hull bound、独立开关约束、全安全 K=0 和不确定 span 回退；
- 原始结果：
  - `benchmark_results/span_certificate_warehouse_20260811/`
  - `benchmark_results/span_certificate_three_pillars_20260811/`
  - `benchmark_results/span_certificate_drawer_to_shelf_20260811/`
  - `benchmark_results/span_certificate_to_drawer_20260811/`

