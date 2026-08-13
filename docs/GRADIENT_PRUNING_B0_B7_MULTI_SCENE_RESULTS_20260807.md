# Gradient Pruning B0--B7 多场景测试（2026-08-07）

## 1. 结论

四个稀疏轴已经可以独立开启：候选、时间、link、B-spline 控制点。以
`A2P-fast + no-grad + 原始 materialized 映射`（B0）为基线，本轮 7 个场景、
8 个变体、3 次重复共完成 **168/168** 次 CUDA 推理，失败数为 0。

当前最值得默认开启的是 **B3 link sparse**：7 场景几何平均 Guide 加速
**1.668x**，完整 inference 加速 **1.561x**，包含 dense check 的端到端加速
**1.341x**；所有场景的 valid/collision 判定均与 B0 相同。

不建议无条件开启 B7：B7 的 Guide 加速为 **1.306x**，明显慢于 B3。原因是
当前时间/候选 selector 的粗扫描、分桶和动态 shape 开销，在 link 投影已经大幅
变快后更难摊薄。推荐当前默认组合为 **B3，仅开启 link sparse**；时间稀疏应按
活跃时间比例有条件开启。

## 2. B0--B7 定义

四个独立开关为：

```yaml
gradient_pruning:
  candidate:
    enabled: false              # 候选稀疏
  temporal:
    enabled: false              # 时间稀疏
  spatial:
    active_link_pruning: false  # link 稀疏
  mapping:
    sparse_bspline_support: false  # 控制点稀疏
```

所有变体都保留 `parent_link_kinematics=true`、
`dense_parent_fast_path=true`、`fused_bspline_integration=false`，所以 B0 确实是
A2P-fast/no-grad/原映射，而不是融合映射基线。

| 变体 | 候选 | 时间 | Link | 控制点 |
|---|---:|---:|---:|---:|
| B0 baseline | 关 | 关 | 关 | 关 |
| B1 candidate | 开 | 关 | 关 | 关 |
| B2 time | 关 | 开 | 关 | 关 |
| B3 link | 关 | 关 | 开 | 关 |
| B4 control-point | 关 | 关 | 关 | 开 |
| B5 candidate+time | 开 | 开 | 关 | 关 |
| B6 candidate+time+link | 开 | 开 | 开 | 关 |
| B7 all | 开 | 开 | 开 | 开 |

## 3. 统一测试协议

- 100 candidates；
- 15 DDIM steps；
- 最后 20% DDIM steps 启用 guide，即 3 个 guide DDIM steps；
- 每个 active DDIM step 6 次 guide iteration，共 18 次 guide call；
- Temporal/Clean-x0 默认关闭，只有 B2/B5/B6/B7 显式打开 temporal；
- dense validation 固定 128 点；
- seed 固定为 2；
- 每个场景、每个变体重复 3 次，时间报告 p50；
- 第 2 次重复反向执行 B7 到 B0，降低固定执行顺序偏差；
- 相同 checkpoint、场景几何和起终点。

场景覆盖 Warehouse simple/medium/hard、ThreePillars，以及 drawer 的两个运动方向：
`open_clearance`、`single_obstacle`、`narrow_020`、`narrow_014`、
`three_pillars_regions`、`drawer_to_shelf`、`to_drawer`。

## 4. 汇总时间

“Guide/Inference/Total 加速”均先对每个场景取 3 次重复的 p50，再对 7 个场景的
相对 B0 加速取几何平均。时间列是 7 个场景 p50 的算术平均。Total 包括独立的
dense validation。

| 变体 | Guide 平均 (s) | Guide 加速 | Inference 加速 | 含 dense 总加速 | Valid rate 变化 |
|---|---:|---:|---:|---:|---:|
| B0 baseline | 0.464636 | 1.000x | 1.000x | 1.000x | +0.00 pp |
| B1 candidate | 0.498269 | 0.933x | 0.942x | 0.958x | +0.00 pp |
| B2 time | 0.456466 | 1.034x | 1.030x | 1.018x | -0.14 pp |
| **B3 link** | **0.279429** | **1.668x** | **1.561x** | **1.341x** | **+0.00 pp** |
| B4 control-point | 0.462273 | 1.005x | 1.007x | 1.006x | +0.00 pp |
| B5 candidate+time | 0.456423 | 1.034x | 1.030x | 1.017x | -0.14 pp |
| B6 candidate+time+link | 0.362479 | 1.292x | 1.254x | 1.162x | -0.14 pp |
| B7 all | 0.358700 | 1.306x | 1.265x | 1.173x | -0.14 pp |

各场景 Guide 加速如下。小于 1 表示负优化。

| 场景 | B1 | B2 | B3 | B4 | B5 | B6 | B7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Warehouse open | 0.941x | 1.322x | **1.797x** | 1.005x | 1.309x | 1.462x | 1.448x |
| Warehouse single | 0.925x | 1.034x | **1.603x** | 1.006x | 1.018x | 1.165x | 1.172x |
| Warehouse narrow-0.20 | 0.934x | 1.341x | **1.596x** | 1.008x | 1.333x | 1.482x | 1.477x |
| Warehouse narrow-0.14 | 0.927x | 0.824x | **1.580x** | 0.998x | 0.816x | 1.171x | 1.178x |
| ThreePillars | 0.924x | 0.904x | **1.665x** | 1.001x | 0.925x | 1.465x | 1.500x |
| Drawer-to-shelf | 0.936x | 0.928x | **1.728x** | 1.018x | 0.927x | 1.225x | 1.281x |
| To-drawer | 0.943x | 0.997x | **1.722x** | 1.001x | 1.015x | 1.134x | 1.141x |

## 5. 准确度与轨迹一致性

| 场景 | B0 valid | B1 | B2 | B3 | B4 | B7 |
|---|---:|---:|---:|---:|---:|---:|
| Warehouse open | 98% | 98% | 98% | 98% | 98% | 98% |
| Warehouse single | 94% | 94% | 93% | 94% | 94% | 93% |
| Warehouse narrow-0.20 | 87% | 87% | 87% | 87% | 87% | 87% |
| Warehouse narrow-0.14 | 84% | 84% | 84% | 84% | 84% | 84% |
| ThreePillars | 5% | 5% | 5% | 5% | 5% | 5% |
| Drawer-to-shelf | 0% | 0% | 0% | 0% | 0% | 0% |
| To-drawer | 1% | 1% | 1% | 1% | 1% | 1% |

注意：ThreePillars 和 drawer 的绝对成功率低是该 checkpoint/场景组合的基线表现，
不能用来证明算法总体准确；本表用于判断 B1--B7 相对同场景 B0 是否退化。
由于三次重复使用相同 seed，它们是计时重复而不是三个独立准确度样本；准确度比较的
有效样本是每场景同一批 100 条候选。若要估计成功率置信区间，还需增加多 seed/
多 start-goal 测试。

- B1、B3、B4 在全部 7 个场景都保持与 B0 相同的 valid mask 和 collision mask。
- B2/B5/B6/B7 在 `single_obstacle` 中有 1/100 条候选从 valid 变为 collision，
  其余场景的离散判定不变。因此跨场景平均 valid rate 下降 0.14 个百分点。
- B3/B4 在数学上保持同一梯度映射，但 packed `index_add`/`scatter_add` 改变了
  浮点累加顺序；细小单步误差经过 18 次迭代会放大，所以最终轨迹张量并非 bitwise
  相同。B3 的最终采样轨迹最大绝对差异最高约 0.137 rad，B4 最高约 0.147 rad，
  但本轮所有 valid/collision 判定不变，EE 误差和平均路径长度也近似不变。
- B2 是有意的近似算法，最终轨迹最大差异最高约 0.304 rad；应以 dense checker
  的最终判定约束安全，而不能宣称轨迹等价。

## 6. 为什么各轴表现不同

### 6.1 候选稀疏 B1

B1 只有在候选能直接进入 K=0 时才省计算。本轮除 `to_drawer` 仅约 0.3% K=0 外，
其他场景都没有 K=0 候选；它仍需运行 selector，因此 Guide 反而平均慢约 7.2%。
当前不应单独默认开启。

### 6.2 时间稀疏 B2

| 场景 | Active time | B2 Guide 加速 |
|---|---:|---:|
| Warehouse open | 29.5% | 1.322x |
| Warehouse narrow-0.20 | 30.5% | 1.341x |
| Warehouse single | 51.0% | 1.034x |
| To-drawer | 50.5% | 0.997x |
| Drawer-to-shelf | 86.5% | 0.928x |
| Warehouse narrow-0.14 | 97.5% | 0.824x |
| ThreePillars | 100.0% | 0.904x |

时间活跃比例低于约 35% 时收益明确；约 50% 时接近盈亏平衡并受场景计算结构影响；
高于 80% 时 selector/分桶开销使其成为负优化。同一 DDIM step 的 selection 已复用，
cache hit 为 15/18 = 83.3%，所以剩余负优化不是“每个 iteration 重扫”造成的。

### 6.3 Link 稀疏 B3

当前实现对碰撞场返回的非零 task-space link gradient 做 packed 投影，只对活跃项执行
`Adjoint + J^T g`，再用 `index_add_` 汇总。这一阶段保持 FK、Jacobian 和 SDF 的原始
语义；它还不是“在 FK 之前通过 broad phase 完全跳过无关 link”。即便如此，本轮已
得到最稳定、幅度最大的收益。下一步若增加保守 link broad phase，可继续尝试跳过部分
FK/Jacobian/SDF，但必须单独做漏检等价测试。

### 6.4 控制点稀疏 B4

B4 利用 B-spline 局部支撑，只把每个相位的梯度散射到 `degree+1` 个非零基函数对应
控制点。映射计算本身在当前总 Guide 中占比很小，7 场景 Guide 几何平均仅 1.005x，
属于测量噪声/近中性。它可作为独立研究开关保留，不建议为性能默认开启。

## 7. 推荐配置与下一步

当前推荐默认：

```yaml
gradient_pruning:
  candidate:
    enabled: false
  temporal:
    enabled: false
  spatial:
    parent_link_kinematics: true
    dense_parent_fast_path: true
    active_link_pruning: true
  mapping:
    fused_bspline_integration: false
    sparse_bspline_support: false
```

后续优化优先级：

1. 把 temporal 变成保守且低开销的条件开关：只有粗扫预测 active ratio 足够低时才进入
   稀疏路径；本轮经验阈值可先从 35% 开始验证，不能直接固化为 autoset。
2. 优化时间 selector 的 GPU 动态 shape、`nonzero` 和 bucket dispatch；当前 B3 已加速
   link 投影，时间基础设施的固定开销占比进一步上升。
3. 在 B3 之后实现真正的 link broad phase，比较“只稀疏 `J^Tg`”与“连 FK/Jacobian/
   SDF 一起保守裁剪”的额外收益。
4. B4 继续保留独立开关和等价测试，但除非映射阶段占比上升，不投入默认路径复杂度。

## 8. 结果与复现文件

- 原始结果：`benchmark_results/gradient_pruning_b0_b7_multiscene_20260807/`
- 场景汇总：`reports/b0-b7-scene-summary.csv`
- 跨场景汇总：`reports/b0-b7-aggregate-summary.csv`
- 自动 Markdown：`reports/b0-b7-summary.md`
- 汇总脚本：`scripts/inference/summarize_gradient_pruning_b0_b7.py`

汇总命令：

```bash
/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  scripts/inference/summarize_gradient_pruning_b0_b7.py \
  benchmark_results/gradient_pruning_b0_b7_multiscene_20260807
```
