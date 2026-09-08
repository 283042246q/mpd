# Panda / Marvin 同数量最终有效轨迹耗时对照

测试日期：2026-09-06。

## 口径

- 两组都从空输出开始，使用 3 个 CPU worker，各生成 30 条最终有效轨迹。
- 任务方向均按 5:5 的 `random_to_placement` / `placement_to_placement` 配额计数。
- 一次 RRT 没有 exact 解时立即放弃端点对，重新采样，不对同一端点重复规划。
- Panda 保留可信单臂流程的默认 PathSimplifier、自动 range、resolution=0.01，并按照其 launch 的 `task_batch_size=1` 每次任务重建场景。
- Marvin 使用生产配置：1035 个细碰撞球、range=0.35、resolution=0.002、无 PathSimplifier；每个 shard 复用一个场景。
- 为使“最终有效”定义可比，Panda 原生 exact 路径后额外执行与 Marvin 同规格的 22 控制点、5 阶 B-spline 拟合，并以 512 点、最大关节步长 0.025 做 PyBullet 稠密检查。Panda 原生数据生成本身不包含这两次额外 audit；报告将其单列。
- 百分比以所有 worker 的累计时间为分母；墙钟时间单列。Marvin 初始化是另行实测 1.7814 秒/worker 后按 3 workers 估算，因此其占比带“估算”标记。
- Panda 使用测试种子 20260906；Marvin 保留生产配置种子 1726484688。两种采样器和状态空间不同，数值相同的 seed 也不能形成逐任务配对。本次 30 条结果用于定位瓶颈，吞吐量仍会受随机性和最慢 shard 影响。

## 总结果

| 指标 | Panda | Marvin |
|---|---:|---:|
| 最终有效轨迹 | 30 | 30 |
| 墙钟时间 | 137.65 s | 451.12 s |
| 墙钟/有效轨迹 | 4.59 s | 15.04 s |
| 吞吐 | 13.08 条/min | 3.99 条/min |
| 任务采样次数 | 100 | 102 |
| 端点成功并进入 RRT | 40 | 85 |
| RRT exact | 38 | 41 |
| 原始路径拒绝 | 7 | 1 |
| B-spline 拒绝 | 1 | 10 |
| 总任务到最终有效率 | 30.0% | 29.4% |

Marvin 的墙钟为 Panda 的 3.28 倍。两组从任务尝试到最终有效的总体成功率几乎相同，速度差主要来自每次尝试的成本结构，而不是总成功率。

## Worker 累计耗时占比

| 阶段 | Panda 时间 | Panda 占比 | Marvin 时间 | Marvin 占比 |
|---|---:|---:|---:|---:|
| 初始化 | 22.73 s | 7.36% | 5.34 s（估算） | 0.54% |
| 端点采样 / IK / 端点筛选 | 235.80 s | 76.32% | 55.99 s | 5.62% |
| RRT solve | 24.64 s | 7.98% | 633.56 s | 63.65% |
| PathSimplifier | 13.96 s | 4.52% | 0 s | 0% |
| 原始路径插值及稠密筛选 | 3.85 s | 1.25% | 113.62 s | 11.41% |
| B-spline 拟合及稠密筛选 | 7.77 s | 2.52% | 186.87 s | 18.77% |
| 清理 / 未归类 | 0.20 s | 0.07% | 0.03 s（估算） | <0.01% |
| 合计 | 308.95 worker-s | 100% | 995.40 worker-s | 100% |

Marvin 端点阶段内部为：目标 region/IK 44.68 秒，随机状态与其余端点筛选 11.30 秒。原始路径筛选内部，1035 球 Torch audit 为 96.33 秒、PyBullet audit 为 17.22 秒；B-spline 阶段内部，拟合仅 0.024 秒，1035 球 Torch audit 为 161.33 秒、PyBullet audit 为 25.36 秒。也就是说 Marvin 的 B-spline 开销几乎全部是碰撞筛选，不是曲线拟合。

## 结论

Panda 本次瓶颈是宽 pose region 下的端点采样/IK。它进入 RRT 的 40 次中有 38 次 exact，条件成功率为 95%；但只有 40% 的任务能采到端点。其每任务重建场景还产生 22.73 秒累计初始化开销。

Marvin 的端点实现反而更快，102 次任务仅用 55.99 秒；瓶颈移到 RRT 和 1035 球的路径/样条筛选。85 次 RRT 中只有 41 次 exact，条件成功率 48.2%，RRT 累计占 63.65%。两个 Torch 球 audit 合计 257.66 秒，占 Marvin 累计总时间 25.88%。

因此当前最值得优化的顺序是：减少 Marvin RRT 中每个 state-validity 调用的 1035 球成本或用粗到细分层检查；随后降低 B-spline 二次筛选的 512 点/稠密重复计算；再调整能提高 exact 率的搜索参数。PathSimplifier 已关闭，不是当前 Marvin 慢的原因；B-spline 数值拟合本身也可以忽略。

原始结果：`benchmark_results/equal_valid_30_seed20260906/panda.json` 和 `benchmark_results/equal_valid_30_seed20260906/marvin/`。
