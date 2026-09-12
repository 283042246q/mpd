# Marvin Warehouse 训练采样区域与轨迹方向设计

日期：2026-09-13

## 1. 数据依据与结论

本次区域更新以两组本地实测为依据：

- `scripts/inference/logs/marvin-workspace-dense-v3/ik-reachability.json`：同侧桌面的大范围 IK 与完整碰撞检查；
- `scripts/inference/logs/marvin-workspace-edges-2400/summary.json`：12 个边缘 cell、2400 个 TCP 目标、955 组 RRT。

边缘测试总计 IK 命中 1452/2400，完整碰撞有效 613/2400；RRT 原始路径有效
345/955，五阶 B-spline 复检有效 226/955。它证明的是离散样本中的可行范围，不能把
成功点的 XYZ 包围盒解释为盒内处处可行。因此生产配置在成功范围内留出小幅边界，
生成时仍逐项执行 IK、完整端点碰撞、RRT、原始路径复检和样条复检。

旧 placement 的桌面高度是 `z=[0.18, 0.28]`。同侧桌面完整碰撞有效点实际为：

| arm/cell | 实测有效 XYZ 包围盒（m） | 新 placement XYZ（m） |
|---|---|---|
| left same-side table | x 0.495–0.794, y 0.026–0.485, z -0.021–0.096 | x 0.50–0.79, y 0.03–0.48, z -0.015–0.09 |
| right same-side table | x 0.418–0.799, y -0.498–-0.039, z -0.022–0.094 | x 0.42–0.79, y -0.49–-0.04, z -0.015–0.09 |

边缘测试中有样条成功的区域按如下范围落入独立 atomic cell：

| placement cell | 新 XYZ（m） | 实测依据 |
|---|---|---|
| left_table_cross_y | x 0.46–0.72, y -0.27–-0.145, z -0.01–0.09 | spline 成功 36/100 |
| left_table_cross_x | x 0.745–0.83, y -0.17–-0.005, z -0.005–0.09 | 32/100 |
| right_table_cross_y | x 0.55–0.79, y 0.15–0.195, z 0.03–0.09 | 8/30；低概率困难 cell |
| right_table_cross_x | x 0.75–0.83, y 0.007–0.17, z -0.005–0.09 | 20/100 |
| left_cabinet_lower_xedge | x 0.505–0.60, y 0.645–0.76, z 0.17–0.29 | 30/100 |
| left_cabinet_lower_yedge | x 0.31–0.49, y 0.765–0.85, z 0.185–0.29 | 35/100 |
| right_cabinet_lower_xedge | x 0.47–0.54, y -0.77–-0.66, z 0.17–0.29 | 17/100 |
| right_cabinet_lower_yedge | x 0.34–0.47, y -0.86–-0.775, z 0.185–0.285 | 14/100 |
| left_cabinet_upper | x 0.365–0.425, y 0.625–0.675, z 0.48–0.55 | 18/100 |
| right_cabinet_upper | x 0.365–0.445, y -0.75–-0.645, z 0.495–0.545 | 16/100 |

左上层 y 外缘虽然 IK 命中 79/200，但完整碰撞有效为 0；右上层 y 外缘只有 5/200
个完整有效端点且样条 0/25，因此二者不进入训练 placement。右臂跨左桌 y 外缘只有
6/200 个完整有效目标，但 30 个可构造 pair 中有 8 个样条成功，所以仅保留一个 5%
低概率困难 cell，避免它主导生成耗时。

`random_regions` 不是抓取/放置面，而是随机起止状态的 TCP 过滤支持。它改成左右分离的
大范围桌面自由空间，明确不包含书架区域：

| arm | random TCP XYZ（m） | 体积 |
|---|---|---:|
| left | x 0.20–0.85, y 0.02–0.60, z -0.05–0.60 | 0.24505 m³ |
| right | x 0.20–0.85, y -0.60–-0.02, z -0.05–0.60 | 0.24505 m³ |

两侧在中线留出 0.04 m 间隔，并在书架前沿 `|y|=0.62` 之前停止。与旧训练配置相比，
它同时向近基座、桌面上方和高位自由空间扩展。random 仍以关节空间均匀候选开始、经
FK 落区和完整碰撞检查后才接收，因此配置盒不表示盒内处处可达或 TCP 均匀分布。
固定随机种子的小规模实现审计中，100/100 个双臂随机状态成功，累计检查 10281 个
关节候选，耗时 7.34 s；实际左/右 TCP 都覆盖了各自 box 的低位、高位和大部分横向范围。

placement atomic cell 不再使用人为指定的最终概率，而按
`平移体积 × difficulty_boost` 归一化。困难系数被限制在 `[1.0, 1.5]`，当前配置只使用
1.10–1.30。由当前坐标计算出的实际提议概率为：

| cell | left | right |
|---|---:|---:|
| same-side table | 58.82% | 76.02% |
| cross table y edge | 15.35% | 3.38% |
| cross table x edge | 7.15% | 6.73% |
| cabinet lower core | 2.49% | 0.90% |
| cabinet lower x edge | 6.75% | 4.82% |
| cabinet lower y edge | 8.28% | 5.77% |
| cabinet upper | 1.17% | 2.37% |

所以大体积桌面仍占主导，困难/边缘 cell 只在体积分配基础上小幅增采，不会把数据重新
限制到少量窄区域。

完整坐标和每个 placement cell 的提议权重以
`data_generation_cfgs/EnvWarehouse-RobotMarvinBimanual-independent.yaml` 为准。

## 2. 四类轨迹及比例

四类都保留，但不是等比例：

| direction | 建议比例 | 主要作用 |
|---|---:|---|
| placement → placement | 45% | 桌面/书架之间的主任务，以及不同 placement cell 间的窄通道模式 |
| random → placement | 35% | 从 home、任意安全状态或实时 replan 的中间状态到任务目标 |
| placement → random | 10% | 撤离、回退和转入预备位，降低方向分布偏置 |
| random → random | 10% | 恢复/预定位，并拓宽起点和 EE goal 的联合支持；不宜占主导以免稀释 warehouse 专项先验 |

这组比例是面向当前 warehouse 工作流的工程先验，不是 MPD 论文给出的最优比例。
论文只明确使用随机上下文、RRTConnect，并从 goal 到 start 再生成一条路径；没有报告
上述四分类的消融或最优配比。配置按 100 条成功轨迹为周期精确得到 45/35/10/10；
每个方向内部仍保持 `dual_independent:left_only:right_only=3:1:1`。失败重试同一个
`task_id`，因此 endpoint/IK/RRT 难度不会改变最终方向配额。

placement cell 的配置权重是任务规格的提议概率。由于失败后固定 cell 重试，而不是换成
容易区域，只要数据生成完整结束，最终比例在有限样本误差内应跟随提议分布；困难 cell
影响的是耗时和整批失败风险。仍需用 manifest 的 `region_counts` 复核。首批建议生成
1000 条 pilot；若某个困难 cell 经常耗尽预算，应先收紧到该 cell 内的实测成功子范围，
而不是通过失败后换区来掩盖问题。

每个 `task_id` 只选择一次 direction、task mode 和各臂的 source/goal placement cell。
随后若随机状态、IK、RRT、原始路径复检或样条复检失败，只在相同类别和相同 cell 内
重新采样连续坐标/关节；不会换区域、换方向或用容易样本替代困难任务。成功后才进入
下一个 `task_id`。`max_attempts_per_task: 30` 是每个 task 独立的硬上限：完成 30 次仍
失败会抛出包含 task ID、mode、direction、source/goal region 和累计统计的
`TaskSamplingBudgetExhausted`。launcher 将它视为确定性 shard 失败，立即终止该 shard，
不会执行通常用于 native worker crash 的自动重试。

## 3. 时间反转扩充

静态 warehouse 的几何路径可以合理地反转，因为关节限位、静态碰撞和路径几何对运动
方向对称；但它不是刚需，也不建议把整个数据集无条件翻倍。反转同一条路径不会增加
通道/同伦类别，容易重复加权已有模式。优先级如下：

1. 先生成原生 45/35/10/10 数据并训练基线；反方向任务让 RRTConnect 重新规划，能够
   获得与简单倒放不同的随机路径模式。
2. 若 `placement → random` 验证集明显欠拟合，可在训练 loader 中以 20% 概率在线
   反转，而不是永久复制 HDF5。对当前基础分布，这对应约 45/30/15/10 的有效方向分布。
3. 做 `0% / 20% / 100%` 反转消融，按四个方向分别报告 MPD 有效率、goal pose 误差、
   碰撞率和多样性；只有验证集改善才保留。

正确反转必须交换 `q_start/q_goal`、source/goal region，并从新的 `q_goal` 重算双 EE goal
context；active joint/EE mask 不变。不要只倒放控制点：应倒放已验证的 dense/raw path，
重新拟合 B-spline 并运行同一套完整验证，或者同时严格变换 knot vector。加入动态障碍物、
时间维度 cost、速度方向约束、抓取/释放或载荷状态后，除非场景时间序列和任务状态也可逆，
否则不得使用这种增强。

## 4. 实现兼容性

- 没有改 Python 入口；仍使用 `generate_marvin_warehouse_bimanual.py --config ...`。
- 未配置 `trajectory_direction_weights` 的旧配置仍保持 50% random→placement、50%
  placement→placement。
- `random_regions` 仍兼容单 XYZ mapping 和 atomic-box list；生产默认使用左右各一个
  大范围、非书架 box。
- placement region 同时支持 list（等权）和 `{region: weight}`（加权）写法。
- `placement_region_weighting: volume` 会根据 atomic cell 体积计算基础权重，再应用受限的
  difficulty boost。
