# Panda 与 Marvin warehouse 数据生成分阶段耗时

## 测试口径

同机、CPU 单进程、`OMP_NUM_THREADS=1`、seed=46。每种机器人测试 10 个任务：
前 5 个为随机关节区域到固定 region，后 5 个为不同固定 region 之间；
每个采样成功的起终点只运行一次 RRTConnect，solve 上限 10 秒。

Panda 使用 `EnvWarehouse-RobotPanda_v01.yaml` 和当前可信单臂生成实现。
其 launcher 设置 `task_batch_size=1`，所以测试按当前行为为每个任务重建 PyBullet/OMPL。
当前 Panda launcher 设置 `fit_bspline=False`；为说明数据质量差异，测试额外执行一次
22 控制点、5 阶、512 点并按 0.025 rad 加密的 PyBullet 样条审查，但单独计时且不算作原生步骤。

Marvin 使用当前 `EnvWarehouse-RobotMarvinBimanual-independent.yaml`：6 个双臂 14D 任务，
2 个仅左 7D、2 个仅右 7D；场景按一个 shard 只初始化一次。
其路径和样条均执行 PyBullet 网格及当前资产实际加载的 1035 个细碰撞球检查。

这不是相同机器人、相同可行空间的算法微基准，而是两个**实际生成入口**在相同任务方向配额和
RRT 时间上限下的端到端比较。一个 seed、10 个任务的成功率置信度很低；耗时结论由阶段计时支持，
成功率不应直接外推到大数据集。

原始结果：

- `benchmark_results/panda_marvin_generation_stages_seed46_detailed.json`
- `benchmark_results/marvin_generation_stages_seed46_no_simplify.json`

复现命令：

```bash
source set_env_variables.sh
OMP_NUM_THREADS=1 python -m scripts.generate_data.benchmark_panda_marvin_generation_stages \
  --tasks 10 --seed 46 --planner-time 10 \
  --output benchmark_results/panda_marvin_generation_stages_seed46_detailed.json

OMP_NUM_THREADS=1 python -m scripts.generate_data.benchmark_panda_marvin_generation_stages \
  --robots marvin --marvin-no-simplify --tasks 10 --seed 46 --planner-time 10 \
  --output benchmark_results/marvin_generation_stages_seed46_no_simplify.json
```

## 原生流程与追加同类样条审查

| 指标 | Panda | Marvin |
|---|---:|---:|
| 测试任务 | 10 | 10 |
| 端点失败 | 5 | 4 |
| RRT 调用 | 5 | 6 |
| RRT exact | 5 | 1 |
| 最终通过 | 5 | 1 |
| 最终通过率 | 50% | 10% |
| 总墙钟时间 | 24.57 s | 101.08 s |
| 平均每个调度任务 | 2.46 s | 10.11 s |
| 串行最终产量 | 12.22 条/min | 0.59 条/min |

产量差约 20.6 倍同时包含成功率和单次计算成本，不能解释成某个函数慢 20.6 倍。
Panda 的“最终通过”包含测试额外增加的样条审查；该审查只有 PyBullet 碰撞，不包含 Marvin 的
细碰撞球，因此仍不是完全相同的安全标准。

## 阶段耗时

| 阶段 | Panda 总计 | Panda 占比 | Marvin 总计 | Marvin 占比 |
|---|---:|---:|---:|---:|
| 场景/机器人初始化 | 2.285 s | 9.3% | 1.788 s | 1.8% |
| 端点采样（含 IK/筛选） | 18.326 s | 74.6% | 5.087 s | 5.0% |
| RRT solve | 包含于下一项 | — | 53.746 s | 53.2% |
| RRT 原生后处理 | 2.681 s（含 solve） | 10.9% | — | — |
| 路径简化 | 未单独暴露 | — | 33.563 s | 33.2% |
| 路径重采样+完整复核 | 包含于原生后处理 | — | 1.378 s | 1.4% |
| 样条拟合+完整复核 | 1.265 s（额外步骤） | 5.2% | 5.513 s | 5.5% |
| 合计 | 24.556 s | 100% | 101.075 s | 100% |

Marvin 端点的 5.087 秒可进一步分为：region IK 4.116 秒、随机状态和最终端点筛选 0.971 秒。
所以本次 Marvin 的主要瓶颈不是 IK，而是 RRT exact 解率和路径简化。

Panda 端点采样反而更慢，是因为 5 个不可行/碰撞位姿会在旧 IK 流程中对同一位姿尝试大量随机初值；
成功端点通常很快。Marvin 每次最多 30 个 IK 提案，并在提案间重新采样 region 位姿，因此失败更早返回，
正式生成器再换一组新任务。

Panda 的 5 个有效端点全部获得 exact 路径，RRT+插值平均 0.536 秒/次。
Marvin 的 6 次 RRT 只有 1 个 exact，solve 平均 8.96 秒/次：
5 个失败基本用满 10 秒，唯一成功为 2.89 秒。
这反映 14D 搜索、双臂/臂间碰撞和窄自由空间；即使 7D 的 Marvin 单臂模式，
3 次 RRT 也只有 1 次 exact，说明细碰撞模型和场景难度同样重要。

更大的 20 条正式 smoke 数据也支持“RRT 是主瓶颈”：为补足 20 条最终轨迹共采样 107 个新任务，
68 次进入 RRT、25 次得到 exact、5 次被样条拒绝；RRT solve 累计 573.18 秒，
固定端点求解累计 58.25 秒。其 exact 率 36.8% 高于本页单 seed 的 16.7%，说明 1/6 不应直接外推，
但累计 RRT 时间仍约为端点求解的 9.8 倍。

## Marvin exact 路径后的细分

本轮只有一条 exact 路径，以下是该路径的实际耗时：

| 子阶段 | 耗时 |
|---|---:|
| OMPL 路径简化 | 33.563 s |
| 128 点重采样/加密 | 0.002 s 以下 |
| 128 点 Torch 细球检查 | 1.168 s |
| 128 点 PyBullet 检查 | 0.208 s |
| 样条拟合及求值 | 0.001 s 以下 |
| 512 点 Torch 细球检查 | 4.670 s |
| 512 点 PyBullet 检查 | 0.839 s |

`PathSimplifier(..., maxTime=0.1)` 没有在 0.1 秒结束，因为一次内部 motion/collision 检查本身不可中断；
它在昂贵的双碰撞模型上成为 33 秒级步骤。真正的矩阵样条拟合几乎不耗时，耗时来自样条安全审查。

关闭 Marvin simplifier 的相同 seed 对照中，任务状态完全相同（4 个端点失败、5 个 RRT 失败、
1 个最终通过），总时间从 101.08 秒降至 67.49 秒，减少 33.6 秒/33.2%；
唯一成功任务从 43.90 秒降至 10.25 秒。完整路径和样条检查仍保留。

后续 100 条 exact 原始路径配对测试已经完成：PathSimplifier 平均耗时 88.2 秒、中位 49.6 秒，
修复 19 条但破坏 9 条原本最终有效的轨迹。综合吞吐和稳定性后，Marvin 生产配置已默认关闭简化；
原始路径与拟合样条的完整碰撞复检保持不变。

## 结论

1. 本次实际差距首先来自 Marvin RRT：更高维、更窄自由空间，exact 率 1/6，而 Panda 为 5/5。
2. Marvin `PathSimplifier` 是已确认的非预期热点，生产配置已经关闭，仅在专项实验中显式开启。
3. 1035 个细碰撞球使 128 点路径和 512 点样条审查合计约 6.9 秒/成功轨迹；
   这是数据质量成本，Panda 当前原生 launcher 没有同等检查。
4. IK 不是 Marvin 的主要瓶颈；本轮仅占总时间约 4.1%。
5. 100 端点消融选择 `state_validity_resolution=0.002`、`planner_range=0.35`；launcher 并行仅降低墙钟时间，
   不改变单次 RRT 成功率。
