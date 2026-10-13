# Warehouse GPU 有效 batch 补齐

2026-10-13。仅修改独立 spacetime 数据生成器；原 GPU forest 搜索、轨迹表示、碰撞与 c/tau-r 验收标准保持相同。

## 调度方式

`--backend torch-forest` 新增 `--torch-batch-policy`：

| 策略 | 每批候选准备 |
| --- | --- |
| `fill`（默认） | 持续尝试候选，直到获得目标数量的有效 source/world，或耗尽 max-attempts |
| `legacy` | 最多尝试目标数量的候选；无效候选筛掉后不补齐，保留旧行为用于对照 |

目标数量为 `min(query-batch-size, 剩余合格轨迹目标, 剩余 attempt 预算)`。有效任务必须通过原来的 source 限位/静态/自碰撞和世界初始动态碰撞检查。求解、拟合失败仍记录原原因并在后续批次继续尝试。

`query-batch-size` 在 fill 下表示希望送入 GPU 的有效 query 数。一个完整 batch 的 attempt 范围可能大于这个值；每个候选仍消耗一个 attempt，不改变 source 行、family 或 job seed 的计算。达到预算时允许最后一批不足目标数量，也允许提交只有失败记录的空 shard。

保存区间仍为连续的 `[attempt_start, attempt_stop)`，恢复从最后完整 shard 的 attempt_stop 继续，不将失败候选漏掉或重复写入。目标附近的 batch 会减小，保证不多生成正样本。

```bash
conda activate mpd-splines-public
cd /home/eric/Projects/MotionPlanningDiffusion/mpd

python -m scripts.spacetime_data.generate_warehouse_spacetime \
  --backend torch-forest --device cuda:0 \
  --output-root data_trajectories_spacetime/Warehouse-TorchSpaceTime-filled-v1 \
  --num-trajectories 100 --query-batch-size 8 \
  --torch-batch-policy fill \
  --edges-per-query 16 --goal-roots 4 \
  --max-iterations 512 --solve-seconds 5 \
  --collision-chunk 512 --obstacles 3 --require-tau-r
```

升级后使用新输出目录，策略和软件 hash 均进入配置校验。OMPL 调度不使用 torch-batch-policy，其 streaming/batch 接口和行为保留。

## 阶段计时

fill 和 legacy 都记录 `report.json.session_pipeline_statistics`，便于同参数对照：

- `prepared/submitted/completed/raw_solved_queries/accepted`：候选数、有效任务数、返回结果数、有原始路径的 query 数、合格轨迹数；均为当前会话。
- `planned_batches/full_batches/batch_size_histogram/effective_batch_size_mean`：非空搜索批次数、达到配置 batch 大小的批次数、有效任务数分布及均值。
- `batch_fill_fraction`：有效任务总数 / `(非空批次数 × query-batch-size)`，包含目标附近的小 batch。
- `preparation_wall_s`：读取/检查 source、抽样世界与初始动态检查，包含无效候选。
- `world_setup_wall_s`：为该批设置 ForecastBatch，包括合并事件时间网格及 GPU 数据准备。
- `planner_wall_s`：创建 forest 并搜索整批任务。
- `postprocess_wall_s`：全部返回结果的 raw 加密检查与 c/tau-r 拟合验收，包含失败结果。
- `shard_write_wall_s`：原子 HDF5 shard 写入。
- `preparation_fraction/accepted_trajectories_per_minute`：候选准备占总会话时间的比例和最终合格轨迹吞吐。

阶段边界同步 CUDA，避免把异步计算算到下一阶段。总会话时间还包含 collision 模型初始化、索引/report 更新和控制开销，不能假定阶段时间之和严格等于总时间。恢复后阶段统计重新计数，顶层 accepted/attempts 为累计值；完成后的重复 resume 保留上次统计。

## 对照方法与预取决策

两种策略依次运行，source、seed、start-index、batch 大小、edges、roots、迭代上限、整批搜索预算、碰撞 chunk、障碍及验收参数相同，仅 torch-batch-policy 不同。

批组成会改变 GPU RNG 序列和同一预算内完成的迭代数，不保证最终轨迹逐点相同。因此比较合格轨迹/秒，同时报告尝试数、raw 成功率、拟合通过率和独立磁盘复检结果，不能只看批次数或纯搜索时间。

若准备时间占比很小，预取的理论收益有限；优先保留简单的同步准备。只有准备成为明显瓶颈时，再增加独立 CPU producer 与有界预取队列，不能让线程修改正在搜索的 collision world。此次先实施有效 batch 补齐，并测量准备开销；未加入候选预取。

测试覆盖拒绝候选补齐、旧模式对照、准确目标数量、连续 attempt 记录、部分/全空批次预算退出、恢复扩展和完整 shard 不变，并回归原 GPU RRT、时空 forest、OMPL streaming 与视频 payload。

## 同参数实测结果

2026-10-13，在 `mpd-splines-public`、RTX 4090 D 上，legacy 和 fill **依次**运行，各生成 100 条验收合格轨迹。两份 config 逐项比较，仅 `torch_batch_policy` 不同：

```text
query-batch-size=8, edges-per-query=16, goal-roots=4
max-iterations=512, solve-seconds=5, collision-chunk=512
obstacles=3, seed=20261013, start-index=0
horizon=14, nominal-duration=8, velocity-scale=0.4
edge-dt=0.02, joint-step=0.04, margin=0.005, require-tau-r=true
```

| 指标 | legacy | fill |
| --- | ---: | ---: |
| 最终合格轨迹 | 100 | 100 |
| 总会话耗时 | 150.404 s | 122.582 s |
| 合格轨迹/分钟 | 39.893 | 48.947 |
| 消耗 attempt | 209 | 219 |
| 有效 query 数 | 153 | 162 |
| 搜索 batch 数 | 28 | 22 |
| 8-query 完整 batch 数 | 2 | 19 |
| 平均有效 batch 大小 | 5.464 | 7.364 |
| batch 填充比例，含尾批 | 68.30% | 92.05% |
| raw 有解 query 比例 | 79.08% | 77.16% |
| raw 有解后最终验收通过比例 | 82.64% | 80.00% |
| 最终执行时长均值 | 6.763 s | 6.919 s |
| 独立磁盘复检，5 ms 加密 | 100/100 | 100/100 |

本次耗时减少 **18.50%**，合格轨迹吞吐提高 **22.70%（1.227 倍）**。fill 虽然消耗了更多 query，仍因有效 batch 更大而减少搜索批次。两种策略最终接受的任务并非全部相同，共同接受的 76 个任务，其起终点和全部冻结 world 数组逐项相同。验收门槛完全相同，但不能据此称轨迹分布相同。

### 阶段耗时与预取决定

| 阶段 | legacy | fill |
| --- | ---: | ---: |
| 候选准备，含无效候选 | 1.432 s | 1.489 s |
| 批量 world 设置 | 0.012 s | 0.012 s |
| 搜索 | 136.906 s | 108.627 s |
| raw 复检与 c/tau-r 拟合验收 | 11.282 s | 11.666 s |
| shard 写入 | 0.342 s | 0.355 s |

候选准备只占 legacy 总耗时的 **0.95%**、fill 的 **1.21%**。即使理想地完全隐藏 fill 的候选准备，本次 100 条最多节省约 1.49 秒。因此**本轮不增加预取**，保持单主进程准备和 GPU 整批搜索；未来更复杂的 source/world 或不同 batch 配置若使准备成为瓶颈，再重新测量并评估独立 CPU producer。

这是一轮完整端到端对照，不是重复实验的统计结论。batch 组成和有限时间搜索会影响成功率与标签分布；没有缩短搜索预算、放宽拟合阈值或降低验收精度。

### 回归与恢复

回归测试 **31 passed**。两组真实输出均重复执行完成后的 `--resume`，legacy 的 28 个 shard、fill 的 22 个 shard 的 SHA256 全部不变，report 阶段统计保持一致。

```text
benchmark_results/warehouse_torch_batch_fill/20261013/
  comparison.json
  matched-worlds.json
  legacy/{config.json,report.json,validation.json,generation.log,shards/}
  fill/{config.json,report.json,validation.json,generation.log,shards/}
```

`comparison.json` 保存配置差异、阶段耗时、成功率、独立复检和恢复校验摘要。
