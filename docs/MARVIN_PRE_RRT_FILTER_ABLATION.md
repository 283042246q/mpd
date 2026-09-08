# Marvin 双臂 pre-RRT 轻量预筛选消融

## 结论

在相同的 100 个有效端点对、完整 1035 球碰撞模型、
`resolution=0.002`、`range=0.35`、RRTConnect 10 秒以及不使用 PathSimplifier 的口径下，
当前两种非空预筛选都没有提高最终有效轨迹吞吐。生产配置应保持：

```yaml
pre_rrt_filter: none
```

`endpoint_clearance` 的 RRT exact 率没有提高，并误拒绝 13/35 条最终有效基线路径；
加入稀疏关节直线后 exact/进入 RRT 从 58.0% 提高到 69.0%，但误拒绝 16/35 条最终有效路径，
尤其将双臂最终样本从 13 条压缩到 3 条。它筛掉的是 RRT 能绕开的困难路径，造成明显选择偏差。

## 方法与口径

- seed 74，100 个端点成功样本，任务计划为双臂/仅左/仅右 60/20/20，方向 50/50。
- 每个端点对先同时计算三种 filter 判定，再只运行一次生产 RRT；因此三组共享完全相同的端点和 RRT 结果。
- 误拒绝是直接观测：filter 拒绝，但该端点对的基线 RRT 后来获得 exact 或最终通过路径及样条审查。
- `final/min` 是配对反事实值：按 3 个原始 shard 累加实测端点采样和 filter 时间，只给被该组放行的端点累计 RRT、路径及样条审查时间，然后取最慢 shard 作为墙钟时间。
- 该吞吐指标比三次独立随机运行更适合隔离 filter 效果，但不是三次独立 launcher 的实测墙钟；生产速度仍应以大样本独立运行复核。
- clearance 阈值为环境额外余量 5 mm、自碰撞距离 1.5 mm；稀疏直线检查 9 个内部点，至少 50% PyBullet 有效。

## 结果

| filter | 进入 RRT | exact/进入 | 最终有效 | 3-worker 反事实墙钟 | 最终轨迹/分钟 | exact 误拒绝 | 最终误拒绝 |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 100 | 58.0% | 35 | 396.99 s | 5.29 | 0/58 | 0/35 |
| endpoint_clearance | 63 | 57.1% | 22 | 261.11 s | 5.06 | 22/58 (37.9%) | 13/35 (37.1%) |
| endpoint_clearance_and_sparse_line | 42 | 69.0% | 19 | 219.65 s | 5.19 | 29/58 (50.0%) | 16/35 (45.7%) |

预筛本身很轻：100 对端点累计分别耗时 1.38 s 和 2.69 s。问题不是 filter 计算昂贵，
而是节省 RRT 时间的同时丢掉了过多可生成样本，所以每条最终轨迹的成本没有下降。

| filter | 平均关节路径长度 | region transition buckets | 基线 bucket 保留率 | 固定 placement region 覆盖 |
|---|---:|---:|---:|---:|
| none | 5.949 rad | 12 | 100.0% | 4/4 |
| endpoint_clearance | 5.206 rad | 10 | 83.3% | 4/4 |
| endpoint_clearance_and_sparse_line | 5.033 rad | 8 | 66.7% | 4/4 |

路径变短不能解释为规划质量提升：三组使用相同 RRT 路径，非空 filter 只是移除了较长、较难的路径。
四个固定 region 名称仍都出现，但 region/方向/模式联合 transition 覆盖下降，且影响集中在双臂任务：

| filter | 双臂最终 | 仅左最终 | 仅右最终 |
|---|---:|---:|---:|
| none | 13 | 13 | 9 |
| endpoint_clearance | 4 | 12 | 6 |
| endpoint_clearance_and_sparse_line | 3 | 11 | 5 |

## 配置与复现

launcher 和单进程入口都支持：

```bash
--pre-rrt-filter none
--pre-rrt-filter endpoint_clearance
--pre-rrt-filter endpoint_clearance_and_sparse_line
```

完整消融命令：

```bash
python -m scripts.generate_data.benchmark_marvin_pre_rrt_filter \
  --count 100 --workers 3 --seed 74 \
  --output benchmark_results/marvin_pre_rrt_filter_seed74
```

原始逐端点记录、机器可读汇总和自动报告分别位于
`benchmark_results/marvin_pre_rrt_filter_seed74/pairs/`、`summary.json` 和 `REPORT.md`。
脚本支持在进程中断且输出目录已存在时添加 `--resume`，只补齐缺失的 `pair_*.json`。
