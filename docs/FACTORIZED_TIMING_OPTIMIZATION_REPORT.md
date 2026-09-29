# Factorized F1/F2/F3 timing 优化与分阶段测试

日期：2026-09-29。测试均关闭 Corridor A，使用同一个 `simultaneous_multi` 冻结请求和世界快照，在 GPU 上直接运行推理。`elapsed_sec` 是离线完整请求时间，不包括 ROS/Isaac Lab 的通信、交接与执行。每组 c 测 3 次，τr 基线与默认配置各测 2 次；样本数不足以估计 p95 或在线动态任务成功率。

## 阶段和启用范围

| 优化 | F1 | F2 | F3 |
|---|---|---|---|
| timing-only 仅对 timing latent 求梯度 | 完整 timing 链 | 初始完整链及每轮交替 timing 子链 | 低噪声 timing 子链；高噪声段没有物理梯度 |
| 固定路径空间量缓存 | 完整 timing 链的引导步骤 | 每条路径固定的 timing 子链，空间更新即失效 | 只在各低噪声 timing 子链内；高噪声段无物理引导，不建 FK 缓存 |
| timing 网络路径编码缓存 | 整条 timing 链 | 每条路径固定的 timing 子链 | 高噪声整段及每个低噪声 timing 子链，空间更新即重新编码 |

固定空间量缓存包含 `q/q_s/q_ss` 和碰撞球位置；到达时刻、速度、加速度、动态障碍距离仍逐步计算。每个 block 用 `try/finally` 清理缓存。joint refinement 更新空间路径，因此不复用 timing block 的缓存。

## 按顺序累计测试（c 权重）

采用同一 seed、请求、世界和轨迹起始时刻，每阶段中位数取 3 次完整请求。将相邻阶段中位数下降至少约 5%、有效候选数和状态无退化作为本次默认开启的操作性标准。这个阈值只用于筛选当前测试配置，并非统计显著性结论。

| 模式 | 全关 | +仅 timing 梯度 | +固定空间缓存 | +路径编码缓存 | 该轮按收益筛选的配置 |
|---|---:|---:|---:|---:|---|
| F1 | 1.140 s | 0.948 s（-16.9%） | 0.794 s（-16.2%） | 0.711 s（-10.4%） | 三项全开 |
| F2 | 1.802 s | 1.433 s（-20.5%） | 1.225 s（-14.5%） | 1.186 s（-3.1%） | 前两项开，编码缓存关 |
| F3 | 0.831 s | 0.756 s（-9.0%） | 0.727 s（-3.9%） | 0.685 s（相对上一阶段 -5.7%） | 梯度、编码缓存开；固定空间缓存关 |

F3 第三项另外从“仅 timing 梯度”直接叠加测试，中位数为 **0.685 s**，比 0.756 s 降约 **9.4%**。该轮按收益筛选时 F3 跳过第二项。此后根据用户要求，**现行 F1/F2/F3 默认均为三项全开**；上表最后一列只记录历史筛选结论，不再代表当前默认配置。所有 c 组均 `success`，F1 每次 8 个有效候选，F2/F3 每次 10 个；结果中的 Corridor `enabled=false`。F1 前两项的导出 NPZ 与基线逐项一致；编码缓存及 F2/F3 的个别轨迹量有小幅浮点差异，选中的候选索引相同。F3 基线同 seed 的重复运行本身也有浮点波动。

## τr 权重交叉验证

同一困难请求，全部优化关闭与该轮按收益筛选的组合各运行 2 次（不是现行全开默认）：

| 模式 | 全关完整请求 | 当时筛选组合完整请求 | 有效候选数 |
|---|---:|---:|---:|
| F1-τr | 1.134 / 1.102 s | 0.765 / 0.744 s | 12 → 12 |
| F2-τr | 1.849 / 1.842 s | 1.288 / 1.251 s | 15 → 15 |
| F3-τr | 0.844 / 0.827 s | 0.725 / 0.701 s | 13 → 13 |

全部返回 `success`，选中的候选索引相同。F3 最终轨迹有小幅数值差异，仍通过原候选安全检查。数据目录为 `scripts/isaaclab/logs/factorized-optimization-study/`。

## 开关和复测

三个开关均可在离线 `infer_factorized.py` 和 resident `infer_factorized_server.py` 中独立覆盖。默认值由 `FactorizedSettings(method=...)` 决定；`--no-timing-grad-only --no-fixed-path-cache --no-path-encoding-cache` 恢复全关闭对照。既有 benchmark/Isaac Lab 命令不需要新参数；从服务健康信息及结果 JSON `factorized.settings` 可查看实际生效值。

例如在仓库根目录复测 F3 现行三项全开默认组合：

```bash
LD_LIBRARY_PATH=/home/eric/anaconda3/envs/mpd-splines-public/lib \
/home/eric/anaconda3/envs/mpd-splines-public/bin/python scripts/inference/infer_factorized.py \
  --request scripts/isaaclab/logs/corridor-speed-first-f1-b1-calibrated-20260929/runs/scenario-002/repeat-00/f1_c_corridor_a/attempt-001/planner-results/request-01790618294350458180/request.json \
  --world scripts/isaaclab/logs/corridor-speed-first-f1-b1-calibrated-20260929/runs/scenario-002/repeat-00/f1_c_corridor_a/attempt-001/planner-results/request-01790618294350458180/world.json \
  --trajectory-start-unix-ns 1790618298209198080 \
  --config scripts/inference/cfgs/config_EnvOpenDrawerShelf-RobotPanda-runtime-to-drawer.yaml \
  --timing-checkpoint data_public/data_trained_models/timing_diffusion/EnvWarehouse/c/warehouse-c-v2/checkpoints/step-00500000.pt \
  --adapt-spatial-basis --method f3 --repeats 3 \
  --output-dir scripts/isaaclab/logs/factorized-optimization-study/f3-default-recheck
```

## Isaac Lab 在线冒烟与边界

另用 `single_crossing` 固定场景、seed 20260928、`motion_aligned`、20 秒上限，测试默认 F1-c/F2-c/F3-c resident 服务，均关闭 Corridor、跳过视频渲染。三者都产出有效规划并进入执行：F1 到 goal 且无 brake，F2/F3 各有 1 次 brake、未到 goal。再次用同一场景文件和 seed、三个优化全关闭，对 F2/F3 做在线对照，两者也各有 1 次 brake。因此这两次 brake 不能归因于本次优化，也不能由该单一场景证明优化提升在线任务成功率。在线记录见 `scripts/isaaclab/logs/factorized-optimization-study/online-smoke/` 和 `online-baseline-f2`、`online-baseline-f3`。

这轮测试尚未覆盖多场景/多 seed 的在线 deadline、首次运动时刻及动态任务成功率分布。正式线上评估仍应使用既有 benchmark，并区分首次 result 与实际开始运动。
