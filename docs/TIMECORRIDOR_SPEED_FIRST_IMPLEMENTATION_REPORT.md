# Time Corridor 1 → 3 → 2 实施与测试记录

日期：2026-09-29。实施方案见 [TIMECORRIDOR_SPEED_FIRST_IMPLEMENTATION_PLAN.md](TIMECORRIDOR_SPEED_FIRST_IMPLEMENTATION_PLAN.md)。本记录区分“首次 planner result 完成”“在线接收”“机械臂开始运动”“到达 goal 且无 brake”，它们不是同一个成功指标。

## 已实施及当前选择

| 阶段 | 实施结果 | 当前选择 |
|---|---|---|
| S0 | 请求/世界/响应逐请求捕获；首次 result、在线状态、Corridor 阶段计时；冻结请求可离线重放 | 保留 |
| A | 候选与分支分块 batch；复用初始 cost；DP 前缀向量化；减少时间样条重复验证与同步 | 推荐显式使用 `batch_exact`, chunk 64 |
| B1 | 同一次请求内复用安全区间、固定路径状态和已验证 timing；保持原精确物理 cost | 保留 |
| B2 | 可选 `[candidate, physical phase, time, link]` 距离表；时间线性插值仅作 surrogate，分支输出精确重排，原 DenseCheck 不变 | 不启用：该冻结请求无端到端收益 |
| B3 | 可选匀速、固定半径/膨胀球体解析碰撞区间，再投影回原 time grid；盒体/胶囊/时变膨胀回退原查询 | 不启用：当前 benchmark 的盒体全部回退 |
| C1 | 可选 DP 到达时刻拟合现有 Phase5 控制点或 F1 六维 latent；精确物理评价不优则回退原初始化 | 默认关闭：尚无覆盖收益 |
| C2 | 可选 K=4/8 有预算的不同窗口序列搜索；原三 preference 保留 | 默认关闭：耗时上升，当前样本覆盖未增加 |

开关由 `--corridor-a-backend`、`--corridor-a-chunk-size`、`--corridor-a-dp-init`、`--corridor-a-k-best` 提供；未显式启用 Corridor 的既有模式不变。在线最终仍由原 DenseCheck、handoff、guard 和 brake 规则决定，不把 surrogate 或 DP 网格可达当成连续安全证明。

## 门槛与冻结请求

B1 在线困难类 `mixed_motion_multi` 首个 result 为 `success`，世界启动后 **2.414568663 s** 完成，达到用户给定的“B 结束前至少一条困难场景首次 plan ≤2.5 s”条件，因此继续 C。但该 episode 随后 brake、未到 goal；它**不**是动态任务成功。`uncertain_motion` 的首次 result 虽为 2.354 s，却是 `no_valid_trajectory`，不用于“首次成功 plan”判定。

冻结复放输入：`scenario-002/repeat-00/joint_corridor_a` 的首个请求；request SHA256 为 `bda3ca6d773d8ad10f39226abd87d12a932440aaf810ea03ab5716fc27f89884`，world SHA256 为 `12d0a6432df6d2da634098c97b7ec7a7c494c27642dabc974d406a16a1bd4701`，trajectory start 为 `1790602042877421312 ns`。同一请求/世界/seed，在 RTX 4090 D 与其他 GPU 任务并行的条件下，每后端运行 3 次，完整计算耗时如下；3 次样本不足以作显著性或 p95 结论。

| 后端 | 3 次完整请求耗时，秒 | 有效候选 | 说明 |
|---|---:|---:|---|
| 串行参考 | 19.860 / 19.894 / 19.875 | 67 | 原三 preference |
| A+B1 `batch_exact` | 1.714 / 1.618 / 1.636 | 67 | 当前选用 |
| B2 `batch_time_table` | 1.714 / 1.657 / 1.695 | 67 | 建表、精确重排后无稳定收益 |
| B3 `batch_event_intervals` | 1.541 / 1.612 / 1.620 | 67 | 两个 chunk 均为盒体回退，差异不可归因于解析算法 |
| C1，仅 DP 初始化 | 1.676 / 1.642 / 1.659 | 67 | 171 分支；首轮 93/171 拟合被精确 cost 接受 |
| C2，仅 K=4 | 1.996 / 1.816 / 1.950 | 67 | 335 分支 |
| C2，仅 K=8 | 2.192 / 2.129 / 2.062 | 67 | 581 分支 |
| C1+K=4 | 2.064 / 1.988 / 1.823 | 67 | 未提升该请求覆盖 |

这些是离线完整计算，不包含在线世界 warm-up、DDS/服务往返、deadline 或执行。离线同一输入成功不等于在线被及时接受。GPU 非独占，尤其 B3 回退组的约 0.1 s 波动不能解释成算法加速。

复现所选 B1 的 easy/hard 在线轮次（在仓库根目录运行；若 ROS 包未构建，删去 `--skip-build`）：

```bash
MPD_CAPTURE_PLANNER_REQUESTS=1 python scripts/isaaclab/benchmark_todrawer_random.py \
  --output-dir scripts/isaaclab/logs/corridor-speed-first-recheck \
  --environment-count-per-category 1 --planner-repeats 1 \
  --timing-protocol motion_aligned --suite-seed 20260928 \
  --modes joint_corridor_a --categories single_crossing simultaneous_multi \
  --corridor-a-backend batch_exact --corridor-a-chunk-size 64 \
  --calibrate-crossing-from-report scripts/isaaclab/logs/corridor-speed-first-b1-hard-range-20260929/report/runs.csv \
  --duration-sec 35 --skip-build
```

若改测 C1 或 C2，分别附加 `--corridor-a-dp-init` 或 `--corridor-a-k-best 4`；每完成一轮，把下一轮 `--calibrate-crossing-from-report` 指向刚生成的 `report/runs.csv`。这会生成新的 suite 和 `crossing-calibration.json`，不在当前运行中改动已经启动的物体轨迹。实际轮次数据保存在 `scripts/isaaclab/logs/corridor-speed-first-*`。

## 在线 easy/hard 复测及 crossing

所有在线轮次均固定 `suite-seed=20260928`、每类 1 环境、每 mode 1 planner seed、`motion_aligned`、原 2.5 s 交接截止。每次新一轮从前一轮 report 的困难场景首次 result 推导该 mode 的 crossing profile，并保存含源哈希的 `crossing-calibration.json`；同一轮内部不即时移动障碍。以下首次时间是从 world start 到首个完成 result，包含失败结果时单独列出状态。

| 轮次 | 简单单物体 | 困难多物体 `simultaneous_multi` | 解释 |
|---|---|---|---|
| B1 Phase5 | 2.308 s success，动态成功 | 2.604 s success，动态成功 | 两者 goal、无 brake |
| C1 Phase5 | 2.598 s success，动态成功 | 2.770 s success，动态成功 | DP 初始化默认不启用 |
| C2 K=4 Phase5 | 2.803 s success，动态成功 | 2.830 s success，动态成功 | 当前两场景未增覆盖 |
| B1 F1-c 初始/首次 plan 校准 | 3.605 / 3.848 s success | 3.647 / 3.877 s success | 首次 plan 多次因 deadline 变 STALE；障碍早于明显运动，`valid_dynamic_success=false` |
| B1 F1-τr 初始/首次 plan 校准 | 3.756 / 3.691 s success | 3.814 / 3.859 s success | 同上 |

另测 Phase5 四种困难类：`fast_crossing` 首次 2.742 s success 且动态成功；`inflated_dense` 首次 1.092 s `invalid_request`，不能当作成功；`uncertain_motion` 首次 2.354 s `no_valid_trajectory`，最终动态成功；`mixed_motion_multi` 首次 2.415 s success，但 episode brake。该组清楚说明首次计算快和任务成功必须分开报告。

F1 的连续 `deadline_expired_after_planning` 导致首个成功 result 约 3.6–3.9 s，而实际明显运动可能从约 4 s 波动到 16 s。仅按首次 plan 校准仍不能保证障碍与机械臂运动重合。测试过按上一轮首次明显运动后移的逻辑（测试时临时启用；现改为 `--calibrate-crossing-motion-floor` 显式开关）：F1-c 在下一轮反而约 4 s 就开始动，晚到障碍使 easy/hard 均 brake；F1-τr 的 easy/hard 均因连续 STALE 没有 replay manifest。因此该修正**默认关闭**，不能作为安全保证。没有修改用户决定暂不调整的 2.5 s deadline/handoff 门限。

## 验证范围与未完成事项

- 相关测试 `113 passed`；另对新增 F1 K-best 测试补跑 `29 passed`。`git diff --check`、脚本 `bash -n` 和 Python 编译检查通过。
- 数值/形状测试覆盖 Phase5、F1-c、F1-τr、精确距离 override 与梯度、插值域拒绝、球体区间密网格不漏接、盒体回退、K-best 窗口去重、DP 初始化回退。B2/B3 没有足够多场景或独立边界 checker 证明可晋级默认配置。
- 本轮在线测试满足至少一个 easy 和一个 hard 的 GPU 测试要求，但**没有执行方案建议的 10 类 × 2 环境 × 3 seed × 3 Corridor modes 全量矩阵**；因此不宣称全量 S4 验收、统计显著提速或所有模式动态成功。共享 GPU 的时变负载也不适合固化正式 timing profile。
- 当前推荐的显式测试配置是 `--corridor-a-backend batch_exact --corridor-a-chunk-size 64`，C1/C2 与 B2/B3 均关闭；CLI 默认后端仍是 `serial`，没有静默更改既有运行。F1 需要独立解决完整请求与在线交接耗时后，才有意义继续优化 crossing/做正式动态成功率比较；不能通过简单平移 Anchor 代替解决。
- 工作树原本包含本任务跨轮次未提交改动及用户的无关未跟踪文档，本轮未自动创建混合 commit，也未触碰无关文件。
