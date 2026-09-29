# ToDrawer 首次 plan 中位数与 crossing 校准（2026-09-29）

## 数据口径与结论

数据源：`scripts/isaaclab/logs/todrawer-median-source/report/runs.csv`；SHA-256：
`4b9c9a1ca32c64618172dfcfc354570f64946e31d7424539f8362fac166139bc`。
冻结 suite seed 为 `20260928`，每类 1 个环境、每 mode 1 个 planner seed，
使用 `motion_aligned`。本表只统计五类 hard 场景
（`simultaneous_multi`、`fast_crossing`、`inflated_dense`、
`uncertain_motion`、`mixed_motion_multi`），每 mode 均有 5/5 个首次完成
planner result。`first_plan_completed_from_world_s` 是从世界开始到首个完成
result 的时间，不要求该 result 规划成功；2.5 秒仅是统计阈值。

非 Corridor mode 的首次 plan 中位数与原 profile 大体接近：F1(c) 约
2.30–2.32 秒，F2(c) 约 3.09–3.12 秒，F3(c) 约 2.19–2.26 秒，
τr 的 F1/F2/F3 分别为 2.23/2.97/2.20 秒。三个 Corridor mode
分别为 joint 2.35 秒、F1-c 2.61 秒、F1-τr 2.61 秒；原先
12.67/8.24/10.12 秒的 Corridor profile 已不适合当前实现。
F2 三个 mode 的全部五次首次 result 均超过 2.5 秒；F1-c 与
F1-τr Corridor 各 4/5 次超过 2.5 秒。

| mode | 首次 plan 中位数 s | >2.5 s / 5 | 首次明显运动早于首次 crossing 至少 1.25 s / 5 | 校准后 shift 区间 s |
|---|---:|---:|---:|---:|
| phase4 | 1.462 | 0 | 4 | 1.132–1.432 |
| phase4_aligned | 1.958 | 0 | 2 | 1.508–1.808 |
| scalar_duration | 1.809 | 0 | 3 | 1.859–2.159 |
| timing_only | 1.782 | 0 | 2 | 1.832–2.132 |
| joint | 1.828 | 0 | 4 | 1.878–2.178 |
| joint_corridor_a | 2.351 | 1 | 5 | 2.401–2.701 |
| f1 | 2.304 | 0 | 3 | 1.454–1.754 |
| f1_c | 2.319 | 0 | 3 | 1.469–1.769 |
| f1_c_corridor_a | 2.614 | 4 | 5 | 1.764–2.064 |
| f1_tau_r | 2.233 | 0 | 3 | 1.703–2.003 |
| f1_tau_r_corridor_a | 2.610 | 4 | 4 | 2.080–2.380 |
| f2 | 3.123 | 5 | 4 | 1.553–1.853 |
| f2_c | 3.088 | 5 | 3 | 1.518–1.818 |
| f2_tau_r | 2.970 | 5 | 2 | 1.710–2.010 |
| f3 | 2.190 | 0 | 3 | 1.850–2.150 |
| f3_c | 2.259 | 0 | 3 | 1.919–2.219 |
| f3_tau_r | 2.197 | 0 | 3 | 2.157–2.457 |

`f1`/`f2`/`f3` 是兼容命名；此次指定的是 c checkpoint，并不等于
τr 模型。本表的 shift 是相对于场景原始 anchor 时间的偏移，
**不是**绝对 `crossing_time_s`。例如 A0/A1/A2 的名义时刻为
4.35/5.25/6.15 秒，实际还带场景 jitter；F1-c Corridor 的 A0
名义 crossing 因而约为 6.114–6.414 秒。

## 使用的旧校准规则

与 `benchmark_todrawer_random.py::_calibrate_crossing_from_report` 一致：

1. 先对每个非 Corridor mode，取五类 hard 场景的首次完成 result 时间中位数
   `m`，把原 profile 的 `first_plan_completed_s` 设为 `m`，其余四项
   （明显运动估计、预计 goal、shift 下/上限）统一加
   `m - old_first_plan_completed_s`。
2. 再对三个 Corridor mode，以**已经更新**的对应非 Corridor profile
   （`joint`、`f1_c`、`f1_tau_r`）为基准，统一加
   `corridor_median - parent_median`。不再沿用旧 Corridor 的大 shift。
3. 对单个场景仍由生成器取
   `shift_lower = max(profile.shift_min, profile.first_plan + 1.25 - min(base_crossings))`、
   `shift_upper = min(profile.shift_max, profile.expected_goal - 0.50 - max(base_crossings))`，
   在这个区间内按冻结随机比例取 shift，然后继续执行物体轨迹与初始机器人安全检查。

校准值已写入 `scripts/isaaclab/run_todrawer_f3c_until_success.py` 的
`MODE_TIMING_PROFILES`，保留到毫秒；后续 benchmark 与 until-success
程序均读取同一份 profile。没有更改 2.5 秒的交接/通信策略，也没有开启
实验性的 `--calibrate-crossing-motion-floor`。

## 解释边界

首次 plan 完成不等于机械臂已经明显运动。本次
`phase4_aligned`、`timing_only`、`f2_tau_r` 各仅 2/5 次满足
“运动至少早于 crossing 1.25 秒”。旧 Corridor crossing 又偏晚，
`f1_tau_r_corridor_a` 有 3/5 次、`joint_corridor_a` 有 2/5 次在首次
crossing 前已经到达 goal。新 shift 用于下一轮验证，不能据此直接宣称
动态交互成功；应同时检查 `first_motion_before_crossing`、goal 时刻、
brake 及碰撞。校准规则还会联动平移 `expected_goal_s`，它只是预测边界，
不是每次运行的实际 goal 时间；下一轮尤其需要复核 Corridor 的保护窗口。

## 离线核验

`tests/test_todrawer_random_benchmark.py`：41/41 通过。用完整源 CSV
运行 `test_todrawer_first_plan_calibration.py` 后，17 个 mode 均得到
5 个返回时间、0 个缺失值；生成 10 场景 × 17 mode 的新 suite，
`validate_todrawer_random_suite.py` 返回 `valid=true`。验证器另报
255 条静态家具 clearance 提示；按本 benchmark 的既定策略，物体穿过
静态家具允许，这些不是底座碰撞或 suite 无效。离线预览在
`scripts/isaaclab/logs/todrawer-median-calibrated-preview/`，未启动 ROS、
Isaac Lab 或在线规划。
