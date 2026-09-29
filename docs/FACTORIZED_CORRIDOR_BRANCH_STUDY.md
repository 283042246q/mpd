# F1 Corridor 多 timing 分支与困难场景计时（2026-09-29）

## 实现与口径

F1/F2/F3 的 timing-only 梯度、固定路径空间量缓存、timing 网络路径编码缓存现均默认开启，可分别用 `--no-timing-grad-only`、`--no-fixed-path-cache`、`--no-path-encoding-cache` 关闭。Corridor A 及本报告的四项扩展仍为独立开关，不改变未启用 Corridor 的行为。

先补模态诊断，再测试以下扩展：

1. `--corridor-a-k-best 4 --corridor-a-branch-fallback`：同一空间候选保留多个 timing 分支，按优化后 exact cost 排序。最低 cost 分支仍经原 DenseCheck；失败才逐批检查下一分支，默认每次请求最多额外检查 64 条、每批最多 32 条。选择第一个通过 DenseCheck 的分支。
2. `--corridor-a-selective-k`：原始 DenseCheck 已确认静态环境碰撞、自碰撞或关节位置越界的固定空间路径不进入 timing 搜索；其他路径只有原时刻不在可达窗口，或有速度/加速度违规时才扩展 K-best。这个过滤不把动态碰撞误当成静态不可修复。
3. 结果 JSON 的 `space_time_guidance.corridor_a` 增加 `optimized_unique_window_sequences`、`validated_unique_window_sequences`、`alternate_branch_rescued_candidates`、`dp_fit_abs_error_s_mean/max`，以及跳过/检查/早停统计。窗口序列按**优化后的连续到达时刻**划分，落在所有安全窗口外记 `-1`，不把 DP 离散路径数冒充有效模态数。benchmark 的 run metrics 与报告也汇总这些字段。
4. `--corridor-a-dp-init --corridor-a-early-stop`：只有窗口约束满足、物理代价连续稳定，且**同一原始 DenseCheck**通过，才提前结束该 timing 分支；最终候选仍执行原 DenseCheck。梯度变小本身不会触发安全判定。

扩展需用 `--corridor-a-backend batch_exact`（也支持其他 batch backend）；不更改模型结构或 checkpoint。所有新增开关默认关闭，便于独立消融。

## 在线困难场景：2.5 秒门限

用 `benchmark_todrawer_random.py`、`motion_aligned`、suite seed 20260928、每类一个环境/一次规划、每次 15 秒，GPU 上测试 5 类困难场景的 F1-c 与 F1-τr Corridor A。这里是**世界开始到首个成功 planner result**，不是规划内核耗时，也不是到达 goal。原结果在 `scripts/isaaclab/logs/factorized-corridor-kbest-study/baseline-hard/`。

| 类别 | F1-c 首次成功 result | F1-τr 首次成功 result |
|---|---:|---:|
| simultaneous_multi | 2.066 s | 2.122 s |
| fast_crossing | 2.004 s | 2.048 s |
| inflated_dense | 2.121 s | 无成功 result |
| uncertain_motion | 1.872 s | 2.117 s |
| mixed_motion_multi | 2.052 s | 1.930 s |

F1-c 的 5/5 类首次成功 result 均在 2.5 秒内；F1-τr 为 4/5，失败类不能计作超时达标。整个 benchmark 进程因该类未生成 replay manifest 返回非零退出码。多数 episode 没有达到 goal 或出现 brake，所以**不能**据此声称任务成功率达标，也不能从单次样本推断 p95。

另以相同 suite seed、类别与 F1-c mode，开启 K4、备选验收、选择性 K、DP 初始化和 C1 早停，在线复测一次，结果在 `combined-hard-online/`。其首个成功 result 依次为 **2.509、2.646、2.484、2.307、2.477 秒**，即 **3/5 类达到 2.5 秒**；`simultaneous_multi` 和 `fast_crossing` 超限。五类优化后有效窗口序列数、备选救回候选数、C1 早停分支数均为 0。这套扩展目前不能默认开启并宣称 2.5 秒稳定达标。`fast_crossing` 到 goal 且无 brake，但另四类有 brake 或未到 goal；这些任务结果与首次规划时间仍须分开评价。

## 冻结请求消融：收益与代价

相同 request、world、轨迹起始时刻和 seed，在 GPU 上每配置运行 3 次。表中为完整离线请求中位耗时；`valid` 是最终原 DenseCheck 的有效候选数。第一份为 `simultaneous_multi`，第二份为另行捕获的 `mixed_motion_multi` 请求。输出在 `scripts/isaaclab/logs/factorized-corridor-kbest-study/`，分别见 `metrics-baseline`、`k4-reference`、`paired-corrected-hard-*`、`paired-corrected-mixed-*`、`dp`、`dp-early`、`corrected-hard-combined`。早期 `k4-fallback-selective` 等目录含一个已修复的静态/动态碰撞分类错误，**不可用于比较收益**。

| 冻结请求 | B1 原版 | K4 | K4+备选验收 | +选择性 K | DP 初始化 | DP+早停 | 组合全部 |
|---|---:|---:|---:|---:|---:|---:|---:|
| simultaneous_multi | 1.074 s / 8 valid | 1.351 s / 8 | 1.976 s / 8 | 1.580 s / 8 | 1.093 s / 8 | 1.076 s / 8 | 1.625 s / 8 |
| mixed_motion_multi | 1.037 s / 15 valid | — | 1.822 s / 15 | 1.430 s / 15 | 1.073 s / 15 | 1.060 s / 15 | 1.384 s / 15 |

首份请求：B1 优化 86 条 timing 分支，K4 变为 335 条，但优化后只有 1 种不同的有效窗口序列，331 条分支落在所选安全窗口之外。正确区分静态/动态碰撞后，备选验收额外 DenseCheck 22 条，救回候选 0；仅启用 K4+备选比 B1 慢约 0.90 秒，其中包括额外的静态专用碰撞查询，不能只归因于备选 DenseCheck。选择性 K 跳过 74 条确认为静态碰撞、自碰撞或关节位置违规的候选路径，将优化分支从 335 降至 39 条；相对 K4+备选节省约 0.40 秒，但仍比 B1 慢约 0.51 秒。第二份请求选择性 K 跳过 44 条，最终无需优化 timing 分支，15 个原有有效候选保持不变；相对 K4+备选节省约 0.39 秒，但仍比 B1 慢约 0.39 秒。**动态碰撞本身不会触发时间不可修复排除。**

DP 时间表拟合在两份请求的平均绝对误差约为 2.68 秒和 2.41 秒。C1 早停的 `early_stop_dense_checks`、`early_stopped_branches`、`early_saved_iterations` 在两份请求均为 0；中位时间的微小变化属于重复运行波动，**目前没有可证实的早停收益**。真实有效窗口序列和备选救回数也均为 0。这提示固定六维 timing latent 对 DP 离散多窗口时间表的表达能力不足，单纯扩大 K 会增加成本而未产生新模态。

组合全部时两份请求分别为 1.625 秒和 1.384 秒，最终有效候选数未降，但比 B1 慢。它只是离线完整请求计时，不能替代在线 world-start 门限或动态任务成功检验。同场景同 planner seed 的独立在线组合冒烟首个成功 result 约 2.476 秒、出现 brake；五类正式复测有 2/5 超限，故不能认定组合配置稳定达标。

## 复测命令

从仓库根目录运行在线 benchmark（不额外开启 K/DP）：

```bash
python scripts/isaaclab/benchmark_todrawer_random.py \
  --output-dir scripts/isaaclab/logs/factorized-corridor-kbest-study/recheck \
  --environment-count-per-category 1 --planner-repeats 1 \
  --timing-protocol motion_aligned --suite-seed 20260928 \
  --modes f1_c_corridor_a f1_tau_r_corridor_a \
  --categories simultaneous_multi fast_crossing inflated_dense uncertain_motion mixed_motion_multi \
  --corridor-a-backend batch_exact --corridor-a-chunk-size 64 \
  --duration-sec 15 --skip-build
```

在上述命令末尾加 `--corridor-a-k-best 4 --corridor-a-branch-fallback --corridor-a-dense-branch-budget 64 --corridor-a-selective-k --corridor-a-dp-init --corridor-a-early-stop` 可测试全部扩展。比较速度时应固定 suite、seed、checkpoint、GPU 负载并查看 `report/summary.json` 的首个成功 result 时间；检查任务成功仍需 goal、无 brake、以及障碍进入前机械臂明显开始运动。
