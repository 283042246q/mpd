# 单 mode、3/5 障碍的 attempt 成功率

2026-10-13。后端 `torch-forest`，采用已随机化形状和交互时刻的世界生成器，以及有效 batch 补齐。

## 统计口径

每个 mode 分别运行 3、5 个障碍，每格固定 **100 次 attempt**。同一场景的所有障碍均使用该 mode；形状、交互时刻、速度、方向和锚点依照现有策略随机化。

主成功率：

```text
最终 c 和 tau-r 均验收合格的轨迹数 / 全部 attempt 数
```

分母包含 source 的静态/自碰撞/限位失败，以及 world 锚点/初始动态碰撞失败。每个 attempt 在场景准备阶段允许原有最多 12 次重采样，这 12 次不另外计为 attempt。全部 mode/count 使用同一 source 前缀、seed 和参数，未按 mode 调整预算。

同时保存进入求解器后的通过率，区分：

1. source 准备失败；
2. world 准备失败；
3. planner 没有原始解；
4. 有原始解但 raw 复检或 c/tau-r 拟合验收失败；
5. 最终正样本。

## 运行

```bash
cd /home/eric/Projects/MotionPlanningDiffusion/mpd
conda activate mpd-splines-public

python -m scripts.spacetime_data.benchmark_warehouse_mode_success \
  --output-root benchmark_results/warehouse_mode_success/20261013/torch-forest-a100 \
  --attempts 100 --device cuda:0 \
  --seed 20261013 --start-index 0 \
  --query-batch-size 8 --solve-seconds 5
```

固定其他参数：edges 16、goal roots 4、512 iterations、collision chunk 512、horizon 14 秒、nominal duration 8 秒、forecast/edge dt 0.02 秒、joint step 0.04 rad、velocity scale 0.4、collision margin 5 mm、require-tau-r。source 使用默认 doubled Warehouse，每次步进两行，因此尝试参考索引 0、2、…、198。

为防止“够了固定成功数就提前停止”改变分母，子生成器使用 `num-trajectories=101、max-attempts=100`。子进程退出码 2、report.complete=false 是预期的固定预算退出；整个统计脚本确认每格确实完成 100 次后，完成全部 20 格才返回 0。

再次运行同一命令跳过已完成的 case，对未完成且已初始化的 case 使用生成器的严格 resume 校验；已完成 case 的报告与数据不重写。汇总还会检查各 case 的共享配置一致性。只重新汇总已完成格子：

```bash
python -m scripts.spacetime_data.benchmark_warehouse_mode_success \
  --output-root benchmark_results/warehouse_mode_success/20261013/torch-forest-a100 \
  --attempts 100 --seed 20261013 --start-index 0 \
  --query-batch-size 8 --solve-seconds 5 --summarize-only
```

脚本校验各 case 的参数、source/资产/软件一致性以及 attempt/失败/求解计数，避免混合旧版本或残缺报告。完整结果与每格失败分类分别保存为 `results.md`、`results.csv`、`results.json`；每完成一格更新一次，原 HDF5 和 generator report 位于 `datasets/n3|n5/MODE/`。

进入求解器及 raw 有解的累计数量从全部失败计数推导；恢复后的 `session_pipeline_statistics` 只属于最后会话，不能当作累计分母。`session_wall_s` 同样是最后会话时间，恢复后不代表整个 case 的总计算时间。

## 解读范围

这是一组固定任务前缀和 seed 的试验，每格 100 次，适合看初步趋势，不代表整个 Warehouse source 的总体成功率。GPU batch 组成和有限时间搜索会随 mode/count 变化。3、5 个障碍的集合不严格嵌套，不能把差异完全归因于“在同一场景添加两个障碍”。

结果是单 mode 场景的成功率，不能直接当作多个不同 mode 混合场景的成功率。所有障碍均要求在最终执行期间经过工作区；没有另外要求 shuttle 完成一整个往返或 orbit 转满一圈。

## 当前结果：按用户要求暂停

已完成 16/20 格、1,600 次 attempt，统计进程及其 GPU 子任务已停止。每个百分数的分母均为 100，成功要求生成器内的 c/tau-r 最终验收通过；本轮尚未另外运行磁盘独立复检。

| mode | 3 个障碍 | 5 个障碍 |
| --- | ---: | ---: |
| linear | 45% | 34% |
| accelerating | 42% | 35% |
| speed_variation | 42% | 35% |
| curved | 43% | 31% |
| zigzag | 43% | 28% |
| shuttle | 40% | 36% |
| orbit | 40% | 24% |
| vertical | 59% | 54% |
| stop_go | 待完成 | 待完成 |
| helix | 待完成 | 待完成 |

已完成的 8 种模式合计：3 障碍为 **354/800 = 44.25%**；5 障碍为 **277/800 = 34.625%**。这些 attempt 重复使用同一批参考任务，不能当作 1,600 个独立任务。

详表位于 `benchmark_results/warehouse_mode_success/20261013/torch-forest-a100/results.md`，失败分类为同目录的 `results.csv/results.json`，暂停汇总为 `paused-summary.json`。以后执行上面的原命令可跳过已完成格子、续跑剩余格子。恢复后的计数使用累计失败记录，避免把新会话计数误当全量统计。
