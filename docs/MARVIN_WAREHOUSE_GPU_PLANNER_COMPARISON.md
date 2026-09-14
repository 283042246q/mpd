# Marvin warehouse 轨迹规划后端比较

## 结论

仓库现在有两条互不影响的路径：原 CPU generate/launch 仍默认使用 OMPL `RRTConnect`；
新增的专用 launcher 使用按模式分桶的 `GpuMultiQueryRRTConnect`。新流程在 80-task active
window 内同时打开最多 8 个十任务 shard，把通过 endpoint gate 的 query 按 14-D dual、
左 7-D、右 7-D 分桶后跨 shard 批量规划。默认 query 上限分别是 12/16/16，endpoint actor
是 4 个；CUDA actor 和 PyBullet actor 各只有 1 个。

每个 task 同时只允许一个已进入规划的 candidate，避免旧实现同一 task 先算出多条有效轨迹
而最终只使用一条。某个 shard 的慢 task 不会阻止其他完整 shard 发布；正式 shard 仍保持
6 dual、2 left-only、2 right-only 的 task id/方向合同。未完整 shard 的已接受 task 会原子
写入 `.inflight`，重启后按 task 恢复，只有十条齐全才发布正式 shard。

10 条旧流水线 smoke、一次 100 条旧同步 multi-query 测试，以及 20 条新跨 shard 流水线
smoke 已成功，actor restart 均为 0，
输出通过训练入口的 hash、模型资产、形状、样条和 EE context 校验。100 条证明了当前
actor/checkpoint 链路的短期稳定性，但还不能替代独占 GPU A/B 和 1,000 条 soak。新的
20 条测试为 8.60 s/条，但样本太少且有外部显存占用，不能直接当作生产吞吐。原来的
单-query 对照仍有参考价值：RTX 4090
上 RRT 核心从 2.052 s 降到 0.599 s（3.43x），而旧单-query GPU 完整生成反而从 CPU 的
15.82 s/条退化到 28.05 s/条。新架构正是针对该 Amdahl 瓶颈，把 query、shortcut 和
raw/spline Torch audit 合并到 GPU 批处理中。

生产默认值暂不切换。CPU 多 worker 已知可能导致整机不稳定，因此稳定性基线应使用 CPU
单 worker，而不是把不可持续的 workers=3 当作吞吐目标。GPU 专用 launcher 先完成
100/1,000 条 soak，再决定是否成为生产入口。

## Multi-query 专用管线

```text
Coordinator（唯一 writer，80-task active window，最多 8 个 shard）
    ├── 4 × endpoint actor（work stealing；每次派 2 个 endpoint candidates）
    │       无 OMPL、无 PyBullet、无 CUDA；只返回 endpoint candidates
    ├── 1 × GPU actor（持久 CUDA context）
    │       endpoint gate → dual 12 / left 16 / right 16 multi-query RRT
    │       → shortcut
    │       → batched raw/spline Torch audit
    └── 1 × PyBullet actor（DIRECT、私有 client、可重启/定期 recycle）
            endpoint mesh gate → 最终 raw/spline mesh gate

三段以有界队列和非阻塞 actor poll 跨 shard 重叠；Coordinator 原子发布完整 shard，
并把已接受但尚未凑齐 shard 的 task 写入 partial spool。
```

这里的 CPU sphere 仅用于 IK 过程的逐臂候选反馈，不是最终 Torch audit。实测完全取消这层
反馈时，一个困难 dual task 的 6,400 个组合中只有 1 个通过 GPU endpoint gate，且随后被
PyBullet 拒绝；因此保留各 endpoint actor 私有的 CPU sphere checker 是成功率所需的
折中。所有 endpoint 都会在 GPU 上重新批量检查，raw path 和 spline 的 Torch 审计也只在
GPU actor 中执行。

actor 采用 `spawn`，请求队列容量为 1，超时或退出后最多自动重启 3 次；PyBullet 默认每
50 条主动 recycle。CUDA 不可用、driver/device 掉线属于 actor 进程无法自行修复的主机级
故障，launcher 会失败并保留已发布 checkpoint 及 task 级 partial spool，未发布内容不会
冒充完整 shard。每次运行还会生成 `pipeline_telemetry.yaml`，记录 mode batch occupancy、
队列等待、actor busy fraction、host peak RSS 和 CUDA allocator peak。

启用方式：

```bash
python scripts/generate_data/launch_generate_marvin_warehouse_gpu_pipeline.py \
  --gpu-device cuda:0 \
  --num-trajectories 100 \
  --output-dir /path/to/output
```

该入口会在 manifest 中写入 `GpuMultiQueryRRTConnect`。原 YAML 的 `planner: RRTConnect`
未修改，原 `generate_marvin_warehouse_bimanual.py` 和
`launch_generate_marvin_warehouse_bimanual.py` 也没有接入这些 actors。新增的
`gpu_pipeline_*` 配置项只被专用 launcher 读取。

首次 10 条成功 smoke 的有效统计为：14 次 task-attempt、285 个 dual-cross 后候选、9 个
GPU endpoint 拒绝、11 个 PyBullet endpoint 拒绝、13 个 RRT 拒绝、7 个 spline Torch
拒绝，最终 10/10；RRT 共 238 轮、5,500 条采样边、48,758 个检查状态。当前版本在每个
shard manifest 中记录 endpoint proposal、GPU endpoint、PyBullet endpoint、GPU plan/audit、
PyBullet trajectory 及整个 checkpoint 的 wall milliseconds，供长时间吞吐分析。

### 新跨 shard 流水线 20 条：window=80、dual=12、left/right=16、4 endpoint actors

2026-09-15 在 RTX 4090 D 上完成 20/20。启动前另一个 Python 进程占用约 2.4 GB 显存；
修正 telemetry query 计数后的复跑总 wall 为 171.91 s，即 8.60 s/条，去除 actor 初始化、
最终 merge/训练格式检查后的 pipeline wall 为 158.93 s，即 7.95 s/条。由于只请求了 20
条，本次 active window 实际占用是 20 个 task/2 个 shard，尚未测到 80-task 满窗吞吐。

| 指标 | 结果 |
|---|---:|
| 完成分布 | 12 dual、4 left-only、4 right-only |
| task attempts / endpoint candidates | 37 / 75 |
| GPU plan queries | dual 43、left 5、right 4 |
| GPU plan batches | dual 17、left 2、right 1 |
| query occupancy | dual 21.1%、left 15.6%、right 25.0% |
| actor busy fraction | endpoint pool 22.8%、GPU 91.9%、PyBullet 6.5% |
| GPU actor CUDA peak | allocated 3.65 GiB、reserved 3.74 GiB |
| host peak RSS | endpoint 每进程最高 0.60 GiB、GPU 0.89 GiB、PyBullet 0.78 GiB |
| checkpoint wall | shard 0：101.40 s；shard 10：158.85 s |

task 0–19 的接受次序明显交错，证明 endpoint/GPU/PyBullet 会跨 shard 推进；本次仍是 shard
0 先完成，乱序发布能力另有确定性单元测试覆盖。训练入口校验通过，task id 为 0–19，
validated spline 为真，left/right-only inactive arm 最大漂移为 0。完成后 `.inflight` 没有
残留，四个 endpoint、GPU、PyBullet actor 都没有重启。

这次小样本里 GPU 已是主瓶颈，增加 endpoint actor 不会提高其 91.9% busy 下的吞吐；真正
打开 80-task 窗口后，重点看 12/16/16 occupancy 是否提高，而不是继续盲目增加 CPU 进程。
telemetry 的全局 plan query 与最终 shard 原始计数逐 mode 完全一致。

### 旧同步 multi-query 100 条：dual=8、left/right=16、3 endpoint actors

2026-09-14 在 RTX 4090 D 上完成 100/100，进程 wall 为 1,392.58 s，即 13.93 s/条。
运行期间另一个推理服务间歇占用约 1.7–8.1 GB 显存，GPU 总利用率观测到 99%，因此这是
“共享 GPU 负载下的稳定吞吐”，不能作为独占 GPU 极限速度或与旧结果做严格 A/B。

| 阶段 | 100 条 wall | 每条 | checkpoint 内占比 |
|---|---:|---:|---:|
| Endpoint/IK proposal（3 CPU actors） | 581.12 s | 5.81 s | 42.18% |
| GPU endpoint audit | 0.87 s | 0.009 s | 0.06% |
| PyBullet endpoint audit | 4.22 s | 0.042 s | 0.31% |
| GPU RRT + shortcut + raw/spline Torch audit | 654.59 s | 6.55 s | 47.51% |
| PyBullet trajectory audit | 136.88 s | 1.37 s | 9.93% |
| actor 启动、两次 PyBullet recycle、写入/merge 等 checkpoint 外开销 | 14.83 s | 0.15 s | — |

这次测试发生在 active-window、candidate wavefront 和跨 shard 事件流水实现之前，各
checkpoint 内部同步完成，因此只作为旧实现基线。10 个 checkpoint 的中位数为 92.27 s，
p90 为 263.73 s，范围 76.70–303.23 s。长尾来自
少数 dual task：例如 task 17 和 task 36 分别到第 23、26 轮才完成。总计 197 次
task-attempt、2,994 个 endpoint candidates、472 个 planning queries、85 个 GPU batches；
平均只有 5.55 query/batch，明显低于配置上限。100 条中另有 175 条已经通过全部审核但因
同 task 先有结果而未采用，说明下一步优先级应是按 task 做 candidate wavefront，而不是
继续放大 batch。

最终分布严格为 60 dual、20 left-only、20 right-only，方向为 45/35/10/10；inactive arm
最大漂移为 0。GPU actor 观测显存约 4.5 GB，3 个 endpoint actors、GPU actor、PyBullet
actor 全程无自动重启，50 条和结束时的 PyBullet recycle 均成功。

## 早期单-query 实现

- 新增纯 Torch/CUDA 的双向 batched RRT，批量做采样、最近邻扩展和边碰撞检测。
- `GpuBatchRRTConnect` 的 batch 是**单条轨迹内部**的并行：每轮同时扩展 64 条候选边，
  不是同时规划 64 条轨迹。新的 `GpuMultiQueryRRTConnect` 才增加独立 query 维度，并在
  每个 query 内默认并行 16 条候选边。
- `dual_independent` 在完整 14-D 空间规划。
- `left_only` 和 `right_only` 只在对应 7-D 子空间规划；另一只手臂在树、边检查、
  返回路径和样条输入中都保持 `q_start`，不会把“终点不动”误当成“路径中间也不动”。
- GPU 规划仍使用 Marvin 的 1,035 个细碰撞球、自碰撞 pair、warehouse analytic
  primitives；早期入口的最终路径仍经过 CPU Torch 与 PyBullet 两套稠密审计，专用入口
  则把 Torch 审计移到 GPU actor。
- launcher 支持 `--planner`、`--gpu-device` 和 `--gpu-batch-size`。当前一个 CUDA device
  只允许一个持久 worker，以避免多个 CUDA context 抢显存和不可控的吞吐退化。
- 该单进程仍串行执行 endpoint/IK 和最终 PyBullet mesh 审计；它移除了实际 RRT solve
  对 OMPL 的依赖，但仍保留 PyBullet，因此必须用 soak 而不能仅凭 10 条 smoke 宣称不崩。
- 新增固定端点 paired benchmark 和独立的 GPU RRT 单元测试。

启用方式：

```bash
python scripts/generate_data/launch_generate_marvin_warehouse_bimanual.py \
  --planner GpuBatchRRTConnect \
  --gpu-device cuda:0 \
  --gpu-batch-size 64 \
  --num-trajectories 100 \
  --output-dir /path/to/output
```

## 实测环境和口径

- GPU：NVIDIA GeForce RTX 4090 D，24 GB
- CPU 与 GPU 使用相同的生成配置、10 s planner timeout、0.35 rad extension、
  0.025 rad 最终碰撞插值步长和 512 点样条审计。
- “固定端点”测试从已有数据集中为 dual、left-only、right-only 各取 10 对端点。
  它隔离 planner 和相同的最终审计，不包含 IK/区域采样。
- “完整生成”测试使用相同 seed，从采样开始，直到真正写出 10 条通过全部审计的轨迹。
  为公平观察单 query pipeline，两者都使用一个 worker：10 条任务只构成一个 shard，
  所以 CPU 配置虽然写着 `workers: 3`，本次也只有 `1/3 active CPU workers`。该测试没有
  重现或衡量 CPU 三 worker 崩溃。

### 固定端点 paired benchmark（每种任务 10 对）

| 后端 / 任务 | 样条有效数 | RRT 均值 | 每请求 wall 均值 | 每条有效样条 wall |
|---|---:|---:|---:|---:|
| CPU / 全部 | 24/30 | 2.052 s | 4.613 s | 5.766 s |
| GPU / 全部 | 22/30 | 0.599 s | 2.974 s | 4.056 s |
| CPU / dual | 8/10 | 2.030 s | 4.624 s | 5.780 s |
| GPU / dual | 7/10 | 0.688 s | 3.081 s | 4.402 s |
| CPU / left-only | 10/10 | 1.554 s | 4.121 s | 4.121 s |
| GPU / left-only | 7/10 | 0.398 s | 2.533 s | 3.618 s |
| CPU / right-only | 6/10 | 2.572 s | 5.093 s | 8.488 s |
| GPU / right-only | 8/10 | 0.711 s | 3.309 s | 4.136 s |

这 30 对样本足够说明 RRT kernel 的方向，但不足以证明成功率等价。两种随机规划器会产生
不同路径，22/30 与 24/30 的差异应在至少数百个 paired requests 上重新估计置信区间。

### 完整生成到 10 条有效轨迹

| 阶段 | CPU RRTConnect | GPU batch RRT |
|---|---:|---:|
| 端点采样 | 13.27 s | 16.99 s |
| RRT（含失败尝试） | 60.56 s / 14 次 | 51.26 s / 25 次 |
| raw path 稠密审计 | 20.74 s | 74.91 s |
| spline 生成和审计 | 58.89 s / 0 次拒绝 | 131.58 s / 14 次拒绝 |
| 进程总 wall | **158.18 s** | **280.54 s** |
| 每条最终有效轨迹 | **15.82 s** | **28.05 s** |

CPU 的本次 15.82 s/条与仓库已有大样本记录的约 15.04 s/条接近。GPU 流程虽然每次
RRT 尝试更快，但为了得到 10 条最终样本进行了更多规划和样条重试，GPU 产生的候选路径
也让 raw/spline 审计成本分别增加。也就是说，只有 planner microbenchmark 会得出错误的
上线结论。

从 CPU 完整测试看，RRT 仅占 wall time 的 38.3%。即使 RRT 变成零耗时，其他阶段完全
不变，Amdahl 上限也只有约 1.62x。历史优化后生产统计中 RRT 占 worker stage time 的
66.0%，对应更乐观但仍仅约 2.94x 的 RRT-only 理论上限。

## 三条路线的工作量与预期

以下是工程人日估算，不是已经测出的性能承诺。

| 路线 | 当前状态 / 主要工作 | 达到可生产比较的剩余工作量 | Marvin 端到端速度判断 |
|---|---|---:|---|
| 仓库内 GPU batch | 已接入跨 shard active window、14-D/7-D multi-query、单 candidate wavefront、GPU shortcut/批量 Torch audit、隔离 actors、partial spool 和 telemetry | 1–3 人日：80-task 满窗的 100/1,000 条 soak、300+ paired 回归和参数调优 | 早期单 query 已测：RRT 3.43x、固定端点有效样条 1.42x、完整生成 **0.56x**；新流水线 20 条为 8.60 s/条，但还不能报告可靠生产加速比 |
| pRRTC | Marvin 不在上游支持列表；需 Foam 球化 URDF、Cricket FKCC CUDA codegen、Marvin/Pika robot struct、编译和 Python 数据管线适配，并分别处理 14-D 与两种锁臂 7-D | 7–12 人日 | 上游报告 RRT 层面约 6–10x 的量级，但没有 Marvin/Pika 数据；若不同时解决路径质量与最终审计，端到端远低于该数字 |
| cuRobo | 本机 runtime 有 `BatchMotionPlanner`，但现有 Marvin 配置不含 Pika gripper collision links，也没有可直接用于每个 query 的 inactive-arm dynamic lock | 6–12 人日：重做 Pika collision YAML、精确 warehouse world、三套/动态锁臂配置、环境桥接、语义与回归验证 | 不应拿缺 Pika 的 smoke test 当结果；需要完成等价几何后才可测。它也会改变数据的规划器/路径分布，不是原 RRT 的无缝替换 |

pRRTC 上游目前列出的机器人是 Panda、Fetch 和 Baxter；增加机器人需要生成近似/精细
球化模型和 FK/碰撞 CUDA 代码并重新编译。其 README 报告平均约 10x，而论文摘要中的
受限 reaching 实验给出最高约 6x 的平均提升，二者都是 planner 层结果，不能直接套到
本仓库的端到端数据生成。

cuRobo 路线更适合“同时批量解决很多目标并接受 trajectory-optimization/roadmap 的路径
分布”，但 left/right-only 必须硬锁非活动关节，而不仅是把目标设成相同值。当前配置缺少
Pika 碰撞几何时，跑出的更快数字会漏碰撞，不具可比性。

## 推荐顺序

1. 先跑 100 条使 80-task active window 真正填满；根据 `pipeline_telemetry.yaml` 比较
   dual/left/right occupancy、队列等待和 GPU peak，再决定是否增大 query batch。20 条 smoke
   的 GPU busy 已达 91.9%，当前不应继续增加 endpoint actors。
2. 随后跑 1,000 条 soak，记录退出码、每个 checkpoint wall、actor restart、partial spool
   恢复、CUDA/host 峰值与 shard 完整性；同时用 CPU `workers=1` 作稳定基线。不要把已知会
   造成主机问题的 CPU `workers=3` 当作可实现吞吐基线。
3. 专门做一次受控中断/恢复测试：至少等一个 task 写入 `.inflight` 后终止专用 launcher，
   重新运行并确认已接受 task 不重算、完整 shard 原子发布、最终 spool 清空。
4. 用每种模式至少 100 对、总计 300+ 固定端点，以及不少于 1,000 条完整生成，比较
   成功率、wall time、路径长度、最小 clearance、样条拒绝率和任务分布。
5. soak 通过后再决定是否把专用 launcher 作为生产入口；原 CPU generator/launcher 和默认
   `RRTConnect` 保持不变。若仍达不到吞吐目标，再优先试 cuRobo；只有在确实要
   保持 RRT 算法族且能接受 C++/CUDA codegen 维护成本时，再投入 pRRTC 适配。

## 复现

固定端点比较：

```bash
python scripts/generate_data/benchmark_marvin_planner_backends.py DATASET_ROOT \
  --per-mode 10 \
  --backends RRTConnect GpuBatchRRTConnect \
  --output benchmark_results/marvin_planner_backends.json
```

测试：

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_gpu_batch_rrt.py \
  tests/test_marvin_gpu_planning_backend.py \
  tests/test_marvin_gpu_pipeline_workers.py \
  tests/test_marvin_gpu_task_contract.py \
  tests/test_marvin_gpu_scheduler.py \
  tests/test_marvin_gpu_streaming.py \
  tests/test_marvin_gpu_partial_spool.py \
  tests/test_marvin_gpu_pipeline_launcher.py \
  tests/test_marvin_warehouse_generation.py
```

## 外部参考

- pRRTC repository: https://github.com/CoMMALab/pRRTC
- pRRTC paper: https://arxiv.org/abs/2503.06757
- OMPL planners（其中 `pRRT` 是共享树并行 RRT，不是 GPU pRRTC）：
  https://ompl.kavrakilab.org/planners.html
