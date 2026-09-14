# Marvin warehouse 轨迹规划后端比较

## 结论

在完成长时间稳定性 soak 之前，暂时不要直接把生产默认值从 OMPL `RRTConnect` 改成
GPU batch。仓库现在已有一个可通过
`--planner GpuBatchRRTConnect` 启用的实验后端，并保留了与原流程相同的最终 Torch 和
PyBullet 稠密校验。RTX 4090 上的固定端点配对测试显示，GPU 后端把 RRT 核心平均耗时
从 2.052 s 降到 0.599 s（3.43x），但在真正“生成到 10 条有效轨迹”为止的测试中，
总耗时从 158.18 s 增加到 280.54 s（15.82 s/条变成 28.05 s/条）。主要原因不是碰撞
查询速度，而是当前 GPU 树产生的路径更曲折，导致样条重试和最终稠密校验显著增加。

这里的 CPU/GPU 端到端数字都是**单 worker**，只回答每个稳定 worker 的延迟，不回答原
CPU launcher 多 worker 并发时的崩溃问题。如果目标是用一个稳定 GPU 进程替代多个会崩的
OMPL/PyBullet worker，那么 GPU 路线仍然合理；判断门槛应改为长时间稳定吞吐，而不是只看
单条延迟。因此本次先把 GPU batch 留作显式 opt-in，配置默认值继续使用 `RRTConnect`，
待 100/1,000 条单进程 soak 通过后再切换默认值。

## 本次实现

- 新增纯 Torch/CUDA 的双向 batched RRT，批量做采样、最近邻扩展和边碰撞检测。
- 当前 batch 是**单条轨迹内部**的并行：每轮同时扩展 64 条候选边，碰撞状态以最多 256
  个一组送入 GPU。它不是同时规划 64 条轨迹，也不是多个 GPU worker。
- `dual_independent` 在完整 14-D 空间规划。
- `left_only` 和 `right_only` 只在对应 7-D 子空间规划；另一只手臂在树、边检查、
  返回路径和样条输入中都保持 `q_start`，不会把“终点不动”误当成“路径中间也不动”。
- GPU 规划仍使用 Marvin 的 1,035 个细碰撞球、自碰撞 pair、warehouse analytic
  primitives；最终路径仍经过 CPU Torch 与 PyBullet 两套稠密审计。
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
| 仓库内 GPU batch | 已接入 14-D/7-D 模式、真实碰撞模型、最终双审计和 benchmark | 3–5 人日：GPU shortcut/平滑、clearance-aware 选择、批量 query pipeline、300+ paired 回归、worker 调度 | 已测：RRT 3.43x；固定端点有效样条 1.42x；完整生成目前 **0.56x**（更慢） |
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

1. 先跑 GPU 单进程 100 条和 1,000 条 soak，记录进程退出码、CUDA/host 峰值内存、每 10/100
   条 wall time、失败重试和 shard 完整性；同时用 CPU `workers=1` 作为稳定基线。不要再把
   已知会崩的 CPU `workers=3` 当作可实现吞吐基线。
2. soak 通过后，如果稳定产数优先于单条速度，可以把 GPU planner 切成默认；如果 GPU 仍崩，
   应进一步让 GPU 分支完全不构造 OMPL native setup，并把 PyBullet 审计隔离成可重启进程。
3. 若目标还包括提速，再改 GPU batch 的路径质量：连接成功后做 GPU shortcut，并用长度、
   clearance 和曲率筛选候选，再进行 512 点样条审计。完整生成测试表明这比继续加大采样 batch 更重要。
4. 把 endpoint/最终 Torch 审计改成 GPU query batches，同时保留 PyBullet 作为最终 CPU
   gate；否则 RRT 加速会受 Amdahl 限制。
5. 用每种模式至少 100 对、总计 300+ 固定端点，以及不少于 1,000 条完整生成，比较
   成功率、wall time、路径长度、最小 clearance、样条拒绝率和任务分布。
6. 如果仍达不到吞吐目标，再优先试 cuRobo（本机已有 fork 和 batch API）；只有在确实要
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
  tests/test_marvin_warehouse_generation.py
```

## 外部参考

- pRRTC repository: https://github.com/CoMMALab/pRRTC
- pRRTC paper: https://arxiv.org/abs/2503.06757
- OMPL planners（其中 `pRRT` 是共享树并行 RRT，不是 GPU pRRTC）：
  https://ompl.kavrakilab.org/planners.html
