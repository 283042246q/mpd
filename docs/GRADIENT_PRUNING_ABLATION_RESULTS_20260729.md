# 梯度裁剪逐步开启消融结果（2026-07-29）

## 1. 测试设置

| 项目 | 设置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D，24 GiB |
| 场景 | Warehouse/Panda：open clearance、single obstacle、narrow 0.14 |
| 候选数 | 100 |
| Context | 每个场景 1 个，所有阶段使用相同 seed 和 start/goal |
| 重复次数 | 3 个独立进程 |
| Diffusion | DDIM 15 steps，配置和 checkpoint 完全相同 |
| Safety oracle | 所有阶段使用相同的 512 点 dense checker |
| 计时方式 | `profile: false`，避免 section 级 CUDA synchronize；保留 active statistics |
| autoset | 未启用；阈值固定，无自动搜索 |

逐步开启顺序：

| 阶段 | 启用内容 |
|---|---|
| A0 | Legacy 原始路径 |
| A1 | 新 dispatcher/稀疏框架，但仍使用全 128 点 collision Jacobian |
| A2 | A1 + EE 只计算末端 |
| A3 | A2 + temporal risk selector + K=0/32/64/128 分桶 |
| A4 | A3 + parent-link collision-sphere 运动学 |

## 2. 完整结果

加速比均为“上一阶段或 Legacy 时间 ÷ 当前阶段时间”；小于 `1.0×` 表示变慢。

| 场景 | 难度 | 阶段 | Guide p50 (s) | 增量加速 | 相对 Legacy | 推理 p50 (s) | 含 dense 总加速 | Active 点 | K0/K32/K64/K128 | Valid |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| open_clearance | simple | A0 Legacy | 0.360 | — | 1.000× | 0.419 | 1.000× | 100.0% | — | 90% |
| open_clearance | simple | A1 新路径/全量 Jac | 0.369 | 0.975× | 0.975× | 0.430 | 0.994× | 100.0% | 0/0/0/100% | 90% |
| open_clearance | simple | A2 +EE 末端 | 0.362 | 1.020× | 0.994× | 0.418 | 1.005× | 100.0% | 0/0/0/100% | 90% |
| open_clearance | simple | A3 +时间分桶 | 0.739 | 0.490× | 0.487× | 0.797 | 0.777× | 48.2% | 8.8/0/86.0/5.2% | 90% |
| open_clearance | simple | A4 +Parent-link | 0.639 | 1.157× | 0.564× | 0.697 | 0.826× | 48.2% | 8.8/0/86.0/5.2% | 90% |
| single_obstacle | medium | A0 Legacy | 0.410 | — | 1.000× | 0.469 | 1.000× | 100.0% | — | 92% |
| single_obstacle | medium | A1 新路径/全量 Jac | 0.424 | 0.967× | 0.967× | 0.481 | 0.987× | 100.0% | 0/0/0/100% | 92% |
| single_obstacle | medium | A2 +EE 末端 | 0.407 | 1.042× | 1.008× | 0.467 | 1.001× | 100.0% | 0/0/0/100% | 92% |
| single_obstacle | medium | A3 +时间分桶 | 0.816 | 0.499× | 0.503× | 0.875 | 0.767× | 59.3% | 0/0/81.3/18.7% | 92% |
| single_obstacle | medium | A4 +Parent-link | 0.730 | 1.119× | 0.562× | 0.787 | 0.808× | 59.3% | 0/0/81.3/18.7% | 92% |
| narrow_014 | hard | A0 Legacy | 0.405 | — | 1.000× | 0.465 | 1.000× | 100.0% | — | 66% |
| narrow_014 | hard | A1 新路径/全量 Jac | 0.402 | 1.009× | 1.009× | 0.460 | 1.002× | 100.0% | 0/0/0/100% | 66% |
| narrow_014 | hard | A2 +EE 末端 | 0.400 | 1.003× | 1.012× | 0.460 | 1.007× | 100.0% | 0/0/0/100% | 66% |
| narrow_014 | hard | A3 +时间分桶 | 0.776 | 0.516× | 0.522× | 0.836 | 0.790× | 85.0% | 0/0/29.9/70.1% | 66% |
| narrow_014 | hard | A4 +Parent-link | 0.682 | 1.138× | 0.594× | 0.742 | 0.832× | 85.0% | 0/0/29.9/70.1% | 66% |

三档场景的 collision rate 在各阶段也完全一致，分别为 `1% / 1% / 22%`。因此本轮没有观察到安全或 coverage 回退。

## 3. 单项功能效果

| 功能 | 三场景平均趋势 | 结论 |
|---|---:|---|
| 新路径全量等价框架（A1/A0） | 约 `0.984×` | 约 1.6% 框架开销，符合等价层预期，但没有加速 |
| EE 末端优化（A2/A1） | 约 `1.022×` | 约 2.2% guidance 加速；收益小但方向正确 |
| 时间风险选择与分桶（A3/A2） | 约 `0.502×` | 虽减少 Jacobian 点数，但 guidance 约慢 2 倍 |
| Parent-link（A4/A3） | 约 `1.138×` | 相对 A3 加速约 13.8%，是当前最明确的正收益 |
| 完整 A4/A0 | 约 `0.573×` | 完整裁剪路径仍约慢 1.75 倍，未达到发布门槛 |

不属于逐步裁剪开关、但同样完成的功能：

| 功能 | open / medium / hard p50 | 归因方式 |
|---|---:|---|
| 512 点 dense checker（A0） | `0.864 / 0.890 / 0.902 s` | 独立安全 oracle，不参与 guidance 加速 |
| 512 点 dense checker（A4） | `0.864 / 0.892 / 0.899 s` | 与 A0 基本一致，证明它未复用 active set |
| Diffusion generator（A0） | `0.055 / 0.054 / 0.056 s` | 所有阶段不变 |
| Diffusion generator（A4） | `0.052 / 0.054 / 0.054 s` | 波动在毫秒级，不归因给裁剪 |
| 显式 self-collision 梯度 | 所有阶段共享 | 当前没有运行时开关，需与改动前 commit 做独立 microbenchmark，不能从 A0/A4 推断 |
| Guidance profiler | 正式计时关闭 | 开启后会在 section 边界同步 CUDA，只用于分段诊断，不用于最终加速比 |

## 4. 难度分布是否符合逻辑

风险分配随场景难度单调增加：

| 难度 | Active 时间点 | K128 候选 | 解释 |
|---|---:|---:|---|
| simple | 48.2% | 5.2% | 有 8.8% 候选进入 K0，大多数为 K64 |
| medium | 59.3% | 18.7% | 障碍物使更多候选升级到 K128 |
| hard | 85.0% | 70.1% | 窄通道中大多数候选保守退化到全量 |

这说明 selector 的难度响应和安全退化方向正确，但当前 bucket 利用不理想：三个场景的 `K32` 均为 0。只要 coarse 32 点之外出现任意风险点，当前实现通常就直接升级到 K64。

## 5. 为什么点数减少但时间变慢

1. selector 当前先对完整 128 点执行 collision-sphere FK 和 SDF/self-clearance 扫描。
2. 随后又对 active 点执行 FK + Jacobian；Legacy 的 `jfk_s_collision_spheres` 一次同时返回 pose 和 Jacobian，因此新路径产生了重复 FK。
3. 每个 bucket 分开执行 TorchKin、cost 和 scatter，增加了 Python 循环、小 kernel 与 tensor 重排开销。
4. simple/medium 场景绝大多数候选落入 K64，而不是预期的 K0/K32；hard 场景 70.1% 退化到 K128。
5. Parent-link 明显降低了 A3 的运动学成本，但无法抵消完整风险扫描和分桶开销。

## 6. 当前启用建议

| 使用场景 | 建议 |
|---|---|
| runtime/生产 | 保持 `gradient_pruning.enabled: false` |
| 只验证等价路径 | 使用 A1/`force_all_active` |
| 实验 EE endpoint-only | 可使用 A2；收益约 0–1.2% 相对 Legacy，需更多 context 后再默认开启 |
| temporal buckets | 暂不用于性能模式；当前只适合作为功能/安全实验 |
| parent-link | 保留；它相对 temporal 阶段稳定加速 11.9–15.7%，但应与下一版 selector 一起验收 |
| hard/narrow | 保持 Legacy 或明确退化到 K128，不能为了速度降低 dense checker |

下一轮优化优先级：

1. 将风险扫描改为真正的 32 点 coarse FK，只对危险区间追加 64/128 点；
2. 让低风险候选实际进入 K32，而不是默认升级 K64；
3. 复用 coarse FK pose，避免 active Jacobian 阶段重复计算 FK；
4. 向量化 bucket cost/scatter，合并过小 bucket；
5. 完成上述修改后，再执行多 context 的 `100 candidates × 3 repeats` 验收。

## 7. 复现

```bash
LD_LIBRARY_PATH=/home/eric/anaconda3/envs/mpd-splines-public/lib \
/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  scripts/inference/benchmark_gradient_pruning_ablation.py \
  --suite warehouse_panda \
  --scenario-ids open_clearance,single_obstacle,narrow_014 \
  --device cuda:0 \
  --contexts 1 \
  --candidates 100 \
  --repeats 3 \
  --dense-points 512 \
  --seed 0 \
  --output-dir /tmp/mpd-ablation-warehouse-timing-20260729
```

原始汇总：

```text
/tmp/mpd-ablation-warehouse-timing-20260729/reports/ablation-summary.csv
/tmp/mpd-ablation-warehouse-timing-20260729/reports/ablation-summary.md
```
