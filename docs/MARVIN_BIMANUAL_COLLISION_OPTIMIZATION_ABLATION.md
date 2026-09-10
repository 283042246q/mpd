# Marvin 双臂碰撞推理优化消融结果

> 日期：2026-09-10；GPU：NVIDIA GeForce RTX 4090 D，24564 MiB；PyTorch float32；
> Warehouse dataset sample 0；seed 12345；128 trajectory points；默认 32 candidates。

## 1. 结论

- 默认 32 candidates 下，全量 pair tensor（A0）OOM；只开启 pair streaming（A1）仍会在未分块的 dense validator OOM。
- `pair_streaming + validator_chunking`（A2）是 23.5 GiB GPU 上保持 32 candidates 的最小可运行组合。
- Parent-bound（A3）把 guide 时间从 10.74 s 降到 5.37 s，但 production geometry 的 scan/cache 峰值比 A2 高；它是速度优化，不是本机的最低显存配置。
- Foam guide（A4）将 guide geometry 从 1035 球/375482 pairs 降到 489 球/76001 pairs，guide 时间降到 4.13 s；最终 validator 仍为 production 1035 球。
- 四项全开（A5）成功，guide 时间 3.46 s，32 个候选中 11 个通过 production validator。
- 不建议因 OOM 直接修改网络、DDIM 步数或 B-spline horizon。先开启两个 exact chunk 开关；`n_trajectory_samples=32` 可保留。batch=8 虽可降低未优化路径的内存，但本次 A5 没有产生有效轨迹，候选多样性明显不足。

## 2. 端到端 A0--A5（batch=32）

开关顺序为 Pair streaming / Validator chunk / Parent bounds / Foam guide。

| Case | 开关 | 状态 | wall (s) | guide (s) | dense validator (s) | 设备峰值占用 (MiB) | valid / 32 |
|---|---|---|---:|---:|---:|---:|---:|
| A0 | 0 / 0 / 0 / 0 | CUDA OOM | 4.28 | - | - | 16659（采样峰值） | - |
| A1 | 1 / 0 / 0 / 0 | CUDA OOM | 16.60 | - | - | 19913（采样峰值） | - |
| A2 | 1 / 1 / 0 / 0 | success | 19.85 | 10.74 | 3.27 | 5271 | 8 |
| A3 | 1 / 1 / 1 / 0 | success | 14.59 | 5.37 | 3.18 | 12169 | 8 |
| A4 | 1 / 1 / 0 / 1 | success | 12.49 | 4.13 | 3.39 | 5271 | 10 |
| A5 | 1 / 1 / 1 / 1 | success | 11.79 | 3.46 | 3.35 | 6827 | 11 |

设备峰值来自 0.2 s 周期的 `nvidia-smi` 采样，包含测试开始前约 799 MiB 的非本进程占用，因此是设备级近似值。成功 case 的 `result.json` 另有进程内 `torch.cuda.max_memory_allocated/reserved`。OOM case 的采样可能错过失败瞬间，错误消息与 kernel 基准是更直接的证据。

所有成功 case 都通过原始 production dense validator。A5 最终 minimum clearance：environment 0.01676 m、left intra-arm 0.02231 m、right intra-arm 0.00106 m、inter-arm 0.24400 m；左右 EE position error 分别为 0.01046 m、0.00759 m。

## 3. batch 容量扫描

| Case | candidates | 状态 | wall (s) | guide (s) | dense (s) | 设备峰值 (MiB) | valid |
|---|---:|---|---:|---:|---:|---:|---:|
| A0 | 16 | CUDA OOM | 4.52 | - | - | 19499（采样峰值） | - |
| A0 | 8 | success | 11.25 | 5.68 | 0.70 | 16637 | 3 |
| A5 | 16 | success | 9.84 | 3.20 | 1.74 | 5271 | 1 |
| A5 | 8 | no valid trajectory | 8.69 | - | - | 5262 | 0 |

单个 sample/seed 不能用于统计成功率，但足以说明：降低 batch 是可选的容量旋钮，不是优先修复；本任务中 32 candidates 带来了更稳妥的有效候选数。动态世界需要为 Isaac Lab、感知或并行节点预留 GPU 时，可先尝试 A5 + batch=16，并用多 request/seed 测成功率；不建议直接降到 8。

## 4. 独立 kernel 消融

### Pair streaming 与 Foam guide

`B=32, H=128, inter-arm`：

| Geometry | spheres | inter-arm pairs | reduction | 状态 | time (s) | PyTorch peak allocated |
|---|---:|---:|---|---|---:|---:|
| production | 1035 | 239868 | full | CUDA OOM | - | 23.65 GB |
| production | 1035 | 239868 | streaming 4096 | success | 0.2226 | 1.01 GB |
| foam_pika_100 | 489 | 42762 | full | success | 0.0531 | 6.35 GB |
| foam_pika_100 | 489 | 42762 | streaming 4096 | success | 0.0383 | 0.98 GB |

两种 geometry 各自的 full/streaming cost sum 与 gradient norm 完全一致。Foam 与 production 的 cost 不要求相等，因为它本来就是不同的 guidance approximation。

### Dense validator chunking

Production 1035 球、375482 pairs、`B=32,H=128`：full validator OOM（峰值已分配约 18.68 GB，并尝试再申请 17.19 GiB）；`candidate=4,time=16,pair=4096` 成功，耗时 2.979 s，PyTorch peak allocated 43.4 MB。最终 mask 和分类 clearance 的等价性由单元测试在可同时执行 full/chunked 的尺寸上验证。

### Parent-bound scan

Production geometry、`B=32,H=128`：fine scan OOM；26 个 physical parents / 228 parent pairs 的 parent scan 成功，耗时 0.0195 s，PyTorch peak allocated 51.8 MB。本批 active fine-pair ratio 为 0.1394。保守覆盖、fine-pair 映射和边界 false-negative=0 由专项测试验证。

### Chunk 参数扫描

Production inter-arm streaming 在 `B=32,H=128` 的单次结果：pair chunk `1024/2048/4096/8192/16384` 分别耗时 `0.166/0.172/0.185/0.254/0.295 s`，peak allocated 分别约 `312/547/1017/1956/3835 MB`。在该 GPU/shape 上小 chunk 同时更快且更省内存，因此 Marvin runtime 的 opt-in 值采用 1024。

Validator 从 `candidate=4,time=16,pair=4096` 的 2.979 s / 43.4 MB，调整到 `candidate=8,time=32,pair=1024` 后为 1.125 s / 53.8 MB。因此 runtime 中保存的建议值更新为 `8/32/1024`；开关仍默认关闭，不改变既有 Phase 入口语义。

用调优参数复跑 A5/batch=32：success，wall 9.38 s、guide 3.41 s、dense validator 1.22 s、设备采样峰值 6906 MiB，9/32 candidates 通过 production validator。相对表中保守 chunk 的 A5，主要收益来自 dense validator 的 3.35 s 降到 1.22 s。

## 5. 复现

端到端：

```bash
conda run --no-capture-output -n mpd-splines-public \
python scripts/inference/benchmark_marvin_collision_optimizations.py \
  --config scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-independent-runtime.yaml \
  --output-dir scripts/inference/logs/marvin-collision-ablation/reproduce \
  --device cuda:0 --batch-size 32 --capacity-batches 16 8 --execute
```

Foam/streaming kernel：

```bash
conda run --no-capture-output -n mpd-splines-public \
python scripts/inference/benchmark_marvin_reduced_guide_geometry.py \
  --device cuda:0 --batch-size 32 --horizon 128 \
  --pair-category interarm --pair-chunk-size 4096 --repeats 1
```

原始详细结果保留在 `scripts/inference/logs/marvin-collision-ablation/`（该目录按仓库规则不提交）。
