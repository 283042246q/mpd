# GridMapSDF GPU index benchmark（2026-08-09）

## 修改

旧实现把 CUDA 计算出的 grid index 强制复制到 CPU：

```python
X_in_map = (...).round().type(torch.LongTensor)
max_idx = torch.tensor(self.points_for_sdf.shape[:-1]) - 1
```

新实现保持 index 和 clamp bounds 在输入设备：

```python
X_in_map = (...).round().to(dtype=torch.long)
max_idx = torch.as_tensor(
    self.points_for_sdf.shape[:-1], dtype=torch.long, device=X.device
) - 1
```

只修改第一行会因为 CUDA index 和 CPU clamp bounds 混用而报 device mismatch，因此两行必须一起修改。

## 同步微基准

输入 shape 模拟 Panda dense collision positions：`[batch, 128, 69, 3]`。每轮前后执行 CUDA synchronize，p50 为 30 次中位数；CPU/GPU 两种 index 和 SDF lookup 结果逐元素完全一致。

| Batch | points | CPU projection+lookup p50 | GPU projection+lookup p50 | 加速 |
|---:|---:|---:|---:|---:|
| 8 | 70,656 | 0.482 ms | 0.049 ms | 9.81x |
| 16 | 141,312 | 0.978 ms | 0.120 ms | 8.14x |
| 32 | 282,624 | 1.458 ms | 0.121 ms | 12.08x |
| 64 | 565,248 | 2.449 ms | 0.125 ms | 19.53x |
| 100 | 883,200 | 3.725 ms | 0.144 ms | 25.83x |

## 同一 planner 连续三个 context

Warehouse single-obstacle，100 candidates、15 DDIM、20% guide、每步 6 iterations、full dense=128。三个 context 在同一个 planner/validator 进程内连续执行。

| Index 实现 | Context | Guide report | Inference report | Dense report |
|---|---:|---:|---:|---:|
| CPU | 1（FK-only 首次） | 473.647 ms | 526.419 ms | 222.910 ms |
| CPU | 2（warm） | 474.838 ms | 527.899 ms | 13.082 ms |
| CPU | 3（warm） | 472.319 ms | 530.630 ms | 13.402 ms |
| GPU | 1（FK-only 首次） | 359.562 ms | 415.514 ms | 208.785 ms |
| GPU | 2（warm） | 364.685 ms | 419.249 ms | 5.513 ms |
| GPU | 3（warm） | 319.177 ms | 372.681 ms | 6.428 ms |

GPU index 的 warm dense 约为旧实现的 `0.42–0.48x`。首次 dense 仍有约 209 ms，因为这是 FK-only/dense 路径第一次调用的固定成本，SDF index 只解释其中约十几毫秒。

Guide 也受益，因为每次 collision guide 的环境 SDF 查询都调用相同 GridMapSDF projection。CPU cast 是每次 SDF query 都会发生的重复同步，不是一次性冷启动。

## 计时口径警告

当前 `TimerCUDA()` 默认 `sync_cuda=False` 且不使用 CUDA events。上述 guide/inference/dense report 是仓库现有口径：

- planner 构造时已对 model 和 Cost Guide warmup 5 轮，所以 guide/inference report 不包含它们的首次调用；
- dense validator 没有 warmup，所以每个新进程的第一个 dense report 包含 FK-only 首次调用；
- 旧 CPU index cast 本身形成隐式 CUDA synchronization，可能把此前排队的 GPU 工作计入当前 section；
- 新 GPU index 移除了这个同步点，因此各 section 数字不能被当作严格的独立 CUDA kernel 总和。

微基准显式 synchronize，因此 CPU/GPU index 投影本身的结果可靠。若要得到严格的 generator/guide/dense 分阶段时间，benchmark 模式应使用 CUDA events，或在每个阶段边界显式 synchronize；生产路径不应为了计时默认加入这些同步屏障。

原始输出：

- 微基准脚本：`scripts/inference/benchmark_grid_index_cast.py`；
- GPU-index 7 场景：`/tmp/mpd-dense-gpu-index-20260809`；
- CPU-index 常驻对照：`/tmp/mpd-dense-warm-sequence-cpu-index`；
- GPU-index 常驻对照：`/tmp/mpd-dense-warm-sequence-gpu-index`。
