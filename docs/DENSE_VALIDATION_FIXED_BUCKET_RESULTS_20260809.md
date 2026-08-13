# Dense validation 固定 GPU 桶实现与多环境结果（2026-08-09）

> 后续发现 GridMapSDF 将 CUDA index 强制复制到 CPU。修复及新一轮结果见
> `GRID_MAP_SDF_GPU_INDEX_BENCHMARK_20260809.md`；本文件下表保留修复前基线。

## 结论

已将 ranked dense validation 从任意动态 batch 改为固定渐进桶：

```text
top-8 -> next-16 -> next-32 -> remaining-in-64
```

100 条候选全部走完时，真实候选数量是 `8 + 16 + 32 + 44 = 100`；最后一次使用固定 64 槽 buffer，前 44 个 slot mask 为 true，后 20 个 padding slot 为 false。

固定桶和 full-dense 在 7 个场景、每个场景 3 次重复中均选择了同一个最终 candidate，没有出现 full-dense 有解但固定桶无解的回归。不过当前冷进程测试中速度收益有限：Warehouse 后处理加速约 `1.04–1.06x`，ThreePillars 基本持平，两个 Drawer 为负优化。

主要原因不是固定桶计算错误，而是每个新进程首次调用 dense validator 时约有 `210 ms` 的 TorchKin/FK/SDF 固定初始化开销。该开销对 batch=8 和 batch=100 接近相同。

## 实现

- 固定 bucket capacity：`[8, 16, 32, 64]`；
- 每个 bucket 持久保存固定地址的 control-point input buffer 和 slot-valid mask；
- planner 构造阶段预先分配四套可控输入 buffer；
- 不足 bucket capacity 的部分复制最后一个真实候选作为数值安全 padding；
- padding 结果在 GPU slot mask 中被排除，不能成为 valid candidate；
- 输出记录真实 checked 数、执行 batch 数、bucket capacity 和 padding 数；
- full-dense 默认仍可独立开启，用于完整候选有效率和碰撞率统计。

TorchKin FK、SDF 和 self-collision 内部产生的中间 tensor 暂时仍由 PyTorch caching allocator/外部库管理；只有启用 CUDA Graph 或给这些接口增加显式 `out=` workspace，才能保证所有中间 tensor 地址完全固定。

## 测试设置

- candidates：100；
- DDIM steps：15；
- guide fraction：20%；
- guide iterations：每步 6；
- dense points：128；
- fixed buckets：8 / 16 / 32 / 64；
- 每个场景相同配置、相同 seed；
- full-dense 和 fixed-bucket 每个场景各重复 3 次；
- 重复间交替执行顺序；
- p50：3 次内部 CUDA 同步计时的中位数；
- accuracy：比较最终 selected candidate index 和是否找到 valid trajectory。

原始结果：`/tmp/mpd-dense-fixed-vs-full-20260809/{runs.json,runs.csv,summary.csv,summary.md}`。

## 结果

| 场景 | Full dense ms | Rank ms | Bucket dense ms | Rank+dense ms | 后处理加速 | 总流程加速 | 实际 checked | batches | 同一候选 | coverage 回归 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Warehouse open | 220.270 | 0.160 | 212.646 | 212.803 | 1.035x | 1.056x | 8 | 1 | 3/3 | 0 |
| Warehouse single obstacle | 224.725 | 0.147 | 212.460 | 212.607 | 1.057x | 1.010x | 8 | 1 | 3/3 | 0 |
| Warehouse narrow-0.20 | 221.076 | 0.163 | 213.031 | 213.194 | 1.037x | 0.997x | 8 | 1 | 3/3 | 0 |
| Warehouse narrow-0.14 | 222.064 | 0.162 | 212.596 | 212.766 | 1.044x | 1.006x | 8 | 1 | 3/3 | 0 |
| ThreePillars | 215.672 | 2.620 | 212.722 | 215.737 | 1.000x | 0.963x | 8 | 1 | 3/3 | 0 |
| To drawer | 217.101 | 2.773 | 218.996 | 221.684 | 0.979x | 1.021x | 24 | 2 | 3/3 | 0 |
| Drawer to shelf | 217.509 | 2.832 | 224.476 | 227.143 | 0.958x | 0.983x | 56 | 3 | 3/3 | 0 |

Warehouse 使用 `shortest_path_length`，因此排序只需要约 `0.15 ms`；ThreePillars 和 Drawer 使用 weighted metrics，排序需要约 `2.6–2.8 ms`。

Drawer-to-shelf 检查了 56 条真实候选，即执行 8、16、32 三个固定 bucket。虽然比第一批多执行两次，dense p50 只从约 213 ms 增加到 224 ms，进一步说明约 210 ms 是首次调用固定成本，后续固定 bucket 的边际成本较小。

本轮没有实际触发最后的 44-in-64 路径，因为所有场景都在前三批内找到 valid；对应 padding 路径已由单元测试覆盖，检查 capacity `[8,16,32,64]`、真实 checked=100、padding=20。

## CUDA Graph 判断

当前 `cuda_graph` 保持关闭。直接跨请求 capture/replay 还需要先解决两类地址稳定性：

1. B-spline 的 start/goal position、velocity、acceleration boundary tensor 会随请求重新绑定；
2. 动态场景更新可能替换 SDF/障碍物 tensor。

直接捕获当前对象地址再处理下一请求存在 replay 旧边界或旧场景的风险。建议下一步按以下顺序实现：

1. runtime 改为持久 planner/validator，而不是每个请求新建进程和模型；
2. 为六个 B-spline boundary tensor 建立固定地址 buffer，每个请求只做 `copy_()`；
3. 为场景 SDF 建立版本号，地址或版本变化时使 graph 失效并重新 capture；
4. 分别 warm/capture 8、16、32、64 四张 graph；
5. 分开报告 cold prepare/capture 时间与 steady-state replay 时间。

在当前“一次请求启动一个新进程”的 runtime 下，即使把 210 ms 移入 warmup，也只是改变计时归属，不会降低端到端首请求延迟。因此固定桶保留，但不应把本轮结果描述为明显加速；明显收益需要持久进程加 CUDA Graph replay。
