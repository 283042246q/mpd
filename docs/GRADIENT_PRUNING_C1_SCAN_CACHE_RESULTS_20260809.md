# C1 细球预扫缓存与 Jacobian-only 测试（2026-08-09）

## 结论

C1 已按“预扫数据直接喂给精确 Cost Guide”的方式完成修改：

1. 全 128 点细球预扫保存 TorchKin 相关链 pose、物理 parent pose、56 个细球 pose/中心、
   环境 SDF distance 和 self-pair distance；
2. 精确阶段按激活 parent 分组，从缓存 pose 构造 spatial Jacobian，不再调用原 J-FK
   内部的 FK；
3. 环境 Cost Guide 复用预扫 SDF distance，GridMapSDF 只查询预计算 gradient；
4. 同一 DDIM step 后续 guide iteration 的轨迹已经改变，只复用 selection mask，不复用
   旧 pose/distance，避免使用过期几何。

实现和数值等价性均通过，但 C1 仍不应默认开启。7 场景的 Guide p50 算术平均为
`B3 0.285201 s / C1 0.454804 s`，C1 相对 B3 只有 `0.627x`；逐场景加速比的几何
平均为 `0.656x`。最好的 ThreePillars 也只有 `0.984x`，没有场景稳定超过 B3。

## 实现

TorchKin 新增 `get_forward_kinematics_pose_cache_fns`：

- FK producer 返回目标 parent 所需的最小 related-link pose tuple；
- Jacobian consumer 直接调用 TorchKin backward helper，从 pose 生成 spatial Jacobian；
- consumer 不接收 `q`，因此接口层面保证不会偷偷重跑 FK；
- C1 为每种激活 parent mask 缓存 Jacobian consumer，只计算该 mask 的 parent 输出。

C1 的 `FineSphereScanCache` 保存：

- `related_poses`：Jacobian-only 所需祖先/parent pose；
- `sphere_poses`：精确 Cost Guide 可直接使用的细球 pose 和中心；
- `environment_sdf_values`：逐 SDF、逐细球的原始 distance；
- `self_pair_distances`：预扫的自碰撞 pair distance，供后续研究复用。

开关为：

```yaml
gradient_pruning:
  spatial:
    link_broad_phase:
      enabled: true
      full_scan: true
      scan_geometry: fine_spheres
      reuse_scan_cache: true
```

`reuse_scan_cache` 只改变 C1 内部执行方式；B3 的 `link_broad_phase.enabled` 仍为
`false`，默认路径不受影响。

## 统一测试协议

- 100 candidates；
- 15 DDIM steps；
- 最后 20%（3 个）DDIM steps 开 guide；
- 每个 guided step 6 iterations，共 18 guide calls；
- dense validation=128；
- seed=0，同场景 B3/C1 使用相同 start/goal；
- 每个变体 3 次重复，报告 p50；
- Temporal、Clean-x0、融合映射均关闭，B3 为 A2P-fast/no-grad/原映射 + sparse
  `J^Tg`。

## 时间与准确度

加速比定义为 `B3/C1`，小于 1 表示 C1 负优化。

| 场景 | B3 Guide p50 (s) | C1 Guide p50 (s) | Guide 加速 | B3 inference (s) | C1 inference (s) | Sphere 保留 | B3/C1 valid | B3/C1 collision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Warehouse open | 0.244601 | 0.785655 | 0.311x | 0.302332 | 0.841056 | 82.3% | 91.0% / 91.0% | 0.0% / 0.0% |
| Warehouse single | 0.312454 | 0.469700 | 0.665x | 0.369337 | 0.525829 | 87.9% | 93.0% / 93.0% | 1.0% / 1.0% |
| Warehouse narrow-0.20 | 0.313690 | 0.364469 | 0.861x | 0.369946 | 0.420960 | 78.9% | 87.0% / 87.0% | 0.0% / 0.0% |
| Warehouse narrow-0.14 | 0.312861 | 0.464415 | 0.674x | 0.368029 | 0.521573 | 86.8% | 75.3% / 75.7% | 11.7% / 11.3% |
| ThreePillars | 0.264958 | 0.269292 | 0.984x | 0.320293 | 0.322675 | 85.7% | 7.0% / 7.0% | 93.0% / 93.0% |
| Drawer-to-shelf | 0.279652 | 0.361654 | 0.773x | 0.337450 | 0.419708 | 96.8% | 1.0% / 1.0% | 99.0% / 99.0% |
| To-drawer | 0.268188 | 0.468440 | 0.573x | 0.328404 | 0.528791 | 94.7% | 25.0% / 25.0% | 73.0% / 73.0% |
| **算术平均** | **0.285201** | **0.454804** | **0.627x** | **0.342256** | **0.511513** | — | **54.19% / 54.24%** | **39.67% / 39.62%** |

每次重复有 18 个 guide calls，cache 只在每个 guided DDIM step 的第一次 iteration
有效，因此 C1 每次重复为 `3/18`、三次共 `9/54` 个 call 使用 pose/distance cache。

## 为什么仍是负优化

缓存已经消除了第一次 iteration 的重复 FK 和 distance lookup，但没有消除 C1 的三个
结构性成本：

1. 每个 guided DDIM step 仍必须先做一次 `100×128×56` 细球全扫描；
2. 实际 sphere 保留率为 78.9%--96.8%，多数场景裁剪量不足；
3. 每个不同 parent mask 形成独立 dispatch，open 场景 mask 碎片化尤其严重，小 kernel
   和 Python 分组开销超过 Jacobian/SDF 节省。

后续若继续研究 C1，优先级应是固定 8-parent bitmask 的单次 GPU kernel/块稀疏执行，
或只有预估 parent 保留率足够低时才启用；继续增加 pose/distance 缓存已不是主要矛盾。
默认仍保持 B3 单开，不加入 autoset。

## 验证与结果文件

- 项目测试：`92 passed, 2 skipped`；
- cached-pose Jacobian 与原 subset J-FK：`rtol=1e-9, atol=1e-10`；
- SDF 单测通过调用计数确认：有缓存时 distance 不再调用，只调用 gradient；
- 10-candidate CUDA smoke 和 7 场景正式测试全部完成；
- 原始结果：`benchmark_results/gradient_pruning_c1_cached_vs_b3_20260809/`；
- 合并表：`benchmark_results/gradient_pruning_c1_cached_vs_b3_20260809/c1-cached-vs-b3-summary.csv`。

