# B3 Link Broad Phase 与 Conditional Temporal 测试（2026-08-08）

## 1. 结论

B3 已设为 Warehouse runtime/paper 配置的默认梯度加速路径。新增两个相互独立的
实验开关：

- `spatial.link_broad_phase.enabled`：在 inflated-margin 全时域扫描后，只对入选
  parent link 执行精确 J-FK、SDF 和 `J^Tg`；
- `temporal.conditional_enabled`：粗扫预测 active ratio，只有不超过 35% 时才进入
  K32/K64/K128 时间稀疏，否则回退 B3 dense-time 路径。

7 个场景、4 个变体、3 次重复共完成 **84/84** 次有效 CUDA 推理。两个新开关均未
在 B3 之后获得额外时间收益，因此继续默认关闭：

| 变体 | Guide 平均 (s) | Guide 加速 vs B3 | Inference 加速 | 含 dense 总加速 | Valid 变化 |
|---|---:|---:|---:|---:|---:|
| B3 sparse `J^Tg` | 0.280101 | 1.000x | 1.000x | 1.000x | 0 |
| C1 B3 + broad phase | 0.406875 | 0.690x | 0.725x | 0.810x | 0 |
| C2 B3 + conditional temporal | 0.313158 | 0.892x | 0.906x | 0.940x | 0 |
| C3 B3 + 两者 | 0.446068 | 0.645x | 0.681x | 0.777x | -0.14 pp |

其中加速小于 1 表示负优化。当前生产推荐仍是 **B3 单开**。

## 2. 开关和默认配置

```yaml
gradient_pruning:
  enabled: true
  candidate:
    enabled: false
  preselection:
    parent_bounds_scan: false  # 当前 C2 消融显式改为 true
  temporal:
    enabled: false
    conditional_enabled: false
    conditional_active_ratio_threshold: 0.35
    coarse_points: 32
    probe_midpoints: true
    reuse_selection_within_ddim_step: true
    buckets: [32, 64, 128]
  spatial:
    parent_link_kinematics: true
    dense_parent_fast_path: true
    active_link_pruning: true
    link_broad_phase:
      enabled: false
      full_scan: true
      scan_geometry: fine_spheres  # 本文 2026-08-08 历史测试语义
      environment_margin: 0.20
      self_margin: 0.10
  mapping:
    fused_bspline_integration: false
    sparse_bspline_support: false
```

缺少整个 `gradient_pruning` 配置时仍走 legacy，以保持旧配置兼容；Warehouse runtime
和 paper-sweep 文件已经显式启用上面的 B3 配置。

四个对照变体：

| 变体 | Sparse `J^Tg` | Link broad phase | Conditional temporal |
|---|---:|---:|---:|
| B3 | 开 | 关 | 关 |
| C1 | 开 | 开 | 关 |
| C2 | 开 | 关 | 开，阈值 35% |
| C3 | 开 | 开 | 开，阈值 35% |

## 3. Link broad phase 实现语义

Broad phase 的流程为：

1. 每个 guide DDIM step 的第一次 iteration 对 128 点执行 FK-only/SDF 扫描；同一
   DDIM step 后续 5 次 iteration 复用选择结果。
2. 环境 clearance 小于 0.20 m、自碰撞 pair clearance 小于 0.10 m 的 sphere/pair
   标为候选。
3. 任一 sphere/pair 入选后扩展到完整物理 parent link；同一 parent 的所有碰撞球均
   保留，环境和自碰撞 parent 集合取并集。
4. 按“时间 bucket × parent-mask”分组；每组只为入选 parent 创建/调用 J-FK，并只
   查询对应 sphere SDF、自碰撞 pair，最后用局部索引 scatter 回关节梯度。
5. 最终轨迹始终经过独立 dense=128 环境碰撞、自碰撞和关节约束检查。

这是真正裁剪精确 Cost Guide 中 FK/Jacobian/SDF 的实现，但“保守”有明确边界：
全 128 点扫描和 inflated margin 对当前 DDIM step 的选择是保守候选生成；选择在同一
DDIM step 的多次 guide update 间复用，0.20/0.10 margin 用来吸收轨迹变化，但不是
连续时间、任意更新幅度下的形式化 Lipschitz 证明。最终安全性仍由 dense checker
保证，不能用 broad-phase mask 替代最终校验。

## 4. Conditional temporal 实现语义

Conditional temporal 使用 32 个粗点加相邻中点，共最多 63 个 probe：

1. 先按现有 risk mask、dilation 和 K32/K64/K128 规则计算拟议 bucket；
2. 计算拟议 `sum(K)/(candidates × 128)`；
3. ratio `<= 0.35` 才采用时间分桶；
4. ratio `> 0.35` 时所有候选恢复 K128，并直接调用规则 dense ParentLinkFast/B3；
5. 同一 DDIM step 的 6 次 iteration 仅首次扫描，后续复用 selection。

因此该开关能阻止已知的 80%--100% active 场景误入时间稀疏，但无法消除“先粗扫才能
知道 ratio”的固定成本。

## 5. 统一测试协议

- 100 candidates；
- 15 DDIM steps；
- 最后 20% DDIM step 开 guide，即 3 个 active DDIM steps；
- 每步 6 guide iterations，共 18 guide calls；
- dense validation=128；
- seed=2；每变体 3 次计时重复，报告 p50；
- 第 2 次重复反向运行 C3→B3；
- 相同 checkpoint、场景、start/goal 和最终 dense checker。

场景：Warehouse open、single、narrow-0.20、narrow-0.14，ThreePillars，
drawer-to-shelf 和 to-drawer。

## 6. 逐场景时间与稀疏率

### 6.1 C1：Link broad phase

| 场景 | Guide p50 (s) | 加速 vs B3 | Sphere 保留 | Self-pair 保留 |
|---|---:|---:|---:|---:|
| Warehouse open | 0.516615 | 0.460x | 87.5% | 56.1% |
| Warehouse single | 0.387036 | 0.790x | 99.7% | 98.6% |
| Warehouse narrow-0.20 | 0.376195 | 0.834x | 78.9% | 30.2% |
| Warehouse narrow-0.14 | 0.387721 | 0.805x | 99.7% | 98.6% |
| ThreePillars | 0.365635 | 0.763x | 86.7% | 42.6% |
| Drawer-to-shelf | 0.369835 | 0.694x | 96.8% | 90.9% |
| To-drawer | 0.445091 | 0.574x | 94.4% | 83.8% |

跨场景累计只裁掉 8.1% sphere 和 28.5% self-pair。parent-mask 分组、动态 subset J-FK、
多个小 kernel 和全时域 broad scan 的固定成本超过节省；即使 narrow-0.20 只保留
30.2% self-pair，仍比 B3 慢 19.9%。

### 6.2 C2：Conditional temporal

| 场景 | Conditional 实际结果 | Active time | Guide p50 (s) | 加速 vs B3 |
|---|---|---:|---:|---:|
| Warehouse open | 3/3 DDIM step 开启 | 29.5% | 0.304477 | 0.781x |
| Warehouse narrow-0.20 | 2/3 DDIM step 开启 | 50.5%（含一次 K128 回退） | 0.324760 | 0.966x |
| Warehouse single | 全部回退 | 100% | 0.343587 | 0.890x |
| Warehouse narrow-0.14 | 全部回退 | 100% | 0.342910 | 0.910x |
| ThreePillars | 全部回退 | 100% | 0.299968 | 0.931x |
| Drawer-to-shelf | 全部回退 | 100% | 0.288382 | 0.890x |
| To-drawer | 全部回退 | 100% | 0.288020 | 0.887x |

35% 门控正确避免了拥挤场景进入时间稀疏，但它在所有场景仍为负优化。原因是 B3 已把
稠密 collision projection 大幅加速，粗 FK/SDF scan 的成本不再能靠减少时间点摊平。
即使 open 的 active time 只有 29.5%，C2 Guide 仍慢约 28.1%。

### 6.3 C3：组合

C3 的跨场景 Guide 加速只有 0.645x。`open_clearance` 同时产生多个 temporal bucket
和 parent-mask group，Guide p50 从 B3 的 0.237734 s 上升到 0.758117 s（0.314x）。
说明时间和 link 的稀疏率可以相乘，但动态分组/kernel 数也会相乘；当前 GPU 实现不适合
组合启用。

## 7. 准确度

- C1 在 7 个场景的 valid mask 和 collision mask 全部与 B3 相同。
- C2 在 7 个场景也全部与 B3 相同；35% 门控单独使用未出现离散准确度退化。
- C3 在 Warehouse narrow-0.20 有 1/100 条候选从 valid 变为 collision，valid rate
  从 87% 降至 86%；跨 7 场景平均变化为 -0.14 个百分点。
- C1/C2/C3 的最终轨迹不保证 bitwise 等价。动态分组和稀疏积分改变浮点累加顺序，
  C1/C2/C3 相对 B3 的最大关节采样点差异分别约 0.137/0.146/0.136 rad；离散安全
  结论以最终 dense checker 为准。
- ThreePillars、drawer 的绝对 valid rate 很低是 checkpoint/场景基线问题；三次重复
  使用同一 seed，只用于计时 p50，不是三个独立准确度样本。

## 8. 推荐

1. 默认继续使用 B3：`active_link_pruning=true`，broad phase 和 conditional temporal
   均为 false。
2. C1/C2/C3 保留为研究开关，不接入自动启用逻辑。
3. 若继续优化 broad phase，应先消除“每个 parent-mask 一个动态 dispatch”的结构，
   例如固定 8-parent mask 的单 kernel/块稀疏 kernel；仅调 clearance margin 不会解决
   当前主要开销。
4. 若继续优化 temporal，应让粗扫描与 B3 已有 FK/SDF 数据共享，或把 probe 与第一
   次精确 guide 融合；在当前独立预扫描架构下，35% 阈值仍不足以盈利。
5. 不加入 autoset；任何默认策略变更必须重新进行多 seed/start-goal 准确度测试。

## 9. 结果文件

- 原始结果：`benchmark_results/gradient_pruning_b3_broad_conditional_multiscene_20260808/`
- 场景 CSV：`reports/b3-broad-conditional-scene-summary.csv`
- 跨场景 CSV：`reports/b3-broad-conditional-aggregate-summary.csv`
- 自动表格：`reports/b3-broad-conditional-summary.md`
- 汇总脚本：`scripts/inference/summarize_gradient_pruning_b3_broad_conditional.py`

## 10. 2026-08-09 C1 预扫描实现更新

本文第 3--8 节的 C1 计时使用全部 56 个细碰撞球预扫描；当前 C1 已恢复相同的
`link_broad_phase.scan_geometry: fine_spheres` 语义，因此该历史结果仍是 C1 的主要
正式参考。

C2 新增独立 `preselection.parent_bounds_scan: true`，无需开启 C1/link broad phase
即可用相同多粗球执行 32+midpoint 粗扫；旧 C2 时间同样不代表当前实现。

多粗球基础设施保留给 C2 独立预扫描和研究对照；生成脚本与数据分别为：

- `scripts/generate_parent_collision_bounds.py`
- `mpd/torch_robotics/torch_robotics/data/configs/panda/panda_parent_collision_bounds.yaml`

## 11. 2026-08-09 C1 Pose/SDF 缓存复测

C1 后续新增了真正的 TorchKin Jacobian-only 接口：细球全扫描保存相关链 pose、细球
中心和 SDF distance；首次精确 iteration 只为激活 parent 构造 Jacobian，环境 SDF 只
补 gradient。同一 DDIM step 后续 iteration 因 q 已更新，不复用旧几何。

在相同 100/15/20%/6/dense128 协议、seed0、3 repeats 的 7 场景复测中，B3/C1 Guide
算术平均为 `0.285201/0.454804 s`，C1 仍只有 `0.627x`。说明重复 FK/distance 已消除，
但全细球预扫、高 sphere 保留率和 parent-mask 多 dispatch 仍占主导。默认结论不变：
B3 单开，C1 默认关闭。详见
[`GRADIENT_PRUNING_C1_SCAN_CACHE_RESULTS_20260809.md`](GRADIENT_PRUNING_C1_SCAN_CACHE_RESULTS_20260809.md)。
