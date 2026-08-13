# 默认 Inference：B3 开关说明

`scripts/inference/inference.py` 默认加载
`config_EnvWarehouse-RobotPanda-config_file_v01_00.yaml`。该配置现在默认使用 B3：

- A2P-fast parent-link kinematics；
- Cost Guide 在 `torch.no_grad()` 路径执行；
- 原始、物化的 B-spline gradient mapping；
- 只对非零 task-space collision gradient 执行稀疏 `J^Tg`；
- 不使用 parent 包络体、时间/候选/span 预筛、link broad phase 或额外缓存。

Python 解析器本身也采用同样的缺省值。因此，即使其他 YAML 完全没有
`gradient_pruning` 段，`resolve_gradient_pruning_config()` 仍会选择这套无缓存 B3，
并不是只有上述 Warehouse YAML 才生效。若需要复现旧版完整 legacy 路径，必须显式写：

```yaml
gradient_pruning:
  enabled: false
```

## 必需开关

```yaml
compute_costs_with_xrecon: false

gradient_pruning:
  enabled: true
  force_all_active: false

  endpoint:
    ee_only_last_point: true

  candidate:
    enabled: false

  preselection:
    parent_bounds_scan: false

  span_certificate:
    enabled: false

  temporal:
    enabled: false
    conditional_enabled: false
    reuse_selection_within_ddim_step: false

  spatial:
    parent_link_kinematics: true
    dense_parent_fast_path: true
    active_link_pruning: true
    link_broad_phase:
      enabled: false
      reuse_scan_cache: false
    environment_link_broad_phase: false
    self_link_pair_broad_phase: false

  mapping:
    fused_bspline_integration: false
    sparse_bspline_support: false

  scheduling:
    enabled: false
```

## 各开关含义

| 开关 | B3 值 | 作用 |
|---|---:|---|
| `gradient_pruning.enabled` | `true` | 进入显式梯度优化路径；字段或整个配置段缺失时也默认 `true` |
| `endpoint.ee_only_last_point` | `true` | EE goal 只在终点计算 FK/Jacobian |
| `spatial.parent_link_kinematics` | `true` | 每个物理 parent 只计算一次运动学，再展开 fine spheres |
| `spatial.dense_parent_fast_path` | `true` | 保持规则 `[candidate, 128, dof]` dense-time 布局 |
| `spatial.active_link_pruning` | `true` | 只稀疏碰撞梯度投影 `J^Tg`；不裁剪 FK/Jacobian/SDF |
| `mapping.fused_bspline_integration` | `false` | 使用原映射，不使用融合映射 |
| `mapping.sparse_bspline_support` | `false` | 不使用控制点稀疏映射 |
| `candidate.enabled` | `false` | 不执行候选预筛 |
| `temporal.enabled` / `conditional_enabled` | `false` | 不执行时间粗扫、分桶或 conditional temporal |
| `preselection.parent_bounds_scan` | `false` | 不使用 parent 粗包络体 |
| `span_certificate.enabled` | `false` | 不执行连续 span lower-bound 预扫 |
| `spatial.link_broad_phase.enabled` | `false` | 不在 FK/Jacobian/SDF 前裁剪 parent/link |
| 两个 `reuse_*cache` | `false` | 不保存或复用 selection、pose、sphere center、SDF distance |
| `scheduling.enabled` | `false` | 不启用候选调度实验路径 |

`active_link_pruning: true` 只省略零梯度项的 `Adjoint + J^Tg`，不是包络体或 broad
phase；B3 仍对全部候选、全部 128 个时间点执行 collision J-FK 和 SDF，因此不会引入
Temporal/C1/span certificate 的预扫描成本。

相同 B3 开关也显式写入：

- `config_EnvWarehouse-RobotPanda-runtime.yaml`；
- `config_EnvWarehouse-RobotPanda-paper_sweep.yaml`。

显式写入的意义是让运行配置自解释；它们与 Python 缺省值一致。只有明确设置
`gradient_pruning.enabled: false` 才会回退 legacy。
