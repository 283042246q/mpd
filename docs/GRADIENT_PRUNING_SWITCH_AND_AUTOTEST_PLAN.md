# MPD 梯度剪枝总开关与自动场景测试方案

本文档补充 `GRADIENT_PRUNING_IMPLEMENTATION_PLAN.md`，明确梯度剪枝功能的生效边界，以及如何在测试时确定性生成简单、一般和狭窄环境，自动完成 baseline/pruning 配对测试。

## 1. 结论

所有会改变 guidance 计算路径或数值结果的改动，只能在：

```yaml
gradient_pruning:
  enabled: true
```

时生效。

当该字段为 `false`，或旧配置中完全没有 `gradient_pruning` 字段时，必须走当前未剪枝的 legacy 路径。

独立 dense checker 不参与梯度计算，可以由单独的 `dense_validation.enabled` 控制。正式 A/B 中 baseline 和 pruning 必须使用相同的 dense checker。

测试环境可以自动生成，但应分成：

1. 2D 算法级环境，用于确定性验证风险选择器不会漏掉粗采样点之间的碰撞；
2. Warehouse/Panda 端到端环境，用同一个 checkpoint 验证真实 FK/Jacobian 加速、有效率和困难场景覆盖率。

## 2. 总开关的严格语义

### 2.1 Legacy 路径

主入口应先解析兼容旧 YAML 的配置：

```python
pruning_cfg = args_inference.get("gradient_pruning", {})
pruning_enabled = bool(pruning_cfg.get("enabled", False))
```

不能因为配置中存在 `temporal`、`spatial` 或 `scheduling` 子字段就隐式开启。

`CostComposite.__call__` 或等价上层入口应明确分流：

```python
if not self.gradient_pruning_enabled:
    return self._legacy_call(
        control_points_normalized,
        cost_weight_overrides=cost_weight_overrides,
        **kwargs,
    )

return self._pruned_call(
    control_points_normalized,
    cost_weight_overrides=cost_weight_overrides,
    **kwargs,
)
```

当 `enabled: false` 时：

- 使用完整 128 点轨迹；
- 对全部时间点计算碰撞球 FK/Jacobian；
- 对全部时间点计算 EE FK/Jacobian；
- 不创建 temporal risk selector；
- 不执行 active-time gather/scatter；
- 不执行 32/64/128 Jacobian 分桶；
- 不启用 parent-link kinematics；
- 不启用 environment link broad phase；
- 不启用 self-collision pair broad phase；
- 不启用候选级 guidance 跳过或提前停止；
- 不启用 dense correction fallback；
- `force_all_active` 和所有剪枝子配置均被忽略。

为了得到严格旧基线，计划中的 EE endpoint-only 优化也放在 `_pruned_call` 中。以后若确认它数值完全等价，再考虑将它提升为两条路径共享的通用优化。

### 2.2 Pruning 路径

建议完整配置：

```yaml
gradient_pruning:
  enabled: true
  force_all_active: false
  profile: false
  record_active_statistics: true

  endpoint:
    ee_only_last_point: true

  temporal:
    enabled: true
    coarse_points: 32
    buckets: [32, 64, 128]
    environment_refine_margin: 0.08
    self_refine_margin: 0.06
    neighbor_dilation: 2
    always_keep_endpoints: true

  spatial:
    parent_link_kinematics: false
    environment_link_broad_phase: false
    self_link_pair_broad_phase: false

  scheduling:
    enabled: false
    skip_safe_candidates: false
    promote_on_stalled_cost: true
```

所有子功能必须同时满足：

```text
gradient_pruning.enabled == true
AND
对应子功能 enabled == true
```

### 2.3 Dense validation 独立开关

建议从剪枝配置中拆出：

```yaml
dense_validation:
  enabled: true
  runtime_points: 256
  benchmark_points: 512
  check_environment: true
  check_self_collision: true
  check_joint_limits: true
  reject_invalid: true
```

原因：

- baseline 和 pruning 必须由同一个安全 oracle 评价；
- dense checker 不产生 guidance gradient；
- 可以单独测量 checker 开销；
- 即使关闭剪枝，也能发现原有 128 点检查遗漏的短时碰撞。

它可能改变最终“接受/拒绝”结果，但不能改变 diffusion 采样、guidance 梯度和生成候选本身。

## 3. 自动测试总体流程

测试脚本建议为：

```text
scripts/inference/benchmark_gradient_pruning.py
```

调用流程：

```text
读取基础 inference YAML
        ↓
读取 scenario manifest
        ↓
根据 seed 生成障碍物和 start/goal
        ↓
在 /tmp 创建 derived baseline YAML
        ↓
在 /tmp 创建 derived pruning YAML
        ↓
两组使用相同 checkpoint、seed、start/goal 和 dense checker
        ↓
分别执行 inference
        ↓
生成 paired report、CSV 和失败轨迹工件
```

临时配置建议写入：

```text
/tmp/mpd-gradient-pruning-<run-id>/
├── configs/
├── scenarios/
├── baseline/
├── pruning/
└── reports/
```

测试结束后默认保留失败 case；成功 case 可以由命令行决定是否保留。

## 4. 场景 Manifest

建议维护少量版本化 manifest，而不是大量完整 inference YAML：

```text
scripts/inference/cfgs/gradient_pruning_scenarios/
├── simple_2d.yaml
├── narrow_2d.yaml
└── warehouse_panda.yaml
```

示例：

```yaml
suite_name: warehouse_panda_gradient_pruning
seed: 0

scenarios:
  - id: open_clearance
    type: warehouse_extra_boxes
    difficulty: simple
    boxes: []

  - id: single_obstacle
    type: warehouse_extra_boxes
    difficulty: medium
    boxes:
      - center: [0.50, 0.14, 0.14]
        size: [0.05, 0.28, 0.28]

  - id: narrow_020
    type: warehouse_narrow_gap
    difficulty: hard
    gap_width: 0.20
    wall_depth: 0.12
    wall_height: 0.70

  - id: narrow_014
    type: warehouse_narrow_gap
    difficulty: hard
    gap_width: 0.14
    wall_depth: 0.12
    wall_height: 0.70
```

manifest 必须保存：

```text
scenario id
scenario seed
生成后的精确障碍物 center/size/orientation
start joint position
EE goal pose 或 goal joint position
checkpoint id
dense checker points
```

这样随机生成的失败场景可以完全复现。

## 5. 2D 算法级自动环境

仓库已经有：

```text
EnvSimple2D
EnvSimple2DExtraObjectsV00
EnvNarrowPassageDense2D
EnvNarrowPassageDense2DExtraObjectsV00
```

它们适合验证时间风险选择器和非均匀积分，不适合替代 Panda 端到端性能测试。

### 5.1 必须生成的 2D 场景

#### A. Open clearance

```text
start → goal
路径与障碍物保持大间隙
```

预期：

- 大多数轨迹进入 32 点 bucket；
- collision Jacobian 调用最少；
- baseline/pruning 均安全；
- 获得接近最大加速。

#### B. Single obvious obstacle

障碍物正好覆盖直线路径，并能被 coarse 点直接观察。

预期：

- 危险区间被激活；
- 邻域点被扩张；
- guidance 能将路径推出碰撞。

#### C. Between-coarse-points collision

将小障碍物放在两个 32-point coarse samples 之间，使区间端点均不碰撞。

预期：

- 仅看端点的实现必须在此测试失败；
- 加入运动幅度、SDF 变化或递归中点后必须识别；
- 512 点 checker 必须检出任何 selector false negative。

#### D. Narrow gap sweep

自动生成：

```text
gap width = [0.30, 0.20, 0.15, 0.10, 0.075]
```

预期：

- gap 越窄，64/128 bucket 占比越高；
- 不能为了保持 32 点比例而漏碰撞；
- 不可行 gap 应稳定返回失败，而不是输出碰撞轨迹。

#### E. Near-margin, no collision

轨迹靠近障碍物但保持正 clearance。

预期：

- refine margin 能提前激活危险区间；
- dense checker 判定安全；
- 不出现由阈值震荡造成的大量 bucket 切换。

### 5.2 2D 测试输出

```text
selector recall
collision waypoint recall
false-negative trajectories
average/p95 selected points
bucket occupancy
minimum clearance
gradient cosine similarity against full gradient
cost relative error
```

## 6. Warehouse/Panda 参数化测试环境

### 6.1 为什么需要新工厂

当前 `EnvWarehouseExtraObjectsV00` 的 extra box 是硬编码的。要自动扫描通道宽度，需要新增测试用参数化工厂，而不是为每个宽度复制一个环境类。

建议新增：

```text
mpd/torch_robotics/torch_robotics/environments/env_gradient_pruning_test.py
```

包含：

```python
class EnvWarehouseGradientPruningTest(EnvWarehouse):
    def __init__(
        self,
        scenario,
        scenario_seed=0,
        tensor_args=DEFAULT_TENSOR_ARGS,
        **kwargs,
    ):
        extra_objects = build_warehouse_test_objects(
            scenario=scenario,
            seed=scenario_seed,
            tensor_args=tensor_args,
        )
        super().__init__(
            obj_extra_list=extra_objects,
            tensor_args=tensor_args,
            **kwargs,
        )
```

或者让 loader 接受：

```yaml
test_environment:
  factory: warehouse_gradient_pruning
  scenario_manifest: /absolute/path/to/generated-scenario.yaml
```

不建议用 `eval` 解析任意 Python 表达式。

### 6.2 Warehouse/Panda 场景

#### A. Simple/open

- 使用原始 Warehouse，不添加额外 box；
- 选择已有高 valid-rate start/goal；
- 用于检查安全轨迹是否大部分进入 32 点 bucket。

#### B. Single extra box

- 复用当前 `EnvWarehouseExtraObjectsV00` 的 box 参数；
- 用于与现有日志直接比较；
- baseline/pruning 使用相同 validation sample indices。

#### C. Narrow passage

用两个或多个 box 形成参数化通道：

```text
gap_width = [0.30, 0.24, 0.20, 0.18, 0.16, 0.14]
```

必须先用 dense geometric checker 检查 start/goal 本身不碰撞，并确认场景不是显然不可行。

Panda 的 task-space gap 不等于 configuration-space gap，因此不能只根据 box 间距标记难度。实际难度还要记录：

```text
baseline success
minimum valid clearance
RRTConnect feasibility
MPD valid candidates / 100
```

#### D. Hidden short collision

障碍物只影响轨迹中的短时间区间，专门测试 32 点 coarse sampling 是否漏检。

该场景最好从基线轨迹反向生成：

1. 运行无 extra obstacle 的 baseline；
2. 选取轨迹中两个 coarse samples 之间的中间 EE/link 位置；
3. 在该位置附近放置小 box；
4. 保证两个 coarse 端点仍有正 clearance；
5. 保存生成后的固定 obstacle 参数。

#### E. Self-collision stress

不通过环境 box 构造，而是保存一组容易产生手臂回折的 start/goal contexts。

预期：

- self-collision active pair recall 达标；
- link-pair broad phase 无 false negative；
- hand 与近端 link 的检查不被剪掉。

## 7. 配对 A/B 生成规则

每个 scenario 自动生成两个 derived config。

Baseline：

```yaml
gradient_pruning:
  enabled: false

dense_validation:
  enabled: true
  benchmark_points: 512
```

Pruning：

```yaml
gradient_pruning:
  enabled: true
  force_all_active: false
  temporal:
    enabled: true
    coarse_points: 32
    buckets: [32, 64, 128]

dense_validation:
  enabled: true
  benchmark_points: 512
```

必须完全相同的字段：

```text
model_dir/checkpoint
planner_alg
DDIM parameters
cost weights
n_trajectory_samples
random seed
start/goal
environment geometry
trajectory duration
dense checker settings
best-trajectory metric
```

唯一允许变化的是 `gradient_pruning` 及其子字段。

## 8. 自动测试层次

### 8.1 CPU 快速测试

不加载 diffusion checkpoint：

```text
test generated geometry is deterministic
test gap width matches manifest
test start/goal are collision-free
test between-coarse collision is present
test selector recall
test nonuniform weights
test enabled=false uses legacy dispatcher
test missing config defaults to false
```

建议测试：

```text
tests/test_gradient_pruning_config.py
tests/test_gradient_pruning_scenario_factory.py
tests/test_collision_risk_selector.py
```

### 8.2 GPU Gradient 等价测试

使用小 batch：

```yaml
gradient_pruning:
  enabled: true
  force_all_active: true
```

比较：

```text
legacy total cost
pruned-path total cost
environment gradient
self-collision gradient
control-point gradient
DDIM updated sample
```

### 8.3 GPU Smoke A/B

每个难度运行 1 个 context、10～20 条候选：

```text
simple
single obstacle
narrow
self-collision
```

用于发现接口和显存问题，不作为最终性能结论。

### 8.4 完整 Benchmark

每个场景至少：

```text
15 contexts
100 candidates/context
3 repeated timing runs
512-point dense validation
```

输出：

```text
paired-results.csv
timing-breakdown.csv
active-statistics.csv
failed-contexts.yaml
summary.md
```

## 9. 自动断言

测试脚本应支持：

```text
--fail-on-regression
```

建议断言：

```text
enabled=false 与 legacy 结果一致
missing gradient_pruning config 等价于 enabled=false
force_all_active 梯度误差在容差内
512-point output false-negative rate == 0
context coverage 不下降
mean valid rate 下降 <= 0.01
EE position error 增加 <= 0.002 m
EE orientation error 增加 <= 0.2 deg
guidance p50 speedup >= 1.8
total p50 speedup >= 1.5
```

阶段开发期间可对速度断言使用较低阈值，但安全断言不能放宽。

## 10. 建议命令接口

生成场景但不运行：

```bash
python scripts/inference/benchmark_gradient_pruning.py \
  --suite warehouse_panda \
  --generate-only \
  --seed 0 \
  --output-dir /tmp/mpd-gradient-pruning
```

快速测试：

```bash
python scripts/inference/benchmark_gradient_pruning.py \
  --suite smoke \
  --device cuda:0 \
  --contexts 1 \
  --candidates 20
```

完整配对测试：

```bash
python scripts/inference/benchmark_gradient_pruning.py \
  --suite warehouse_panda \
  --device cuda:0 \
  --contexts 15 \
  --candidates 100 \
  --repeats 3 \
  --dense-points 512 \
  --fail-on-regression
```

重放自动保存的失败场景：

```bash
python scripts/inference/benchmark_gradient_pruning.py \
  --replay-scenario /path/to/failed-scenario.yaml \
  --device cuda:0
```

## 11. 实施顺序

自动测试支持应随主方案逐步实现：

```text
1. 配置默认值和 enabled=false legacy dispatcher
2. scenario manifest schema
3. 2D deterministic scenario factory
4. dense checker
5. force_all_active gradient A/B
6. Warehouse/Panda parameterized extra-box factory
7. paired config generator
8. smoke A/B runner
9. full benchmark and regression assertions
10. failure artifact replay
```

在 `enabled=false` 等价测试、dense checker 和自动失败重放尚未完成前，不应在 runtime 配置中默认开启梯度剪枝。
