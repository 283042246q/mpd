# MPD + World Model Environment Generalization Plan

本文档规划如何在当前 MPD 代码库基础上，逐步加入环境泛化能力。核心目标是把现有的

```text
p(control_points | q_start, ee_goal)
```

扩展为

```text
p(control_points | q_start, ee_goal, environment)
```

并让 collision guidance 可以使用当前环境的 SDF，来源可以是显式几何直接计算，也可以是预训练 world-model SDF。

## 0. 当前代码状态

当前 MPD 训练和推理的关键事实：

- 训练阶段只使用 diffusion denoising loss，不使用 collision cost / SDF guidance。
- 推理阶段才使用 cost guidance，Warehouse DDIM 配置里通常在最后一部分采样步启用。
- 当前 `ContextModel` 只编码 `qs` 和可选 `ee_goal_pose`，没有环境输入。
- 当前 Warehouse 训练集基于固定 `EnvWarehouse`，`EnvWarehouseExtraObjectsV00` 更像 inference/evaluation 时替换环境的固定 extra objects。
- 当前代码中已有 SDF，但它不是 learned SDF。它由显式障碍物 primitive / grid 在环境构建时生成。

相关代码位置：

- Cost guidance: `mpd/inference/cost_guides.py`
- DDIM guide 触发: `mpd/models/diffusion_models/diffusion_model_base.py`
- 当前 SDF grid: `mpd/torch_robotics/torch_robotics/environments/grid_map_sdf.py`
- 环境构建 fixed/extra SDF: `mpd/torch_robotics/torch_robotics/environments/env_base.py`
- Collision field: `mpd/torch_robotics/torch_robotics/torch_planning_objectives/fields/distance_fields.py`
- Planning task 创建 collision fields: `mpd/torch_robotics/torch_robotics/tasks/tasks.py`
- Context model: `mpd/models/diffusion_models/context_models.py`
- B-spline dataset context 字段: `mpd/datasets/trajectories_dataset_bspline.py`
- 训练入口: `scripts/train/train.py`

## 1. 总体实施顺序

推荐按以下顺序推进：

```text
Stage A: 抽象 SDF / collision field 接口
Stage B: 支持当前环境直接计算 SDF 的多环境 baseline
Stage C: 可选预训练 neural SDF / world-model SDF
Stage D: 把 world latent 加入 ContextModel
Stage E: 评估环境难易度，做 curriculum 分阶段训练
```

原因：

- Stage A/B 可以最快得到可靠 baseline，且不改变 MPD prior 训练逻辑。
- Stage C 解决只有观测、点云、参数化随机环境时的环境理解问题。
- Stage D 让 diffusion prior 本身适配环境，而不只依赖最后几步 guidance 修正。
- Stage E 控制多环境数据难度，降低训练不稳定和 hard-case 数据浪费。

## 2. Stage A: 抽象 SDF / Collision Field 接口

### 目标

当前 `CostTaskSpaceCollisionObjects` 通过 `planning_task.get_collision_objects_field()` 获得 collision field，然后调用：

```python
cost, grad_cost_wrt_x = collision_objects_field.compute_distance_field_cost_and_gradient(x_positions)
```

因此最小改动是保持这个接口不变，新增不同来源的 collision field：

```text
ExactGeometryCollisionField
  显式 object primitives / GridMapSDF

LearnedSDFCollisionField
  world latent + query x -> sdf, grad sdf
```

### 建议实现

保留原有 `CollisionObjectDistanceField` 作为 exact geometry baseline。新增一个 learned 版本，例如：

```text
mpd/torch_robotics/torch_robotics/torch_planning_objectives/fields/learned_sdf_fields.py
```

接口：

```python
class LearnedSDFCollisionField:
    def __init__(self, sdf_model, world_context_provider, robot, link_margins, cutoff_margin):
        ...

    def compute_distance_field_cost_and_gradient(self, link_pos, **kwargs):
        # link_pos: [B, H, n_links, 3]
        # returns:
        #   cost: [B, H, n_links]
        #   grad_cost_wrt_x: [B, H, n_links, 3]
```

collision cost 保持和当前代码一致：

```text
margin = link_radius + cutoff_margin
cost = relu(margin - sdf(x))
grad cost wrt x = - grad sdf(x) where cost > 0
```

这样 `CostTaskSpaceCollisionObjects` 基本不用重写。

## 3. Stage B: 当前环境直接计算 SDF

### 适用场景

当每个 Warehouse 环境都能拿到完整 obstacle 参数，例如 box/rack/extra object 的位置、尺寸、朝向，应优先使用直接计算 SDF：

```text
env_params -> object primitives -> GridMapSDF / analytic SDF -> collision guidance
```

### 优点

- 几何精确，不需要训练 SDF。
- 梯度可靠，适合 MPD 最后几步 cost guidance。
- 适合作为所有 learned SDF 的强 baseline。

### 需要做的代码改动

1. 扩展 Warehouse 环境生成器

   当前 `EnvWarehouse` 偏固定环境。需要支持随机化：

   ```text
   env_seed
   obstacle_count
   obstacle_size_range
   obstacle_pose_range
   min_clearance_to_start_goal
   narrow_passage_width
   clutter_level
   ```

   每个环境生成后保存：

   ```text
   env_id
   env_params
   obstacle_params
   difficulty_metrics
   ```

2. 数据生成阶段保存环境参数

   当前 dataset 主要保存 trajectory/control points/context qs/ee pose。多环境训练需要在 HDF5 中额外保存：

   ```text
   env_id: int
   obstacle_params: [N, object_dim]
   obstacle_mask: [N]
   difficulty_score: float
   difficulty_level: int
   ```

3. 推理时根据 env 参数构造当前环境 SDF

   对于 exact geometry：

   ```text
   env_params -> EnvWarehouseRandomized -> EnvBase.build_sdf_grid()
   ```

   然后复用现有 `CollisionObjectDistanceField`。

### 注意点

- 如果环境很多，不建议每个 sample 都重新构建高分辨率 3D grid SDF。可以按 `env_id` 缓存。
- 对 Panda 3D Warehouse，SDF grid 分辨率需要在速度和精度之间折中。
- 如果 obstacle 是 box/sphere 等简单 primitive，也可以先用 analytic signed distance，避免频繁构建 grid。

## 4. Stage C: 可选预训练 SDF / World-Model SDF

### 目标

训练一个模型：

```text
E_world(scene_obs or env_params) -> z_world
F_sdf(z_world, x) -> sdf(x)
```

其中：

```text
scene_obs 可以是 obstacle params / occupancy grid / SDF grid / point cloud
x 是 workspace query point, shape [B, M, 3]
sdf(x) 是 query point 到障碍物表面的 signed distance
```

### 为什么有效

专家轨迹很贵：

```text
new env + start/goal -> RRT/OMPL solve -> trajectory label
```

SDF 标签便宜：

```text
random env -> random workspace points -> analytic SDF label
```

因此预训练 SDF 可以把一部分数据需求从昂贵 trajectory label 转移到便宜 geometry label。

### 输入形式选择

优先级建议：

```text
1. obstacle params encoder
2. SDF / occupancy grid encoder
3. point cloud encoder
4. RGB-D / image encoder
```

当前 Warehouse 最适合从 obstacle params 做起。

### 模型结构建议

#### obstacle params encoder

```text
object_i = [type_onehot, center_xyz, size_xyz, yaw_or_quat]
object_encoder: MLP(object_i) -> e_i
pooling: masked mean/max(e_i) -> z_world
```

#### SDF decoder

```text
input = concat(z_world, positional_encoding(x))
decoder MLP -> sdf_pred
```

### 训练数据

对每个随机环境采样 query points：

```text
uniform points: 覆盖全 workspace
near-surface points: 障碍物表面附近，训练 collision boundary
trajectory-near points: 机器人 link 常出现区域
hard-negative points: narrow passage / low clearance 区域
```

标签：

```text
sdf_gt = exact_geometry_sdf(x)
occupancy_gt = sdf_gt <= margin
```

损失：

```text
L_sdf = smooth_l1(clamp(sdf_pred), clamp(sdf_gt))
L_occ = BCE(sigmoid(-sdf_pred / tau), occupancy_gt)
L_eikonal = (||grad_x sdf_pred|| - 1)^2   # 可选
```

第一版可以只做：

```text
L = L_sdf + lambda_occ * L_occ
```

### 推理接入

learned SDF 接入后，inference guidance 变成：

```text
MPD sample control points
-> dense q trajectory
-> FK 得到 link positions x
-> F_sdf(z_world, x) 得到 sdf 和 grad sdf
-> collision cost gradient
-> guide 更新 control points
```

### 使用方式

可以有三种模式：

```text
sdf_mode: exact
  使用当前显式几何 / GridMapSDF

sdf_mode: learned
  只使用 neural SDF

sdf_mode: hybrid
  已知几何用 exact；未知/部分观测区域用 learned；或两者取 conservative min
```

hybrid 可先定义为：

```text
sdf = min(sdf_exact_known, sdf_learned)
```

这会更保守，降低 learned SDF 漏障碍物的风险。

## 5. Stage D: 把 World Latent 加入 ContextModel

### 目标

让 diffusion prior 在生成轨迹时已经感知环境，而不是完全依赖 inference guide 修正：

```text
p(control_points | q_start, ee_goal, z_world)
```

### World latent 怎么算

第一版建议用 obstacle params：

```text
obstacle_params: [B, N, object_dim]
obstacle_mask: [B, N]

object_encoder: MLP(object_dim -> hidden_dim)
pooling: masked mean/max
world_mlp: hidden_dim -> world_latent_dim

z_world: [B, world_latent_dim]
```

如果已经训练了 world-model SDF，则优先复用它的 encoder：

```text
z_world = E_world(env_params)
sdf = F_sdf(z_world, x)
context_emb = ContextModel(qs, ee_goal, z_world)
```

这样 latent 同时服务：

- MPD prior 环境条件化。
- learned SDF guidance。

### 代码改动

1. 新增 `ContextModelWorld`

   文件：`mpd/models/diffusion_models/context_models.py`

   形式：

   ```python
   class ContextModelWorld(nn.Module):
       def __init__(self, object_dim, out_dim=128, hidden_dim=128):
           ...

       def forward(self, obstacle_params_normalized=None, obstacle_mask=None, **kwargs):
           ...
           return z_world
   ```

2. 扩展 `ContextModelCombined`

   当前只融合：

   ```text
   context_model_qs
   context_model_ee_pose_goal
   ```

   增加：

   ```text
   context_model_world
   ```

   最终：

   ```text
   context_emb = MLP(concat(z_qs, z_ee, z_world))
   ```

3. 扩展 dataset

   在 `TrajectoriesDatasetBspline` 里增加 context 字段：

   ```text
   field_key_context_obstacle_params = "obstacle_params"
   field_key_context_obstacle_mask = "obstacle_mask"
   ```

   并在 `get_context()` 中返回：

   ```text
   obstacle_params
   obstacle_params_normalized
   obstacle_mask
   ```

4. 扩展训练入口

   在 `scripts/train/train.py` 增加参数：

   ```text
   context_world: bool = False
   context_world_out_dim: int = 128
   context_world_n_layers: int = 2
   world_context_type: "obstacle_params" | "sdf_encoder" | "pretrained_sdf_encoder"
   pretrained_world_model_path: Optional[str]
   ```

5. checkpoint 热启动

   当前训练脚本没有完整 resume 逻辑。建议加：

   ```text
   pretrained_mpd_path
   strict_load: bool = False
   ```

   当新增 world context 时：

   ```text
   load old UNet weights
   load old qs / ee context weights
   skip new world encoder and fusion layers
   ```

### 训练策略

建议先冻结或小学习率训练旧 MPD：

```text
old UNet/context: lr = 1e-5 ~ 5e-5
new world encoder/fusion: lr = 1e-4 ~ 3e-4
```

待 validation 稳定后再全量 fine-tune。

## 6. Stage E: 环境难易度评估与分阶段训练

### 为什么需要难易度

多环境训练会显著增加数据分布复杂度。如果直接混合大量 hard env：

- RRT/OMPL 生成数据成本高。
- diffusion prior 容易学到高方差、多模态、不稳定分布。
- hard case 数量不足时可能过拟合。
- easy case 可能被遗忘。

因此建议做 curriculum，但必须混入 replay，避免纯顺序训练导致遗忘。

### 难度指标

每个 `(env, start, goal)` 生成时记录：

```text
straight_line_collision_ratio
min_clearance_along_expert_path
planner_solve_time
planner_retry_count
path_length_ratio = path_length / distance(start, goal)
goal_region_clutter
narrow_passage_width
num_obstacles_near_robot_reachable_space
```

推荐难度分数：

```text
difficulty =
  w1 * straight_line_collision_ratio
+ w2 * normalized_planner_solve_time
+ w3 * normalized_path_length_ratio
+ w4 * normalized_inverse_min_clearance
+ w5 * normalized_retry_count
+ w6 * goal_region_clutter
```

第一版可以简化为：

```text
difficulty =
  0.35 * straight_line_collision_ratio
+ 0.25 * solve_time_norm
+ 0.20 * path_length_ratio_norm
+ 0.20 * inverse_min_clearance_norm
```

### 难度等级

按分位数或人工阈值分桶：

```text
L0: 原始 EnvWarehouse / 无随机 extra objects
L1: 少量障碍，直线路径多数无碰撞
L2: 中等 clutter，直线路径部分碰撞
L3: 目标附近或主通道附近有障碍，需要明显绕行
L4: narrow passage / dense clutter / RRT solve time 高 / clearance 极低
```

### 分阶段数据混合

不要使用：

```text
stage 1: easy only
stage 2: medium only
stage 3: hard only
```

推荐：

```text
stage 0:
  100% L0/L1

stage 1:
  60% L0/L1 + 40% L2

stage 2:
  30% L0/L1 + 40% L2 + 30% L3

stage 3:
  20% L0/L1 + 30% L2 + 30% L3 + 20% L4

stage 4 fine-tune:
  15% L0/L1 + 25% L2 + 30% L3 + 30% L4
```

### 热启动训练

每个阶段使用上一阶段 EMA checkpoint 初始化：

```text
stage_k/checkpoints/ema_model_current.pth
-> stage_k+1 初始化
```

如果模型结构不变，直接加载完整 state dict。若新增 world latent 模块，则 partial load：

```text
UNet: load
qs context: load
ee goal context: load
world encoder: random init or pretrained init
combined fusion layer: random init if input dim changed
```

### 学习率建议

当前默认 `lr=3e-4`，训练器默认没有打开 scheduler。建议：

```text
from scratch:
  lr = 3e-4
  steps = 500k ~ 1M

warm start easy -> medium:
  lr = 1e-4
  steps = 200k ~ 500k

medium -> hard:
  lr = 5e-5 ~ 1e-4
  steps = 200k ~ 500k

hard fine-tune:
  lr = 1e-5 ~ 5e-5
  steps = 100k ~ 300k
```

加入 world latent 后：

```text
old MPD modules:
  lr = 1e-5 ~ 5e-5

new world encoder / fusion:
  lr = 1e-4 ~ 3e-4
```

建议同时启用：

```text
use_ema = True
clip_grad = True
clip_grad_max_norm = 1.0
```

## 7. 评估指标

每个难度等级都单独评估，不只看整体平均：

```text
success_rate
collision_free_rate
dense_collision_rate
mean_min_clearance
path_length
path_length_ratio
smoothness / acceleration cost
guide_steps_needed
inference_time
failure_by_level
```

还需要分开比较：

```text
MPD prior only
MPD + exact SDF guidance
MPD + learned SDF guidance
MPD + world latent
MPD + world latent + learned/exact SDF guidance
```

关键 ablation：

```text
without world latent
with obstacle-param latent
with pretrained SDF latent
exact SDF guide
learned SDF guide
curriculum vs mixed-from-scratch
```

## 8. 推荐最小可行版本

第一阶段不要直接做完整 world model。建议 MVP：

```text
1. 做 randomized Warehouse env generator
2. 数据集中保存 obstacle_params / obstacle_mask / difficulty_score
3. 使用 exact geometry SDF 做 inference guidance
4. 加入 ContextModelWorld(obstacle_params)
5. 做 L0-L3 curriculum + replay
6. 对比 no-world-latent vs world-latent
```

此时还不需要预训练 neural SDF。

如果 world latent 已经能明显提升：

```text
prior collision rate 降低
guide 修改幅度变小
hard env success rate 提升
```

再进入 neural SDF：

```text
7. 预训练 E_world + F_sdf
8. 用 F_sdf 替换/增强 collision guidance
9. 复用 E_world 的 z_world 加入 ContextModel
```

## 9. 风险与处理

### Learned SDF 梯度错误

风险：

```text
guide 把轨迹推向错误方向
```

处理：

```text
先 exact SDF baseline
learned SDF 使用 conservative margin
hybrid 模式下 sdf = min(exact_known, learned)
对 learned SDF 的 grad 做 clip
```

### World latent 被模型忽略

风险：

```text
模型仍主要依赖 q_start / ee_goal，环境条件不起作用
```

处理：

```text
训练中使用多环境同 start/goal 对照样本
增加 hard env 占比
评估 no-world-latent ablation
在 ContextModelCombined 中确保 z_world 输出维度足够，例如 128
```

### Curriculum 导致遗忘

风险：

```text
hard fine-tune 后 easy env 性能下降
```

处理：

```text
每阶段混入旧难度 replay
每个 level 独立 validation
保留 best checkpoint by weighted success rate
```

### 数据量膨胀

风险：

```text
环境维度增加后 trajectory dataset 需求大幅增加
```

处理：

```text
先用 exact/learned SDF guidance 降低对 trajectory prior 的要求
SDF 预训练使用便宜 geometry labels
active hard-case mining，只对高价值 env-task 调 OMPL/RRT
```

## 10. 建议开发里程碑

```text
Milestone 1:
  exact SDF multi-env baseline
  输出: 多环境推理可跑，collision guidance 使用当前 env SDF

Milestone 2:
  dataset 增加 env_params / obstacle_params / difficulty
  输出: 训练数据可按 difficulty 分桶

Milestone 3:
  ContextModelWorld(obstacle_params)
  输出: MPD prior 支持 environment-conditioned generation

Milestone 4:
  curriculum trainer / sampler
  输出: 分阶段训练和跨难度 validation

Milestone 5:
  neural SDF pretraining
  输出: E_world + F_sdf 可预测 SDF

Milestone 6:
  learned SDF guidance
  输出: MPD inference 可在 learned SDF 下避障

Milestone 7:
  hybrid exact + learned SDF
  输出: 对已知/未知环境均有保守 collision guidance
```

推荐先完成 Milestone 1-4，再决定是否投入 Milestone 5-7。
