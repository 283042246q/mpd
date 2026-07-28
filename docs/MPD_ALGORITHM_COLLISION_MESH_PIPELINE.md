# MPD 算法测试、碰撞模型与 Mesh 仿真导入实施方案

本文只讨论当前 `mpd-splines-public` 仓库，不涉及 `physical_ai_runtime`。

目标分为三个任务：

1. 建立可重复的 MPD 算法测试流程；
2. 明确机器人、环境和抓取物碰撞模型的实现与验证方法；
3. 将新的 mesh 资产通过本仓库 pipeline 同步到 MPD、PyBullet/OMPL 和 IsaacLab 仿真。

## 1. 当前能力和关键结论

### 1.1 当前已经实现

- Panda 的 URDF、FK、关节限制和 collision-sphere 模型；
- Warehouse 的 sphere/box primitive 环境；
- 基于 Torch SDF 的碰撞 cost 和最终轨迹有效性过滤；
- PyBullet/OMPL 数据生成中的 sphere/box 障碍物；
- IsaacLab evaluator/replay 中的 Panda 轨迹执行；
- MPD scene payload 到 IsaacLab 的 sphere/box 障碍物导出；
- B-spline 边界条件、抓取物碰撞索引、IsaacLab 子进程清理单元测试；
- inference report 和 `infer_once.py` 的中性结果输出。

### 1.2 当前尚未实现

- 通用环境 `MeshField` 的可微 signed-distance 实现；
- mesh 障碍物从 `EnvBase` 自动导出到 scene payload；
- mesh 障碍物在 PyBullet/OMPL 数据生成中自动创建；
- scene payload 中的 `type: mesh`；
- IsaacLab evaluator/replay 对任意 mesh/USD 障碍物的 spawn；
- MPD 机器人 mesh 修改后自动同步到 IsaacLab 的 Panda asset；
- 抓取 mesh 在 MPD 和 IsaacLab 之间的统一附着与碰撞表示。

因此，当前最稳妥的第一版不是直接让 MPD 对三角 mesh 求梯度，而是：

```text
visual mesh
    +
sphere/box collision proxy
    ↓
MPD Torch collision
PyBullet/OMPL collision
IsaacLab primitive collision
```

视觉 mesh 和碰撞代理必须是同一个版本化资产的两个表示。

## 2. 当前 pipeline 总览

```text
Robot / Environment source
        │
        ├─ Robot URDF + collision sphere YAML
        │      ↓
        │   RobotBase / RobotPanda
        │      ↓
        │   Torchkin FK + CollisionSelfField
        │
        ├─ EnvWarehouse ObjectField
        │      ├─ MultiSphereField
        │      └─ MultiBoxField
        │             ↓
        │          GridMapSDF
        │             ↓
        │   CollisionObjectDistanceField
        │
        ├─ GenerateDataOMPL
        │      └─ sphere/box → PyBullet body
        │
        └─ export_isaaclab_scene_payload
               └─ sphere/box JSON payload
                      ↓
            evaluate/replay _spawn_scene_obstacles
                      ↓
                 IsaacLab scene
```

需要特别注意：

- URDF 的 collision mesh 不等于 MPD 最终使用的碰撞模型；
- MPD 机器人碰撞主要依赖 collision-sphere YAML；
- MPD 环境碰撞主要依赖 `ObjectField → GridMapSDF`；
- IsaacLab 当前机器人来自 `FRANKA_PANDA_HIGH_PD_CFG`，不是直接加载 MPD 的 Panda URDF；
- 修改 MPD URDF 不会自动修改 IsaacLab Panda；
- MPD 最终候选过滤使用 Torch collision fields，不会让 PyBullet 逐条复核所有最终轨迹。

## 3. 任务一：算法测试

算法测试应分为五层，不能只看一次 inference 是否成功。

### 3.1 L0：静态和配置测试

目的：在加载模型或 GPU 前发现语法、YAML 和接口错误。

建议命令：

```bash
/home/eric/anaconda3/envs/mpd-splines-public-cu128/bin/python \
  -m compileall -q mpd scripts tests
```

Runtime YAML 检查：

```bash
/home/eric/anaconda3/envs/mpd-splines-public-cu128/bin/python -c "
from dotmap import DotMap
from mpd.utils.loaders import load_params_from_yaml
from scripts.runtime.infer_once import _validate_runtime_config
cfg = DotMap(load_params_from_yaml(
    'scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-runtime.yaml'
))
_validate_runtime_config(cfg)
print('runtime_config_ok')
"
```

通过条件：

- Python 编译成功；
- YAML 可加载；
- runtime 固定的机器人、场景、轨迹长度和候选数量未被意外修改。

### 3.2 L1：CPU 单元测试

当前仓库不要求 pytest，已有测试可以直接通过 `unittest` 运行：

```bash
CUDA_VISIBLE_DEVICES='' \
/home/eric/anaconda3/envs/mpd-splines-public-cu128/bin/python \
  -m unittest discover -s tests -p 'test_*.py' -v
```

当前覆盖：

| 测试 | 主要验证内容 |
|---|---|
| `test_bspline_boundary_conditions.py` | 非零/零速度加速度边界、EE context 终点行为 |
| `test_grasped_object_collision_pairs.py` | Panda sphere 顺序、抓取物 sphere 边界、object-link/self-collision pair |
| `test_isaaclab_subprocess_utils.py` | IsaacLab launcher 及子进程组能被完整终止 |

本次核对时该命令运行 5 个测试并全部通过。

新增算法功能时，先在 CPU 层增加小规模确定性测试，不应直接从 GPU inference 开始定位错误。

### 3.3 L2：单请求 inference smoke test

使用 `scripts/runtime/infer_once.py`，测试完整模型加载、FK、cost guide、有效性过滤、最佳轨迹选择和中性导出。

```bash
/home/eric/anaconda3/envs/mpd-splines-public-cu128/bin/python \
  scripts/runtime/infer_once.py \
  --request /absolute/path/to/request.json \
  --output-dir /tmp/mpd-runtime-smoke \
  --device cuda:0
```

验收 `result.json`：

- `status == "success"`；
- `request_id` 与输入一致；
- `candidates.generated == 100`；
- `candidates.valid > 0`；
- `trajectory.horizon == 128`；
- `trajectory.duration_s == 10.0`；
- `trajectory.start_max_abs_error_rad <= 1e-5`；
- `trajectory.velocity_limit_utilization <= 1.0`；
- `trajectory.acceleration_limit_utilization <= 1.0`。

验收 `trajectory.npz`：

- `positions/velocities/accelerations` 都是 `[128, 7]`；
- 全部数值有限；
- `time_from_start` 从 0 开始、严格递增、在 10 秒结束；
- 第一轨迹点与请求起点一致。

### 3.4 L3：多 context 算法回归

一次 smoke test 只能证明调用链可用，不能证明算法退化与否。

至少固定：

- model checkpoint；
- runtime/inference YAML；
- start/goal context 索引或请求文件集合；
- seed；
- GPU 型号和软件环境；
- Git commit 和 dirty 状态。

统计以下指标的均值、中位数和分位数：

- planning success coverage；
- valid trajectory rate；
- EE position error，单位 m；
- EE orientation error，单位 degree；
- path length；
- smoothness；
- velocity-limit utilization；
- acceleration-limit utilization；
- inference time；
- collision trajectory rate。

当前 `inference-report-*.txt` 已提供上述主要数据。参数 sweep 可使用：

```bash
/home/eric/anaconda3/envs/mpd-splines-public-cu128/bin/python \
  scripts/inference/run_ee_pose_sweep.py \
  --output-root /tmp/mpd-sweep \
  --contexts 15 \
  --contexts-per-process 3 \
  --seed 0 \
  --device cuda:0
```

8GB 显存下不要把 15 个 context 放入同一 inference 进程，保持 `--contexts-per-process 1` 或 `3`。

回归阈值不应凭感觉写死，应先保存一份已验收 baseline，再设置相对容差。例如：

| 指标 | 推荐回归规则示例 |
|---|---|
| pose metric coverage | 不低于 baseline |
| valid-rate mean | 不低于 baseline - 5 percentage points |
| position median | 不高于 baseline × 1.2 |
| orientation median | 不高于 baseline × 1.2 |
| inference median | 不高于 baseline × 1.25 |

这些只是建立 CI 的起始规则，最终阈值应由项目安全和任务精度要求决定。

### 3.5 L4：IsaacLab 跨后端复核

目的：验证同一条 MPD 输出轨迹在独立物理/接触后端中是否发生接触。

IsaacLab 必须继续使用独立环境：

```bash
conda run -n env_isaaclab_ori \
  /home/eric/IsaacLab_ori/isaaclab.sh -p \
  scripts/isaaclab/evaluate_mpd_trajectories.py \
  --input /absolute/path/to/isaaclab-trajectories.pt \
  --output /tmp/isaaclab-statistics.json \
  --device cuda:0 \
  --headless
```

验收：

- `n_trajectories_collision == 0`；
- `first_collision_step` 全部为 `-1`；
- `n_unsupported_obstacles == 0`；
- IsaacLab joint 名和 MPD Panda joint 顺序一致；
- contact-force threshold 有明确记录。

当前独立 evaluator/replay 路径不应视为真机安全证明；它是额外复核层。

## 4. 任务二：模型碰撞

### 4.1 机器人碰撞模型

Panda 机器人有三套相关但不同的几何：

1. URDF visual mesh：用于显示；
2. URDF collision mesh：PyBullet/某些仿真后端可使用；
3. `panda_sphere_config.yaml`：MPD Torch collision 的主要机器人近似。

`RobotPanda` 把 URDF、collision sphere YAML 和 joint-limit YAML 一起传给 `RobotBase`。

`RobotBase` 随后：

- 解析非固定关节，生成 `q_pos_min/q_pos_max`；
- 生成 Torchkin FK；
- 为 collision spheres 生成 FK；
- 按 sphere 索引建立 self-collision tuples；
- 加载 `dq_max/ddq_max`。

修改机器人 mesh 后，必须同步检查 collision spheres。只替换 STL/OBJ 而不更新 sphere YAML，会导致 MPD 仍按旧几何避障。

### 4.2 sphere 顺序约束

collision sphere 的顺序是结构性接口，不只是显示顺序。

当前约定：

```text
[Panda spheres][grasped-object spheres]
                 ↑
       n_robot_collision_spheres
```

以下模块共享同一索引顺序：

- `fk_map_collision()`；
- `CollisionSelfField`；
- object-link collision pairs；
- `CostTaskSpaceCollisionObjects`；
- 最终 Torch collision validity check。

增加、删除或重排 sphere 后，必须运行并扩展：

```text
tests/test_grasped_object_collision_pairs.py
```

不要只检查整数 tuple；测试应把 sphere index 反查到 parent link，再验证实际 link pair。

### 4.3 环境碰撞模型

Warehouse 当前由 `ObjectField` 包含的 primitive 构成：

- `MultiSphereField`；
- `MultiBoxField`。

环境创建时：

```text
EnvWarehouse
  → obj_fixed_list / obj_extra_list
  → GridMapSDF
  → CollisionObjectDistanceField
  → robot collision-sphere centers 查询 SDF
```

cost guidance 使用 SDF 及其梯度；最终有效性过滤使用 occupancy 语义，即检查 signed distance 是否小于 collision-sphere radius 加 margin。

新增环境物体后必须同时检查：

- `obj_fixed_list` 还是 `obj_extra_list`；
- position 和 orientation 是否以 Panda base/world 为基准；
- `sdf_cell_size` 是否足以表达小尺寸结构；
- `obstacle_cutoff_margin_extra`；
- `margin_for_dense_collision_checking`；
- scene payload 是否完整导出；
- PyBullet 和 IsaacLab 是否使用同一 pose/scale。

### 4.4 抓取物碰撞

当前抓取物实现是 `GraspedObjectBox`：

- 几何由 `MultiBoxField` 表示；
- 相对 `panda_hand` 定义 pose；
- 采样点加入机器人临时 URDF；
- object spheres 放在 Panda spheres 后面；
- 与 `allowed_self_collision_links` 之外的机器人 link 建立 collision pair。

新增抓取 mesh 时不能只把 mesh 显示在手上，还需要：

- 统一 attached frame；
- 生成 collision proxy 或 mesh SDF；
- 定义 object-to-robot 允许接触 link；
- 定义 object-to-environment 碰撞；
- 在 IsaacLab 中创建固定附着或等价刚性连接；
- 验证质量、惯量和接触参数。

## 5. 任务三：新 Mesh 通过 pipeline 导入仿真

首先区分三种 mesh：

| 类型 | 例子 | 需要改动的主要链路 |
|---|---|---|
| 机器人 link mesh | 新机器人、修改手爪 | URDF、Robot class、sphere config、IsaacLab robot asset、训练数据 |
| 环境 obstacle mesh | 货架、机器、工装 | Env/ObjectField、SDF/proxy、PyBullet、scene payload、IsaacLab spawn |
| grasped-object mesh | 箱体、工具 | GraspedObject、URDF attachment、object-link pairs、IsaacLab attachment |

三类资产不能用同一套最小改动处理。

### 5.1 资产目录建议

建议新增统一目录：

```text
mpd/torch_robotics/torch_robotics/data/meshes/custom/<asset_id>/
  source/
    asset.obj
  visual/
    asset.usd
  collision/
    asset_collision.usd
  proxy/
    collision_proxy.yaml
  asset.yaml
```

`asset.yaml` 建议记录：

```yaml
schema_version: 1
asset_id: warehouse_machine_v01
units: m
up_axis: Z
source_mesh: source/asset.obj
visual_usd: visual/asset.usd
collision_usd: collision/asset_collision.usd
scale: [1.0, 1.0, 1.0]
source_sha256: "..."
collision_representation: boxes
collision_proxy: proxy/collision_proxy.yaml
```

强制要求：

- 统一使用米；
- 统一 Z-up；
- 固定 mesh 原点；
- visual 和 collision mesh 分离；
- collision mesh 必须简化；
- 所有资产有 SHA-256；
- payload 使用 repo-relative asset ID，不直接保存只在一台机器有效的任意绝对路径。

### 5.2 第一阶段：visual mesh + primitive collision proxy

这是当前仓库最容易稳定落地的方案。

#### Step 1：准备 mesh

- 清理重复顶点、退化三角形和非法法线；
- 确认单位、坐标轴和原点；
- visual mesh 可保留较高面数；
- collision mesh 或 proxy 应低面数、无自交；
- 动态/抓取物 collision mesh 优先使用 convex decomposition；
- 静态环境才考虑 triangle-mesh collider。

#### Step 2：生成 MPD collision proxy

把 mesh 近似成：

- 多个 box；
- 多个 sphere；
- 或 box+sphere 混合。

示例 `collision_proxy.yaml`：

```yaml
boxes:
  centers:
    - [0.0, 0.0, 0.25]
  sizes:
    - [0.8, 0.4, 0.5]
spheres:
  centers: []
  radii: []
```

将这些数据构造成 `MultiBoxField`/`MultiSphereField`，再包装为 `ObjectField`。

#### Step 3：接入 MPD 环境

新增环境类或扩展现有环境，例如：

```text
mpd/torch_robotics/torch_robotics/environments/env_warehouse.py
```

推荐新增独立类而不是静默修改旧训练环境：

```text
EnvWarehouseMeshV01
EnvWarehouseMeshExtraObjectsV01
```

随后在：

```text
mpd/torch_robotics/torch_robotics/environments/__init__.py
```

导出新类，并在训练/推理 YAML 中使用新的 `env_id` 或 `env_id_replace`。

如果 mesh 改变训练环境的可行空间，应生成新数据并重新训练；不能默认旧模型已经学会新场景。仅作为 inference extra object 时，可以先依赖 cost guidance，但必须单独评估有效率和目标误差。

#### Step 4：接入 PyBullet/OMPL

当前 `GenerateDataOMPL.add_obstacles()` 只接受 `MultiSphereField` 和 `MultiBoxField`。

第一阶段直接复用同一 proxy，因此无需 PyBullet mesh API：

```text
ObjectField proxy
  → get_all_single_primitives()
  → add_sphere()/add_box()
```

这样 Torch、OMPL 和数据生成至少使用同一碰撞近似。

如果后续要使用 PyBullet triangle mesh，需要：

- 新增 `add_mesh()`；
- 使用 `GEOM_MESH` 或 mesh URDF；
- 明确 mesh scale；
- 静态 mesh 设置 fixed base；
- 为非凸 mesh 选择分解后的 collision mesh；
- 增加 PyBullet 与 Torch proxy 的采样一致性测试。

#### Step 5：扩展 scene payload

当前 `scripts/isaaclab/scene_payload.py` 只序列化 sphere 和 box。

第一阶段建议同时导出：

- collision proxy primitives，供 IsaacLab evaluator 做碰撞；
- visual asset metadata，供 replay 显示 mesh。

建议 schema v2：

```json
{
  "schema": "mpd_isaaclab_scene",
  "schema_version": 2,
  "assets": [
    {
      "asset_id": "warehouse_machine_v01",
      "visual_type": "usd",
      "visual_asset": "data/meshes/custom/warehouse_machine_v01/visual/asset.usd",
      "asset_sha256": "...",
      "position": [0.5, 0.1, 0.0],
      "orientation": [1.0, 0.0, 0.0, 0.0],
      "scale": [1.0, 1.0, 1.0],
      "collision_proxy_names": ["MpdObstacle_000", "MpdObstacle_001"]
    }
  ],
  "obstacles": [
    {
      "type": "box",
      "name": "MpdObstacle_000",
      "position": [0.5, 0.1, 0.25],
      "orientation": [1.0, 0.0, 0.0, 0.0],
      "size": [0.8, 0.4, 0.5]
    }
  ]
}
```

必须保留 schema version，旧 evaluator 遇到 v2 时应显式检查，而不是静默忽略新字段。

#### Step 6：扩展 IsaacLab spawn

需要同步修改：

```text
scripts/isaaclab/evaluate_mpd_trajectories.py
scripts/isaaclab/replay_mpd_trajectory.py
```

建议拆出公共函数，避免 evaluator 和 replay 各维护一份 `_spawn_scene_obstacles()`：

```text
scripts/isaaclab/scene_spawn.py
```

职责：

- 校验 schema version；
- 解析 repo-relative asset path；
- 校验 SHA-256；
- spawn sphere/box collision proxy；
- replay 时额外 spawn visual USD；
- evaluator headless 时可以关闭 visual mesh，仅保留 collider；
- 记录实际 spawn 的 asset、pose、scale 和 collider 类型。

USD 资产建议通过 IsaacLab/Isaac Sim 的 `UsdFileCfg` 加载。不要假定 OBJ/STL 在所有 IsaacLab 版本中都能以相同方式直接加载；先离线转换为 USD，并把转换工具版本写入 `asset.yaml`。

### 5.3 第二阶段：MPD 精确 mesh SDF

只有 primitive proxy 的误差不能满足任务时，才实现这一阶段。

当前 `primitives.py` 中的 `MeshField` 只是注释占位，`distance_fields.py` 末尾的 `MeshDistanceField` 示例也没有可用类定义，不能直接启用。

推荐实现路径：

```text
watertight collision mesh
  ↓ offline voxelization / signed-distance generation
dense SDF grid + metadata
  ↓ load as torch tensor
trilinear interpolation
  ↓
compute_signed_distance(x, get_gradient=True)
```

需要新增：

```text
mpd/torch_robotics/torch_robotics/environments/mesh_field.py
mpd/torch_robotics/torch_robotics/data/meshes/.../collision/sdf.npz
```

`MeshField` 至少实现：

- `compute_signed_distance_impl()`；
- 对查询点的可微三线性插值；
- 越界处理；
- object pose/rotation；
- `get_all_single_primitives()` 或明确禁止 primitive-only 后端；
- asset hash、grid resolution、bounds；
- CPU/GPU dtype/device 转换。

SDF grid 验证：

- mesh 表面采样点距离接近 0；
- mesh 内点为负、外点为正；
- 数值梯度与自动微分梯度方向一致；
- 旋转和平移后距离保持一致；
- proxy 与 mesh SDF 的 false-negative rate 为 0 或在明确安全规则内；
- grid resolution 改变时结果收敛。

注意：如果 MPD 使用精确 mesh SDF，而 PyBullet/IsaacLab 使用另一份 collision mesh，仍然存在后端差异。三者必须共享同一 collision asset 版本和 pose/scale metadata。

### 5.4 机器人 link mesh 的特殊处理

如果新 mesh 属于机器人 link：

1. 更新或新增 URDF；
2. 新增对应 Robot class；
3. 更新 joint limit YAML；
4. 重新拟合 collision sphere YAML；
5. 更新 self-collision pair；
6. 更新 PyBullet 侧使用的 URDF；
7. 为 IsaacLab 创建/转换对应 USD robot asset；
8. 修改 evaluator/replay，不再固定使用 `FRANKA_PANDA_HIGH_PD_CFG`；
9. 按 joint name 显式解析 DoF；
10. 重新生成训练数据并训练模型。

机器人 link mesh 改变后不能继续把旧 Panda checkpoint 当成同一模型。

### 5.5 抓取 mesh 的特殊处理

如果新 mesh 是抓取物：

1. 新增 `GraspedObjectMesh`；
2. 指定 `attached_to_frame`；
3. 提供相对手部的 pose；
4. 生成 collision proxy 或 mesh SDF；
5. 生成 object collision points/spheres；
6. 保持 Panda spheres 在前、object spheres 在后；
7. 配置允许接触的手部 link；
8. 测试 object-to-link 和 object-to-environment；
9. 在 IsaacLab 中使用 fixed joint 或等价 attachment；
10. 验证抓取物 contact sensor、质量和惯量。

## 6. Mesh pipeline 测试矩阵

新增 mesh 后至少增加以下测试。

### 6.1 资产静态测试

- 文件存在；
- SHA-256 与 manifest 一致；
- scale 为正；
- quaternion 已归一化；
- collision proxy 非空；
- asset path 不能逃离仓库资产根目录；
- 单位和 up-axis 有明确定义。

建议新增：

```text
tests/test_mesh_asset_manifest.py
```

### 6.2 MPD collision 测试

- 已知自由点返回非碰撞；
- 已知内部点返回碰撞；
- 机器人已知姿态与新 obstacle 碰撞；
- 邻近姿态无碰撞；
- collision margin 增大后碰撞集合只能扩大；
- sphere index 和 parent-link 映射不变。

建议新增：

```text
tests/test_mesh_collision_proxy.py
tests/test_mesh_sdf.py
```

### 6.3 PyBullet/Torch 一致性测试

固定采样一组 Panda joint states：

```text
q samples
  ├─ Torch collision result
  └─ PyBullet is_state_valid result
```

统计 confusion matrix：

- true positive；
- true negative；
- false positive；
- false negative。

安全优先时，Torch collision proxy 相对精确 mesh 不允许出现 false negative；允许少量 false positive，但应报告其对 valid rate 的影响。

### 6.4 Payload round-trip 测试

验证：

```text
Env object
  → scene payload
  → IsaacLab spawn metadata
```

pose、orientation、scale、asset ID、hash 和 proxy 数量必须一致。

建议新增：

```text
tests/test_scene_payload_mesh.py
```

该测试应保持无 IsaacLab 依赖，只检查 payload 和 path/hash 解析。真正 spawn 测试放到 IsaacLab 环境。

### 6.5 IsaacLab smoke test

在单环境中：

- spawn 新 mesh visual；
- spawn collision proxy/collider；
- 检查 prim 存在；
- 检查 pose 和 scale；
- 让 Panda 进入一条已知碰撞轨迹，必须检测到 contact；
- 运行一条已知自由轨迹，不能误报；
- 输出 screenshot 和统计 JSON 作为证据。

## 7. 推荐实施顺序

### Phase A：测试基线

1. 保存当前 CPU unittest 输出；
2. 固定一组 inference requests/contexts；
3. 保存 baseline CSV/JSON；
4. 固定 IsaacLab 版本和环境调用命令。

### Phase B：资产 manifest 和 primitive proxy

1. 建立 `data/meshes/custom/<asset_id>`；
2. 增加 `asset.yaml`；
3. 生成 visual USD；
4. 生成 sphere/box proxy；
5. 加入 MPD 新环境类；
6. 加入资产静态和 collision proxy 测试。

### Phase C：scene payload v2

1. 增加 asset metadata；
2. 保持旧 sphere/box payload 兼容；
3. 增加 asset path resolver 和 hash 校验；
4. 增加 payload round-trip 测试。

### Phase D：IsaacLab visual/collision spawn

1. 抽取公共 `scene_spawn.py`；
2. evaluator spawn collision representation；
3. replay spawn visual USD；
4. 记录实际 collider；
5. 完成 contact smoke test。

### Phase E：数据与算法回归

1. PyBullet/OMPL 使用同一 proxy 或 collision mesh；
2. 如环境定义改变，重新生成训练数据；
3. 重新训练或明确只使用 cost-guided adaptation；
4. 执行 L0-L4 全部测试；
5. 对比 baseline 指标。

### Phase F：可选精确 mesh SDF

1. 生成 watertight simplified collision mesh；
2. 预计算 SDF grid；
3. 实现可微 `MeshField`；
4. 增加数值/梯度测试；
5. 比较 proxy、mesh SDF、PyBullet 和 IsaacLab。

## 8. 完成定义

新 mesh 只有同时满足以下条件，才算真正通过本仓库 pipeline：

- 有版本化 source/visual/collision 资产；
- 有单位、坐标轴、原点、scale 和 SHA-256；
- MPD Torch collision 能识别；
- PyBullet/OMPL 数据生成能识别；
- scene payload 不报告 unsupported obstacle；
- IsaacLab evaluator/replay 能 spawn；
- pose/scale 在三个后端一致；
- 已知碰撞和自由样例测试通过；
- inference 回归没有不可接受的 success/valid-rate/EE-error 退化；
- 如果机器人或训练环境发生结构变化，已重新生成数据并训练模型。

仅在 IsaacLab 中“看见 mesh”，或者仅在 URDF 中引用 STL，都不能证明 MPD 已经使用该 mesh 做碰撞规划。
