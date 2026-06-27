# IsaacGym 到 IsaacLab 迁移须知

本文档说明 MPD 仓库中 IsaacGym 仿真/回放部分迁移到 IsaacLab 时的边界、步骤、接口和风险点。目标不是重写 MPD，而是把旧的 IsaacGym 验证层替换为一个 IsaacLab 适配层。

## 总体原则

MPD 的训练、采样、轨迹优化和指标计算应保持不变。IsaacGym 在当前仓库中的主要作用是把已经生成的关节空间轨迹放进物理仿真中执行，并返回哪些轨迹发生接触/碰撞。

推荐迁移方式：

```text
MPD 生成轨迹 q_trajs_pos
  -> IsaacLab adapter 执行轨迹
  -> 返回碰撞统计、可选视频、可选状态日志
  -> MPD 保存 results_single_plan
```

不推荐把所有 MPD 代码改成 IsaacLab 风格，也不推荐把 IsaacLab 直接塞进原 MPD 的 Python 3.8 环境。

## 当前 IsaacGym 调用点

核心调用点如下：

| 文件 | 作用 | 迁移处理 |
| --- | --- | --- |
| `mpd/torch_robotics/torch_robotics/isaac_gym_envs/motion_planning_envs.py` | IsaacGym 环境、资产创建、reset/step、接触统计、视频录制 | 不建议原地重写；新增 IsaacLab adapter |
| `scripts/inference/inference.py` | inference 后创建 IsaacGym env 并执行 valid trajectories | 改为 backend 选择或调用 IsaacLab evaluator |
| `scripts/inference/launch_inference-experiments.py` | 实验启动参数 | 增加 IsaacLab 相关 flag |
| `scripts/generate_data/visualize_trajectories.py` | 轨迹可视化 | 第二阶段再迁移 |
| `mpd/motion_planning_baselines/examples/panda_isaac_replay.py` | Panda replay 示例 | 第二阶段再迁移 |
| `scripts/train/train.py` | 顶层 `import isaacgym` | 删除或 lazy import，训练本身不应依赖仿真 viewer |
| `mpd/parametric_trajectory/trajectory_bspline.py` | 顶层 `import isaacgym` | 若未使用，删除 |
| `mpd/parametric_trajectory/trajectory_waypoints.py` | 顶层 `import isaacgym` | 若未使用，删除 |

## 可以动的范围

可以新增这些文件/目录：

```text
mpd/torch_robotics/torch_robotics/isaac_lab_envs/
mpd/torch_robotics/torch_robotics/isaac_lab_envs/__init__.py
mpd/torch_robotics/torch_robotics/isaac_lab_envs/motion_planning_envs.py
scripts/isaaclab/evaluate_mpd_trajectories.py
scripts/isaaclab/export_mpd_scene.py        # 可选
scripts/isaaclab/README.md                  # 可选
```

可以改这些现有文件：

```text
scripts/inference/inference.py
scripts/inference/launch_inference-experiments.py
scripts/generate_data/visualize_trajectories.py
scripts/train/train.py
mpd/parametric_trajectory/trajectory_bspline.py
mpd/parametric_trajectory/trajectory_waypoints.py
```

改动原则：

- IsaacGym import 必须 lazy import，只在真的使用 IsaacGym backend 时导入。
- IsaacLab import 也必须延迟到独立脚本或 adapter 内部，且必须在 `AppLauncher` 启动 Isaac Sim 后再导入大部分 IsaacLab/Omniverse 相关模块。
- `run_evaluation_issac_gym` 可以保留兼容旧配置，但建议新增 `sim_backend` 或 `run_evaluation_isaac_lab`。
- 新逻辑应通过同一份 trajectory contract 接 MPD，不要让 MPD 的 planner 知道底层是 IsaacGym 还是 IsaacLab。

## 不要动的范围

除非明确要重做算法，不要改这些部分：

```text
mpd/models/
mpd/trainer/
mpd/inference/inference.py
mpd/metrics/
mpd/parametric_trajectory/               # 除删除无用 isaacgym import 外
mpd/torch_robotics/torch_robotics/tasks/
mpd/torch_robotics/torch_robotics/robots/
data_trajectories/
data_trained_models/
```

不要做这些事：

- 不要为了 IsaacLab 修改 MPD 数据集格式。
- 不要改变模型输入输出张量定义。
- 不要改变 `q_trajs_pos` 的维度语义。
- 不要把 IsaacLab 的 RL env API 强行塞进 MPD planner。
- 不要在原 `environment.yml` 中直接替换成 IsaacLab 依赖。IsaacLab 应单独环境运行。
- 不要把 IsaacGym 文件夹删除；保留旧 backend 便于对照。

## 推荐环境结构

建议两个环境分离：

```text
mpd-splines-public 环境
  Python 3.8
  MPD 原依赖
  PyBullet/OMPL/Theseus/torch_robotics
  负责数据、训练、采样、轨迹优化

IsaacLab 环境
  Isaac Sim / IsaacLab 推荐 Python 与 PyTorch 栈
  负责仿真执行、接触验证、视频录制
```

推荐用 subprocess 或文件交换：

```text
MPD inference
  1. 保存 trajectories.pt
  2. 调用 /home/eric/IsaacLab_ori/isaaclab.sh -p scripts/isaaclab/evaluate_mpd_trajectories.py ...
  3. 读取 isaaclab_statistics.json
```

这样可以避免 Python、Torch、CUDA、Isaac Sim 版本互相冲突。

## MPD 与仿真层的接口 contract

MPD 输出给仿真层的核心张量：

```python
q_trajs_pos: torch.Tensor  # shape: [H, B, D]
q_pos_starts: torch.Tensor # shape: [B, D]
q_pos_goal: torch.Tensor   # shape: [D]
```

含义：

- `H`: trajectory horizon。
- `B`: 候选轨迹数量，通常等于 valid trajectory 数量。
- `D`: 机器人 arm joint 维度，Panda 通常是 7。

IsaacLab adapter 应提供与旧 controller 接近的接口：

```python
class MotionPlanningControllerIsaacLab:
    def execute_trajectories(
        self,
        trajectories,          # [H, B, D]
        q_pos_starts=None,     # [B, D]
        q_pos_goal=None,       # [D]
        n_pre_steps=0,
        n_post_steps=0,
        stop_robot_if_in_contact=False,
        make_video=False,
        **kwargs,
    ):
        ...
```

最小返回值：

```python
{
    "n_trajectories_collision": int,
    "n_trajectories_free": int,
    "n_trajectories_free_fraction": float,
}
```

推荐额外返回：

```python
{
    "collision_mask": list[bool],        # length B
    "first_collision_step": list[int],   # length B, 无碰撞可用 -1
    "video_path": str | None,
    "backend": "isaaclab",
}
```

为兼容旧结果字段，第一版可以继续写入：

```python
results_single_plan.isaacgym_statistics = isaaclab_statistics
```

后续再统一重命名为：

```python
results_single_plan.sim_statistics
```

## IsaacGym 到 IsaacLab API 映射

| IsaacGym | IsaacLab |
| --- | --- |
| `gymapi.acquire_gym()` | `isaaclab.app.AppLauncher` |
| `gym.create_sim(...)` | `isaaclab.sim.SimulationContext` |
| `gym.create_env(...)` | `isaaclab.scene.InteractiveScene` |
| `gym.load_asset(... urdf ...)` | `ArticulationCfg`，优先用 `FRANKA_PANDA_HIGH_PD_CFG` |
| `gym.create_sphere(...)` | `RigidObjectCfg` + `sim_utils.SphereCfg` |
| `gym.create_box(...)` | `RigidObjectCfg` + `sim_utils.CuboidCfg` |
| `gym.set_dof_state_tensor(...)` | `robot.write_joint_state_to_sim(joint_pos, joint_vel)` |
| `gym.set_dof_position_target_tensor(...)` | `robot.set_joint_position_target(joint_pos, joint_ids=...)` |
| `gym.simulate(...)` / `gym.fetch_results(...)` | `scene.write_data_to_sim(); sim.step(); scene.update(dt)` |
| `gym.acquire_dof_state_tensor(...)` | `robot.data.joint_pos`, `robot.data.joint_vel` |
| `gym.acquire_rigid_body_state_tensor(...)` | `robot.data.body_state_w` |
| `gym.get_env_rigid_contacts(...)` | `ContactSensorCfg` 或 PhysX contact view |
| `gym.create_viewer(...)` | Isaac Sim viewer 或 headless + camera sensor |

## IsaacLab adapter 的实现步骤

### 1. 先解耦 IsaacGym import

先让 MPD 在不使用 IsaacGym 时也能正常 import：

- 删除训练脚本的顶层 `import isaacgym`。
- `scripts/inference/inference.py` 中只在 `run_evaluation_issac_gym=True` 时导入旧 IsaacGym backend。
- `trajectory_bspline.py`、`trajectory_waypoints.py` 中无用的 `import isaacgym` 应删除。

这一步不改变算法，只是解除旧仿真库对训练/采样的阻塞。

### 2. 新增 backend 参数

建议新增：

```python
sim_backend: str = "none"  # none, isaacgym, isaaclab
```

兼容旧参数：

```python
if run_evaluation_issac_gym:
    sim_backend = "isaacgym"
```

推荐逻辑：

```python
if sim_backend == "isaacgym":
    from torch_robotics.isaac_gym_envs.motion_planning_envs import ...
elif sim_backend == "isaaclab":
    # 第一版建议 subprocess 调用 IsaacLab 脚本
else:
    # 不做物理仿真验证
```

### 3. 新增 IsaacLab evaluator 脚本

建议脚本：

```text
scripts/isaaclab/evaluate_mpd_trajectories.py
```

输入：

```text
--input /path/to/trajectories.pt
--output /path/to/isaaclab_statistics.json
--num_envs B
--device cuda:0
--headless
--make_video
```

`trajectories.pt` 建议包含：

```python
{
    "q_trajs_pos": q_trajs_pos.cpu(),   # [H, B, D]
    "q_pos_starts": q_pos_starts.cpu(), # [B, D]
    "q_pos_goal": q_pos_goal.cpu(),     # [D]
    "robot_name": "panda",
    "env_name": planning_task.env.name or type(planning_task.env).__name__,
    "dt": planning_task.parametric_trajectory.dt,
}
```

输出 JSON：

```json
{
  "backend": "isaaclab",
  "n_trajectories_collision": 0,
  "n_trajectories_free": 16,
  "n_trajectories_free_fraction": 1.0,
  "collision_mask": [false, false],
  "first_collision_step": [-1, -1]
}
```

### 4. 构建 IsaacLab scene

IsaacLab 脚本必须先启动 app：

```python
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

之后再导入 IsaacLab 仿真模块：

```python
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
```

Panda 优先使用 IsaacLab 内置资产：

```python
robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

如果必须使用 MPD 的 URDF，则走 IsaacLab 的 URDF converter，但第一版不建议这样做。优先用内置 Panda 可以少处理 mesh、material、joint drive 和 USD 转换问题。

### 5. 映射 MPD 障碍物

MPD 障碍物主要来自：

```python
planning_task.env.obj_fixed_list
planning_task.env.obj_extra_list
```

常见 primitive：

- `MultiSphereField`
- `MultiBoxField`
- `ObjectField`

映射规则：

```text
MultiSphereField(center, radius)
  -> RigidObjectCfg(spawn=sim_utils.SphereCfg(radius=...))

MultiBoxField(center, size)
  -> RigidObjectCfg(spawn=sim_utils.CuboidCfg(size=(x, y, z)))
```

注意：

- 2D 环境中的 center 只有 `(x, y)` 时，需要补 `z=0`。
- 2D box 的 size 只有 `(x, y)` 时，需要补一个合理厚度，例如 `z=0.1`。
- MPD 中 object 可能有 `obj.pos` 和 `obj.ori`，IsaacLab 中要转成 world pose。
- 四元数顺序要统一，IsaacLab 常用 `(w, x, y, z)`，MPD/旧 IsaacGym 代码中有些地方使用 `xyzw`。

### 6. reset/step 等价实现

旧 IsaacGym `reset(q_pos_starts, q_pos_goal)` 做了这些事：

- 设置每个 env 的机器人起点关节位置。
- 速度清零。
- 如果画 goal configuration，则额外放一个 goal robot。
- 打开 gripper。
- 清空可视化状态。

IsaacLab 等价实现：

```python
joint_pos = robot.data.default_joint_pos.clone()
joint_vel = robot.data.default_joint_vel.clone()
joint_pos[:, arm_joint_ids] = q_pos_starts
joint_pos[:, finger_joint_ids] = 0.04
robot.write_joint_state_to_sim(joint_pos, joint_vel)
robot.set_joint_position_target(joint_pos)
scene.write_data_to_sim()
sim.step()
scene.update(sim_dt)
```

旧 IsaacGym `step(actions)` 做了这些事：

- actions 是 `[B, D]`。
- 下发 arm joint position target。
- gripper 保持打开。
- step 若干次 physics。
- 读取 joint state。
- 检查哪些 env 发生接触。

IsaacLab 等价实现：

```python
target = robot.data.joint_pos.clone()
target[:, arm_joint_ids] = actions
target[:, finger_joint_ids] = 0.04
robot.set_joint_position_target(target)
scene.write_data_to_sim()

for _ in range(action_repeat):
    sim.step()
    scene.update(sim_dt)

joint_states = torch.stack([robot.data.joint_pos, robot.data.joint_vel], dim=-1)
collision_env_ids = get_collision_env_ids()
```

### 7. 接触/碰撞统计

原 IsaacGym 使用：

```python
gym.get_env_rigid_contacts(env)
```

IsaacLab 第一版可以使用 `ContactSensorCfg`：

```python
ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",
    update_period=0.0,
    history_length=1,
)
```

然后通过：

```python
forces = scene["contact_forces"].data.net_forces_w
collision_mask = torch.linalg.norm(forces, dim=-1).amax(dim=-1) > threshold
```

需要注意：

- 如果只想统计 robot-object，不想统计 self-collision 或 ground contact，需要配置 filter。
- `ContactSensorCfg.filter_prim_paths_expr` 对单个 prim 更可靠；如果 sensor prim 匹配多个 link，过滤行为可能不符合预期。
- 要做精确对齐，建议第二版改用 per-link contact sensors 或 PhysX contact view。
- 2D 环境如果补了地面或障碍厚度，要避免机器人和 ground plane 产生无关接触。

### 8. 视频和可视化

第一版不要强求完全复刻旧 IsaacGym viewer。

优先级：

1. headless 碰撞统计。
2. 可选保存相机视频。
3. 可选画 EE path、goal marker。
4. 可选复刻旧颜色逻辑。

IsaacLab/Isaac Sim 的 viewer 与 IsaacGym viewer 不是同一套 API，不应在第一版为 viewer 细节消耗太多时间。

## 迁移顺序建议

### 阶段 A：解除 IsaacGym 强依赖

目标：

- 不使用 IsaacGym 时，MPD 能 import、训练、采样、inference。
- 当前 5060 环境不被 `import isaacgym` 卡住。

验收：

```bash
python scripts/inference/inference.py --run_evaluation_issac_gym False
```

至少不应在 import 阶段加载 `gym_38.so` 或 `gymtorch`。

### 阶段 B：IsaacLab 独立执行器

目标：

- 单独运行 IsaacLab 脚本。
- 输入一批 `[H, B, D]` 轨迹。
- 返回碰撞统计 JSON。

验收：

```bash
/home/eric/IsaacLab_ori/isaaclab.sh -p scripts/isaaclab/evaluate_mpd_trajectories.py \
  --input logs/trajectories.pt \
  --output logs/isaaclab_statistics.json \
  --headless
```

### 阶段 C：接入 inference

目标：

- `scripts/inference/inference.py` 在 `sim_backend=isaaclab` 时自动调用 IsaacLab evaluator。
- `results_single_plan` 中保存 IsaacLab statistics。

验收：

- `metrics` 仍由 MPD 原 `PlanningMetricsCalculator` 计算。
- IsaacLab statistics 只作为额外物理验证结果。

### 阶段 D：可视化和批量实验

目标：

- 支持视频。
- 支持 `launch_inference-experiments.py` 批量运行。
- 支持多环境、多轨迹并行验证。

## 关键注意事项

### 版本隔离

IsaacGym Preview4 只适合 Python 3.6/3.7/3.8。IsaacLab/Isaac Sim 是另一套现代环境。不要试图用一个 conda env 同时满足两者。

### IsaacLab import 顺序

IsaacLab 脚本中必须先启动 `AppLauncher`。不要在 app 启动前导入大量 Omniverse/IsaacLab runtime 模块。

### 关节顺序

MPD 的 `q_trajs_pos[..., :D]` 必须和 IsaacLab robot 的 arm joint order 对齐。Panda 通常应解析：

```python
SceneEntityCfg("robot", joint_names=["panda_joint.*"])
```

不要假设 `robot.data.joint_pos[:, :7]` 永远就是正确顺序，第一版可以打印 joint names 并断言。

### Gripper 维度

MPD 轨迹通常只包含 Panda 7 个 arm joints。IsaacLab Panda 资产包含 finger joints。执行时需要补齐 finger joint target，通常设为打开：

```python
panda_finger_joint.* = 0.04
```

### 坐标系和四元数

需要统一：

- world frame 原点。
- z-up。
- quaternion 顺序。
- 2D 环境补 z 的方式。

特别注意旧代码中 `quat_xyzw=True` 的路径，IsaacLab 多数接口使用 `(w, x, y, z)`。

### 碰撞统计口径

MPD 自身 metrics 通常基于几何/SDF/robot collision model。IsaacLab statistics 是物理仿真接触结果，两者不一定完全一致。

记录实验时要区分：

```text
MPD collision metric: algorithmic/geometric validation
IsaacLab collision statistic: PhysX execution validation
```

### 不要让仿真结果反向污染训练数据

第一版 IsaacLab 只做验证，不改训练集、不改先验轨迹、不改 diffusion loss。等验证稳定后，再考虑用 IsaacLab 采集新先验数据。

### 性能

IsaacLab/Isaac Sim 启动成本高。不要每条轨迹启动一次 IsaacLab。应一次传入 `[H, B, D]`，用 `num_envs=B` 并行验证。

### 命名

旧代码里有拼写：

```python
run_evaluation_issac_gym
```

不要在第一版大规模重命名，避免配置文件失效。可以新增正确命名：

```python
run_evaluation_isaac_lab
sim_backend
```

并保留旧参数兼容。

## 最小实现伪代码

```python
class MotionPlanningIsaacLabEnv:
    def __init__(self, env, robot, num_envs, device="cuda:0", headless=True, **kwargs):
        self.env_tr = env
        self.robot_tr = robot
        self.num_envs = num_envs
        self.device = device
        self._build_sim()
        self._build_scene()
        self._resolve_robot_indices()

    def reset(self, q_pos_starts=None, q_pos_goal=None):
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel)
        joint_pos[: q_pos_starts.shape[0], self.arm_joint_ids] = q_pos_starts.to(self.device)
        joint_pos[:, self.finger_joint_ids] = 0.04
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.set_joint_position_target(joint_pos)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim_dt)
        return torch.stack([self.robot.data.joint_pos, self.robot.data.joint_vel], dim=-1)

    def step(self, actions):
        target = self.robot.data.joint_pos.clone()
        target[: actions.shape[0], self.arm_joint_ids] = actions.to(self.device)
        target[:, self.finger_joint_ids] = 0.04
        self.robot.set_joint_position_target(target)
        self.scene.write_data_to_sim()
        for _ in range(self.action_repeat):
            self.sim.step()
            self.scene.update(self.sim_dt)
        joint_states = torch.stack([self.robot.data.joint_pos, self.robot.data.joint_vel], dim=-1)
        envs_with_contact = self.get_envs_with_contacts(first_n_envs=actions.shape[0])
        return joint_states, envs_with_contact
```

## 验收清单

迁移完成前至少检查：

- `sim_backend=none` 时 MPD inference 不导入 IsaacGym/IsaacLab。
- `sim_backend=isaacgym` 仍可在旧兼容环境运行。
- `sim_backend=isaaclab` 可通过 IsaacLab 环境执行。
- 输入 `q_trajs_pos` 的 shape 是 `[H, B, D]`。
- IsaacLab 中 `num_envs == B`。
- Panda arm joint order 与 MPD 一致。
- finger joints 被补齐。
- 2D 障碍物补 z 后不产生额外地面接触。
- 返回 statistics 字段与旧 IsaacGym statistics 兼容。
- 无碰撞轨迹和故意碰撞轨迹各做一个 smoke test。
- headless 模式能运行，viewer/video 是可选功能。

## 推荐结论

短期目标是让 MPD 在 5060 上不被 IsaacGym 阻塞：先做 lazy import 和独立 IsaacLab evaluator。

长期目标是扩展新仿真、新资产、cuRobo 或 SkillGen：保留 MPD 算法核心，把 IsaacLab 作为独立的现代仿真验证后端。
