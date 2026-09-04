# Marvin 双臂实现说明

本文记录当前 MPD 仓库和 ROS2 仓库中已经加入的 Marvin 双臂内容。双臂代码采用
独立模块、独立包名和独立入口，不替换原有 Franka/Panda 单臂链路。

## 1. 总体结构

```text
ROS2 PlanBimanualTrajectory.action
          │
          ▼
mpd_bimanual_planner_adapter
          │  Marvin request/result v1
          ▼
MPD Marvin runtime / worker
          │
          ▼
14-joint JointTrajectory → bimanual_arm_jtc
```

所有边界都使用固定关节顺序：

```text
[Joint1_L ... Joint7_L, Joint1_R ... Joint7_R]
```

## 2. MPD 侧

### 2.1 机器人模型

新增 [`RobotMarvinBimanual`](../mpd/torch_robotics/torch_robotics/robots/robot_marvin_bimanual.py)，支持：

- 14 个关节、左右 slice、`split_q`/`merge_q`；
- `fk_left`、`fk_right`、`jfk_left`、`jfk_right`、`jfk_bimanual`；
- 左右末端 `flange_L` 和 `flange_R`；
- `model_hash`，用于部署时发现模型副本过期。

模型资产位于 `mpd/torch_robotics/torch_robotics/data/` 下的 `marvin/` 目录，包括：

- `marvin_bimanual_mpd.urdf`；
- `joint_limits.yaml`；
- `collision_spheres.yaml`；
- `collision_parent_bounds.yaml`；
- `self_collision_pairs.yaml`；
- `grasp_profiles.yaml`。

`marvin_bimanual_mpd.urdf` 由 ROS2 的 arm-only xacro 导出，关闭了 `ros2_control`，并将
`marvin_description` mesh 路径本地化到 MPD 的 `marvin/meshes/`。关节限制、碰撞球、
self-collision pair 和 parent bounds 已从 ROS2/CuRobo 配置转换；导出源 checksum 保存在
同名 `.urdf.sha256` 文件中。重新生成时使用 [`export_marvin_mpd_model.py`](../scripts/robots/export_marvin_mpd_model.py)
并指定 `--asset-root`，不要使用包含 Pika 夹爪的 18-DoF bringup xacro。

### 2.2 双臂任务和代价

新增 `mpd/bimanual/`：

- `planning_task.py`：`left_only`、`right_only`、`dual_independent`、`cooperative_rigid`；
- `goal_ik.py`：pose/object goal 到 `q_goal` 的 IK 注入点；
- `costs.py`：冻结臂保持、硬投影、闭链残差、臂间距离代价；
- `cost_guide.py`、`dynamic_cost_guide.py`、`space_time_cost_guide.py`：静态、动态和推理期 timing façade；
- `trajectory_validator.py`：14D、有限值、关节/速度/加速度和闭链硬验证；
- `runtime_contract.py`：请求/结果 JSON contract、joint order、world version、deadline 校验。

协同模式在整条稠密轨迹上检查左右末端闭链，不只检查终点。抓取搜索、力控、负载
分配和滑移检测不在当前实现范围内。

### 2.3 数据与训练

新增：

- `data_generation_cfgs/EnvMarvinTable-RobotMarvinBimanual-independent.yaml`；
- `data_generation_cfgs/EnvMarvinTable-RobotMarvinBimanual-cooperative.yaml`；
- [`generate_marvin_bimanual_trajectories.py`](../scripts/generate_data/generate_marvin_bimanual_trajectories.py)；
- `scripts/train/cfgs/marvin_bimanual_independent.yaml`；
- `scripts/train/cfgs/marvin_bimanual_cooperative.yaml`；
- [`train_marvin_bimanual.py`](../scripts/train/train_marvin_bimanual.py)。

数据生成器写入 `sol_path`、`q_start`、`q_goal`、`task_mode`、`active_joint_mask`、
scene/closure/clearance/grasp metadata、`args.yaml` 和 `manifest.yaml`。

训练薄入口只验证 `state_dim=14`、`context_q_dim=28` 和 task family，然后复用原有
Temporal U-Net、loss 和 optimizer 流程，不复制 Franka 训练代码。`--dry-run` 可只检查配置。

> 当前路径生成函数是 smoke scaffold。生产数据需要替换为 Marvin 复合模型上的
> RRTConnect、双臂 IK、payload 碰撞和逐 waypoint 闭链验证。

### 2.4 静态/动态 runtime

静态入口：

- `scripts/inference/inference_marvin_bimanual.py`；
- `scripts/runtime/infer_once_marvin_bimanual.py`。

动态入口：

- `runtime_engine_marvin_bimanual.py`；
- `dynamic_runtime_engine_marvin_bimanual.py`；
- `infer_dynamic_server_marvin_bimanual.py`；
- `space_time_runtime_engine_marvin_bimanual.py`；
- `infer_space_time_server_marvin_bimanual.py`。

支持三种运行语义：`snapshot_no_time`、`fixed_time_dynamic`、
`inference_time_optimized`。三者都不把物理时间加入 diffusion 训练张量；space-time
时间参数只属于推理阶段。当前静态入口提供确定性插值 fallback，用于 contract 和 ROS 链路 smoke。

## 3. ROS2 侧

### 3.1 接口包

新增 [`manipulation_planning_interfaces`](../../../physical_ai_runtime/src/interfaces/manipulation_planning_interfaces)：

- `PlanBimanualTrajectory.action`：四种任务模式、左右目标、物体目标、抓取 profile、预算、执行开关；
- `DynamicObject.msg`：形状、位姿、twist、协方差、inflation 和有效期；
- `DynamicWorld.msg`：header、单调 `world_version`、有效期和对象数组。

### 3.2 双臂 adapter

新增 [`mpd_bimanual_planner_adapter`](../../../physical_ai_runtime/src/motion_planning/motion_planners/mpd_bimanual_planner_adapter)：

- 固定顺序提取 14D joint state；
- 校验 action goal、数值、时间预算和 mode；
- Unix socket IPC；
- latest-only generation 和动态世界版本管理；
- worker result 到单个 14-joint `JointTrajectory` 的转换；
- collision guard、统一停止和协同 handoff 门限 façade。

配置分别对应静态、动态快照、固定时间和 space-time，socket 使用
`/tmp/mpd_marvin_bimanual_*.sock`，不复用 Franka socket。

### 3.3 bringup 和控制器

[`marvin_mpd_bimanual_bringup`](../../../physical_ai_runtime/src/apps/marvin_mpd_bimanual_bringup)
新增：

- Marvin stand、左右 7DoF arm、adaptors 和 Pika gripper 的双臂 xacro；
- fake hardware 默认路径，真实硬件需显式关闭 fake 参数；
- `bimanual_arm_jtc`，一次 claim 14 个关节；
- `execution_bimanual.yaml`、`planning_static.yaml` 和 Pika 挂载标定文件；
- `marvin_mpd_bimanual_fake_hardware.launch.py`，默认 `execute:=false`。

## 4. 典型命令

### MPD 配置检查

```bash
cd /home/eric/Projects/MotionPlanningDiffusion/mpd
PYTHONPATH=. python3 scripts/train/train_marvin_bimanual.py \
  --config scripts/train/cfgs/marvin_bimanual_independent.yaml --dry-run
```

### 静态单次推理

```bash
python3 scripts/inference/inference_marvin_bimanual.py \
  --config scripts/inference/cfgs/config_EnvMarvinTable-RobotMarvinBimanual-independent.yaml \
  --request /tmp/marvin_bimanual_request.json \
  --output /tmp/marvin_bimanual_result.json
```

### ROS2 fake hardware / plan-only

```bash
cd /home/eric/Projects/physical_ai_runtime
ros2 launch marvin_mpd_bimanual_bringup \
  marvin_mpd_bimanual_fake_hardware.launch.py execute:=false
```

## 5. 兼容性边界

本次新增没有修改：

- `scripts/inference/inference.py` 和原有 runtime 入口；
- `mpd_planner_adapter`、`mpd_dynamic_planner_adapter`；
- `marvin_motion_planning_bringup`；
- Franka/Panda 模型、数据、checkpoint 和 socket。

双臂使用新的模块、ROS2 包、action、controller、socket 和 checkpoint 目录。当前工作区
中原先存在的 `pixi.lock`、时序配置和 space-time 数据变更不属于本双臂提交。

## 6. 当前限制与后续

当前仍需在完整 ROS2/训练环境中完成：

1. ROS xacro 与 torchkin 的随机状态 FK/Jacobian 数值门禁；
2. 真实 RRT/IK independent/cooperative 数据和正式 diffusion checkpoint；
3. 完整 worker IPC、Top-K 稠密验证和统一 14D retiming；
4. fake hardware 下的 action cancel、controller ownership、world revalidation；
5. 仿真、HIL、无载荷和低速载荷真机测试。

协同动态重规划第一版应采用“停稳/保持后换轨”；连续运动中的闭链 bridge 尚未实现。
