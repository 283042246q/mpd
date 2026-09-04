# Marvin 双臂 Warehouse 数据与训练

本实现新增 Marvin 专用 Warehouse 环境和独立运动数据生成入口，不修改
`EnvWarehouse`、`RobotPanda`、Franka 数据生成器或原有训练入口。

## 场景布局

`EnvWarehouseMarvinBimanual` 使用与单臂 Warehouse 相同的桌子/货架 primitive，重新
布置为：

- Marvin 基座前方中央为台面；
- 左侧为左臂优先访问的柜体，右侧为右臂优先访问的镜像柜体；
- 环境 limits 扩展到左右臂和底座的完整活动范围；
- PyBullet 规划和 torch_robotics 的 393 个碰撞球都继续参与碰撞检查。

环境实现位于
`mpd/torch_robotics/torch_robotics/environments/env_warehouse_marvin_bimanual.py`，
并以 `EnvWarehouseMarvinBimanual` 导出。原 `EnvWarehouse` 的 Panda-only planner 参数
和几何不变。

## 数据任务

配置文件：

```text
data_generation_cfgs/EnvWarehouse-RobotMarvinBimanual-independent.yaml
```

生成入口：

```bash
cd /home/eric/Projects/MotionPlanningDiffusion/mpd
PYTHONPATH=. conda run -n mpd-splines-public \
  python scripts/generate_data/generate_marvin_warehouse_bimanual.py \
  --config data_generation_cfgs/EnvWarehouse-RobotMarvinBimanual-independent.yaml
```

每条轨迹是完整的 14-DoF Marvin 状态，固定顺序为：

```text
[Joint1_L ... Joint7_L, Joint1_R ... Joint7_R]
```

数据生成器使用 PyBullet/OMPL 的 14 维 `RRTConnect`，并在状态采样、IK/区域过滤和
每条规划路径上做碰撞检查。目标区域位于中央台面、左柜体和右柜体的开口内。

任务分布如下：

- 50%：随机无碰撞状态 → 台面/柜体放置区域；
- 50%：台面/柜体放置区域 → 另一个放置区域；
- 双臂同时运动 : 左臂 only : 右臂 only = 3 : 1 : 1。

例如生成 1000 条轨迹时，模式计数为 600/200/200，方向计数为 500/500。奇数条数
使用前半段向上取整。

输出目录默认为：

```text
data_trajectories/EnvWarehouse-RobotMarvinBimanual-independent-v1/
```

其中包含 `dataset_merged.hdf5`、`args.yaml`、`manifest.yaml`、生成配置和摘要。除
标准 `sol_path`、`q_start`、`q_goal`、`task_id` 外，还写入 `task_mode`、`direction`、
左右 active mask、源/目标区域和规划耗时，便于按任务重采样或评估比例。

可先只检查配置和区域：

```bash
PYTHONPATH=. conda run -n mpd-splines-public \
  python scripts/generate_data/generate_marvin_warehouse_bimanual.py \
  --config data_generation_cfgs/EnvWarehouse-RobotMarvinBimanual-independent.yaml \
  --dry-run
```

## 训练

训练配置：

```text
scripts/train/cfgs/marvin_bimanual_warehouse_independent.yaml
```

训练入口只做 Marvin 14D 配置校验，然后复用已有的 B-spline、Temporal U-Net、loss、
EMA 和优化器实现：

```bash
PYTHONPATH=. conda run -n mpd-splines-public \
  python scripts/train/train_marvin_warehouse_bimanual.py \
  --config scripts/train/cfgs/marvin_bimanual_warehouse_independent.yaml
```

训练前可使用 `--dry-run` 检查 `state_dim=14`、`context_q_dim=28` 和 Warehouse 数据
目录。训练目标仍是关节轨迹控制点，环境碰撞统计由 `PlanningTask` 使用 Marvin 的
torchkin/碰撞模型完成；不会读取或覆盖 Panda checkpoint。

## 注意事项

- `Link5_R` 与 `Link7_R` 的导出 mesh 在 PyBullet 中存在永久零距离接触，因此仅在
  Warehouse 数据生成器的 PyBullet link-pair 检查中禁用该 mesh artifact；MPD 的碰撞
  球配置、torchkin 和运行时碰撞验证没有修改。
- 生成正式数据前建议先用少量轨迹验证柜体坐标和 IK 可达率，再增大到 1000 或更多。
- 当前入口是独立运动任务；双臂共同抓取仍使用现有 cooperative 数据协议，需要额外
  payload、左右抓取变换和闭链约束。
