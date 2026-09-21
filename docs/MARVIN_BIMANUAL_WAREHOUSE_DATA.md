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

数据生成器使用 PyBullet/OMPL 的 14 维 `RRTConnect`，并在状态采样、末端区域过滤和
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

Warehouse v3 训练入口会先严格校验 manifest、完整 HDF5 hash、机器人资产、双 EE schema
和已验证 B-spline 标记。校验通过后，首次加载会按 HDF5 chunk 批量读取序列化控制点，
并跳过生成阶段已经完成的重复碰撞统计。预处理缓存使用同目录临时文件完整写入后原子
替换；中断不会留下被误认为有效的半成品 pickle。后续运行会直接复用文件名包含
`marvin_dual_ee_v2_batched` 的缓存。

### 多电脑数据集合并与精确去重

多电脑生成目录存在重叠时，在合并命令中显式增加：

```bash
--deduplicate-exact-trajectories
```

该选项先以 `sol_path` 的 BLAKE2b-256 哈希定位候选，再逐字段精确比较路径、起终点、
B-spline、EE goal、mask、任务模式、方向、区域和路径长度。`task_id` 和
`planning_time` 仅作为来源信息，不参与轨迹语义判重。因此，完全相同的后续副本只保留
一次，而相同原始 task ID 下由随机规划得到的不同路径都会保留。建议先同时加
`--dry-run` 核对 `exact_duplicates_dropped`，确认后仅去掉 `--dry-run` 正式写出。
去重明细（被删除行、保留行、原 task ID 和摘要）会写入输出的
`merge_report.json` 和 `manifest.yaml`。

### 正反轨迹增广

先完成所有 shard 和多电脑数据集合并，再对最终 Warehouse v3 数据集运行：

```bash
PYTHONPATH=. conda run -n mpd-splines-public \
  python -m scripts.generate_data.augment_marvin_warehouse_reverse \
  data_trajectories/EnvWarehouse-RobotMarvinBimanual-independent-v3-gpu-combined \
  --output-dir \
  data_trajectories/EnvWarehouse-RobotMarvinBimanual-independent-v3-gpu-combined-reversed \
  --dry-run

# 确认输入、输出和轨迹数量后去掉 --dry-run
```

该入口以新目录原子写出两倍数据，源数据不变。它同步反转原始路径、B-spline knot/
控制点，交换关节起终点和 source/goal region，更新 direction，并用规范 Marvin/Pika
模型重新计算两个 TCP 的 goal pose。正反行相邻且共享原 `task_id`；训练器按 `task_id`
划分 train/validation，因此一对轨迹不会跨集合。输出还包含
`augmentation_pair_id`、`augmentation_is_reversed` 和 `augmentation_source_row` 供审计。

不要对 Marvin v3 使用通用的 `flip_solution_paths.py`。也不要在增广后继续调用多数据集
合并器；后者要求输入 task ID 唯一。正确顺序是：每机 shard 合并 → 多电脑数据集合并 →
正反增广 → 训练。

## 注意事项

- `Link5_R` 与 `Link7_R` 的导出 mesh 在 PyBullet 中存在永久零距离接触，因此仅在
  Warehouse 数据生成器的 PyBullet link-pair 检查中禁用该 mesh artifact；MPD 的碰撞
  球配置、torchkin 和运行时碰撞验证没有修改。
- 生成正式数据前建议先用少量轨迹验证柜体坐标和 IK 可达率，再增大到 1000 或更多。
- 当前入口是独立运动任务；双臂共同抓取仍使用现有 cooperative 数据协议，需要额外
  payload、左右抓取变换和闭链约束。
