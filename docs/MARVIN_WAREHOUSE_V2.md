# Marvin 双臂 warehouse：独立运动数据与训练

本轮对照 `generate_trajectories.py`、Panda warehouse 的 pose-region YAML、
`launch_generate_trajectories.py` 和共享 MPD trainer，修正 Marvin/Pika 的独立运动链路。
单臂场景、单臂数据生成入口和原有 v1 数据保留；新数据使用 v2 目录。

## 场景与可达区域

场景类为 `EnvWarehouseMarvinBimanual`，版本 `marvin_warehouse_v2`。
两侧书架朝内，中央桌面范围 x=[0.30,0.80]、y=[-0.50,0.50]、z=-0.16 m。
书架第一可用格层的净空为 z=[0.015,0.40] m，替代旧配置跨越多个隔板的连续 z 范围。
所有坐标相对机器人 `base_link`，末端使用真实资产中的左右 **Pika TCP**，不使用 flange 代替。

场景预览：[双臂进入书架的碰撞有效姿态](../benchmark_results/marvin_warehouse_scene_v2.png)。

| TCP region | x (m) | y (m) | z (m) | 相对姿态 XYZ (度) |
|---|---|---|---|---|
| left_table | 0.54–0.66 | 0.35–0.45 | 0.18–0.28 | 每轴 ±15 |
| right_table | 0.54–0.66 | -0.45–-0.35 | 0.18–0.28 | 每轴 ±15 |
| left_cabinet | 0.36–0.48 | 0.66–0.72 | 0.14–0.24 | 每轴 ±15 |
| right_cabinet | 0.34–0.40 | -0.73–-0.68 | 0.18–0.24 | X/Z ±15，Y 0–15 |

沿用单臂的姿态约定：`R_target = R_base @ EulerXYZ(angles)`。
桌面接近方向为 TCP +Z 朝世界 +X；左右书架分别朝世界 +Y/-Y。
这些是接近/预抓取的 TCP 体积，不是物体底面坐标，也不代表已完成夹爪闭合或抓取执行。
完整旋转矩阵见 `data_generation_cfgs/EnvWarehouse-RobotMarvinBimanual-independent.yaml`。

右臂的腕部限位、自碰撞几何不能简单镜像左臂。初始右书架框的边界测试只有 9/15 成功；
收紧后通过 15/15。左右桌面及左书架也各通过 15/15。
测试包含 8 个内缩 1 mm 的位置角点、中心、以及中心处 6 个姿态轴向极值。
原始解和误差记录位于 `benchmark_results/marvin_warehouse_region_audit.json` 及
`benchmark_results/marvin_right_region_audit_v2.json`。
有限测试不能证明盒内所有位置与姿态的笛卡尔积均可行，尤其另一臂改变后仍可能相碰；
生成时对每个端点重新执行完整筛选。

随机区域比固定 region 宽：x=[0.15,0.85]、z=[0.05,0.55]，
左 y=[0.25,0.95]、右 y=[-0.95,-0.25]，姿态不设限。
分别随机采样两臂关节角，经 FK 和完整双臂碰撞筛选后组合成随机起点；
左右采样顺序随机，避免同时抽取两臂再拒绝时两个低命中率相乘。零位只是中间参考，不作为返回兜底。

## 两种生成方法与数据约束

- `joint_fk`：在活动臂的关节限位内均匀采样，FK 后检查位置、相对欧拉角、网格和球体碰撞。
- `region_ik`：在同一 region 内采样位姿，用带关节限位的最小二乘 IK 求解，
  再执行与 FK 方法相同的筛选。默认一半初值来自参考状态、一半均匀随机，
  可用 `ik_reference_seed_fraction: 0.0` 对照纯随机重启。

IK 的停止残差不能代替最终 FK 检查：求解后仍须落在 region 内。
另一臂在端点求解时固定，最终检查整个 14 维状态。
Pinocchio、PyBullet、TorchKin 的公开顺序全部检查为左 7 + 右 7。

每 10 条**成功写入**的轨迹组成一个配额块：5 条随机区域到固定区域，5 条固定区域到另一固定区域；
两类方向内部都保留原双臂入口的双臂同时独立运动/仅左/仅右 = 3:1:1。
所以同时运动总占 60%，仅左、仅右各占 20%。独立运动不施加共同持物的相对位姿约束。

每组采样起终点只调用一次 RRTConnect。没有 exact solution（包括 approximate solution）即丢弃，
下一次使用新起终点；不对原起终点重复 RRT。失败不改变目标数据配额。
`max_attempts_per_trajectory` 只是新任务采样总预算的倍率。

仅左/仅右任务在 7 维子空间内规划，整条轨迹的未激活臂保持固定；双臂任务在 14 维空间规划。
RRT 扩展、最终稠密路径检查均使用网格与球体碰撞标准。
沿路径按最大单关节增量 0.025 rad 插值审查；这属于离散稠密检查，不是连续碰撞检测证明。
RRT 的时间预算沿用单臂 10 秒，扩展步长显式设为 0.35 rad，避免 14 维空间默认约 3.47 rad 的大步扩展。
100 个固定端点消融后，有效性分辨率默认设为 0.002；保持完整 1035 个碰撞球，并禁用
PathSimplifier。除非显式实验配置重新开启，否则生产数据生成不会执行路径简化。
三组 100 对端点配对消融也表明当前 clearance/稀疏直线预筛选会大量误拒绝最终有效轨迹，
没有提高最终轨迹/分钟，因此 `pre_rrt_filter` 默认保持 `none`。完整结果见
[MARVIN_PRE_RRT_FILTER_ABLATION.md](MARVIN_PRE_RRT_FILTER_ABLATION.md)。
OMPL 在一次状态/边检查期间不会中断，实际 solve 时间可能略超预算。

B-spline 在数据生成阶段拟合：degree=5、22 个控制点、端点零速度/零加速度；
至少 512 个采样点及其间的关节插值经过同样的碰撞和限位检查后，保存 knots/coefficients/degree。
训练直接加载这些已验证的控制点，防止“折线路径无碰撞，但训练时重新拟合后穿板”。
HDF5 同时保存模式、方向、左右 source/goal region、活动关节 mask、真实 task_id 和起终点。
manifest 保存数据文件、URDF、资产版本的校验信息。

保留原配置中的 PyBullet `Link5_R`/`Link7_R` 网格接触例外；球体自碰撞仍检查该对。
没有新增碰撞豁免、缩小球半径或降低 0.02 m 环境余量。

碰撞 FK 优化只改变计算方式：用真实连杆姿态批量变换球心，避免遍历一千多个固定球体坐标系。
32 组随机状态与原始 FK 的最大误差为 1.19e-7 m；本机 100 次单状态 FK 为 0.193 s，原来为 2.126 s。
这不是整个规划器的加速倍数。

## 运行

在算法仓库根目录，并使用已有 `mpd-splines-public` 环境：

```bash
conda activate mpd-splines-public
source set_env_variables.sh

# 检查配置，不生成数据
python -m scripts.generate_data.launch_generate_marvin_warehouse_bimanual --dry-run

# 默认 1000 条，3 个 CPU workers，500 条/分片，BLAS 每进程 1 线程
python -m scripts.generate_data.launch_generate_marvin_warehouse_bimanual

# 使用另一种端点采样方法，另存目录
python -m scripts.generate_data.launch_generate_marvin_warehouse_bimanual \
  --sampler joint_fk --output-dir data_trajectories/EnvWarehouse-RobotMarvinBimanual-independent-v2-fk

# 显式选择实验性预筛选；不传此参数时为 none
python -m scripts.generate_data.launch_generate_marvin_warehouse_bimanual \
  --pre-rrt-filter endpoint_clearance --output-dir data_trajectories/marvin-clearance-test

# 有限位姿边界审查
python -m scripts.generate_data.validate_marvin_warehouse_regions \
  --output benchmark_results/marvin_regions.json

# 可复现的对比入口；输出目录必须是新目录
python -m scripts.generate_data.benchmark_marvin_warehouse_sampling \
  --output-dir benchmark_results/marvin_comparison \
  --seeds 41 42 43 --endpoint-seconds 10 --task-seconds 15 --tasks 10 --equal-task-time

# 同一批 100 个端点对上的三组预筛选配对消融
python -m scripts.generate_data.benchmark_marvin_pre_rrt_filter \
  --count 100 --workers 3 --seed 74 \
  --output benchmark_results/marvin_pre_rrt_filter_seed74

# 小规模生成/训练链路检查，避免先启动百万步训练
python -m scripts.generate_data.launch_generate_marvin_warehouse_bimanual \
  --num-trajectories 20 --tasks-per-shard 10 --workers 3 \
  --output-dir data_trajectories/EnvWarehouse-RobotMarvinBimanual-independent-v2-smoke-new
python -m scripts.train.train_marvin_warehouse_bimanual \
  --config scripts/train/cfgs/marvin_bimanual_warehouse_independent.yaml \
  --dataset-subdir EnvWarehouse-RobotMarvinBimanual-independent-v2-smoke-new \
  --device cpu --num-train-steps 2 --no-summary --results-dir /tmp/marvin_smoke_new

# 验证数据/资产/样条契约，再启动训练
python -m scripts.train.train_marvin_warehouse_bimanual \
  --config scripts/train/cfgs/marvin_bimanual_warehouse_independent.yaml --check-dataset
python -m scripts.train.train_marvin_warehouse_bimanual \
  --config scripts/train/cfgs/marvin_bimanual_warehouse_independent.yaml
```

生成数量、分片上限和 worker 生命周期必须为 10 的倍数以保证两种配额。
实际小分片上限取 `tasks_per_shard` 与 `worker_lifetime_trajectories` 的较小值；默认后者为 10，
所以 1000 条会形成 100 个短生命周期分片。本例 20/10/3 只有两个分片，因此实际使用 2/3 个 worker。
并发输出带有 `[shard 000000000]` 前缀；各分片分别从 0/分片总数计数，不能把交错日志当成全局进度。
完成的分片按配置及文件 hash 跳过；输出先写入带 PID 的隐藏 staging 目录，完整后才原子改名。
采样预算耗尽或 native worker 退出只会重算当前十条块，不会损坏已完成分片。
合并文件通过临时文件完成后改名，task_id 跨分片不重号。
要重新生成已完成的数据，请选新输出目录。

## 网络与训练评析

保留联合 14 维 Temporal U-Net 和 28 维 `q_start + q_goal` 条件。
时序卷积会混合两臂关节通道，能够表示相互避碰关系；独立任务不意味着必须拆成两个互不通信的网络。
`unet_input_dim=32` 是卷积特征宽度，不是机器人关节数。
22 个控制点减去 6 个边界固定点后得到 16 个可学习点，满足当前 U-Net 的下采样长度约束。
在没有充分训练和验证集证据前，不据此宣称 32 通道一定够用或盲目增加网络规模。

修正的训练问题：

- 拒绝把只有一个 TCP 的共享 EE 编码器当作双臂目标编码器；本阶段使用关节起终点条件。
- 检查真正加载的 robot/state/context 维度为 14/14/28，而不只检查 YAML 里的数字。
- 训练前校验环境版本、数据和资产契约、样条 degree/控制点数；旧 v1 数据不能直接作为新场景数据训练。
- 直接调用训练函数时补全并保存默认参数，保证 checkpoint 带有完整扩散和网络配置。
- Pika 大量碰撞球使原来 2000 条轨迹的碰撞统计批次不合适，Marvin 改为小批次；训练统计同时报告两 TCP 及静止臂漂移。
- 开启梯度裁剪。修正不足一个 batch 时的 epoch 计算，以及达到训练步数后外层循环未立即停止的问题。

扩散先验本身不能硬保证静止臂、无碰撞或共同持物约束；这些仍需推理阶段的约束投影和规划检查。
当前 `inference_marvin_bimanual.py` 还是插值占位入口，不是已接入 checkpoint 的完整 MPD 推理，
不能把本轮的数据/训练验证表述为 ROS 实机闭环验证。
共同协作抓取需要另行实现物体状态、双 TCP/相对位姿条件与闭链轨迹约束，本轮未混入独立任务数据。

详细采样测试数据和最终结论见同目录 `MARVIN_WAREHOUSE_SAMPLING_RESULTS.md`。
Panda/Marvin 生成阶段耗时对照见 `PANDA_MARVIN_GENERATION_TIMING.md`。

## 原生 worker 崩溃与恢复

OMPL/PbOMPL/PyBullet 都包含原生 C/C++ 对象。实际长时间生成中曾出现
`malloc(): unsorted double linked list corrupted`，随后 Python 主进程只能观察到
`BrokenProcessPool`。这不是 RRT approximate solution 本身的正常失败，而是 worker 非正常退出；
没有 core dump 时不能进一步断言具体是 OMPL、绑定层还是 PyBullet 的写越界。

launcher 因此默认令每个新 spawn 的 worker 只生成一个完整的 10 条配额块后退出，
并原子发布小分片。某个进程 native abort 时，只重试缺失块；其他完整块可在再次执行同一命令时直接复用。
相关参数为：

```yaml
worker_lifetime_trajectories: 10
max_worker_restarts_per_shard: 3
```

`tasks_per_shard` 仍是上限，实际分片上限为它与 `worker_lifetime_trajectories` 的较小值。
150000 条、默认安全值会产生 15000 个十条小分片，最后再合并为一个 HDF5；这增加进程启动和文件数量，
但把 native 生命周期和单次故障损失限制在一个配额块内。失败进程的隐藏
`.000000000.incomplete-<pid>` staging 目录会保留用于排查，不会被当成完整分片。
若停止/掉电留下了已编号但校验失败的目录，恢复时会先将其改名为
`.000000000.corrupt-<time>-<pid>` 隔离，再以相同 task_id 重生成，不会静默删除故障证据。

150000 条的正确命令中，输出目录必须位于同一个 shell 参数内：

```bash
python -m scripts.generate_data.launch_generate_marvin_warehouse_bimanual \
  --num-trajectories 150000 --tasks-per-shard 500 --workers 3 \
  --output-dir data_trajectories/EnvWarehouse-RobotMarvinBimanual-independent-v3-res002-nosimplifier-150k
```

如果在 `independent-` 后直接换行且上一行末尾没有反斜杠，shell 会先把截断路径传给
`--output-dir`，再把下一行的 `v3-res002-nosimplifier-150k` 当作新命令，因而报告“未找到命令”。
