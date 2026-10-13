# 单臂 Warehouse batch spacetime 数据生成

2026-10-09。独立入口，不修改旧 RRT、Marvin、MPD 推理或 ROS 请求/响应接口。

## 后端与范围

| `--backend` | 实际实现 | batch 的含义 |
| --- | --- | --- |
| `ompl-strrtstar` | 原版 OMPL `STRRTstar`，保留其条件采样、rewiring 和到达时间优化 | 常驻 `spawn` CPU worker 并行独立任务，每进程有自己的机器人/world/RNG |
| `torch-forest`（默认） | 复用本仓库 `_BatchedTree` 的 Torch 时空 RRT-Connect，带共享前向树、多到达时间后向根、显式等待和根回收 | GPU 上 query × 候选边 × 边采样点同时处理；可同时处理不同动态世界 |

`torch-forest` 是 ST-pRRTC 思路启发的实用 teacher，**不是 GPU 版 OMPL ST-RRT*，不宣称原论文的理论保证**。没有将两种算法混用同一个名称。主要借鉴 [ST-pRRTC 的目标时间森林和 root recycling](https://arxiv.org/html/2609.30533v1)，并增加显式等待，不依赖高频循环近似等待。

环境严格使用 `EnvWarehouse` 的桌面、大货架、小货架；机器人为现有 `RobotPanda`、7 关节、`panda_link0` frame。没有读取 drawer 的 scene、锚点或 ROS 参数。

初始/目标 joint state 来自已有 warehouse `sol_path`。程序也读取原空间路径，用于静态有效性检查、障碍交互锚点和部分搜索 proposal；新的 q-t 解由时空 planner 重新搜索。当前入口没有重新做 IK/pose-region 采样，目标类型为 joint goal。

## 运行

在仓库根目录，使用 `mpd-splines-public`：

```bash
cd /home/eric/Projects/MotionPlanningDiffusion/mpd
conda activate mpd-splines-public
```

### GPU batch 时空 teacher

```bash
python -m scripts.spacetime_data.generate_warehouse_spacetime \
  --backend torch-forest \
  --device cuda:0 \
  --output-root data_trajectories_spacetime/Warehouse-TorchSpaceTime-v1 \
  --num-trajectories 100 \
  --query-batch-size 8 \
  --edges-per-query 16 \
  --goal-roots 4 \
  --solve-seconds 5 \
  --obstacles 3 \
  --require-tau-r
```

### 原版 ST-RRT* 多进程 batch

```bash
python -m scripts.spacetime_data.generate_warehouse_spacetime \
  --backend ompl-strrtstar \
  --device cuda:0 \
  --output-root data_trajectories_spacetime/Warehouse-STRRTstar-v1 \
  --num-trajectories 100 \
  --query-batch-size 8 \
  --cpu-workers 4 \
  --solve-seconds 5 \
  --obstacles 3 \
  --require-tau-r
```

OMPL worker 自动使用仓库 `deps/pybullet_ompl/ompl/py-bindings` 和当前 conda library path，运行环境需确有编译好的 `SpaceTimeStateSpace/STRRTstar`。CPU worker 不创建 CUDA context；主进程可用 GPU 做碰撞复检和拟合验收。纯 CPU 小试可显式设置 `--device cpu`。

2026-10-13 更新：OMPL 默认使用持续任务队列，补齐有效候选，单条结果返回后即可验收，候选准备与 worker 求解重叠。`--ompl-queue-size 0` 自动取 `max(query-batch-size, 2 × cpu-workers)`；此模式下 `query-batch-size` 控制 shard 的连续 attempt 提交数。旧批处理对照使用 `--ompl-scheduling batch`。参数、恢复与实测结果见 [WAREHOUSE_STRRT_STREAMING.md](WAREHOUSE_STRRT_STREAMING.md)。

两后端的预算含义不同：GPU 的 `solve-seconds` 是一个 batch 的搜索预算，OMPL 是每个 query 的预算。这里的命令用于生成数据，不是同预算性能对照。

默认 source：

```text
data_trajectories/EnvWarehouse-RobotPanda-config_file_v01-joint_joint-one-RRTConnect/dataset_merged_doubled.hdf5
```

可用 `--source PATH --start-index N` 更换任务来源。默认 `--source-group-size 2` 将 doubled 数据的正反两行归为同一 family，并每次前进两行；未做 reverse augmentation 的 source 应设为 1。source 必须包含 `sol_path[N,W,7]`、`task_id[N]`。

## 动态障碍如何设计

默认每个 world 有三个物体，形状依次为 sphere、轴对齐 box、z 轴 capsule；可用 `--obstacles 5` 增加数量。尺寸与方向随机，形状和完整未来位置一并保存。障碍相对静态物体和相互之间可以穿模，**这些重叠不会拒绝场景或改变运动**；机械臂相对静态环境、自身和动态障碍的碰撞仍须检查。

| 模式 | 运动 |
| --- | --- |
| `linear` | 匀速穿越 |
| `accelerating` | 沿主方向平滑变速，保持前进 |
| `speed_variation` | 周期速度变化 |
| `curved` | 前进叠加侧向正弦 |
| `zigzag` | 前进叠加三角波侧摆 |
| `shuttle` | 往返横扫 |
| `orbit` | 环绕运动 |
| `vertical` | 纵向穿越并轻微侧摆 |
| `stop_go` | 在交互点停留后继续移动 |
| `helix` | 前进叠加螺旋侧向运动 |

默认按 attempt 轮换模式，使多个场景覆盖整个集合。指定子集，例如：

```bash
--motion-modes linear curved shuttle stop_go
```

每个物体的交互锚点取自参考路径中位于 warehouse 工作区的机械臂碰撞球。工作区范围为 `[0.12,-0.78,0.08]` 到 `[0.88,0.68,1.00]` m；排除大部分底座区域。物体在设计事件时刻经过该锚点，事件时刻默认落在 nominal duration 的约 25%–65%。

默认 `--nominal-duration 8 --horizon 14`。最终接受时再次检查：**每个物体都必须在最终执行时长的 5%–95% 内发生事件，并在执行期间进入工作区。** 如果轨迹太早结束，或减速后的时长不再满足交互条件，该样本拒绝。环绕/往返模式可能多次经过工作区，不强求从工作区外起步。

这个条件保证障碍在执行期间经过工作区，不保证安全解必须与每个障碍近距离交会。planner 可以抢先、等待或绕开。

未来世界是冻结的 oracle forecast，时间从执行起点 0 开始，规划计算耗时不移动障碍。解析运动先转换成包含精确事件 knot 的分段线性预测，默认 `forecast-dt=0.02` s；保存的预测就是搜索、拟合复检和回放共同使用的世界。

## 搜索与数据验收

GPU 后端在 q-t 中执行定向最近邻与扩展，使用各关节速度下界筛选可达边。后向树增长时时间递减，最终输出时间严格递增。等待边也沿时间采样；compact 后保留 query ID，避免不同 world 混用。

第一次找到解后，搜索可以继续改进到达时刻，保留最多三条改进解；根回收只替换新的目标时间根，旧节点时间不平移。原始 q-t 解是第一阶折线 teacher，不能直接宣称满足加速度限制。

正样本必须经过：

1. source 的关节限位、静态与自碰撞检查；初始动态无碰撞。
2. raw q-t 的边界、单调时间、速度和加密碰撞检查。
3. 保留时间的混合 phase：`s=(1-eta)*chord/chord_total+eta*t/T`，`eta=0.05`。重复 q 的等待段不会直接删除。
4. 拟合 29 控制点、五阶空间 B-spline，硬约束 rest-to-rest 边界；拟合八控制点、六独立维的 c timing。
5. 检查拟合误差、joint limits、速度、加速度和 horizon。任何 slowdown 都重新查动态碰撞。
6. 2049 点 phase 检查与时间加密检查，默认 5 mm collision margin；再次验证所有物体的工作区事件。
7. 拟合并**单独复检** tau-r 的解码轨迹。`--require-tau-r` 要求两种表示都有效；否则以 c 有效为正样本，并用 `tau_r_valid` 标注 tau-r 是否可用。

两条路径中 raw search 和 c/tau-r 解码的时序可能不同；因此都保存而不覆盖。没有把不能表达的精确 dwell 当作已经拟合成功，也不会把碰撞 teacher 当作正 imitation 数据。

终点策略为“轨迹执行结束即任务结束”，没有检查到 horizon 的额外 terminal hold。当前验收是球/primitive 模型上的加密采样，不是连续碰撞证明；不包含额外 jerk 门槛或 PyBullet mesh gate。

## 输出与恢复

```text
OUTPUT/
  config.json                冻结配置、source identity、代码/机器人/静态几何 hash
  manifest.yaml              schema、机器人资产、codec 和验收范围
  robot/                     导出的 URDF、joint limits、碰撞球和 parent bounds
  shards/shard-000000.hdf5    每个 batch 原子提交，包含可变长度轨迹与 world
  index.json                 按 source family 分组的 train/validation/test 索引
  report.json                接受数、失败原因、运动模式覆盖、时长、tau-r 有效数
```

schema 为独立的 `warehouse_spacetime_joint_v1`，不冒充旧 `spacetime_mpd_v1`。单条样本包含：

```text
condition/q_start, q_goal
world/times, positions, shape, size, event_times, anchors
raw/solution_00, ...          [N,8]，最后一列为时间
raw/fitted_solution_index
spatial/control_points       [29,7]
spatial/knots
timing/control_points        [8]，c 的端点成对约束
timing/tau_r                 [6]，使用前检查 tau_r_valid
timing/T_min
timing/reference_phase, reference_time
execution/q, dq, ddq, time   最终 c codec 的时间均匀执行参考
```

shape 编码：0 sphere，size=`[radius,0,0]`；1 box，size 为 half extents；2 z-capsule，size=`[radius,half_length,0]`。位置单位 m、关节 rad、时间 s；障碍朝向固定，不隐含时间变化的旋转。

质量、planner statistics、motion modes、seed、source row/family、tau-r 是否有效保存在 sample attrs。normalizer 尚未拟合；新增网络的 loader 应使用此 schema，不能直接把它交给旧 timing loader。无效 tau-r 不作为训练目标。

`num-trajectories` 是最终合格样本数。默认最多尝试其 20 倍任务，可用 `--max-attempts` 调整；预算耗尽返回退出码 2，已接受样本保留，不能把它误当作完成目标。失败原因进入 report。

恢复时使用相同生成参数加 `--resume`，可增加 `num-trajectories` 或 `max-attempts`。一个 output 只允许一个 writer。每批 shard 原子重命名；index/report 默认每 10 批和退出前更新。恢复时以完整 shard 为准，重建可能滞后的索引，不重复写已提交样本。`.inprogress` 不算完整 shard。

source、代码、资产、算法参数不一致时恢复会拒绝，应使用新的 output root。GPU seed 的复现范围是固定 batch 配置/任务组成；OMPL 记录进程 RNG seed 与 job seed，不能保证改变 worker 数或调度后轨迹逐点相同。

## 独立复检与查看

```bash
python -m scripts.spacetime_data.validate_warehouse_spacetime \
  data_trajectories_spacetime/Warehouse-TorchSpaceTime-v1

python -m scripts.spacetime_data.render_warehouse_spacetime \
  data_trajectories_spacetime/Warehouse-TorchSpaceTime-v1 \
  --sample 0 --output /tmp/warehouse-spacetime-preview.png
```

validator 从磁盘重建 world、c/tau-r 轨迹与碰撞模型，以更细的 5 ms 时间采样复检，同时检查 raw 边、边界、索引和 source-family split。renderer 展示 warehouse 静态几何、机器人、末端路径和动态障碍轨迹。

测试命令：

```bash
python -m pytest -q \
  tests/test_warehouse_spacetime_planner.py \
  tests/test_warehouse_spacetime_fit.py \
  tests/test_warehouse_spacetime_generation.py \
  tests/test_gpu_batch_rrt.py
```

2026-10-09 在 `mpd-splines-public` / RTX 4090 D 上完成最终版本验收：

| 验收 | 结果 |
| --- | --- |
| 新增测试与原 GPU RRT 回归测试 | 16 passed |
| `Warehouse-BatchSpaceTime-verify-20261009-gpu` | 26 次尝试，12 条接受；磁盘独立复检 12/12；c 和 tau-r 均通过 |
| `Warehouse-BatchSpaceTime-verify-20261009-ompl` | 7 次尝试，4 条接受；磁盘独立复检 4/4；c 和 tau-r 均通过 |
| 运动覆盖 | GPU 接受样本覆盖全部 10 种运动、3 种形状 |
| 真实输出恢复 | 两后端 `--resume` 成功，已提交 shard 的 SHA256 全部保持一致 |
| 中断恢复与写入互斥 | 测试覆盖提交 shard 后丢失索引、扩展目标数、配置不一致及第二 writer 拒绝 |

这两份数据位于 `data_trajectories_spacetime/`。GPU 使用 query batch 4、256 iterations、3 s/batch；OMPL 使用 query batch 2、2 workers、2 s/query、start-index 2。两者均启用 `--require-tau-r`。预览位于 `benchmark_results/warehouse_spacetime/preview.png`。

这些小批用于功能与数据正确性验收，不能替代大规模 soak 或同任务、同预算的吞吐比较。小数据集按 family 哈希划分，某个 split 可能为空；生成训练规模数据后应检查各 split 的数量与场景覆盖。
