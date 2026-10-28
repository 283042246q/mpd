# Warehouse / ToDrawer：已知障碍运动的单次规划 benchmark

实现日期：2026-10-25。入口：[oracle_single_plan.py](../scripts/benchmarks/oracle_single_plan.py)。默认配置：[oracle_single_plan.yaml](../scripts/benchmarks/cfgs/oracle_single_plan.yaml)。

## 1. 实验定义

两个场景使用同一套生成和评测协议：给定障碍物完整运动信息，在执行起点 `t=0` 发起一次规划，生成完整轨迹。计算期间世界时钟不前进，规划耗时独立统计。无 Kalman、恒速外推、预测膨胀、滚动 replan 或多项式 handoff。

冻结的 `world.npz` 保存分段线性轨迹的全部节点，是碰撞世界的唯一依据。障碍可以穿过静态家具，也可以相互穿过；其运动不会因为物理接触而改变。机器人与静态环境、自身和动态障碍的碰撞仍需分别通过检查。

默认固定碰撞间隙为 5 mm，各模型和公共验证器一致。这个几何判定间隙与不确定性相关的障碍膨胀不同；障碍尺寸、状态协方差及膨胀率不会随预测时间增加。

## 2. 场景和起终点

| 场景 | 静态环境 | 起终点来源 |
|---|---|---|
| warehouse | `EnvWarehouseExtraObjectsV00`，包含原单次推理中的额外静态 box | [Warehouse 单次配置](../scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-config_file_v01_00.yaml)及其 regions YAML |
| todrawer | `EnvOpenDrawerShelf` | [ToDrawer 单次配置](../scripts/inference/cfgs/config_EnvOpenDrawerShelf-RobotPanda-regions-to-drawer.yaml)及其 regions YAML |

直接复用原 `get_planning_task_and_dataset` 加载流程和 `EvaluationSamplesGenerator` 的区域姿态采样、IK、状态碰撞检查、旋转规则、采样次数和最小关节距离设置。包括数据集 args 继承的 `min_distance_robot_env=0.02 m` 和原入口的 `gripper=True`，不会另写一套更容易的起终点分布。Warehouse 当前最小距离为 3.5 rad 的关节空间范数，ToDrawer 为 0.5。实际采样机器人 URDF 及净空写入 scene 元数据。

每个 case 先在没有动态障碍的同一静态环境中，通过 OMPL `RRTConnect` 求出参考路径。参考路径还需通过 Torch 连杆球模型的静态／自碰撞复验。静态求解失败与重采样原因进入 `generation.jsonl`。

参考路径用于放置障碍和定义到达事件，不作为模型 warm-start，不提供给某个特定算法。它按离散路径的均匀 phase 参数化，参考标称时长采用单次 MPD 设置的 10 秒。这里不声称参考路径已经完成动力学最优定时。

所有模型获得相同的起点和原采样的末端位姿目标。参考 RRT 所用的关节终点只保存在 `reference.npz` 中。请求采用 `goal_type=cartesian, ik_candidates=0`：旧 API 的 q_goal 占位量不提供额外目标 IK 信息，模型按照共同 EE goal 求解。目标容限统一为位置 15 mm、方向 1.5°，起终速度及加速度为零。

## 3. 障碍分布与到达模式

默认每场景 60 个 case，使用最大余数配额法先分配数量，再打乱顺序：

| 障碍数量 | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| 目标比例 | 10% | 15% | 20% | 25% | 20% | 10% |
| 每场景默认 case 数 | 6 | 9 | 12 | 15 | 12 | 6 |

比例可通过 `obstacle_count_weights` 修改；总数变化时按最大余数分配，实际配额可从 `suite.json` 审计。不会对每个模型重新随机数量。

障碍锚点在 RRT 路径的移动连杆球上选择，排除 `panda_link0/1/2`；空间锚点与事件时刻使用同一 phase，沿用 JointDual 数据生成的基本范式：

\[
t_i=\phi_i T_{ref},\qquad a_i=FK_{sphere}(Q_{RRT}(\phi_i)),\quad T_{ref}=10s.
\]

默认 phase 范围为 `[0.25, 0.80]`，按精确 phase 插值参考连杆球位置。事件时刻作为额外节点插入保存时间轴，保证障碍在该时刻精确经过锚点。

| 到达模式 | 配置权重 | 行为 |
|---|---:|---|
| simultaneous | 30% | 全部障碍使用同一到达 phase，空间锚点可属于不同连杆 |
| sequential | 40% | 不同到达 phase 至少间隔 0.06，即参考时间至少间隔 0.6 秒 |
| grouped | 30% | 两个到达组，组内同时、组间先后 |

0 个障碍标记为 `empty`，1 个障碍标记为 `single`。到达模式在这些 case 上不具有多障碍含义，因此最终多障碍子集的实际模式比例应读取报告，不把全套配额权重误称为多障碍子集的精确比例。两障碍的 grouped 是两个单成员组，与 sequential 在“组内同时”方面没有区别。

形状默认等概率采样：sphere、轴对齐 box、沿 z 轴 capsule。半径范围 `[0.04, 0.075] m`，box 半边长为 `[r,1.3r,1.1r]`，capsule 轴段半长为 0.09 m。基础速度参数范围为 `[0.22, 0.42] m/s`；它是运动公式的参数，并非所有曲线模式的最大实际速度。

运动方式默认等概率采样 10 类：linear、accelerating、speed_variation、curved、zigzag、shuttle、orbit、vertical、stop_go、helix。公式参考 [JointDual multiworld 采样器](../scripts/spacetime_data/multiworld_sampling.py)，当前独立实现位于 [oracle_sampling.py](../scripts/benchmarks/oracle_sampling.py)。

保存时间窗为 `[0,14] s`，基础间隔 20 ms，再加入精确事件节点。序列化后的分段线性轨迹定义真实世界，规划器不会重新计算某个连续解析公式而产生不同插值。

## 4. 底座与初始状态检查

从同一 RRT PyBullet 机器人读取 base link 的 AABB，针对整个 14 秒世界逐段检查。每个障碍使用包围球半径扩张底座 AABB，采用线段 slab 相交检查；拒绝任何可能穿入扩张 AABB 的整段运动。

这是保守的连续线段筛选，可能排除一些实际不碰底座的路径，但不会仅凭保存帧无碰撞就接受中间穿过底座的运动。障碍 box 和 capsule 也按包围球处理，尺寸没有被忽略。

另外拒绝 `t=0` 时与完整初始机械臂发生碰撞的世界。世界筛选只使用这些生成规则，不使用任何待比较模型的推理成败，不要求机器人等到所有障碍事件结束才到达，也不要求所有事件落在某个模型最终执行窗口内部。

## 5. 比较模型

原空间网络统一使用单次 runtime 中默认解析出的 checkpoint，并记录实际 SHA-256；不切换为其他训练权重。所有 corridor 开关关闭。

| ID | 方法 |
|---|---|
| phase4 | 固定 10 秒，原 Phase4 动态碰撞引导与选择 |
| phase5joint | 推理时空间／时间联合优化，保持旧模型先验 |
| phase4_aligned | 固定时间的动态风险引导及风险排序 |
| f1c / f1taur | F1，分别使用 c / tau_r timing checkpoint |
| f2c / f2taur | F2，分别使用 c / tau_r timing checkpoint |
| f3c / f3taur | F3，分别使用 c / tau_r timing checkpoint |

Phase4 aligned 在这里包含算法层面的引导和候选选择。ROS 中的偏离惩罚、切换 hysteresis 等闭环执行机制不属于单次 plan。

默认 timing checkpoint 与原 ToDrawer benchmark 相同：

- c：`data_trained_models/timing_diffusion/EnvWarehouse/c/warehouse-c-v2/checkpoints/step-00500000.pt`
- tau_r：`data_trained_models/timing_diffusion/EnvWarehouse/tau_r/warehouse-tau-r-v2/checkpoints/step-00060000.pt`

另外自动读取 [JointDual checkpoint 清单](../diagnostics/jointdual_inference_round1_20261025/checkpoints.json) 中全部 12 个候选，不按测试成功率再次挑选。因此总计 21 项。c 与 tau_r、KIN 与 WORLD、main 与 multiworld、补充训练配方均保留原标记。

F 系列显式启用已有的 spatial-basis adaptation，记录训练和运行时控制点维度；不伪装成在当前表示上重训的匹配模型。F3 的现有 rollout-finetuning 验证状态也保留在 engine 元数据中。

## 6. 已知轨迹如何进入各个模型

[PrescribedDynamicWorld](../mpd/inference/prescribed_world.py) 复用动态形状 SDF 与梯度，只将中心位置的查询改为冻结节点插值：

- Phase4 / aligned：在其固定时间网格查询真实障碍位置。
- Phase5 / F：按各候选自己的物理时间查询，保留对时间参数的梯度。
- JointDual：相同轨迹用于碰撞引导和验证；WORLD encoder 按训练契约读取 32 个物理时间采样的 context。

没有用当前速度替代未来轨迹，也没有在采样预算不足时回退到恒速模型。时间超出保存覆盖范围直接拒绝。

所有算法均可通过碰撞查询访问同一真实未来。KIN/F 不把世界编码进网络，是这些方法本身的结构差异；不能为了看起来“输入一样”去修改其训练网络。

JointDual checkpoint 的训练静态场景是 `EnvWarehouse`。本实验 Warehouse 使用单次配置的 `EnvWarehouseExtraObjectsV00`，ToDrawer 使用另一类柜体，因此显式记录 `static_scene_transfer=true`。通过专用 collision backend 接入真实静态场景，保留 checkpoint 的训练几何指纹核对，不重新训练、不冒充场景分布内结果。

旧在线 runtime 的默认世界仍为恒速预测；已知轨迹 backend 和可变 benchmark 预算均由 benchmark 显式注入。

## 7. 预算与公平性

默认统一候选数 64、每分支去噪步数 32、引导比例 0.3、最终 refinement 5 次、DDIM eta=0、cost 网格 128 点、最多返回 8 个候选。

旧 Phase4/5 每个后段去噪步进行一次引导，再执行 5 次最终修正；F 和 JointDual 的最终修正由各自 sampler 负责，避免重复添加。F 的空间和时间分支各自有去噪工作量，F3 还涉及交替更新，因此相同每分支步数不等于相同 FLOPs 或严格相同墙钟预算。

这是一组固定预算协议下的比较，不声称已经完成等耗时曲线。各模型的实际分支统计、guidance 统计和耗时保留在 `result.json` 及 `engine.json`。要得到预算曲线，可复制配置到新 suite，修改预算后重跑；不能在同一冻结 suite 中按模型单独增加候选数。

公平性措施包括：同一请求和 world 文件、相同 seed 规则、相同起终状态及 EE 容限、相同静态场景指纹、相同几何间隙、相同公共验证器、相同执行起始时刻。不同架构消耗随机数的顺序不同，因此配对 seed 不表示每个候选初始噪声逐元素相同。

可变时间方法共同允许 `[2,14] s`；Phase4 保留其固定 10 秒算法定义。默认终端保持时间为 0，任务在到达后结束。若增加保持，需要保证轨迹最大时长加保持时间不超过世界覆盖范围。

## 8. 公共验证与结果解释

每个模型的本地验证先按其原生方式运行。公共验证进一步检查最终选中的第一名轨迹，使用导出的 `q/dq/ddq` 构造物理时间下的分段五次 Hermite 插值，统一按最多 10 ms 间隔检查，并加入全部原生轨迹节点及执行期间的障碍节点。

检查包含静态／自／动态碰撞、关节位置、速度、加速度、起终运动边界和末端目标误差。公共验证是对统一导出执行插值的数值验证，不等于对每个模型内部连续曲线的无碰撞证明。

默认不因为公共验证失败而重新采样，也不为某个模型额外调用一次 plan。检查第一名轨迹反映该方法实际候选选择的输出；不会用隐藏的其他候选替换失败结果。

状态区分：

- `success`：本地输出存在，且通过公共验证。
- `no_valid_trajectory`：该模型没有产生本地认可的可行输出。
- `common_validation_failed`：已有输出，但没有通过更密的共同判据。
- `native_validation_failed`：原 runtime 的输出边界或导出检查拒绝结果，保留其原始容差。
- `inference_error`：请求处理、数值或接口异常。
- worker 初始化失败：保留日志及 `run_status.json`，没有推理行，不能记成已运行成功或悄悄删除模型。

报告包含按障碍数和到达模式分组的成功数、平均规划耗时、成功轨迹执行时长及逐 case 的失败原因。只在各自成功样本上求平均的时长，不能直接作为整体质量排名。

CPU 结果可验证功能并描述该 CPU 上的运行情况；不能当作 GPU 性能结论。正式计时应在生成完成后独占相应计算资源，避免把并行生成或测试的资源竞争计入算法差异。

默认 `--jobs 1` 顺序运行。CPU 可使用 `--jobs N` 加快完整成功率评测：每个 worker 分配互不重叠的物理核组，将同一物理核的 sibling threads 保持在同一组，并限制 Torch 线程数。优先选取相同且最大的 sibling 数量类别，避免本机 i9 的 SMT P 核与单线程 E 核混用。这种并行模式仍共享内存与 I/O，报告其耗时应注明资源协议；用于延迟排名时应另外顺序复测。GPU 禁止多 worker 并行。

## 9. 运行命令

在仓库根目录使用 `mpd-splines-public` 环境：

```bash
conda run --no-capture-output -n mpd-splines-public \
  python -m scripts.benchmarks.oracle_single_plan generate \
  --output benchmark_results/oracle_single_plan_20261025

conda run --no-capture-output -n mpd-splines-public \
  python -m scripts.benchmarks.oracle_single_plan audit \
  --suite benchmark_results/oracle_single_plan_20261025

conda run --no-capture-output -n mpd-splines-public \
  python -m scripts.benchmarks.oracle_single_plan run \
  --suite benchmark_results/oracle_single_plan_20261025 \
  --run-name gpu_round1 --device cuda:0 --repeats 3 --resume

conda run --no-capture-output -n mpd-splines-public \
  python -m scripts.benchmarks.oracle_single_plan report \
  --suite benchmark_results/oracle_single_plan_20261025 --run-name gpu_round1
```

先做小样本真实推理可使用 `run --case-limit 1`。它只限制每场景运行的 case 数，不改变 checkpoint 或采样预算；报告中需要保留它属于子集实验的说明。可使用 `--models phase4 phase5joint f1c` 或 `--scenes warehouse` 排查某组入口。

小预算接口检查可以在新的输出目录生成：

```bash
conda run --no-capture-output -n mpd-splines-public \
  python -m scripts.benchmarks.oracle_single_plan generate \
  --output diagnostics/oracle_single_plan_small \
  --cases-per-scene 2 --candidates 4 --steps 4 --refinement-steps 1
```

4 步等小预算结果主要用于接口检查，不能据此评价 checkpoint 的正常推理质量。主配置不会因这个命令而改变。

`generate` 拒绝覆盖已存在的 suite。`run --resume` 跳过已完成且匹配 suite/device 的逐 case 结果。重新运行时旧轨迹文件会在新请求前清除，避免失败目录残留上一次成功轨迹。若 checkpoint 或冻结世界文件变更，运行会拒绝继续。

`--run-name` 为同一冻结场景池创建独立实验目录，例如 `cpu_round1` 与 `gpu_round1`，避免设备结果混合。每个实验首次运行时冻结推理代码 hash、runtime 配置和资源设置；修改这些条件后应换一个 run-name。CPU 完整运行示例：

```bash
conda run --no-capture-output -n mpd-splines-public \
  python -m scripts.benchmarks.oracle_single_plan run \
  --suite benchmark_results/oracle_single_plan_20261025 \
  --run-name cpu_round1 --device cpu --jobs 8 --threads 1 --resume
```

## 10. 产物目录

```text
suite.json                         # 配置、全部模型与 hash、全部 case、协议
execution_contract.json            # 实际推理代码 hash、设备与冻结 runtime 参数
generation.jsonl                   # 静态任务失败与重采样审计（有失败时创建）
audit.json                         # 对冻结数据的独立配额、事件、底座扫掠与 hash 审计
warehouse/static_scene.json
warehouse/warehouse-0000/
  request.json                     # 所有模型共用的原始任务
  reference.npz                    # 静态 RRT，phase，参考时间轴
  world.npz                        # times / positions / shapes / sizes
  case.json                        # phase、事件时刻、运动公式参数、拒绝次数
todrawer/...
results/<scene>/<model>/
  runtime.yaml                     # 实际使用的参数
  engine.json                      # 实际加载的 checkpoint、场景与预热信息
  worker.log
  <case>/repeat-00/
    request.json                   # 本次 seed 和内部世界版本
    trajectory.npz                 # 仅成功导出后存在
    result.json                    # 模型原生结果
    benchmark.json                 # 公共判据与统一统计
run_status.json
report.json
report.md
experiments/<run-name>/             # 指定 run-name 后，结果与执行契约位于这里
```

`suite.json` 和每个数据文件都有内容 hash；checkpoint 同样冻结 hash。`report` 拒绝混合不同 suite 或不同设备的结果。缺失运行与已经运行失败分别呈现，主表只描述实际已完成样本。
