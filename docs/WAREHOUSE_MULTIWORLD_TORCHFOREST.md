# Warehouse 按 family 一次生成 0～5 障碍物轨迹

独立入口：`scripts/spacetime_data/generate_warehouse_multiworld.py`。
启动脚本：`scripts/spacetime_data/run_warehouse_multiworld.sh`。

这里按列出的 **0、1、2、3、4、5 共六个 world** 实现。每个 world 目标保留一条验收通过的轨迹，无额外早/晚变体。

## 1. 抽样与方向

- 默认读取已随代码导出的 `scripts/spacetime_data/assets/warehouse_solved_families_ge3_v1.npz`，当前为 **104,319** 个 family，约 157 KiB。生成时不再打开历史动态数据目录。
- 清单来自历史 `Warehouse-TorchSpaceTime-v1-{3,4,5}obstacle` 三个目录的成功 family 并集；1/2 障碍目录不参与。旁边的同名 `.json` 保存源数据身份、各目录数量、split 统计和来源哈希。统计为 train 83,650、validation 10,214、test 10,455。
- `--num-families X` 从并集均匀、无放回抽取 X 个；不会按历史成功轨迹条数给 family 加权。
- 默认检查清单格式、family ID 排序/去重/范围/内容哈希，以及源文件大小/mtime、源行数和双向分组大小。来源中的旧目录路径仅作为记录，不需要存在。源文件内容身份沿用已有大小/mtime 合约，不新增全文件内容认证。
- 同一 seed 和候选池，抽样结果可复现；较小 X 是较大 X 抽样清单的前缀。生成过程中不补换失败 family。
- `--direction random` 为每个 family 独立指定正向或反向，**该 family 的六档全部使用同一个方向**。总体约一半正向一半反向，不保证精确各半。
- `--direction forward` 或 `reverse` 可以使全部 family 使用指定方向。
- source 行号为 `2*family_id + direction`，0 为正向，1 为反向。读取该行参考路径，重新创建 world 并规划，不反转已有动态轨迹。
- 沿用旧数据的 family 哈希 80/10/10 split；随机抽样不保证各 split 精确占比。同 family 的全部新旧样本归属相同 split。

抽样结果保存在 `families.json`，包含 family、方向、source row、source task ID 和 split。

### 预先统计与跨机器使用

已完成一次统计并提交清单，通常不必再执行导出。其他机器只需此 NPZ、原始 `dataset_merged_doubled.hdf5` 参考路径与原本依赖的机器人/静态环境资产，不需要复制五个旧动态数据目录。生成仍须从原始 HDF5 读取参考轨迹及起终点；清单不包含参考路径。

若历史数据更新，重新导出只读取三个旧目录的 `config.json`、`index.json`，不读取动态轨迹 shards、不运行 GPU 规划：

```bash
python -m scripts.spacetime_data.warehouse_family_catalog
```

可加 `--output /path/to/catalog.npz` 另存，再通过 Python `--family-catalog /path/to/catalog.npz` 或 Bash `FAMILY_CATALOG=/path/to/catalog.npz` 使用。NPZ 内含完整来源元数据，运行时不依赖旁边的统计 JSON。

仍可通过 `--eligible-roots DIR3 DIR4 DIR5` 恢复现场扫描历史目录的方式；它与显式 `--family-catalog` 互斥。现场扫描时会检查旧 config/index 哈希与 source 合约。

## 2. 障碍物与时间政策

固定 `nominal_duration=8s`，预测 `horizon` 默认 14s；**最终轨迹时长仍然可变**。

| 障碍数 | 每对象独立采样的 phase fraction | 事件时间 |
|---|---|---|
| 0 | 无 | 无 |
| 1、2 | U(0.25, 0.65) | 2～5.2s |
| 3、4、5 | U(0.25, 0.80) | 2～6.4s |

同一个 fraction 同时用于选择参考 waypoint 对应的空间 anchor，以及计算 `event_time = 8 * fraction`。参考 FK 最多使用 64 个均匀 waypoint 索引；不额外 retime，不按实际参考运行时间或弧长计算比例。

其余政策与原程序一致：

- mode 对每个对象独立、等概率、有放回采样，形状独立随机；保留原来的十种运动公式、速度和尺寸范围。
- 物体与静态环境/其他物体相交视为穿模，不修改其运动；机械臂仍检查静态、自碰撞与动态碰撞。
- scene 必须初始无碰撞；每槽位最多 `MAX_WORLD_ATTEMPTS` 次 scene 候选，使用第一个通过初始检查的 world。
- 有效 world 仅提交一次规划；求解失败或拟合失败后不自动换 world 重来，不降障碍数。
- 原始路径的速度/边界/加密碰撞检查、P29/c8 拟合、可选 tau-r 独立检查均复用原函数。
- GPU 原始规划的最早终点时间保留 `max(event_times)+0.25s` 下界与关节速度下界；最终所有事件仍须在 `(0.05T, 0.95T)`，且障碍物在运行期间进入 workspace。
- 0 障碍仍走 torch-forest 静态规划，再做相同拟合与验收；不会自动改成 RRT+retime。

独立 sampler `multiworld_sampling.py` 保留原运动公式，仅扩展高数量的 fraction 范围。原 `warehouse_world.py` 和原生成器文件均保持不变，避免改变旧程序的 software hash 和 resume 合约。

## 3. GPU 组批

六个队列分别对应 M=0～5，每队列按冻结清单顺序跨 family 提取任务。每轮依次从每个非空队列生成一个 batch，然后继续下一轮。

```text
第 1 轮：M0 满 batch → M1 满 batch → … → M5 满 batch
第 2 轮：M0 满 batch → M1 满 batch → … → M5 满 batch
……直到所有 family/count 槽位处理完
```

- scene 无效的槽位记失败，并继续从该数量队列取任务，尽量补齐有效 batch。
- **一个 batch 只有一种障碍数量**，不产生跨数量 padding；各队列末尾允许不足一个 batch。
- 同一进程复用机器人和静态环境；参考路径静态检查与 anchor FK 使用有界缓存。
- `--solve-seconds` 是每 batch 的规划预算，不是每条轨迹的预算，也不包含准备、拟合和写盘时间。
- Python 与 Bash 均默认 batch=128、collision_chunk=32768、edges=16、roots=4、solve=10s、max_world_attempts=12；max_iterations=512。Bash 可通过 `QUERY_BATCH_SIZE`、`COLLISION_CHUNK`、`EDGES_PER_QUERY`、`GOAL_ROOTS`、`SOLVE_SECONDS`、`MAX_WORLD_ATTEMPTS` 分别覆盖。
- 不额外增加候选预取或多进程 GPU 并发。

## 4. Seed 与恢复

world seed 按 `global seed + source namespace + family + direction + obstacle count + scene draw` 派生，独立于队列顺序和 batch size。不同数量的 world 不是嵌套物体集合。

solver 使用独立 namespace 派生每任务的 seed key。当前底层 forest 仍是 **每 batch 一个 RNG**，实际使用该 batch 第一个任务的 solver seed，保存为 `batch_seed`；因此更改 batch 组成、GPU 或时间预算，不保证原始解完全相同。`solver_seed` 保存任务的派生 key，`batch_seed` 才是该次实际传入规划器的 seed。

每个 batch 将成功样本、失败槽位和计数一起原子提交到 shard。恢复时从已提交 shard 的 `task_outcomes` 重建六个队列游标与 report/index；未提交 batch 重新计算。已提交成功或失败槽位都不会重复执行。

`--resume` 要求源文件、候选清单文件（现场扫描模式为历史 config/index）、X、方向、所有生成配置与代码哈希一致。改变 X 或其他参数需使用新输出目录。

## 5. 运行命令

从 MPD 根目录运行：

```bash
conda activate mpd-splines-public

NUM_FAMILIES=10000 \
DIRECTION=random \
QUERY_BATCH_SIZE=128 \
MAX_WORLD_ATTEMPTS=12 \
SOLVE_SECONDS=10 \
SEED=20261015 \
OUTPUT_ROOT=data_trajectories_spacetime/Warehouse-TorchForest-Multiworld-f10000-v1 \
bash scripts/spacetime_data/run_warehouse_multiworld.sh
```

这是 10,000 个 family、**60,000 个目标槽位**，最终合格轨迹可能少于此数。bash 默认要求 tau-r 验收；直接 Python CLI 的 `--require-tau-r` 为可选项。

统一正向只需 `DIRECTION=forward`，统一反向使用 `DIRECTION=reverse`。

先检查抽样清单而不初始化 GPU：

```bash
NUM_FAMILIES=10000 \
OUTPUT_ROOT=data_trajectories_spacetime/Warehouse-TorchForest-Multiworld-f10000-v1 \
bash scripts/spacetime_data/run_warehouse_multiworld.sh --dry-run
```

同一参数正式启动或中断后恢复，追加 `--resume`：

```bash
NUM_FAMILIES=10000 \
OUTPUT_ROOT=data_trajectories_spacetime/Warehouse-TorchForest-Multiworld-f10000-v1 \
bash scripts/spacetime_data/run_warehouse_multiworld.sh --resume
```

默认启动不需要旧动态数据目录。可通过 `FAMILY_CATALOG` 指定另一个已导出的清单；或通过 `--eligible-roots DIR3 DIR4 DIR5` 现场扫描其他历史成功目录，但必须满足相同 source 合约且障碍数至少 3。

## 6. 输出、失败与验收

输出沿用 `warehouse_spacetime_joint_v1` 数据结构，可使用原有 validator 和 renderer：

- `config.json`：新入口的独立政策、软件/资产哈希与候选来源。
- `families.json`：冻结抽样及方向清单。
- `shards/*.hdf5`：原有成功轨迹结构，增加 `world_id`、`obstacle_count`、`direction`、seed 与 scene draw 属性；同 shard 中的 `task_outcomes` 包含所有已处理槽位。
- `index.json`：仅合格样本，按 family 分组划分。
- `report.json`：总体及各障碍数成功/失败/待处理数、batch 大小、阶段耗时、完整 family 数和每 family 成功条数分布。

失败状态：`invalid_scene`、`teacher_unsolved`、`fit_rejected`。teacher 未解不等于不可行；没有监督轨迹的槽位不会出现在训练 index 中。

`run_finished=true` 表示所有目标槽位已处理；`complete=true` 表示所有槽位均成功。退出码 0 为全部成功/成功 dry-run，2 为有限预算处理完但存在失败；其他异常正常抛出。再次 resume 不会把这些失败自动重跑。

独立读取磁盘并验收：

```bash
python -m scripts.spacetime_data.validate_warehouse_spacetime \
  data_trajectories_spacetime/Warehouse-TorchForest-Multiworld-f10000-v1 \
  --device cuda:0 --limit 100
```

正式合并训练时仍使用 object mask；如把 family 六槽位组成固定 tensor，另设 solution-valid mask。普通逐条训练只读取 index 中的合格样本。

## 7. 实际验证（2026-10-15）

38 项相关测试通过，覆盖 sampler 低数量与原版完全一致、高数量的 phase/event 范围、family 无放回抽样、方向、有效 batch 补齐、有限失败记录、原子 shard、断点恢复和 dry-run。原生成器与原 sampler 无修改。

开发期间进行了真实 GPU 小样本，全部要求 tau-r：

| 候选池 / 配置 | 0 / 1 / 2 / 3 / 4 / 5 障碍成功数 | 合格合计 | 磁盘独立 5ms 复验 |
|---|---|---|---|
| 3/4/5 并集，4 family，随机方向，batch=4，默认 10s 预算 | 4 / 4 / 3 / 3 / 1 / 0 | 15/24 | 15/15 |
| 历史 5 障碍池，4 family，全正向，batch=4，5s 预算 | 4 / 3 / 3 / 3 / 2 / 2 | 17/24 | 17/17 |

这些仅用于确认入口、六种数量、导出和验收链路，不用于估计总体成功率或 GPU 吞吐；两个候选池和配置不同。有限失败均保留，未启动正式数据生成。

验证产物位于 `benchmark_results/warehouse_multiworld_smoke/20261015/`，对应目录 `gpu-f4-default-budget`、`gpu-f4-five-obstacle-pool`，独立复验记录分别为 `verification-default-budget.json`、`verification-five-obstacle-pool.json`。开发期间的代码快照哈希保存在各自 config 中；按严格合约，当前代码与快照不一致时不能直接 resume 这些 smoke 输出，正式运行使用新目录。

清单解耦另有两项测试：移走历史目录后，抽样清单与原扫描方式完全相同，仍可完成生成和 resume；源文件不匹配或清单内容损坏会拒绝使用。导出的真实 104,319 个 ID 还与前次统计的 `eligible-families.npz` 全量对照。
