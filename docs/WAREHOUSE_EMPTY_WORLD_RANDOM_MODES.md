# Warehouse：0 动态障碍与独立随机运动 mode

2026-10-15。适用于 `scripts.spacetime_data.generate_warehouse_spacetime` 的两个后端：`torch-forest` 与 `ompl-strrtstar`。保持已有 MPD、RRT 与 ROS 接口。

## 0 动态障碍

两个后端现在均可设置 `--obstacles 0`。静态 Warehouse 货架/桌面、自碰撞、关节限位、速度/加速度、raw 边与 c/tau-r 拟合验收仍执行；没有动态锚点或交互事件要求。仍由选定 planner 求解时空轨迹，不会自动切换成原参考路径 retime。

```bash
cd /home/eric/Projects/MotionPlanningDiffusion/mpd
conda activate mpd-splines-public

START_INDEX=0
MAX_WORLD_ATTEMPTS=12

python -m scripts.spacetime_data.generate_warehouse_spacetime \
  --backend torch-forest --device cuda:0 \
  --output-root data_trajectories_spacetime/Warehouse-TorchSpaceTime-v2-0obstacle \
  --start-index "$START_INDEX" --max-world-attempts "$MAX_WORLD_ATTEMPTS" \
  --obstacles 0 --num-trajectories 100 --query-batch-size 8 \
  --solve-seconds 5 --require-tau-r
```

使用原版 OMPL：将 backend 改成 `ompl-strrtstar`，另设新的 output-root，例如 `Warehouse-STRRTstar-v2-0obstacle`，并按 CPU 核心数设置 `--cpu-workers 6`。OMPL 搜索在 CPU worker 内；主进程可用 `--device cuda:0` 或 `cpu`。

`MAX_WORLD_ATTEMPTS` 是动态世界采样上限；0 对象世界一次即可构造。无对象时不做 FK 锚点准备，GPU earliest-arrival 仅依据关节速度可达性，不附加动态事件时间下界。最终执行时长仍满足现有 2–14 s 和运动学验收。

### 保存与读取

schema 仍为 `warehouse_spacetime_joint_v1`：

```text
world/times         [2] = [0, horizon]
world/positions     [2,0,3]
world/shape         [0]
world/size          [0,3]
world/event_times   [0]
world/anchors       [0,3]
metadata.modes      []
metadata.dynamic_known = true
metadata.object_count  = 0
```

这是已知没有动态对象，不是缺失预测。ForecastBatch 支持全空及空/非空混合 batch，通过对象 mask 保持 world 隔离，空场景仍拒绝预测时间区间之外的查询。

动态 clearance 在计算中为 +inf，质量记录 JSON 中第三项为 `null`，并附 `dynamic_collision_active=false`；静态和自碰撞 clearance 必须有限且大于 margin。不要对 `null` 做数值归一化/平均，也不要将其当作零间隙。独立磁盘 validator 与静态预览 renderer 均支持这种情况。

## 每个对象独立抽取运动 mode

默认 pool 为全部十种 mode。每个对象从 pool 独立等概率、有放回抽样，允许同一场景中重复，例如：

```text
linear + orbit + stop_go
shuttle + shuttle + vertical
```

不再按 attempt 轮换列表，也不按对象序号分配连续 mode。每个对象的形状、事件时刻仍独立随机，运动参数继续随机；事件时刻策略保持 nominal_duration 的 25%–65%，本次没有扩大事件时间窗口。

```bash
python -m scripts.spacetime_data.generate_warehouse_spacetime \
  --backend torch-forest --device cuda:0 \
  --output-root data_trajectories_spacetime/Warehouse-TorchSpaceTime-v2-3obstacle \
  --obstacles 3 --num-trajectories 100 --require-tau-r
```

`--motion-modes linear orbit stop_go` 限定抽样 pool；`--motion-modes orbit` 仍让所有对象使用 orbit，方便单 mode 统计和视频。随机抽样不会强制每批覆盖所有 mode；接受集还受到初始碰撞、搜索和拟合筛选影响。

`config.json.world_sampling.motion_mode` 记录 `iid_uniform_with_replacement_from_motion_modes`。冻结 forecast 是搜索、拟合复检和回放的共同世界。job seed 仍由 seed 与 attempt_id 派生；相同新版本与参数复现 proposal，不承诺与旧顺序 mode 版本相同。

## 旧数据与恢复

旧数据保留原 forecast 和 mode 标签，仍可读取、验收。生成器、world sampler 和拟合代码的 hash 已变化，不能用新版 `--resume` 继续旧版本数据目录；新生成使用新的 output-root。同版本和相同参数的空/非空场景恢复继续严格校验配置并保持已提交 shard。

本次只提供数据生成、验收和预览支持，没有新增 JointDual 网络或训练 loader。训练端读取 M=0 时仍需正确处理 padding mask、已知空世界状态及全空 attention。

## 验证

- 单元/回归测试：44 passed，覆盖空/混合 forecast、时间越界、静态拒绝、c/tau-r 拟合、两个后端生成/恢复流程、独立抽样及允许重复、全部单 mode 锚点检查。
- 真实 Warehouse：两个后端在 0 和 3 障碍下各生成 4 条 c/tau-r 合格轨迹，共 16 条，独立 5 ms 磁盘复检全部通过。共享 accepted attempt 的世界抽样一致；GPU/CPU 主进程 FK 锚点产生的最大数组浮点差异约 9e-8，记录见 `benchmark_results/warehouse_empty_random_modes/20261015/verification.json`。
