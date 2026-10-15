# Warehouse 同任务多世界轨迹生成方案

日期：2026-10-15。状态：设计方案，尚未实现 family/world 任务调度入口，未启动新一轮正式生成。当前生成器已支持两个后端的 0 动态障碍和各对象独立 mode 抽样；本方案新增的 family 清单、world 槽位、成对事件变体与 seed 分层仍需实施。

## 1. 候选池：历史 3/4/5 障碍至少成功一次的 family

这是合适的优先生成池：参考路径已经通过当前静态/self/limit 门槛，且对应任务至少在一个较高障碍世界下得到过 c/tau-r 合格解，能减少重新筛选明显无效任务的成本。

统计来自旧五目录 index/config，而非只按 source 行号去重：

| 项目 | family 数 |
|---|---:|
| 候选并集 | 104,319 |
| train | 83,650 |
| validation | 10,214 |
| test | 10,455 |
| 仅正向成功 | 70,569 |
| 仅反向成功 | 15,950 |
| 两个方向均成功 | 17,800 |

这里的方向统计仅指旧数据中有成功记录的方向，不代表另一方向曾被尝试但失败。

候选按最高成功障碍数分为：3 障碍 48,233，4 障碍 33,586，5 障碍 22,500。建议在原 split 内按这些层及方向证据分层抽样，兼顾起终关节距离/路径长度覆盖，不只选择 5 障碍成功或最短最容易的任务。

历史成功不保证新 world 或另一方向也成功。该候选池定义的是“历史 teacher 在至少一个较高障碍世界成功过的任务”，评价结论应说明此筛选，不能当成任意随机关节端点分布。

可复用的候选清单：`benchmark_results/warehouse_multiworld_plan/20261015/eligible-families.npz`。统计与字段说明：同目录 `pool-statistics.json`。

## 2. 推荐规模：每 family 8 个 world 槽位，先固定一个方向

推荐首轮每个 family 固定一个有序 task `(q_start,q_goal)`，规划以下 8 个 world，每个 world 首先保留一个合格解：

| slot | 障碍数 | 场景 |
|---|---:|---|
| 0 | 0 | 静态基线 |
| 1 | 1 | 独立随机 mode、形状和运动参数 |
| 2 | 2 | 同上 |
| 3 | 3 | 同上 |
| 4 | 4 | 同上 |
| 5 | 5 | 同上 |
| 6 | n_pair∈{3,4,5} | 对相应 base world 的选定对象构造较早事件变体 |
| 7 | 同一 n_pair | 对同一对象构造较晚事件变体 |

每个 family 的 n_pair 从 3/4/5 中稳定分配，在各 split 内总体近似各 1/3，并与方向独立。slot 6/7 与对应 slot 3/4/5 形成同数量的三世界对照；它们不是额外两个任意随机世界。

该配置下每个有序 task 有 8 个 world，目标数量比例约为 0/1/2 各 12.5%，3/4/5 各 20.83%。训练采样可按部署需求重新加权，不强求生产与训练配额相同。

**8 是目标槽位，不是保证每个 family 最后都有 8 条成功轨迹。** 每槽位失败必须保留记录；不允许通过复制已有解、删物体、降障碍数量来填满。

### 为什么首轮不默认 16 条

两个方向都做 8 world 意味着每 family 16 条。它有助于成对研究方向效应，但会将生成与存储成本翻倍。首轮更优先在同一个有序起终点下建立世界变化，整体方向平衡即可；后续对代表性子集补另一方向的完整 8 槽位。

同 world 下多条解是另一个轴。首轮先每 world 一条；对早/中/晚变体出现明显策略变化的场景，再用不同 solver seed 保留 2–3 条经过去重与验收的解，不对所有 world 无条件生成很多近似重复解。

## 3. 正反方向

- 在每个 split 内，按独立稳定哈希/分层排名将候选约一半分配正向、一半反向。分配不能依赖障碍数量或 slot。
- family f 的 source 行为 `2*f + direction`，direction=0 正向、1 反向；source_task_id 从该行读取，不能用 family_id 替代。
- 同一 family 的 8 个 world 使用相同 direction，从而真正保持 q_start/q_goal 相同。
- 不沿用“仅在正向成功就永远正向”的规则；否则历史 70,569/15,950 的偏斜会保留。另一方向可能有新失败，应记录而非偷偷改回。
- 若以后补另一方向，归属原 family split，分别使用独立 world；参考路径先反向，再创建世界并重新求解。不直接反转已有动态成品。

## 4. train/valid/test：沿用原 family 划分

继续使用当前规则：

```text
bucket = int(sha256(str(family_id)).hexdigest()[:8], 16) % 100
0..79   -> train
80..89  -> validation
90..99  -> test
```

同 family 的所有数量、正反方向、时刻变体和多解均留在同一 split。当前 source 固定，family_id 在此 source 内唯一；将来换 source 要增加 source identity 命名空间，不能跨 source 直接合并整数 family_id。

若全池每 family 的 8 槽位全部成功，目标上限为：

| split | family | 轨迹上限 |
|---|---:|---:|
| train | 83,650 | 669,200 |
| validation | 10,214 | 81,712 |
| test | 10,455 | 83,640 |
| 合计 | 104,319 | 834,552 |

这不是已生成数量，也不意味着要立即生成全池。

### 分阶段

| 阶段 | family 目标 | split family 配额 | 8 槽位目标 | 用途 |
|---|---:|---|---:|---|
| 小试 | 200 | 160/20/20 | 1,600 | 检查配对、seed、恢复、失败类型与耗时 |
| 首轮训练 | 10,000 | 8,000/1,000/1,000 | 80,000 | 与旧数据联合训练，评估 world utilization |
| 扩展 | 全部 104,319 | 原有固定划分 | 834,552 | 仅在首轮学习曲线与消融支持后进行 |

阶段子集在各原 split 内稳定排名、分层抽取，并使小试嵌套在首轮子集中。不能重新随机切分一万 family，再与旧数据混合。

80k 新样本与旧 202,500 条可虚拟合并；使用 root/shard/group 全局样本键，保留各配置 hash、teacher 来源及 source identity。normalizer 只用合并 train 拟合。

## 5. world 采样与早/中/晚变体

### 普通世界

- 继续每对象独立、有放回抽取 mode；形状、尺寸、方向、速度独立随机。
- 不要求枚举所有 mode×数量的组合，也不按对象序号绑定模式。
- 采样种子与 family/direction/world slot 绑定，不能依赖该任务排到第几个 batch 或跑在哪台机器。

### 同数量的时刻对照

对选定 n_pair 的 base world，保存完整 scene parameters：anchor、mode、shape/size、方向、速度、事件时刻。slot 6/7 保持这些几何及运动参数不变，只移动选定对象的事件时刻；其他对象保持不变。

建议用静态参考路径的 retime 估计通过 anchor 的时刻 t_ref(anchor) 和总时长 T_ref，设置中心事件靠近该通过时刻，较早/较晚事件在其两侧，初始可测试约 0.15–0.25*T_ref 的偏移。具体幅度由小试的有效场景率与实际解策略差异决定。

事件与 horizon、有效起点及最终执行期 5%–95% 条件冲突时记录失败或重新采样整个成对模板，不通过静默 clipping 伪装成相同间隔。更宽时刻策略需要新配置版本。

当前代码的事件仍为 nominal_duration=8 的 25%–65%，即 2–5.2 秒，且 anchor phase 与该比例相关。本方案若扩展到真正较早/较晚事件，需要将 anchor phase 与 event time 分离，再按冻结 scene parameters 重新计算分段线性 forecast。不能直接平移旧 forecast 数组后隐式外推 [0,14] 以外的值。

只变化事件时刻也不保证教师一定产生“先过/等待/绕行”三种不同决策。应根据实际生成轨迹统计策略变化，不能把预设 slot 名称当作机器人 mode 标签。

## 6. 重试、失败与验收

### 分开两类重试

1. scene/geometry 重采样得到新 world：world_id 必须变化。
2. 同一冻结 world 的 solver 重试：只改变 solver seed，不改变 world。

训练槽位可设置有界的 scene 补充，例如最多 3 个有效 world 候选，每个 world 最多 2 次 solver 尝试。现有 MAX_WORLD_ATTEMPTS=12 只处理每个候选内部的 anchor/初始碰撞抽样，不能与 solver 重试混为一谈。成对时刻模板的 geometry 作为一个整体管理，不能只替换早事件失败的一侧成任意新世界。

valid/test 先冻结世界清单，不能为补足每 family 8 条而不断替换困难世界。保存初始无效、teacher 未解、拟合失败与成功状态；通过验收的轨迹用于监督验证，固定有效起点世界清单用于规划评估。teacher 超时/未解不自动等于该世界不可行。

全局与每 family 都应有总预算；允许部分槽位没有正样本，并报告每数量、方向、变体的 attempted/accepted/失败原因。不要删除整个部分成功 family 后只保留最容易完成 8 槽位的 family。

### 质量规则

沿用目前机器人与环境、margin=5 mm、joint goal、rest-to-rest、P29 degree5、c8/tau-r6 和 2–14 s。

所有 accepted 均要求 require-tau-r、raw 边检查及两种 codec 分别验收。0 障碍写 known-empty，不以远处假障碍代替；其余静态/运动学验收不减少。

本轮可统一使用当前 torch-forest（与旧正式数据一致），OMPL 作为可选 teacher 对照。0 槽当前可用同一后端的 --obstacles 0；若改用原参考+retime，应记录 teacher=reference-retime，并通过相同导出/验收 adapter，不能直接混入另一 schema。

## 7. 任务身份、seed 和恢复

建议独立 manifest 记录：

```text
source_identity, family_id, source_row, source_task_id, direction, split
slot_id, base_scene_id, world_id, phase_variant, obstacle_count
scene_seed, solver_seed, solver_retry, teacher_backend
status, rejection_reason, shard/group
```

seed 使用稳定的 source namespace、global seed、family、direction、slot/template 与 scene retry 派生。为相位配对分别管理 geometry seed 和事件变体，确保只改变预期参数。

solver seed 由 world_id 与 solver retry 派生。GPU 的实际路径仍受 batch 组成、时间预算和硬件影响；任务/世界可复现不等于改变 batch/机器后 q-t 路径逐点相同。保存 batch seed/job 顺序与代码/配置版本供审计。

manifest 先确定 family/split/direction/槽位；写出保持原子 shard 和连续 attempt 持久化。多机按唯一 job/world ID 领取互不重叠任务，不能再用 START_INDEX 加 accepted 数或 attempts 数分配物理行范围。

## 8. 成本与验收指标

旧生成机在 query batch=128、10 s 搜索预算下约 125–150 accepted/min。这只是历史参考，新独立 mode、成对时刻和候选池会改变接受率：

- 80,000 条约 8.9–10.7 h。
- 834,552 条约 92.7–111.3 h。

先用 200 family 小试重新测量；不能把上述数值当作性能保证。

小试/首轮至少输出：每 slot/count/direction 接受率、部分完成 family 比例、相位配对同时成功率、同 task 世界间的路径/时长变化、同 world 多解去重、full/known-empty mask、固定世界下模型/teacher 成功率，以及 source family split 无泄漏检查。

训练时可以先均匀抽 family，再在该 family 中抽有效 world，或按障碍数分层，避免成功槽位多的容易任务被无意重复加权。slot/world/source ID、anchor 与 teacher 决策标签保留为元数据；网络读取 task 和公开 forecast。

## 9. 实施顺序

1. 冻结候选目录及 source identity，导出 family/split catalog，生成嵌套阶段清单。
2. 增加独立 family/world manifest 入口，不改旧 START_INDEX 入口或 MPD/ROS 接口；先支持 0–5 普通世界及方向平衡。
3. 引入 scene parameters 与独立事件时刻，支持同几何早/中/晚变体。
4. 增加冻结 world solver 重试、逐 world 失败日志、恢复和多机任务领取。
5. 200 family 小试及磁盘复检，核对世界配对与分组划分。
6. 10k family 首轮，接入统一 loader 训练；对比旧数据、新普通 world、新增配对 world 的实际收益后决定扩展。
