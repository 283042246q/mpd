# 同任务多世界：失败槽位、采样关联审计与文献核对

日期：2026-10-15。核查对象为当前 `torch-forest` / `ompl-strrtstar` Warehouse 数据管线及 [多世界方案](WAREHOUSE_MULTIWORLD_GENERATION_PLAN.md)。本次修改文档和诊断工件，未修改生成算法，未启动正式数据生产。

## 1. 有界尝试后没有轨迹：少一条监督样本，保留 world 记录

是的。例如 8 个 world 槽位中 slot 4 最终失败，该 family 可以保留 7 条合格监督轨迹。不能用其他数量、更容易 world 或重复解冒充该槽位成功。

需要分清三张表：

1. **计划 manifest**：所有预先指定的 family / direction / slot / world，失败也保留。
2. **状态日志**：尝试次数、固定 world 的 solver retry、初始无效、teacher 未解、codec 失败等。
3. **正样本索引**：只收通过验收的轨迹，可直接展开训练，不要求每 family 等长。

建议状态区分 `invalid_scene`、`teacher_unsolved`、`fit_rejected`、`accepted`。限时未解不是不可行证明，不能给空轨迹当作“不可行标签”。即便 family 新一轮 0 条成功，manifest 中也保留它；正样本训练中自然没有该 family。

若训练按独立样本展开，无需为了缺失 world 填假轨迹；若按 family 组成固定 8 槽位 tensor，增加 `solution_valid[B,8]`，loss 仅平均有效目标。该 mask 不同于动态对象 mask：失败的 5 障碍 world 仍是 5 障碍，不能编码成 0 障碍。配对指标/配对 loss 使用对应 base/early/late 全部成功的 group mask；普通 denoising loss 仍可使用组内单独成功的轨迹。

训练可以按 family 或数量分层采样，避免成功槽位多的任务被无意重复加权。valid/test 固定 world，失败留在评估记录，不为补齐数目无限换世界。只保存成功训练轨迹有文献先例：DiffusionSeeder 的数据生成过滤了 cuRobo 未找到可行解的问题；这不意味着评价时可以隐去方法失败。[DiffusionSeeder §4.1、§7.1](https://arxiv.org/html/2410.16727v1#S4.SS1)。

## 2. 审计结论：没有要求所有变量无条件独立

应消除的是与数组位置、调度、目录编号意外绑定的关联。物理约束、mode 本身的定义，以及成对对照共享几何，是有意义的关联。通过有效性与质量筛选后的 accepted 数据通常不再保持 proposal 的独立均匀分布。

### 2.1 已消除的旧规则

当前 `sampled_modes = rng.integers(0, len(modes), size=object_count)`，每对象有放回抽样。已无 attempt 轮换 mode 列表和对象序号选择连续 mode 的规则；单 mode pool 仍固定该模式。形状同样按对象抽样。旧数据的顺序 mode 不会被代码升级追溯改变。

### 2.2 剩余关联与处理建议

| 关联 | 当前证据 | 判断与建议 |
|---|---|---|
| 事件时刻与 anchor phase 共用 fraction | warehouse_world.py 中 event=fraction*nominal_duration，同时 phase=round(fraction*(K-1)) | 新多世界时刻对照前优先解耦；同一几何只改时刻时不能让 anchor 跟着移动 |
| 事件范围依赖固定 nominal duration | 默认 8 s × [0.25,0.65] = [2,5.2] s | 训练分布较窄；引入 T_ref 与早/中/晚覆盖，记录新策略版本 |
| GPU 终点到达下界与最晚事件挂钩 | generator 向 forest 传 max(event_times)+0.25；OMPL 没有相同参数 | 作为显式数据交互策略处理，不能误称纯运动学下界；公平 teacher 对照需区分该规则 |
| accepted T 与全部事件强耦合 | interaction_audit 要求每个事件在最终 T 的 5%–95% 内 | 这是原先“所有物体执行期经过 workspace”的任务定义；保留时明确声明，并与允许提前结束的物理可行任务区分 |
| seed 绑定 attempt，未绑定 family/direction/world ID | SeedSequence([global_seed,current]) | 新 manifest 必须改为稳定 job identity；避免同目录任务顺序变化重定义 world |
| 跨数量的同 seed 事件共享前缀 | 同 seed 分别 M=3/5 调用 make_forecast，前 3 个 event 完全相同，但其他 RNG 字段错位 | 既非完全独立世界，也非保留相同对象的数量对照；普通世界用独立 world seed，数量消融显式用同一对象模板增删 |
| GPU solver 与 batch 顺序关联 | 整 batch 使用 jobs[0].seed，random tensor 按 query 维消耗 | 不影响冻结 world 的身份，但改变 batch 会改变求解结果；记录 batch 顺序，不承诺任务 seed 相同就逐点同解 |
| world 几何依赖参考路径 | anchor 从参考机器人碰撞球选取 | 有助于构造相关交互，不是错误；新数据应另含少量不定向命中参考路径的背景世界，评价泛化另做固定清单 |
| 历史方向与数量绑定 | 1/4 奇数行、2/3/5 偶数行 | 已存在于旧数据；新清单在每 split 内独立平衡方向，不能依靠更新代码自动修复 |
| mode 与物理参数的语义关联 | vertical 固定 +Z；box 固定比例，capsule 固定半长，朝向不变 | 属于模板/几何定义；不必为了独立性打散，但应声明支持范围并逐步拓宽 |
| 到达时间根与 slowdown 离散取值 | forest 初始 roots 用 linspace；拟合尝试 1/1.15/1.35 | teacher 分布可能有算法造成的时间峰；记录 raw/fitted duration 与 scale 后再归因，不把峰直接当真实 temporal mode |
| mode pool 可重复列出同一名称 | CLI 当前只检查 choices，不去重 | 默认 pool 唯一，不影响默认 iid；自定义重复条目会形成隐式权重，应在新入口去重/拒绝重复，或显式使用概率字段 |

跨数量前缀诊断见 `benchmark_results/warehouse_multiworld_plan/20261015/audit/sampling-coupling-probes.json`。该诊断比较纯 proposal，没有被场景/解的拒绝筛选干扰。共享部分随机数本身不等于完整 world 或轨迹泄漏。

### 2.3 终点时间的两层限制，不能混成 crossing time

当前最终交互条件给出：

```text
T > max(event_times) / 0.95
```

GPU raw 搜索另有 `earliest_arrival >= max(event_times)+0.25`，同时仍受关节速度下界限制。这些都是终点时长限制，不直接禁止机械臂在较早时刻抢先通过某处。

如果每对象 t_event 独立均匀分布于 [2,5.2]，则 proposal 的 E[max t_event] = 2 + 3.2*M/(M+1)。因此对象更多时，仅该事件规则就提高最低允许 T；不能将跨数量的时长增加全部归因于真实避障难度。

这保留了用户此前要求的活跃交互场景，不能自动当成错误删除。建议将 `all_events_during_execution` 写入显式策略，并设计独立对照：按 T_ref 生成潜在相关事件，最终仅依据物理安全验收，允许提前完成。两类数据分别标识，先测教师覆盖与模型效果再决定比例。

ST-RRT* 的核心问题是动态障碍、速度约束与未知到达时间，并非必须等待所有障碍事件后才能结束。因此上述附加规则属于本项目的采样/验收政策。[ST-RRT* 摘要与 §III](https://arxiv.org/html/2203.02176)。

## 3. 文献支持到哪一步

以下只采用论文、作者实现或官方方法文档；“本项目建议”与文献事实分别列出。

| 来源 | 核对到的做法 | 对本项目的启示/边界 |
|---|---|---|
| MPD | Appendix A 明确每个 start/goal context 一条轨迹；§IV 展示未见 context 上的多模态生成；数据含反向连接 | 单 context 多解不是扩散训练的硬前提；此前“单解必然退化成 delta”的说法过强，应修正 |
| CAMPD | 静态球场景数量 1–10，113,469 条轨迹覆盖 2,400 环境；结构化 context 支持变数量，测试未见环境 | 支持数量/几何变化的 context-conditioned 学习；不直接证明本项目动态事件模板或 8 条配额最优 |
| DiffusionSeeder | 在场景内扩充起终点问题，过滤 expert 未解的问题；测试同类但不同配置场景 | 支持重复利用场景构造多问题与只收有效监督；这里重复 task 改 world 是另一条互补轴 |
| Motion Policy Networks 官方数据管线 | 训练清理仅留合格 expert 问题，控制问题类型比例；推理问题也标明 expert 可解的筛选 | 历史 teacher-success 候选池合理，但必须明确评价分布经过筛选 |
| scikit-learn grouped CV 官方说明 | 组内样本相关时，检测未见 group 泛化应保持 train/validation group 不交叉 | 将正反、多世界、多解按 source+family 一起划分符合此原则；80/10/10 及 hash 形式是本项目选择 |

来源链接分别为：[MPD](https://arxiv.org/html/2412.19948v3)、[CAMPD §IV-A](https://arxiv.org/html/2510.14615v1#S4.SS1)、[DiffusionSeeder §4.1](https://arxiv.org/html/2410.16727v1#S4.SS1)、[MπNets 作者数据管线](https://github.com/NVlabs/motion-policy-networks#data-cleaning)、[Grouped CV](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)。

这些来源没有规定每个 family 必须同时有 0–5 障碍，也没有证实每 family 8 条是最佳配额；这属于围绕 world conditioning 的实验设计，应通过数据量受控的小试决定。

## 4. 对新方案的判断与修订

### 合理部分

- 相同有序起终点配不同 world，为 JD-WORLD 学习环境响应提供直接对照。
- 数量变化、对象独立 mode、形状/运动随机化符合环境条件生成的方向。
- 基础 world 与时间变体共享几何属于有意控制变量，应保留，不追求全部参数无条件独立。
- source+family 统一 split 能防止同一路径正反和相关世界跨集合，适合主测试的未见任务评估。
- 每 world 一条有效 teacher 可作为首轮起点，随后对重点 world 补解和保留不同策略。

### 生成前优先补齐

1. job identity seed 与独立 scene/solver RNG；direction、n_pair、geometry、event 使用不同 namespace，不能复用一个取模结果决定多个属性。
2. 解耦 anchor phase 和 event time，保存 scene parameters，支持固定几何的时间变体；扩大时间覆盖时新增策略版本。
3. 显式记录/区分活跃交互政策和物理终点条件，避免把 max(event) 引起的 T 变化解释成学习到的避障策略；两个后端评价口径对齐。
4. per-world 状态日志、有限重试和固定验证/测试清单，允许 partial family。
5. 不依赖 slot 编号的任意物体顺序：加载时可共同置换 shape/size/整条 object forecast，时间顺序保持；mask 跟随对象同步置换。

当前代码/旧数据不应因此被一次性否定。缺乏独立性的部分有些是原任务设计，有些是旧采样机制；新方案逐项解耦并记录版本，比追求所有数组统计独立更合适。

## 5. 测试集应回答三个不同问题

1. **主测试：未见 family + 新 world**。沿用现有 family split；正反和所有变体不跨 split。
2. **world 使用诊断：见过 task + 未见 world**。从 train family 预留世界，只用于独立诊断，不混称未见任务测试；这些 world 的监督不参加训练。
3. **明确的 world OOD**。在 test family 上预先指定未训练相位范围/运动参数/组合，数量外推另列。仅 family 分组不能自动证明这些泛化，更不能证明跨静态环境。

随机 0–5 世界中所有参数都变化，适合训练覆盖，但不是纯数量因果消融。若要测数量效应，增加显式嵌套对象测试：同一 task、同一已有对象，逐个添加新对象并重新求解；不能靠“相同 seed、不同 M”假定已有对象保持一致。

同样，时间变体改变 world 事件，最终机器人 crossing time/等待/绕行由求解结果决定。slot 名称只描述世界变化，不能作为机器人决策的提前标签。

## 6. 最终建议

继续使用历史 ≥3 障碍成功 family 作为首轮候选，保留每 family 8 个目标槽位，接受变长成功数量；先做 200 family 小试。生成前优先完成上述 seed、anchor/time、交互策略和日志改动。

在相同冻结测试集上比较普通随机 world、加入配对 world 的收益；再通过预算相近的 4/8 槽位数据试验检查“更多 family vs 每 family 更多 world”的取舍。对生成接受率、数量/方向/相位覆盖、部分配对率及模型成功率分别统计；不以凑满 8 条作为数据质量指标。
