# Time Corridor 先提速、后扩展模态：1 → 3 → 2 分阶段实施文档

- 日期：2026-09-28。
- 代码基线：MPD 提交 6d8c57f；首次 plan 时间语义由 778c753 引入。
- Python 环境：mpd-splines-public。
- 文档状态：实施方案；实际实施和验收边界见 [实施记录](TIMECORRIDOR_SPEED_FIRST_IMPLEMENTATION_REPORT.md)。
- 范围：joint_corridor_a、f1_c_corridor_a、f1_tau_r_corridor_a；保留对应无 Corridor mode 作为对照。
- 与[总体时空规划设计](SINGLE_ARM_SPACETIME_TIMECORRIDOR_JOINTDUAL_FLOWMATCHING_DESIGN.md)衔接；其中较早的实现状态不作为当前代码状态。

## 1. 决策与阶段顺序

采用用户指定的 1 → 3 → 2 顺序。这里的数字对应上一轮建议，而不是本文小节号：

| 顺序 | 阶段 | 对应原建议 | 本阶段改变什么 | 本阶段固定什么 |
|---|---|---|---|---|
| 准备 | S0 | 测试前置条件 | 冻结输入、补全计时和失败样本记录 | 算法、checkpoint、crossing profiles |
| 第一阶段 | A | 1：GPU batch 与减少同步 | 候选和分支并行、分块、等价冗余消除 | nominal/early/late、原始物理 cost、迭代数 |
| 第二阶段 | B | 3：减少重复碰撞查询 | 请求内缓存、时间距离表、适用时的解析区间 | 三种 preference、checkpoint、空间候选 |
| 第三阶段 | C | 2：改善多模态 | 有预算的 K-best、窗口序列去重、DP timing 初始化 | 已验收的计算后端与安全检查 |
| 收尾 | S4 | 综合验证 | 完整在线矩阵、独立校准、报告 | 同一轮对照的冻结环境和 seed |

这个顺序可行：A 先降低执行开销，B 再减少重复几何计算，最后 C 利用释放出的预算探索更多时间模态。先扩展 K 会同时改变搜索覆盖和计算量，不利于解释提速收益。

B 中的距离插值和区间近似会改变优化梯度，即使 preference 数量不变，也不属于严格数值等价改造。必须与 B 中的精确缓存分开测试；近似分支不通过时保持关闭，记录结论后再进入 C，不能把它当作必须启用的依赖。

每个小阶段单独实现、测试、记录结果并 commit。阶段门槛失败时回到上一个已通过的配置，不把多个未验收改动叠加。

## 2. 当前实现及不可改变的约束

### 2.1 已核对的实现

关键代码：

- [time_corridor.py](../mpd/inference/time_corridor.py)：网格、CPU DP、逐候选逐分支 Adam、阶段 profiler。
- [space_time_guidance.py](../mpd/inference/space_time_guidance.py)：物理时间 mean/CVaR 动态风险、速度、加速度、duration、timing smoothness。
- [dynamic_collision.py](../mpd/inference/dynamic_collision.py)：批量动态 SDF 查询和障碍物维度归约。
- [factorized_guidance.py](../mpd/inference/factorized_guidance.py)、[learned_timing.py](../mpd/inference/learned_timing.py)：固定空间路径和六维 timing latent。
- [space_time_runtime_engine.py](../scripts/runtime/space_time_runtime_engine.py)：原始/调整后 DenseCheck、回退与运行身份。
- [factorized_runtime_engine.py](../scripts/runtime/factorized_runtime_engine.py)：representation 身份及后处理时间。
- [benchmark_todrawer_random.py](../scripts/isaaclab/benchmark_todrawer_random.py)、[until-success runner](../scripts/isaaclab/run_todrawer_f3c_until_success.py)：冻结场景、两种时间协议和 timing profiles。

当前默认设置为 phase_points=32、time_step_s=0.2、refinement steps=20、learning_rate=0.04、corridor weight=0.1。实际测试还要保存运行时生效配置，不能只引用默认值。每条分支最多执行 21 次 forward 和 20 次 backward，可提前退出。

当前实现对每条候选分别检查初始 timing；动态 cost 接近零时跳过后续 Corridor。这个跳过条件不是连续碰撞安全证明。需要调整时构建联合安全网格，对 preference 0、-1、+1 分别运行 DP，按窗口序列去重，再串行优化剩余分支。

DP 已使用前缀最小值，一次搜索约 O(P·T)。它只检查采样阶段之间的单调性和关节位移/速度下界，不保证连续空间段安全、加速度可行或可被六维 timing 表达。

refinement 内已经复用固定的 q、q_s、q_ss 和碰撞球位置；提速不能重复宣称“首次实现 FK 缓存”。初始评估、重复物理 cost 和 GPU/CPU 同步仍有优化空间。

### 2.2 历史 smoke 数据仅用于定位瓶颈

历史目录：scripts/isaaclab/logs/todrawer-gpu-corridor-smoke-fixed-20260928-183214。场景为 single_crossing，一个匀速盒体；该次 GPU 同时运行其他训练，不能作为独占 GPU 的性能基准。

| 指标 | 历史值 |
|---|---:|
| f1_c request_total_sec | 2.174 s |
| f1_c_corridor_a request_total_sec | 4.810 s |
| grid_build | 0.627 s |
| branch_selection | 0.073 s |
| refinement_forward | 1.007 s |
| refinement_backward | 0.848 s |
| final_validation | 0.027 s |
| 原始候选 / 进入 refinement / 分支数 | 100 / 12 / 23 |

该次 Corridor 的 result 为 success，但在线响应被 deadline_expired_after_planning 拒绝，没有 replay manifest。它不是一次在线任务成功。只有这一条样本，也不能据此给出耗时分布或多障碍缩放结论。

### 2.3 所有阶段的约束

1. 保留原 Franka 单臂入口、无 Corridor 默认行为和现有 checkpoint；不重训网络。
2. A/B 不改变三种 preference、候选数、refinement 步数、物理 cost 权重和随机 seed。
3. 保持 F1 的六维 latent 和 Phase5 的现有 timing controls；空间路径在 Corridor 内固定。
4. 保持现有碰撞球、margin、风险聚合和最终 DenseCheck；不能通过减少碰撞球或降低检查分辨率获得表面加速。
5. 原候选经验证有效、refinement 后无效时，保留现有回退；两者均无效时返回规划失败。
6. 正式在线对照保留当前 deadline、handoff、guard 和制动规则。离线完整计算不受在线 deadline 截断，但必须单独标记。
7. 各小阶段不重新调整 crossing time 掩盖性能差异；完整功能稳定后才单独校准。

## 3. S0：先建立可解释、可复现的测试基线

### 3.1 两层冻结输入

第一层是 Corridor 内核测试：保存进入 Corridor 之前的空间候选、初始 timing、完整世界快照、机器人/样条配置、dtype/device、seed 和输入哈希。所有后端消费完全相同的输入，避免 diffusion 随机性和在线闭环分叉掩盖差异。

第二层是完整 planner 请求测试：固定原始 request、世界预测、checkpoint 哈希和随机 seed，从 resident worker 执行完整规划。分别测模型加载、首次请求和预热后请求；不能只报告 kernel 时间。

离线重放必须保持 world_version、plan_start、预测有效期和障碍轨迹之间的相对时间。如果需要平移历史时间戳，统一平移所有相关绝对时间并记录偏移，不重新采样世界。物理代价仍按同一世界快照求值，不运行 ROS 执行。

### 3.2 计时口径与数据缺口

当前 _StageProfiler 将 CPU 提交耗时与 CUDA event 区间取 max，输出单个阶段值；它不是纯 GPU kernel 时间，profiled_total_s 也不能自动等同于请求墙钟时间。新增字段应保留旧字段兼容性，并至少分开：

| 指标 | 用途 |
|---|---|
| stage host wall time | Python 调度、CPU DP、显式同步等开销 |
| stage CUDA event elapsed | 当前设备/stream 上的事件区间；可能包含等待和其他负载干扰 |
| synchronized Corridor wall time | Corridor 整段端到端耗时，单独处理最终验证 |
| request_total_sec / server round trip | 完整 planner 与服务开销，不与分项重复相加 |
| first_plan_completed_from_world_s | world start 至最早完成 result，不论成功失败 |
| inference_total_sec / dense_validation_sec | 保留原字段；解释各自实际覆盖范围 |

历史 F1 样本的 inference_total_sec 约 2.157 s，而 request_total_sec 约 4.810 s，说明前者不覆盖全部 Corridor 后处理。最终报告必须使用完整请求时间比较提速，不能只看 inference_total_sec。

CUDA event 在对应设备/stream 记录，只在适当边界同步。另抽取少量 profiler trace 检查 kernel、拷贝和同步；重型 profiler 与无 profiler 延迟测试分开，避免每步同步改变原始性能。CPU/GPU 分项可能重叠，不能机械求和解释墙钟时间。

当前 benchmark 的 Corridor 分项主要聚合成功 result。新增逐请求性能记录必须覆盖失败、有结果但超 deadline、无 manifest 等情况，并保留以下两组状态：

- planner_result_status：轨迹计算结果，例如 success / no_valid_trajectory。
- online_acceptance_status：客户端/服务端是否及时接受，例如 deadline_expired_after_planning。

进程崩溃、超时杀死且没有完成结果的样本标记为 censored/error，不能填入 0 秒，也不能当作已完成请求。首次完成时间没有 manifest 时从可靠 world-start 日志或新增元数据获取；缺失则明确为 missing。同时报告每种失败率，避免“更快地失败”被解释为规划性能提高。

### 3.3 样本、预热与 GPU 条件

- 使用三种 Corridor mode，以及 joint、f1_c、f1_tau_r 对照；记录实际 checkpoint 路径、step 和内容哈希，不推断默认路径版本。
- 单元/合成场景覆盖无障碍、单障碍、多障碍、窄窗口、无可达分支、原 timing 已安全、near-obstacle 等边界。
- 完整冻结集使用现有 10 类随机环境，初始建议 2 环境/类 × 3 planner seeds × 3 Corridor modes，共 180 个配对输入。该数量是计划，不是已完成测试。
- 微基准额外控制障碍数 0/1/2/4/8、不同候选数和不同活跃候选比例；这是拟新增 harness 功能，现有 benchmark 没有对应的障碍数 CLI。
- resident worker 至少 3 次预热；内核微基准建议每输入 10 次计时，完整 planner 至少 3 次。首次请求单独保留，预热不混入首次计划分布。
- 交错运行参考/优化版本，或用固定随机顺序排列配对执行顺序；禁止一边跑满参考后端、一边在不同负载条件下跑优化后端。
- 持续记录 GPU 型号、驱动、PyTorch/CUDA、显存、利用率、功耗和外部进程。共享 GPU 可做功能 smoke，但单独标记，不用于更新正式 timing profiles；不停止无关训练。
- 同一场景的重复请求有相关性。按 mode、类别、障碍数和 attempted/branches 分层报告；区间估计以场景为重采样单位，不把所有迭代当独立样本。

S0 验收：同一输入可重复运行；成功、失败、超时样本都可定位；完整请求计时覆盖 Corridor；基线输入和元数据有哈希。

## 4. A：GPU batch 与等价提速（原建议 1）

### A1：批量初始评估和固定状态复用

一次或分块计算候选 q、q_s、q_ss、碰撞球位置和初始 cost，生成逐候选 active mask；避免在 Python 循环中逐候选读回标量。Phase5 初始 evaluate_control_points 已计算过的 cost 应按实际返回值复用，避免紧接着重复调用同一 evaluator。F1 的 duration penalty 仍按原规则添加。

默认保持当前采样点、网格和三种 DP 结果。先将每个候选的 safe mask、初始 cost、最低时间间隔与串行基线逐一比较，再测初始评估耗时。

### A2：候选 × 分支批量 refinement

将 ragged 分支展平为 R=sum(K_b) 行，维护 row_to_candidate 和 row_to_branch。F1 参数形状为 [R,6]，Phase5 为 [R,C_t]；每行持有独立窗口、优化状态和最优结果。DP 暂时仍在 CPU 执行。

必须保持的逐行语义：

- objective 先按候选内部原规则归约，再对行求和以执行 backward；不能因 batch 大小改变权重。
- 梯度裁剪逐行执行，不能对整个参数矩阵做一个全局范数裁剪。
- Adam 的一阶/二阶矩、bias correction 所需计数、停止条件和 best-so-far 保持独立；停用行不能继续被 momentum 更新。
- duration 非法、NaN 或无梯度的行独立退出，其他行继续。避免对 NaN 简单乘零，需在危险计算前隔离或替换为有效占位状态。
- Phase5 固定控制点及端点导数约束保持原规则；不能把“固定控制点”解释为总 duration 被锁定。
- 对齐原先 forward-before-update、最后一次 forward、严格小于时更新 best 和相同目标值时的选择顺序。
- 最后按 candidate 归约选择分支，并保留原来的无可达分支和回退行为。

先支持 branch chunk，再支持 candidate chunk；改变 chunk size 应只影响执行方式，不改变有效结果。OOM 时降低本请求 chunk，并记录重试开销；不能静默减少候选、分支或碰撞检查。

### A3：网格批量查询与减少同步

把候选/phase 维合并后调用现有 [batch,time,links,3] 动态 SDF 接口；按显存预算分块。距离查询的临时数据还含障碍物维度，不能只根据最终 safe mask 大小估算显存。

集中一次或少数几次传输 DP 所需 safe masks；清理逐步 float(tensor)、CPU 拷贝和 Python bool。边界/非法值检查改为逐行 mask 后，严格公共 API 的验证约束仍需保留。

本阶段不 batch 不同 checkpoint、不同世界或依赖未来观测的多次 replan。优化迭代和 DP 的 phase 递推存在依赖，不能直接并成独立 batch。

### A 的测试与门槛

1. 比较串行、batch=1、多个 chunk size；覆盖 c、tau_r 和 Phase5。
2. 比较每行目标、梯度、一次 Adam 更新、完整迭代轨迹、最终 timing 与候选有效性；加入提前退出、等分 tie、NaN 行隔离和端点冻结测试。
3. 初始浮点容差建议 atol=1e-5、rtol=1e-4；按输出物理单位另设误差界。边界附近的布尔判定不能用放宽浮点容差直接豁免，应精确复核并记录。
4. 冻结集的 DenseCheck 接受/拒绝及回退结论应一致；差异定位清楚前不升级默认后端。
5. 无 Corridor 路径和 F2/F3 原有功能回归。
6. 建议性能目标：活跃分支较多场景的 Corridor p50 至少 1.5× 加速；完整请求 p95 不退化超过 5%。无障碍/少活跃候选另报，不用混合均值掩盖退化。目标需由实测验证。

A 完成后先 commit 和保存报告，再开始 B；不同时扩展 K 或改变初始化。

## 5. B：减少重复 SDF 查询（原建议 3）

### B1：精确的请求内复用

在 A 已缓存固定路径状态的基础上，检查同一 world、同一时间网格下可复用的障碍预测位置、旋转、inflation 和查询配置。只复用与 timing 参数无关的内容；新的查询时间仍须得到正确时间梯度。

缓存身份至少包含不可变路径内容、world_version、plan_start_unix_ns、预测有效期、运动参数、uncertainty/inflation、机器人和 margin 配置、phase/time 网格、dtype/device。优先把缓存限制在单次请求内；跨请求复用作为独立后续设计。

缓存原样距离和完全相同查询的结果属于等价优化，先验证数值/梯度，再计入 B1 收益。不能将带计算图的 tensor 跨 backward 错误复用。

### B2：时间距离表作为可选近似后端

当前 cost 在障碍物维度取最小距离后，仍保留每个碰撞球的 penetration，再计算物理时间 mean/CVaR。因而距离表至少保留 [candidate,phase,time,link]，不能只保存一个“全身最小 clearance”就声称与原风险等价。

固定路径上预计算 D[b,h,t,l]，refinement 在每个实际到达时间插值，仍按当前时间权重计算 mean/CVaR、运动学和 duration cost。先在全部 H 个物理 cost 采样点缓存；如果只用 P 个点，phase 插值是另一个近似，必须单独消融。

预计算约 O(M·H·T·L·O)，每轮动态查询可降为约 O(R·H·L) 次插值；这不包括 timing 解码、风险聚合和其他物理 cost。只有复用收益超过建表和显存开销才启用，不能假定所有场景都会更快。

约束与回退：

- 不越过表的时间域和世界预测有效期外推；出界候选停止或切回精确查询。
- 把插值代价明确标记为 surrogate；精确缓存与近似插值分开统计。
- 如果有可信的时间 Lipschitz 上界 L_t，均匀时间线性插值可采用 L_t·Δt/2 的保守距离误差余量；上界须覆盖障碍运动及 inflation 的变化。未建立上界时不得宣称保守保证。
- 插值梯度不是精确 SDF 梯度；窗口边缘和最接近障碍切换处重点测试。
- 分支输出用原始 evaluator 重新打分，再执行原 DenseCheck；记录精确复查开销。必要时保留少量中间候选供精确复查，不能只看 surrogate 最优值。
- 原有效候选的回退规则保留；距离表导致可行解减少时，按门槛关闭该近似后端。最终 DenseCheck 通过仍不等于数学上的连续无碰撞证明。

### B3：解析碰撞时间区间，按运动模型逐步支持

先把匀速、固定半径的球—球模型用于验证基础算法：对每个固定 phase 解二次碰撞不等式，合并各 link/obstacle 的碰撞区间并取安全补集。正式场景使用盒体，球体试验的加速不能直接声称适用于当前 benchmark。

固定姿态、匀速盒体需要处理球—盒膨胀几何；简单扩大 AABB 是保守近似，可能缩小可行窗口，应独立标记。加速、曲线、时间相关 inflation 和不支持形状继续走原网格后端。

为隔离模态变化，B3 首先把解析区间投影回原 time grid，仍执行原三种 preference 和原 margin 规则；直接换成 interval-state 搜索留到 C 的独立实验。粗网格比较之外还要增加更密/独立边界查询，检查窄碰撞区间和擦边情况。

### B 的测试与门槛

| 子阶段 | 必测内容 | 进入默认配置的条件 |
|---|---|---|
| B1 精确复用 | cache on/off 数值与梯度、world/time/path/config 变更失效 | 与 A 一致，计入缓存构建后确有收益 |
| B2 距离插值 | 距离/梯度误差、mean/CVaR、精确重排、边界/出界、多障碍 | 无新增独立 checker 漏检，困难场景覆盖不下降，完整请求有收益 |
| B3 区间构建 | 解析边界与独立密采样、保守近似误差、不支持模型回退 | 支持范围明确、原 DP 输入可对照、总耗时有收益 |

建议 B2/B3 的晋级目标为相对 A 完整请求 p50 至少降低 10%，p95 不退化超过 5%；冻结集成功覆盖不下降，新增非法接受数为 0。样本不足以判断时保持实验开关，不通过降低碰撞 margin 或删除失败样本放行。

每个子阶段单独 commit、保存消融报告。可保留 B1 并关闭无收益的 B2/B3；在实施记录中说明，而不是把未启用方案标成已完成提速。

## 6. C：有预算的多模态搜索（原建议 2）

只有 A 和 B 的选定后端稳定后才开始 C。先评估 DP 到达时间初始化，再评估更多分支，避免同时更改两项造成归因困难。

### C1：复用 DP 找到的到达时间

让搜索同时返回窗口序列及选中的时间单元；用其拟合现有 timing 参数作为额外初始化。F1 保留六维表示，Phase5 保留现有控制点和端点规则。拟合失败或精确物理评价更差时使用原初始化。

这是启发式初始化变化，不能要求与 A/B 数值等价；单独比较拟合开销、实际优化步数、精确可行率和覆盖。直接 arrival-time QP 可作为试验初始化，但不把它的线性可行性视为真实加速度或 spline 可行性。

### C2：K-best 与窗口序列去重

在当前前缀前驱约束和可分离代价下，对每个状态维护有限 K 个标签；按窗口 ID 序列去重，优先保留瓶颈位置决策不同的路径。有限 top-K 标签传播的主要工作量约 O(K·P·T)，去重/多样性筛选必须另外记录开销。

不同网格路径可能对应同一窗口序列。精确 top-K 网格路径和有预算的多样化搜索不是同一保证：设置 oversampling/label 上限后，只保证有限预算，不宣称得到全部模态或精确 top-K 不同窗口类别。

先固定 K=4 对比原三 preference；再测试 K=8 和有上限的 4→8 扩展。限制每候选标签数、请求总分支数及总优化工作量。保留原三种分支作为候选来源之一；有限预算内替换分支时需记录被舍弃的原分支，不能暗示其覆盖必然保留。

第一次配对实验使用确定性工作预算，便于复现；墙钟截止策略随后单独测试。自适应扩展的判断应使用已完成的精确评价/验证信息，并计入该判断的开销。

### C 的测试与门槛

- 小型人工网格可穷举所有路径，用于验证 K-best 的正确性；同时测试 top-K 路径重复属于同一窗口序列。
- 构造至少四种不同可行窗口序列、局部先早后晚/先晚后早、窄窗口和低维 timing 无法拟合的场景。
- 分别比较：原三分支、仅新初始化、仅 K-best、两者组合。
- 同时报告相同分支预算和相同时间预算下的 coverage、任务成功率、独特且通过验证的窗口序列数。
- 报告搜索节点/标签、实际分支数、拟合失败数、精确验证通过数和 p50/p95；不能只报告 K 或 changed 数。
- 晋级要求：预算内取得可重复的覆盖收益、没有新增非法接受、原困难场景无未解释退化；单纯增加分支导致耗时上升而无覆盖收益不启用。

## 7. 每一步怎么测试、保存什么

### 7.1 实验顺序

| 运行标签（拟议） | 后端/变化 | 直接对照 |
|---|---|---|
| reference | 冻结串行实现 | 基线 |
| batch_exact | A，原三 preference、原 cost | reference |
| batch_cache_exact | A+B1 | batch_exact |
| batch_time_table | A+B1+B2，近似开关 | batch_cache_exact |
| batch_event_intervals | 选定 A/B 后端+B3 | 同后端的原网格 |
| dp_initialized | 选定 A/B+C1 | 选定 A/B |
| diverse_k4 / diverse_k8 | 选定 A/B+C2 | 原分支；再与 C1 组合 |

这些是实验标签，不是当前已支持的 CLI 参数。后续配置需要显式声明，写入 result/run-spec，并在默认关闭状态下接通 server、直接推理入口及 benchmark；禁止根据 mode 后缀隐式猜测优化版本。

每次报告至少包含输入哈希、代码提交、完整参数、checkpoint 哈希、GPU 状态、有效/失败/删失数量、p50/p90/p95/max、分阶段耗时、峰值 allocated/reserved 显存、候选/分支/迭代数、cache hit/fallback、精确 clearance、速度/加速度超限、最终接受率。没有足够数据不报告稳定 p99。

### 7.2 现有可执行检查命令

从 MPD 仓库根目录运行；以下是后续实施时可使用的检查命令，不表示本次文档提交已运行它们。

~~~bash
/home/eric/anaconda3/envs/mpd-splines-public/bin/python -m pytest -q \
  tests/test_time_corridor.py \
  tests/test_space_time_guidance.py \
  tests/test_factorized_guidance.py \
  tests/test_todrawer_random_benchmark.py \
  tests/test_todrawer_mode_motion_start.py \
  tests/test_run_todrawer_f3c_until_success.py \
  tests/test_dynamic_demo_pipeline.py

nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,power.draw --format=csv
~~~

下面仅生成现有 baseline 的双协议冻结 suite，不运行 GPU/ROS，也不包含尚未实现的后端切换：

~~~bash
/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  scripts/isaaclab/benchmark_todrawer_random.py \
  --output-dir scripts/isaaclab/logs/corridor-speed-first-baseline-v1 \
  --environment-count-per-category 2 \
  --planner-repeats 3 \
  --timing-protocol both \
  --modes joint joint_corridor_a f1_c f1_c_corridor_a f1_tau_r f1_tau_r_corridor_a \
  --duration-sec 35 \
  --suite-seed 20260928 \
  --dry-run

/home/eric/anaconda3/envs/mpd-splines-public/bin/python \
  scripts/isaaclab/validate_todrawer_random_suite.py \
  --suite scripts/isaaclab/logs/corridor-speed-first-baseline-v1/suite.json
~~~

完整矩阵为 20 环境 × 3 repeats × 6 modes × 2 协议 = 720 次在线运行/后端版本。先完成离线及小规模 GPU smoke，再运行完整矩阵；不为每个未通过的小改动重跑全部 720 次。旧目录存在时先核对 suite/config，使用新版本目录，避免混合不同实验。

离线 snapshot/harness、后端选择和逐请求分布报告仍需在 S0/A 实现；不能把上面的 dry-run 当作性能测试。

### 7.3 时间协议与 deadline

性能 A/B 使用完全一致的冻结请求/世界；在线 paired suite 固定版本和 crossing profile。motion_aligned 当前是按历史首次结果时间 profile 生成，不会依据本轮实际 plan 耗时即时移动障碍，也不保证各 mode 的相对 crossing 延迟严格相等。

absolute_world_time 保持相同物体世界轨迹，反映真实启动延迟；motion_aligned 保留基于 profile 的模式差异。两者分别报告，不能混合后只给一个成功率。

S4 校准使用与验证集分开的场景，统计任意完成状态的首次 result；固定软件、checkpoint 和 GPU 条件后更新 profile。profile 变更单独提交、升级 generation revision，并在所有被比较版本上生成一致版本的 suite。旧 profile 下的提速对照和新 profile 下的难度对照保留为两份实验。

离线完整规划可不施加在线 deadline；在线测试保留原 deadline。若要试验更大 planning budget，需独立配置并重新检查 handoff 和预测 horizon，不能为了让 Corridor 通过而在正式对照里隐式放宽。

## 8. 交付、提交与完成标准

建议提交边界：

1. 文档提交：本实施方案。
2. S0：冻结输入与逐请求计时/状态记录，附 baseline 报告。
3. A1：批量初始评估与精确复用。
4. A2：候选×分支 batch、独立优化状态、对应回归测试。
5. A3：网格分块和同步优化，附 A 的性能报告。
6. B1、B2、B3 各自独立提交；实验后端保留开关和适用范围。
7. C1 初始化、C2 多模态搜索各自独立提交。
8. S4：完整报告及独立 timing profile 校准。

每份阶段报告包含：改动、对照输入、运行命令、实际样本数、GPU 条件、正确性结果、耗时分布、失败与回退、是否达到门槛、下一阶段所选配置。数据与结论分开，未执行项保留为待执行。

| 事项 | 当前状态 |
|---|---|
| 确定 1 → 3 → 2 顺序及分步测试方案 | 本文已定义 |
| S0 新测试/报告工具 | 已实现逐请求记录与冻结复放；全量样本待补 |
| A 等价 batch 提速 | 已实现并通过定向回归和在线 easy/hard 测试 |
| B 缓存/距离表/解析区间 | B1 选用；B2/B3 作为关闭的实验后端 |
| C 多模态与初始化 | 已实现独立开关；当前测试未证明收益，默认关闭 |
| S4 全量在线验证与重新校准 | easy/hard 烟测和跨轮校准完成；全量矩阵待执行 |

完成标准是可复现的完整请求提速、可解释的失败率与经过验证的输出，而不是只有某个 kernel 变快。先关闭模态扩展完成 A/B 的性能归因，再用固定预算评价 C 的覆盖收益。
