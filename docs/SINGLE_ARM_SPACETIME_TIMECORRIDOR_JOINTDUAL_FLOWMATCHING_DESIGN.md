# 单臂 Space-Time MPD：时间走廊、时间先验、JointDual 与 Flow Matching 设计

- 核对日期：2026-09-26。
- 代码基线：MPD 仓库提交 `b0f86da`；运行环境 `mpd-splines-public`。
- 范围：Panda/Franka 单臂的静态单次推理、动态世界多次 replan。本文的 JointDual 指空间/时间双分支，不指双臂。
- 本文是研究与实施方案。本次仅新增文档，未训练新模型、未实现新 cost、未重新运行 benchmark。建议超参数和预期收益均不是已测结果。
- 与既有《Space-Time MPD Timing 训练与数据工程实施方案》衔接；对现有功能以实际代码为准。论文依据见第 12 节；本文的组合设计、公式推导与实施判断会与论文原有结论区分。

## 1. 建议实施顺序与主要判断

优先顺序建议为：

~~~text
时间语义/指标与公平基线
    → 固定路径的 phase-time 安全走廊 + 多时序候选 + timing 优化
    → 用走廊优化器生成多模态标签，训练 corridor-conditioned timing
    → 联合训练 JointDual，同步联合采样
    → 按剖析结果进行 Timing Flow Matching / 联合 Flow Matching
~~~

Timing Flow Matching 可以在第二阶段数据稳定后独立做小规模试验，不必等待 JointDual 完成；不建议首次实验同时修改数据、时序表示、网络耦合和生成目标，否则无法归因。

核心判断：

1. **Time-corridor cost 可先实施，而且无需重训。** 它给局部梯度增加“从哪个时间窗口通过”的结构，但直接改善的是推理结果，并没有修改网络先验。
2. **提高时间先验本身，需要增强条件与标签。** 当前 timing 网络主要学 `p(z|P)`，无法从同一空间路径判断本次障碍会在何时经过。建议先学 `p(z|P,C,W_b,m)`，其中 `C` 为 phase-time 走廊，`W_b` 为必要的预测/时间边界，`m` 为离散通行模式。
3. **JointDual 可行，但不是自动提升。** 它的价值是让空间分支也学习时间可行性与联合相关性；若数据仍只是任意全局慢放和随机 near-wait，联合训练不能自动学到动态让行。
4. **Flow Matching 可以替换生成模型训练/采样机制。** 最有希望减少 denoiser/velocity-field 调用与迭代开销；不能补齐缺失的动态世界条件、修复时间戳错误，也不能自动保证无碰撞。
5. **必须比较无需 learned timing 的低维优化器。** 当前时间自由度只有六维；若走廊 + 多初值优化已足够快且可靠，时间先验应通过更低计算量或更高覆盖率证明其增益。

## 2. 当前代码实际具备什么，时间先验可能弱在哪里

### 2.1 已实现能力与边界

| 项目 | 核对结果 | 本地依据 |
|---|---|---|
| `phase5_joint` | 空间 diffusion + 推理时优化时间变量；并非 JointDual 联合训练 checkpoint | `mpd/inference/space_time_guidance.py` |
| F1/F2/F3 | 独立空间与时间模型，区别主要是完整分离/部分重采样/低噪声交替的 sampler | `mpd/inference/factorized_sampler.py` |
| Timing 网络 | 路径特征编码 + 六维 noisy timing + diffusion step；没有当前动态场景编码 | `mpd/timing_training/model.py` |
| 路径条件 | 64 个 phase 点上的 `q,q_s,q_ss,s`，保序 CNN 编码后进入 FiLM MLP | `scripts/train/TIMING_DIFFUSION.md` |
| 时间表示 | `c` 与 `tau_r` 各自标准化为六维、独立 checkpoint | `mpd/inference/learned_timing.py` |
| 已有 cost | 动态碰撞的物理时间 mean/CVaR、速度/加速度超限、时长、timing 平滑 | `SpaceTimeCostEvaluator` |
| 数据 | 从空间路径转 canonical HDF5，TOPP-RA/全局缩放/local slowdown/near-wait，并展开 `tau_r` duration modes | `scripts/spacetime_data/README.md` |
| 数据碰撞验收 | 当前 converter 不实例化 warehouse 碰撞世界，`quality/static_clearance_min` 可为 NaN；`accepted` 不能解释为完整静态/动态碰撞通过 | 同上 |
| JointDual | 已有设计文档和可复用 canonical 字段，未找到专用 JointDual 网络、训练入口、联合 sampler | `docs/Space-Time MPD Timing 训练与数据工程实施方案.md`，`mpd/datasets/spacetime_schema.py` |
| 在线边界 | learned timing 的 `full_path` 显式要求零端点速度/加速度 | `mpd/inference/factorized_guidance.py` |
| ROS2 时间输出 | 每候选独立 `time_from_start`、起点为 0、严格递增、显式 duration 检查 | `scripts/runtime/timing_contract.py` |

因此，“已有 SpaceTimeCostEvaluator”与“已有 time-corridor cost”是两件事；后者需新增。

### 2.2 不输入动态世界，无法学习本次应该先过还是后过

设相同空间路径 `P` 遇到两个世界 `W_1,W_2`：前者要求早通过，后者要求晚通过。如果网络只看 `P`，它最多拟合训练世界边缘化后的分布：

$$
p(z\mid P)=\int p(z\mid P,W)\,p(W\mid P)\,dW.
$$

它可以输出多个时间模式，却不能知道本次应选哪一个。这是条件缺失，不是简单增加训练步数或扩大 MLP 就能消除的问题。

还要区分：现有数据主要由静态路径 retiming 与人工 timing augmentation 产生，尚不能假定它覆盖了真实动态世界的 `p(z|P,W)`。

### 2.3 标签多样不一定意味着有用的决策多样

现有 `tau_r` 按 `y={0.01,0.05,0.15,0.35,0.60}` 展开 duration fraction。它增加时长覆盖，但如果没有 deadline、代价偏好、障碍交互条件，网络没有依据判断哪个 `y` 更适合当前任务。

同理，随机 near-wait 是“会减速”的样本，只有等待位置、等待时长与世界风险相联系时，才是“知道何时让行”的样本。应同时监督可行性与选择原因。

### 2.4 表示、训练分布与在线执行不一致

- 当前 canonical 数据常用 `P[29,7]`，公共 spatial prior 可使用另一控制点数量。现有 basis adaptation 保证按新 basis 求特征，不等价于训练分布完全一致；必须检查 `q_s,q_ss` 和 timing 可行性分布。
- F3 中 timing 看到空间 clean estimate；只在 teacher clean paths 上训练会产生条件分布偏移。对生成/修复路径应重新 retime，而非复制原标签。
- 静态 warehouse timing 数据向 drawer 动态任务迁移，存在几何曲率、目标区域和所需时序分布变化。
- 零端点导数模型与运动中 handoff 的非零速度/加速度有结构差异；现有 bridge 能承担一部分衔接，但不能据此声称任意起始导数已被模型支持。
- 六维时间表示可能难以同时表达多个尖锐等待/加速区；只有在 oracle timing 拟合误差与安全窗口丢失率证明其不足后，再增加 knots/显式 dwell。

### 2.5 首先建立三个诊断实验

| 实验 | 控制条件 | 要回答的问题 |
|---|---|---|
| 固定 `P`，随机/均匀 timing、TOPP-RA、learned timing 使用相同 refinement | 世界、总候选预算、优化次数一致 | 时间先验是否提供有效初始化？ |
| 固定 `P`，高预算走廊 retiming teacher 对比模型 | 使用相同表示和限制 | 差距来自模型，还是六维表示/空间路径本身不可行？ |
| 先验原始输出、guided 输出、最终执行分别统计 | 同一采样 batch 和同一 snapshot | 增益到底来自先验、cost，还是延迟/执行策略？ |

只比较 epsilon-MSE、只看最终成功视频，都不足以回答这些问题。不同表示/normalizer 的 loss 不能直接比较大小并据此排序模型。

## 3. 统一数学表示、物理时钟与可行性

### 3.1 明确区分三种时间

- `s∈[0,1]`：路径相位，无量纲。
- `t(s)`：从本条候选轨迹开始起算的物理秒数。
- `λ∈[0,1]` 或 `k`：生成过程的 flow time / diffusion step，不是机器人时间。
- `τ`：`tau_r` 中的 duration logit，不是上面的任一种时间。

空间 B-spline 与 timing density：

$$
q(s)=B_P(s)P,\qquad u(s)=\frac{dt}{ds}>0,\qquad
 t(s)=\int_0^s u(v)\,dv,\qquad T=t(1).
$$

两个已有 codec 的核心为：

$$
\text{c: }u(s)=u_{\min}+\operatorname{softplus}(B_c(s)c),
$$

$$
\text{tau\_r: }\pi(s)=\epsilon_\pi+(1-\epsilon_\pi)
\frac{e^{g_r(s)}}{\int_0^1e^{g_r(v)}dv},\quad u(s)=T\pi(s),
$$

$$
T=T_{\min}(P,r)+(T_{\max}-T_{\min}(P,r))\sigma(\tau)
\quad\text{仅在 }T_{\min}<T_{\max}\text{ 时可行。}
$$

当前 codec 对 `T_min≥T_max` 不掩盖超期；新设计也必须保留显式不可行处理。

速度、加速度链式法则：

$$
\dot q=\frac{q_s}{u},\qquad
\ddot q=\frac{q_{ss}}{u^2}-\frac{q_su_s}{u^3}.
$$

归一化 density 下的采样时长下界为：

$$
T_v=\max_{s,j}\frac{|q_{s,j}|}{v_{\max,j}\pi},\qquad
T_a=\sqrt{\max_{s,j}\frac{|q_{ss,j}/\pi^2-q_{s,j}\pi_s/\pi^3|}{a_{\max,j}}},
$$

$$
T_{\min}=\max(T_{\rm floor},T_v,T_a).
$$

这是给定路径与 timing shape 的采样下界，不是整条连续轨迹的严格动力学证明，也不是对所有路径的全局最短时间。

### 3.2 走廊必须使用执行时间，不得拿推理开始时间代替

令 `t_snap` 为世界 snapshot 的时间，`t_exec` 为本条轨迹计划开始/衔接时刻：

$$
\Delta=t_{\rm exec}-t_{\rm snap},\qquad
 t_{\rm world}(s)=t_{\rm snap}+\Delta+t(s).
$$

风险查询须使用 `Δ+t(s)` 相对 snapshot 的预测。等待前缀、bridge、控制提交提前量，以及继续执行旧轨迹的时间都必须纳入契约。

当前 `dynamic_runtime_engine.py` 已要求 `trajectory_start_unix_ns`。新走廊生成器应消费同一契约，避免在 helper 内再加一次 offset。计划 handoff 改变、snapshot 更新、结果过 deadline，都需要重验或重建走廊。

不能假定 `Δ=神经网络推理耗时`：当前 ROS2 的 planning budget、lead/commit margin 与 handoff search 也会决定轨迹起始时刻。

### 3.3 全机器人风险场

令第 `ℓ` 个碰撞球中心为 `x_ℓ(q)`，半径为 `r_ℓ`，第 `j` 个预测障碍的 SDF 为 `d_j(x,t)`，额外安全裕度为 `μ_ℓj(t)`：

$$
D_P(s,t)=\min_{\ell,j}\left[d_j(x_\ell(q(s)),t_{\rm exec}+t)-r_\ell-\mu_{\ell j}(t)\right].
$$

走廊可行集：

$$
\mathcal F_P=\{(s,t):D_P(s,t)\ge0\}.
$$

必须覆盖机械臂全身与抓取物，不能仅用 EE anchor。静态碰撞、自碰撞、关节限制仍单独检查；固定 `P` 时 timing 不能修复静态几何碰撞。

注意当前动态世界查询可能已包含机器人球半径或障碍 inflation。实际实现先统一返回值语义，避免按上述概念公式重复减半径/重复膨胀。

## 4. 多种 time-corridor 与相关 cost 设计

以下 A–G 为组合建议，并非宣称某篇论文原样采用。第 12 节解释对应文献启发。

### 4.1 A：phase-time 安全区间 + 分支走廊 cost（首选）

固定候选 `P`，在相位点 `s_i` 查询未来风险，得到若干不相连的安全区间：

$$
\mathcal I_i=\bigcup_k[a_{ik},b_{ik}].
$$

为候选选定连通且运动学可达的区间序列 `m=(k_0,…,k_N)`。在这个分支内使用：

$$
J_{\rm corridor}(z;m,P)=\sum_i w_i\left[
\left(\frac{[a_{ik_i}+\delta_t-t_z(s_i)]_+}{t_{\rm scale}}\right)^2+
\left(\frac{[t_z(s_i)-b_{ik_i}+\delta_t]_+}{t_{\rm scale}}\right)^2\right].
$$

`δ_t` 为时间余量；若区间宽度小于 `2δ_t`，直接去除。`t_scale` 用于量纲归一，`w_i` 用 quadrature 或归一权重避免网格越密 cost 越大。边界处可用 Huber 等平滑替代。

**有用之处：** 即使当前 SDF 时间梯度很弱，该 cost 也给出明确的提前/延后方向；远离碰撞时仍能保持通行时序。

**必须防止：** 将“先过”和“后过”两个区间平均。例如可通行时间为 `[1,2]∪[4,6]`，平均时间 `3` 可能最危险。应对候选分别分配区间分支，不能取一个包住二者的区间。

可用下式作近似分支评分：

$$
J_{\rm softmin}=-\beta^{-1}\log\sum_m e^{-\beta J_m}.
$$

但 softmin 不证明区间可行、可能在对称处出现弱梯度。建议搜索/枚举少量 `m` 后固定各候选分支优化，而不是依靠 softmin 解决全部离散决策。

**走廊搜索与验证：**

1. 用当前空间 clean candidate 生成全身 `D_P(s,t)` 网格。
2. 合并相邻安全单元，边界加密与收缩。
3. 搜索若干通行序列，保留早过/后过/局部慢行等不同解。
4. 检查相邻 phase 之间的边，而不仅是节点。
5. 以相应时间曲线初始化或引导 `c/tau_r`，最后重新做候选对应物理时间的稠密检测。

SIPP 提供了“状态 + 安全区间”的搜索思想 [R5]；本方案将状态限制到某一空间路径相位上。对有加速度约束的机械臂，需传播速度可达范围，例如节点 `(s_i,I_k,v_s)`；仅按最早到达时间保留一个标签，不一定保持足够的未来可达性。不能直接继承原始 SIPP 的完整性结论。

**离散检查的边界：** 若有可信的 Lipschitz 界 `L_s,L_t`，安全网格点还可要求

$$
D_P(s_i,t_j)\ge L_s\Delta s/2+L_t\Delta t/2+\delta_d.
$$

才可能覆盖相应邻域。没有可验证的界时，应明确称为采样检查，并对高速物体、近碰撞段加密，而不是声称连续安全。

**算量：** 风险网格粗略开销 `O(BN_sN_tN_{\rm sphere}N_{\rm obj})`；不能为所有 100 个候选、每个 denoise step 都无条件重建。先以分块评估、少量空间代表候选、低噪声构建、缓存失效规则和分项计时验证收益；任何省算近似不能替代最终完整验证。

### 4.2 B：冲突事件的先后顺序 cost（较轻量）

把风险场中的连通冲突区域提取为事件 `e`。设机器人进入/离开该空间冲突段的相位为 `s_e^-、s_e^+`，障碍占用时间近似为 `[a_e,b_e]`。两个保守选择为：

$$
J_e^{\rm before}=[t(s_e^+)-(a_e-\delta_t)]_+^2,
\qquad
J_e^{\rm after}=[b_e+\delta_t-t(s_e^-)]_+^2.
$$

它约束的是“整个相关臂段通过冲突区”，比只对齐 EE 在 anchor 的一个时刻合理。多个事件用离散序列 `m_e∈{before,after}` 描述；筛去无法在速度/加速度/截止时间内实现的序列。

适合单次 crossing、连续过门和 conveyor 交互。对于回转、曲线、反复进出等复杂运动，事件可能不止一个；上述矩形占用近似须重新分段或回退到完整 phase-time 走廊。

### 4.3 C：直接连续时空风险 + mean/CVaR（已有基础，保留）

定义非负风险 `ℓ(s)=[-D_P(s,t_z(s))]_+²`。物理时间 mean 为：

$$
J_{\rm mean}=\frac1T\int_0^1\ell(s)u(s)\,ds.
$$

对“最危险的 `ρ` 比例物理时间”，可写为：

$$
J_{\rm CVaR}=\min_\eta\left\{\eta+\frac1{\rho T}
\int_0^1[\ell(s)-\eta]_+u(s)\,ds\right\}.
$$

现有 `mean_cvar_dynamic_risk` 已沿此方向实现；新增走廊时应复用并核对其时间权重，而不是重复增加同名风险项。

时间梯度来源为：

$$
\frac{\partial D_P(s,t_z(s))}{\partial z}
=\frac{\partial D_P}{\partial t}\frac{\partial t_z(s)}{\partial z}.
$$

对纯平移障碍，`∂_t d≈−∇_x d·v_obs`；旋转、尺度膨胀等还要加相应项。最近接近时 `∂_t d` 可能接近零；多个障碍最小值切换也会使梯度不平滑。这是走廊/先后序列可以帮助局部优化的原因。

此处 CVaR 是对时间尾部风险，不等于对预测误差分布做风险控制。如果需要后者，应另对未来轨迹样本 `W^(k)` 聚合风险，并分别报告两种 CVaR 的含义。

### 4.4 D：信赖域、原始先验与引导权重

规划可写为目标分布：

$$
p^*(P,z\mid x,W)\propto p_\theta(P,z\mid x)\exp[-J(P,z;W)/\gamma].
$$

这是建模目标；使用 clean-estimate 梯度修正并不自动等于精确后验采样 [R1]。

在一个原始 timing sample `z^(0)` 附近可增加局部信赖域：

$$
J_{\rm trust}=\tfrac12(z-z^{(0)})^\top M(z-z^{(0)}),
$$

`M` 取标准化 latent 的对角尺度或经验局部精度。它约束过大跳变、减少极端 timing，但不是 learned prior 的精确负对数似然；单个 sample 的二次惩罚也不能代表完整多模态分布。

建议高噪声阶段保留分布探索，在可信 clean estimate 出现后增加走廊/物理 cost。空间、时间分别裁剪梯度，监测每分支梯度范数和修正距离。长时间无可行解时应换分支或空间路径，不要无限增大 cost 权重。

### 4.5 E：让 cost 转化为更好的训练先验（最关键）

推荐条件：

$$
p_\phi(z\mid \operatorname{Enc}_P(P),\operatorname{Enc}_C(C),b,m,w).
$$

其中 `b` 包括计划执行相对 snapshot 的 offset、有效预测 horizon、初始/终端边界；`m` 是通行模式；`w` 是速度/时长/风险偏好。

三种可选训练方案：

| 方案 | 做法 | 价值与限制 |
|---|---|---|
| E1 多模态 teacher imitation | 优化器在同一 `P,W` 下产出多个有效时序；按模式均衡训练 | 最易评估；teacher 若单一，先验会继承其覆盖不足 |
| E2 低噪声物理辅助损失 | epsilon loss 外，对去噪 clean estimate 的 arrival、动力学与碰撞增加辅助项 | 可减少解码误差；高噪声 cost 梯度不可靠，权重不能压倒生成训练 |
| E3 cost-guided 样本蒸馏 | 离线运行高质量 guided planner，复核后把 guided outputs 作为有条件标签 | 把推理计算摊到训练；需要保存失败与数据来源，避免只保留容易成功样本 |

E1/E3 默认优先。一个可实施的 loss 为：

$$
L=L_\epsilon+\lambda_t\mathbb E_k\|\hat t(s_k)-t^*(s_k)\|^2/t_{\rm scale}^2
+\lambda_vL_v+\lambda_aL_a+\lambda_CL_C+\lambda_mL_{\rm mode}.
$$

其中 arrival 重建项只与当前配对 teacher 比较；不能先把同条件的 before/after 标签平均后做回归。辅助项只在低噪声区启用或按 SNR 加权，并在可行样本上保持模式覆盖。

若做质量加权，权重可以是截断的 `exp(−J/β)`，并在同任务/模式内归一。不得把负 clearance、错误时间戳或不满足表示约束的样本作为正样本；失败可供独立可行性分类器学习。

### 4.6 F：时长、jerk 与跨 replan 稳定性

总 cost 建议分清目标：

$$
J=w_CJ_C+w_RJ_R+w_vJ_v+w_aJ_a+w_jJ_j+w_TT/T_{\rm ref}+w_HJ_H+w_SJ_S.
$$

- 无时间 cost：设 `w_T=0`，但保留最大时长、预测有效时域、动力学限制与安全终端保持条件。否则“无限慢/推迟到预测结束之后”会成为投机解。
- 有时间 cost：正的 `w_T` 鼓励较早完成；与 clearance/动力学做 Pareto sweep，而非单独追求最短时间。
- `J_H` 验证到达后指定 hold 窗口的风险，避免候选停在即将被障碍扫过的位置。
- `J_S` 可约束重规划后相同未来物理时刻的命令变化；比较对齐后的 `q_new(t_abs)` 与 `q_old(t_abs)`，不能比较不同时间轴的同序号 waypoint。

物理 jerk 由链式法则得：

$$
\dddot q=\frac{q_{sss}}{u^3}-\frac{3q_{ss}u_s}{u^4}
-\frac{q_su_{ss}}{u^4}+\frac{3q_su_s^2}{u^5},
\quad J_j=\int_0^1\|\dddot q\|^2u\,ds.
$$

`u_s/u` 的平滑惩罚不能替代物理 jerk。还应检查 knots 附近与实际插值器；B-spline 平滑不自动意味着所有阶数的物理导数都连续或限幅。

### 4.7 G：预测与启动延迟鲁棒性

建议对执行 offset 误差 `δ∈[−δ_-,δ_+]` 收缩窗口，或优化：

$$
J_{\rm robust}=\max_{\delta\in\mathcal D}J(P,z;\Delta+\delta,W).
$$

有限 `δ` 采样是近似，要记录其范围。若用线性化高斯距离约束：

$$
\mathbb E[D]-\Phi^{-1}(1-\varepsilon)\sqrt{n^\top\Sigma n}\ge0,
$$

需明确 `n` 为距离敏感方向、`Σ` 为误差协方差，且这仅是局部/点时刻近似，不是完整路径碰撞概率保证。与现有 inflation 相同来源的 uncertainty 不应重复计入。

预测更新后，仅当路径、预测模型、scene timestamp、execution offset 和 margin 均未失效时才能复用走廊。机器人的停等位置必须在整个等待区间安全。

## 5. 一个实用的最小可实施版本

### 5.1 固定路径六维优化基线

先保留 spatial MPD，比较四类初始化：均匀 timing、TOPP-RA、随机 `tau_r`、learned timing。每类用相同总候选数、相同风险场和相同 refinement 预算。

优化变量为 `z∈R^6`，采用多初值 L-BFGS/Adam 或小批量 CEM；对离散 before/after 分支分别优化。CEM/MPPI 平均只在同一可行模式内做，避免跨互斥时窗平均后穿入危险区。最后选择实际通过验证的个体。

另一个 teacher baseline 可直接用正 density 控制点 `a`：

$$
u(s)=B_u(s)a,\quad a\ge\epsilon,\quad t=Aa.
$$

固定分支的 arrival 区间 `l≤Aa≤h` 和采样速度下界 `u(s_i)≥|q_{s,j}|/v_max,j` 是线性的，可构造带平滑正则的 QP。加速度条件

$$
|q_{ss,j}u-q_{s,j}u_s|\le a_{\max,j}u^3
$$

仍非线性，完整问题不能被称为“一次凸 QP 即解决”；QP 输出需非线性 refinement、投影与复核。拟合回现有 `c/tau_r` 后也必须再次验证。

### 5.2 推理流程

~~~text
输入：机器人边界、当前世界 snapshot、计划轨迹起始时刻、冻结候选预算 B
1. 生成空间候选 P，保留有效的空间多样性。
2. 对选定 P 构建全身 phase-time 风险网格与安全区间。
3. 搜索最多 M 个可达时序分支，按预算分配候选；总数仍为 B。
4. 在每分支内生成/初始化 z；最初可使用现有 timing checkpoint。
5. 在 clean (P,z) 上用走廊与现有物理 cost 做有限次修正。
6. P 改变超过允许范围时重建走廊；失效分支不继续沿用旧窗口。
7. 对完整轨迹、前缀/bridge、终端 hold 和最新世界进行验证。
8. 导出每候选时间轴；在线 guard 继续监测执行。
~~~

`B=100、M=2–4、N_s=32/64、时间网格步长=0.05/0.1 s` 仅是初始性能 sweep 建议，不是安全分辨率承诺。按速度与最小几何间隙决定加密；最终验证独立于走廊网格。

第一轮可以固定 `P` 只优化时间，把“走廊帮助选时”与“空间绕行改变”分开验证。随后再接 F3 的联合 refinement。

### 5.3 exact wait 的表示问题

当前严格单调、有限正 `u(s)` 支持 near-wait，但一般不能在路径中间表达 `s` 完全不变的有限时间 dwell。

启动等待 `w_0` 可先由 runtime 在 `q_start` 处显式保持，后续轨迹的起始绝对时间改为 `t_exec+w_0`，同时检测整个保持段与退出段。

中途等待可采用分段运动 + 显式 dwell：减速到等待状态、保持 `w_e`、再出发，并让相邻段满足导数连接条件。状态停住时必须满足 `dq=0`；不能将运动中一个 waypoint 简单复制若干次就宣称动力学可行。

若扩展轨迹格式，使用连续递增时间和恒定位置保持段是可行的；需要新增分段 metadata 与解析器测试，不能把不连续 `t(s)` 偷塞进当前六维 codec。

## 6. JointDual 的数学依据与网络设计

### 6.1 它与 factorized 的实际区别

Factorized 本身已可定义联合分布：

$$
p(P,z\mid x)=p_\theta(P\mid x)p_\phi(z\mid P,x).
$$

因此不能把 JointDual 的卖点写成“原来没有联合分布，现在才有”。关键在于采样时双向信息如何作用：

$$
\nabla_P\log p(P,z\mid x)=\nabla_P\log p_\theta(P\mid x)
+\nabla_P\log p_\phi(z\mid P,x).
$$

当前 factorized 空间 denoiser 不输入 timing，时间主要通过 cost 梯度影响空间；上述概率分解的第二项并没有因此被完整实现。JointDual 通过配对联合加噪学习耦合 score，在相应噪声分布下让两分支互相影响。

理论可行不等于经验一定更好：若路径和时间耦合很弱，或条件 timing 已足够好，JointDual 可能只增加训练和推理成本。

### 6.2 建议结构

~~~text
P_k [B,H_free,7] → Spatial TemporalUnet/1D Transformer → ε_P
                              ↕ cross adapters
z_k [B,6]        → Timing residual MLP / timing tokens → ε_z
                              ↑
       start/goal、边界、noise-level embedding、可选 world/event tokens
~~~

- 保留两个输出头与独立 normalizer，不把六维 timing 粗暴广播成第八个关节。
- 空间到时间：保序的空间 token cross-attention 或 pooling，提供路径形状/导数线索。
- 时间到空间：timing embedding 对空间层做 FiLM/零初始化 residual adapter。
- 必须显式编码两分支的噪声级别。旧 timing 的 clean-path encoder 直接读取 noisy `P_k` 会放大导数噪声，需针对带噪条件重新训练，或使用带置信度的 self-conditioned clean estimate。
- 第一版采用同步 diffusion schedule；若空间和 timing 原 schedule 不同，对齐归一噪声程度/log-SNR，不能仅使用相同整数 step。

关于 world 条件有两条路线：

| 路线 | 定义 | 推荐用途 |
|---|---|---|
| J0 运动学联合先验 | `p(P,z|start,goal,boundary)`；当前世界只进入 guidance | 首先验证双向耦合是否有效，与原有设计文档一致 |
| J1 动态条件联合先验 | `p(P,z|start,goal,boundary,Enc(W),m)` | 学习绕行与让行的联合选择，作为下一步主模型 |

JointDual 不能预先把“最终真实路径对应的走廊”作为可用输入，因为最终路径尚未生成。J1 优先使用不依赖目标轨迹的 obstacle/event tokens；低噪声阶段再从预测的 clean `P` 在线构建走廊。

如果训练时使用 teacher `P*` 计算的走廊，而推理时使用 noisy/预测路径走廊，必须显式做 rollout/curriculum 和混合条件训练，否则会泄漏目标信息并产生明显 train–test mismatch。

可直接用于第一版实现的张量接口建议如下；这些维度是试验配置，不是现有网络已经支持的接口：

| 输入/分支 | 建议形状与编码 | 必须保留的语义 |
|---|---|---|
| 空间控制点 | `[B,H_free,7]`，各关节标准化；投影到 128/256 维序列 token | 控制点位置、两分支 log-SNR、边界 mask |
| 时间 latent | `[B,6]`，MLP 或 6 个带变量身份的 token | c/tau_r 类型、独立尺度、噪声级别 |
| 固定路径特征 | 64 个 phase 点的 `q,q_s,q_ss,s`，沿 phase 编码 | 用于条件 timing；不把高噪声导数当真实动力学 |
| 走廊 | `[B,N_s,K_max,2]` 的区间端点，加 validity mask 与 phase/time embedding | 保留不相连区间；截断会丢失分支，不能合并成大区间 |
| 动态世界 | `[B,M,K,F]` 的障碍预测 token；先沿预测时间编码，再跨物体 attention | 机器人坐标系下的位姿/形状、相对秒数、速度、有效 mask、预测不确定性 |
| 全局边界 | 起终点、execution offset、horizon、速度/加速度上限、偏好权重 | 使用推理时可获得的信息；绝对时间戳只作 provenance |

例如先 sweep `K=16/32` 个预测时刻、128/256 维 hidden size、少量 cross-adapter 层；实际 M 与 K 按场景上限及预测 horizon 设定并使用 mask。姿态采用连续、约定明确的编码，静态场景若不再固定，也需独立几何条件或显式 guidance。为公平评估，先让 factorized 与 JointDual 使用相同 world encoder。

离散模式 `m` 可由走廊分支搜索分配，也可增加 `p(m|P,W,b)` 分类头，在可行分支中采样；JointDual 尚未确定 P 时优先使用任务/世界级模式或低噪声在线分支搜索。训练使用 teacher mode label，推理不能直接读取该标签：必须评估预测/搜索 m 的完整流程，并保留少数模式的候选配额。

### 6.3 联合加噪与损失

在训练样本的标准化自由坐标上：

$$
P_k=\alpha_k^P P_0+\sigma_k^P\epsilon_P,\qquad
z_k=\alpha_k^z z_0+\sigma_k^z\epsilon_z.
$$

已知起终点与固定导数由 mask/边界解码器施加，不把硬约束维度随意加噪后再计入主损失。

$$
(\hat\epsilon_P,\hat\epsilon_z)=F_\psi(P_k,z_k,k_P,k_z,c),
$$

$$
L_{\rm joint}=\underbrace{\operatorname{mean}_{H,D}|\epsilon_P-\hat\epsilon_P|^2}_{L_P}
+\lambda_z\underbrace{\operatorname{mean}_{6}|\epsilon_z-\hat\epsilon_z|^2}_{L_z}
+\lambda_{\rm phys}L_{\rm phys}(\hat P_0,\hat z_0).
$$

分支分别取 mean，防止空间维数淹没时间。`λ_z∈{0.5,1,2}` 可作起始 sweep；物理辅助项只在低噪声区施加。

`tau_r` 的 `T_min(P,r)` 同时依赖两分支；guidance 里要保留正确依赖，或者显式声明交替冻结近似。不能直接在高噪声 `P_k` 上计算物理 `T_min`、碰撞或加速度，再把爆炸梯度解释成模型能力不足。

### 6.4 warm start 与同步推理

建议先移植空间 backbone 和 timing 分支，cross adapter 零初始化；先训 timing/adapter，再低学习率解冻空间 backbone。旧 timing 权重只能是初始化，不能未经训练就认为适应了 noisy-space 条件。

同步采样的每一步：

~~~text
读取同一旧状态 (P_k,z_k)
    → 同一轮计算 εP,εz
    → 重建 clean estimates (P̂0,ẑ0)，施加边界
    → 在低噪声时做 time-corridor/风险/动力学 refinement
    → 用各自 noise schedule 更新到 (P_{k-1},z_{k-1})
~~~

不要先更新 `P` 再让 `z` 使用更新后的 `P`，却把算法称作同步 JointDual；那是额外的交替 sampler 消融。

先只训练同步噪声级别时，也不要在部署时任意采用异步噪声组合。若需要异步/交替，需要训练覆盖 `(k_P,k_z)` 组合和 clean/noisy 条件边界。

### 6.5 非零边界与 ROS2 的后续扩展

令候选起始物理状态为 `(q_0,v_0,a_0)`，则必须满足：

$$
q_s(0)=u(0)v_0,\qquad
q_{ss}(0)=u(0)^2a_0+v_0u_s(0).
$$

空间端点控制点与 timing 边界因此耦合。扩展时需要 boundary-conditioned decoder 或联合投影，不能仅加一项 endpoint penalty。

建议 J0/J1 初版继续保持现有 rest-to-rest 语义及 bridge；再单独加入 motion-to-motion 数据、边界解码器、接续验证与 timing-dependent endpoint Jacobian。新功能用独立配置开启。

## 7. 轨迹数据：从静态 retiming 到动态决策配对数据

### 7.1 分开建立三层数据，不要直接废弃现有数据

| 数据层 | 每条样本的主要内容 | 能训练的能力 | 不能据此宣称的能力 |
|---|---|---|---|
| D0：运动学 retiming | `P,z,t(s)` 与边界、限速、限加速度 | 路径形状与速度分配的相关性；J0 warm start | 对当前动态障碍正确让行 |
| D1：固定路径动态 timing | 同一 `P`、不同世界 `W`、多个可行时间分支 | corridor-conditioned timing；before/after、near-wait 选择 | 必须改变空间绕行才能成功的任务 |
| D2：联合动态轨迹 | 同一任务/世界下多个不同 `P,z` 配对 | J1 的空间绕行与时序联合决策 | 未覆盖场景的可靠泛化或无条件安全保证 |

现有 canonical HDF5 可以作为 D0 原料，但要补碰撞验收；不能只读取旧 `accepted` 字段就把它当成 D1/D2 的正样本。

**D1 是最先值得新增的数据。** 固定空间路径，改变 crossing time、方向、速度、障碍尺寸、运动模型和执行 offset，使相同 `P` 既出现“早过更好”的世界，也出现“晚过更好”的世界。这样才能检验模型是否使用世界条件，而非只记住路径对应的平均时长。

### 7.2 场景设计与 teacher 生成流程

建议覆盖以下场景，并记录生成参数而不只保存 seed：

1. 无动态障碍：检查加入新模型后基础能力是否退化。
2. 单次 crossing：两种时间分支都存在、只存在早过、只存在晚过，分别保留。
3. 双次/多物体 crossing：不同顺序组合、重叠危险窗口和窄安全窗口。
4. 周期或转弯运动：测试单事件近似不成立时的完整走廊。
5. 必须改变空间路径：最短路径不可 retime，但另一条空间通道可行。
6. 等待位置危险：起点或中途等待会被扫过，防止模型学会无条件停等。
7. 不可行/截止时间不足：保存失败原因，供拒绝决策与可行性评估，不冒充正轨迹。
8. 在线预测变化：启动延迟、预测误差、目标/障碍更新，用于鲁棒性与 replan 测试。

每个任务的推荐生成流程：

~~~text
静态场景、起终点、边界与机器人限制
    → 多条静态空间候选，保留不同绕行模式
    → 采样并冻结动态世界、snapshot、execution offset、horizon
    → 构建全身 phase-time 走廊，搜索多个事件/区间分支
    → 固定路径多初值 retiming，生成 D1
    → 空间-时间联合优化或独立时空搜索，补充 D2
    → 拟合回目标 B-spline 与 c/tau_r 表示
    → 在拟合后的轨迹上重新检查全部限制
    → 去重、按模式保留、记录失败与完整 provenance
~~~

Teacher 至少保留两个相互独立的来源，以检查自己的 cost 是否有系统偏差：

- 主来源：走廊分支搜索 + 高预算 timing 优化 + 联合 refinement。
- 对照来源：在适用约束下用 ST-RRT* 等时空搜索产生路径，再平滑/retime/拟合并重新验证。ST-RRT* 的速度约束与到达时间目标并不自动包含本仓库全部加速度、jerk、B-spline 和执行边界要求 [R8]。

TOPP-RA 适合提供给定几何路径的可行/快速 retiming 初始化，不能直接把它的输出视为满足绝对时间障碍约束的解 [R7]。遇到动态障碍后，增加全局时长也不保证更安全，必须按新的物理时间重验。

如果 teacher 使用真实未来轨迹，而部署只有带误差预测，应把这类样本标记为 `oracle_future`：可以评估上界，但不能让推理输入偷偷读取真实未来。用于部署训练的数据应使用实际可获得的预测条件；预测与真实世界分别存储，验收说明使用的是哪一个。

### 7.3 保持模式，不只保留一个最优解

建议每个可行任务保留少量不同的 `(spatial_mode,event_order,duration_bin)` 解，再在各组内择优，而不是只保留全局最短时间解。三个重要区分：

- 不同 random seed 并不必然产生不同运动模式。
- 同一路径的全局时间缩放增加 timing 覆盖，但不等于学到了新的避障决策。
- 相近 cost 的 before/after 或上下绕行，不应在监督标签中做平均。

可按以下距离与语义共同去重：

$$
d_P(P_a,P_b)=\left(\int_0^1\|q_a(s)-q_b(s)\|_2^2ds\right)^{1/2},
\qquad d_t=\left(\int_0^1|t_a(s)-t_b(s)|^2ds\right)^{1/2}.
$$

空间比较前统一相位约定；相同几何路径的不同参数化不能被误当作不同绕行。语义模式由事件顺序、空间通道或任务标签确定，不能只靠欧氏距离。

若要求“偏快”和“偏稳”均可控，可在训练条件中加入 cost preference `w`，离线对一组权重生成 Pareto 解。没有偏好条件却只以混合标签训练，推理时仍需搜索/排序，不能期待网络自动选中用户当前想要的策略。

### 7.4 数据字段与版本建议

以下是**拟新增**的动态扩展字段，不是声称当前 schema 已具备。保留 `spacetime_mpd_v1` 的读取能力；新增有版本号的 sidecar 或新 schema，并由显式 loader 选择，不能悄悄改变旧字段语义。

~~~yaml
schema_version: spacetime_dynamic_mpd_v2_proposed
trajectory:
  P: [H, 7]                    # 与声明的 basis/knots 一致
  timing_representation: tau_r # 或 c，分别训练与统计
  z: [6]                      # 未归一化编码，同时保存 normalizer 版本
  reference_phase: [N]
  time_from_start: [N]         # 秒，首项为零，严格递增
  duration: scalar
boundary:
  q_start: [7]
  q_goal: [7]
  dq_start: [7]
  ddq_start: [7]
  terminal_derivative_policy: rest_to_rest
world:
  scene_id: string
  static_asset_hash: string
  robot_collision_model_hash: string
  frame_id: string
  prediction_model_version: string
  obstacle_ids_and_parameters: variable_length
  prediction_time_from_snapshot: [K]
  predicted_obstacle_states: [M, K, state_dim]
  prediction_valid_mask: [M, K]
  prediction_uncertainty: optional
  snapshot_timestamp_ns: integer
  execution_offset_s: scalar
  prediction_horizon_s: scalar
decision:
  spatial_mode_id: string
  event_order: variable_length
  selected_safe_intervals: variable_length
  cost_preference: vector
quality:
  static_clearance_min: scalar
  dynamic_clearance_min: scalar
  self_clearance_min: scalar
  max_velocity_ratio: scalar
  max_acceleration_ratio: scalar
  jerk_integral: scalar
  boundary_residual: scalar
  timing_fit_error: scalar
  hold_and_prefix_valid: boolean
  validation_resolution: object
  accepted_dynamic: boolean
provenance:
  task_group_id: string
  base_path_id: string
  source_pair_id: string
  world_family_id: string
  teacher_and_config_hash: string
  failure_reason: optional
  oracle_future: boolean
~~~

训练网络通常使用相对秒数和机器人参考系；绝对纳秒时间戳留作可追溯信息，不应直接当作有意义的数值特征。

如果 source 文件可在多台机器间移动，资产引用采用仓库相对路径与内容 hash，不依赖开发机的绝对路径。训练与 benchmark 使用的机器人球模型、附着物、margin 和 frame 必须可核对。

### 7.5 表示拟合、验收和拆分

从 teacher 高分辨率时间曲线拟合六维 codec 时，联合最小化 arrival error 与约束违例：

$$
\min_z \sum_i w_i|t_z(s_i)-t_i^*|^2
+\lambda_CJ_C(z)+\lambda_vJ_v(z)+\lambda_aJ_a(z).
$$

只做 latent MSE 不够：小的 latent 误差也可能跨越一个窄时间窗口。若六维拟合丢失可行性，记录 `representation_infeasible`，不能通过降低 margin 或放宽碰撞检查伪造训练成功。

对每个最终正样本检查：时间单调性、起终点、静态/动态/自碰撞、关节位置/速度/加速度、有效预测时域、前缀/bridge/hold，并记录实际插值后的结果。若要声称 torque-feasible，应另有载荷、惯性与逆动力学验证；目前仅有速度加速度验收时，应称为运动学可行。

拆分要求：

- 同一 source path、反向配对、timing augmentation 和对应动态 variants 放在同一个 split；现有 source-group 的约定继续保留。
- 另外按 world family、静态布局、障碍运动组合保留 OOD 测试集，区分已知场景新任务与真正新布局。
- 如路径组与世界组交叉复用，要按关联分组或设计独立测试集，不能分别随机 split 后产生交叉泄漏。
- normalizer 只在 train split 拟合；统计有效样本比例与各模式覆盖，避免大量 near-duplicate 压倒少数困难模式。

Pilot 可先用约 1,000 个基础任务，每任务 2–4 个世界、每世界最多保留 2–4 个验证通过的不同模式；这是工程试验规模建议，不是已生成数量或训练效果保证。先看 teacher 成功率、表示损失率和条件可辨识性，再决定扩到多少数据。

### 7.6 训练组织与在线分布适配

建议新增 `JointDatasetView`，从同一 canonical 样本同时返回 `P,z,condition,masks`。禁止从空间 dataset 与 timing dataset 各自随机取样后拼成“联合样本”。

训练阶段可为：

1. D0：复用既有路径与 timing 数据，建立 J0 与 factorized 的同数据对照。
2. D1：固定 spatial backbone，训练 corridor/world-conditioned timing；报告 unguided 的可行样本率。
3. D2：zero-init cross adapter + 联合训练，随后小学习率解冻 spatial backbone。
4. Rollout：采集当前模型输出的空间候选，重新生成 corridor 与 retiming 标签，修正 clean-teacher 条件到实际候选条件的偏移。

以“raw world + 当前预测路径生成的 corridor”作为训练条件时，保存其生成版本。仅对路径加随机扰动但继续沿用旧 timing/corridor 标签，可能直接制造错误监督。

## 8. Diffusion 换成 Flow Matching：可行，但要按层替换

### 8.1 最小改造边界

保留以下模块：B-spline 表示、`c/tau_r` codec、FK/碰撞模型、boundary decoder、cost、最终验证与 ROS2 输出契约。替换的是生成过程的训练目标与 sampler，不是物理时间模型本身。

推荐先对六维 timing 做 FM，对照同数据的 timing diffusion。若收益成立，再训练 JointDual-FM；空间 backbone、条件编码器可复用结构，但已有 epsilon checkpoint 不能只换函数名就当作训练好的 velocity-field checkpoint。

Flow Matching 学习沿生成时间的向量场，推理仍通常需要数值积分 [R9][R10]。生成时间不是机器人执行时间。Diffusion 训练通常也直接抽取噪声级并构造 noisy sample，而不是逐步运行整个正向加噪链；不能用“省去训练时逐步加噪”来推断 FM 的加速收益。

另一个不同的选项是将已有 diffusion 的 epsilon/score 按原 schedule 转为 probability-flow ODE 或使用少步 diffusion sampler。这是更换已有模型的采样方法，不等于训练了 FM，应单独作为低成本加速基线；其与新 CFM 模型的质量、NFE 和 guidance 接口分别验证 [R10]。

### 8.2 条件线性桥与训练目标

设 `X_1` 是标准化的有效训练样本；timing-only 时 `X=z`，joint 时 `X=(P_free,z)`。令源噪声 `X_0~N(0,I)`，生成时间 `λ∈[0,1]`：

$$
X_\lambda=(1-\lambda)X_0+\lambda X_1,
\qquad U^*=X_1-X_0.
$$

一个可实施的 conditional flow-matching 目标为：

$$
L_{\rm CFM}=\mathbb E_{X_0,X_1,\lambda,c}
\left[\|V_\psi(X_\lambda,\lambda,c)-(X_1-X_0)\|_M^2\right].
$$

这里 `c` 表示条件集合而非 timing codec 的 `c` 控制点；实现建议命名为 `context` 避免歧义。`M` 对空间/时间分支分别做维度归一与权重平衡。

采样求解：

$$
\frac{dX_\lambda}{d\lambda}=V_\psi(X_\lambda,\lambda,context),
\quad X_0\sim\mathcal N(0,I),\quad \lambda:0\rightarrow1.
$$

上述是标准线性条件桥的具体选择，不是所有 FM 的唯一形式 [R9][R10]。若做 minibatch OT pairing，只能在条件相同或严格兼容的组内耦合；不能把不同任务/世界的目标任意配对后忽略其条件一致性。

多模态时，即使每个训练 pair 的桥是直线，学到的边缘向量场及实际积分轨迹也不必是直线。因此不能从该公式推导“一步必定准确”或“比 diffusion 必定更少步”。Rectified Flow/reflow 可以作为进一步压缩采样步数的研究选项，但需要额外数据、训练与质量验证 [R11]。

### 8.3 约束、guidance 与物理时间解码

硬边界优先通过自由控制点与 decoder 施加；在生成过程中固定已知端点对应自由度的速度为零。非零起始导数仍需第 6.5 节的联合边界解码，不因换 FM 而消失。

保持 `tau_r` 的时长下界与 `c` 的正 density 解码；在 noisy latent 上保持这些结构，不等于 noisy physical trajectory 已有意义。碰撞和动力学 cost 仍优先作用于可信的终态估计或最终样本。

在线性桥定义下，可使用以下局部终态估计：

$$
\widehat X_1=X_\lambda+(1-\lambda)V_\psi(X_\lambda,\lambda,context).
$$

当向量场是理想条件期望时，它对应条件期望形式的终态估计；对实际神经网络与多模态分布，它不是该条 ODE 轨迹的精确终点。建议仅在生成后段做轻量修正，或先完整生成再 refinement。

一种启发式 cost-guided ODE 为：

$$
\frac{dX_\lambda}{d\lambda}
=V_\psi(X_\lambda,\lambda,context)
-\eta(\lambda)\nabla_{X_\lambda}J(\widehat X_1;W).
$$

这不是无需推导即可成立的精确 posterior sampler；`η` 的量纲、分支尺度、边界 mask、梯度裁剪和启用区间均需验证。原 diffusion sampler 中与噪声方差相关的 guidance 系数不能原样搬用。

终态估计梯度可需要经过 velocity network 的 VJP，成本可能显著；直接 detach 网络再修正是另一种近似，应作为独立实现与消融。先比较：

- FM 原始输出 + 统一后处理；
- FM 后段 clean-estimate guidance；
- FM + 受约束投影/小 QP。

SafeFlow（早期版本称 SafeFM）[R13] 与 UniConFlow [R14] 提供 flow 约束处理思路，但生成过程中的 barrier/投影保证依赖约束、初始化与可行性等假设。实际机器人还存在插值误差、预测误差、控制延迟与 tracking error，不能据此取消全身轨迹验证或在线 guard。

### 8.4 能解决什么，不能解决什么

| 问题 | FM 的可能作用 | 不能替代的工作 |
|---|---|---|
| 多次网络调用占主要延迟 | 较少 NFE 的 ODE、reflow/蒸馏可能加速 | 必须比较等质量、等预算，不只比较步数 |
| diffusion schedule 与低噪声调参复杂 | 提供另一种训练目标、积分器与时间采样方式 | 新的积分误差和 guidance 稳定性仍需调试 |
| joint 两分支连续耦合 | 同一 vector field 中输出两个分支 | 配对数据、双向条件与边界耦合仍必须正确 |
| 时间先验不知道障碍何时经过 | 单独换 FM 没有针对性解决 | 动态 world/corridor 条件与相应训练标签 |
| before/after、多条绕行覆盖不足 | FM 也能表示多模态，但不自动增加训练覆盖 | 多模式 teacher、均衡训练与候选预算 |
| 需要 exact waiting 或多尖锐 slowdown | 不改变六维 codec 的表达上限 | 显式 dwell/分段表示或增加自由度 |
| FK/SDF/梯度/DenseCheck 占主要延迟 | 换生成器的上限可能很小 | 碰撞计算批处理、缓存、低噪声 guidance 调度 |
| OOD、安全与全局完备性 | 不自动解决 | 独立验证、fallback、预测误差处理 |

用 Amdahl 定律估算收益：若总延迟中网络部分占 `f`，网络加速 `a` 倍，则端到端加速上限约为

$$
S=\frac{1}{(1-f)+f/a}.
$$

例如 `f=0.2,a=4` 时，仅约 `1.18×`。这是算例，不是本仓库 profiling 结果。MPD 原论文也明确指出 cost 梯度是其推理的重要开销 [R1]。

### 8.5 推荐公平实验

用完全相同的 D1/D2、normalizer、条件、机器人限制、候选预算与后处理，对照现有 diffusion、适用的少步 diffusion sampler、CFM。先保持网络容量相近，另做最优工程配置对照。

FM 可先 sweep `NFE={4,8,16,32}`，使用 Euler/Heun；Heun 一次步进一般需要两次场评估，必须按真实 NFE 记账。对每个候选还记录 guidance 次数、反向传播次数、FK/SDF 查询数和最终检查时间。

先比较 unguided 质量，再比较固定预算 guided 质量；另报告等成功率下的端到端延迟。只测一个世界、仅成功样本耗时或只报平均 NFE，都不能证明能替代现有模型。

FlowMP 已直接研究条件 FM 机器人规划与二阶运动相关设计，可作为实现对照 [R12]。本仓库应保留物理时间链式法则与样条导数评估：`一阶生成 ODE`不意味着`机器人轨迹只有一阶光滑`，二者属于不同时间轴。

## 9. 文献对照：哪些解法值得借鉴，哪些结论不能直接搬用

### 9.1 先消除版本与名称歧义

本文检索截止日期为 2026-09-26，区分以下对象：

- cuRobo 原论文：2023 年，GPU 并行轨迹优化 [R2]。
- cuRoboV2：2026-03-05 首发，本文核对 2026-04-16 的 v2 修订版 [R3]。不能只用旧 cuRobo 的特征评价现有研究。
- RAMP-2023：Reactive Action and Motion Planner，机械臂分层规划与局部反应控制 [R4a]。
- RAMP-2025：点云条件、energy-based diffusion 与 potential fields 的实时自适应规划论文，本文核对 v3 [R4b]。它不是 RAMP-2023 的一次模型升级。
- SafeFlow：本文按 2025-11-12 的 v3 标题引用；早期版本名为 Safe Flow Matching/SafeFM [R13]。

这些论文的速度、成功率来自各自硬件、数据、任务定义与碰撞标准。本文不把它们拼接成一个跨论文排行榜，也不把论文结果当作本仓库已经复现的收益。

### 9.2 对照表

| 方法 | 核心机制 | 对当前问题的可借鉴之处 | 本项目采用时需要补的部分 |
|---|---|---|---|
| cuRobo [R2] | 多种子并行优化、L-BFGS、粒子方法与几何规划辅助 | 强化无学习多初值 baseline；统一优化器后测 learned seed 的价值 | 本仓库的预测时间轴、时序分支、执行契约仍需对接 |
| cuRoboV2 [R3] | B-spline 优化、力矩限制、GPU 运动学/逆动力学与深度融合 ESDF | 高效碰撞与导数计算、带边界轨迹表示、载荷下可执行性 | 不能从 dynamics-aware 一词推导已具备本文的动态障碍时间走廊与 learned timing |
| RAMP-2023 [R4a] | MPPI 轨迹生成 + 异步局部向量场 follower；构型空间 SDF | 全局规划与快速反应分层；生成器不承担全部在线安全职责 | 接入当前预测、时序优化及控制边界；重审安全论证的假设 |
| RAMP-2025 [R4b] | 点云条件 diffusion、CFG/APF、对旧轨迹少步扰动修复 | 感知条件、低噪声 warm start、只修复未执行部分 | 不能把导航/追逃实验直接外推成七轴全身碰撞、力矩与时间调度结论 |
| SIPP / SIPP-IP [R5][R6] | 安全时间区间搜索；考虑加减速时需扩展状态/传播 | 给多模态 before/after 提供显式决策结构 | phase 图、机械臂约束及连续验证；不直接继承原图搜索保证 |
| TOPP-RA [R7] | 给定路径上的可达/可控速度集合传播 | 构造快且可行的 retiming 初始化与约束基线 | 已知路径优化不等于动态障碍下的绝对时间规划 |
| ST-RRT* [R8] | 时空双向采样搜索、速度限制、到达时间优化 | 独立 teacher、困难模式与可行性对照 | 样条拟合、加速度/jerk、执行接口需额外处理和重验 |
| FlowMP [R12] | 条件 FM、样条轨迹先验、二阶运动相关设计 | 直接相关的 MPD→FM 实验基线 | 保留本仓库非均匀 timing、动态世界与在线边界处理 |
| SafeFlow / UniConFlow [R13][R14] | 在 flow 生成过程中施加 barrier 或约束引导 | 比单纯增加碰撞 penalty 更结构化的约束处理 | 逐项检查假设、离散积分误差、预测误差和运行时安全链 |
| Time-Triggered Corridor [R15] | 对通道切换时间建立结构化优化与可行性检测 | 离散通道选择与连续轨迹优化分层 | 原问题是飞行器通道；全机械臂 FK 约束一般不保持凸性 |

### 9.3 cuRobo：重点借鉴优化与计算，不制造不公平对比

原 cuRobo 的关键不是单个局部优化器，而是 GPU 上多个初始化、优化与几何规划的组合 [R2]。因此，“MPD 比一个直线初值优化器好”不足以证明胜过 cuRobo；必须包括相同预算的多种子优化器。

cuRoboV2 已使用 B-spline 并处理力矩约束、非静止起始边界和实时深度 ESDF [R3]。**样条、平滑性和并行候选不再能被列为 MPD 独有优势。** MPD 更值得验证的是任务分布先验、示范偏好、多模式候选覆盖和更好的优化初始化。

该论文第 4 节式 (10) 使用固定 `dt_u` 的样条时间缩放；这不同于本文按路径相位学习非均匀 `dt/ds`。同时，机器人 dynamics-aware 指惯性/力矩等动力学可执行性，不等同于预测运动障碍的时间调度。这是问题定义差别，不是断言 cuRobo 无法扩展相关功能 [R3]。

工程建议：若以后接入其 FK/SDF 或优化器，先做独立 backend adapter 与数值对齐测试，核对球半径、signed distance、附着物、关节顺序、坐标系、梯度及 margin；不要为追求速度悄悄换掉碰撞语义。

### 9.4 两类 RAMP：分别借鉴控制分层与轨迹修复

RAMP-2023 将 MPPI 生成器与异步局部向量场 follower 分开，并在构型空间使用 SDF [R4a]。对当前 ROS2 架构的启发是：长时域 MPD 提供候选与通行模式；近时域控制/guard 负责执行偏差与突发变化。换一个更强的 diffusion 或 JointDual 不应取消这一层次。

RAMP-2025 在旧轨迹上加小噪声、少步 refinement，并保留执行历史约束；这与本仓库 F2/F3 和 replan warm start 的研究方向接近，但网络条件、目标和约束不相同 [R4b]。可借鉴点云/world condition 和局部修复，不应照抄 APF 位移后跳过全身动力学与碰撞重验。

本项目的限制分析：局部反应更适合短时突发障碍；跨多个交叉窗口的提前让行仍需要预测与显式时序选择。APF/局部修复可能保留错误通行模式，故应在失败或风险上升时触发新分支/新空间候选，而非无限修补旧解。这是本文针对目标任务的推断，不是两篇 RAMP 的统一实验结论。

### 9.5 “time corridor”不必直接等于安全凸多面体

本文首版走廊位于固定空间路径的 `(s,t)` 平面；它不是笛卡尔空间中只包住 EE 的通道。SIPP 的 safe intervals 与该构造直接相关，而飞行器的 time-triggered corridor [R15] 更适合借鉴通道切换时间的结构化搜索。

对全机械臂，`x_link(q)` 是非线性映射、全身可行集通常非凸；即使 workspace corridor 是凸的，控制点、关节和 timing 的联合约束也不自动凸。本文第 5 节只把固定分支的部分 arrival/速度约束用于 QP 初始化，不把完整问题包装成凸优化。

## 10. MPD 可以在哪些场景特别有效

以下为结合 MPD 论文定位与当前代码提出的**待验证假说**，不是已经在本仓库得到的性能结论。原 MPD 明确聚焦结构变化不大的环境，并讨论从采样规划/示范学习后用 cost 适应新增约束；原论文固定轨迹时长，当前仓库的 learned timing 属于后续扩展 [R1]。

### 10.1 优先验证的优势场景

| 场景 | 预期 learned prior 能提供什么 | 验证方式 |
|---|---|---|
| 固定 warehouse 布局、很多重复抓放任务 | 摊销跨任务的空间搜索，提供高质量可优化初值 | 与多初值优化、几何规划同预算比较 time-to-first-valid |
| 狭窄货架与多个绕行通道 | 在训练覆盖的情况下直接提出不同空间模式 | 对照无先验 seed，统计可行模式数和困难任务成功率 |
| 同一通道中存在早过/晚过/绕行选择 | world-conditioned timing/JointDual 学习常见决策组合 | 固定几何、改变 crossing time，测试模式是否随条件切换 |
| 动态障碍只改变局部可行性 | 保留既有路径结构，仅修复局部空间或时间 | 对比全量重采样与 warm start 的成功率、延迟和连续性 |
| 有人类示范或任务偏好 | 在避障同时保留特定姿态、接近方向、动作风格 | 几何与动力学达标后比较示范偏差/任务约束满足率 |
| 多次 replan 的世界预测相对稳定 | 复用候选模式与已验证的局部结构 | 记录 replan 抖动、切换次数、失效缓存比例与执行成功 |

其中，时间先验的独特价值应表现为“更少候选/更少 refinement 就找到正确时间分支”，而不只是同一预算下输出更慢的轨迹。JointDual 的价值应表现为“需要换空间路径时也能作出更好的选择”，而不只是 timing loss 降低。

### 10.2 不应优先押注的场景

- 开阔、简单、一次性任务：传统规划或多初值优化可能已经足够快，训练成本难以摊销。
- 与训练结构差异很大的布局、机器人或载荷：先验可能给错模式，不能把 OOD 能力当作默认属性。
- 极突然、不可预测且反应时间很短的障碍：执行端安全与快速反应比长时域生成器更关键。
- 主要瓶颈是感知更新、FK/SDF 或通信：扩大网络或换 FM 可能几乎没有端到端收益。
- 固定路径 retiming 已能可靠解决的简单 crossing：六维多初值优化可能是性价比更高的主方案。

建议最终架构保留互补关系：`学习先验提出模式 → 走廊/优化器修正 → 独立验证 → 在线跟踪与安全反应`。MPD 可以是强 proposal，也可以是主规划器，不必以替代所有经典模块为目标。

## 11. 实施计划、实验矩阵与验收门槛

### 11.1 分阶段顺序与是否继续的决策

| 阶段 | 主要交付 | 是否训练 | 进入下一阶段前的条件 |
|---|---|---|---|
| P0 时间/指标基线 | snapshot→execution→candidate 时间对齐；分项 profiling；冻结任务回放 | 否 | 同一轨迹在 MPD/执行侧查询一致；无 offset 重复计算；旧入口回归通过 |
| P1 走廊 + 低维 timing | phase-time 风险场、离散分支、多初值优化、完整验证 | 否 | 在同预算下对 raw risk baseline 有可重复增益；额外开销可接受 |
| P2 条件时间先验 | D1 数据与多模态 teacher；world/corridor-conditioned timing | 是，时间分支 | 未见世界上的 unguided 可行率/模式选择优于旧 timing；guided 结果不退化 |
| P3 JointDual | D2、J0/J1、双向 adapters、同步 sampler | 是，联合 | 与同数据/同条件的 factorized 相比有统计可靠的增益，而非容量或预算差异 |
| P4 Flow Matching | timing-CFM，再按结果决定 joint-CFM/reflow | 是，生成目标变化 | 等质量下减少端到端延迟或提高同预算成功率；模式覆盖与安全不退化 |
| P5 在线边界增强 | motion-to-motion 数据、边界解码、handoff/bridge 回归 | 是，视表示而定 | 非零边界、延迟扰动与预测更新下接续和执行均通过验证 |

P5 的时间契约与边界测试要从 P0 开始；完整非零边界学习可以后做。P4 的 timing-only 小实验可在 P2 数据稳定后独立开展，不需要等到 P3 结束；但不要把模型耦合、数据扩展、codec 升级与 FM 切换合在一个无法归因的实验里。

**止损规则：** 如果 P1 已达到目标且 P2 在等预算下无明显价值，保留优化器主导的 timing；如果 P2 已解决主要问题且 D2 中空间-时间耦合很弱，不强上 JointDual；如果网络不是主要延迟，先优化 cost/FK/SDF，而非重训全部 FM。

### 11.2 文件级落点：只新增或显式 opt-in

以下新文件名是建议，不是已创建接口。公共组件改动必须保持原默认行为，原 `scripts/inference/inference.py`、Franka 配置和现有 checkpoint 加载路径不重定向。

| 工作包 | 建议新文件/入口 | 复用或对接现有模块 |
|---|---|---|
| 时间走廊构建 | `mpd/inference/time_corridor.py` | `dynamic_collision.py`、现有全身 FK、snapshot/执行时间契约 |
| 走廊与事件 cost | `mpd/inference/time_corridor_guidance.py` | `space_time_guidance.py`、`factorized_guidance.py` |
| 低维无学习 baseline | `mpd/inference/timing_optimizer.py` | `learned_timing.py` 中 codec、样条、动力学限制与验证器 |
| 动态 teacher 数据 | `scripts/spacetime_data/generate_dynamic_spacetime_dataset.py` | 现有 canonical HDF5、validator，增加动态 sidecar/version |
| 动态数据视图 | `mpd/datasets/dynamic_spacetime_dataset.py` | `spacetime_schema.py`、`spacetime_hdf5.py`；兼容旧文件 |
| 条件 timing | `mpd/timing_training/corridor_model.py`；`scripts/train/train_timing_corridor.py` | 旧 path encoder、normalizer、trainer 的通用部分 |
| JointDual | `mpd/joint_training/`；`scripts/train/train_joint_spacetime.py`；`mpd/inference/joint_spacetime_sampler.py` | 空间 backbone、时间 codec、共享 cost，不替换 F1/F2/F3 |
| FM 试验 | `mpd/timing_training/flow_model.py`；`mpd/inference/timing_flow_sampler.py` | 同一 dataset/context/validator，使用独立 checkpoint 类型 |
| 单次离线推理 | `scripts/inference/infer_time_corridor.py`、`infer_joint_spacetime.py` | 现有加载与可视化辅助；新增独立配置 |
| ROS2 对接 | 新 backend/mode 的显式选择与日志字段 | `scripts/runtime/timing_contract.py`、runtime engine；ROS2 adapter 的 selector/bridge/guard |

模型 metadata 至少记录：`model_family`、`timing_representation`、normalizer、basis、机器人/球模型 hash、world condition schema、边界类型、schedule/flow convention、训练数据版本。必须拒绝把 diffusion checkpoint 当 FM、把 c 当 tau_r、或把 rest-to-rest 当 motion-to-motion 静默加载。

ROS2 侧首版无需重写执行链：让新 backend 输出已有逐候选时间契约，继续使用原来的有效性筛选、handoff、bridge 和 guard。若新增 exact dwell 或非零边界，另开协议/测试变更；不能仅在模型输出层假装兼容。

### 11.3 测试建议

在 `mpd-splines-public` 下新增针对性测试，先小型 CPU/张量例子，再 GPU 场景回放；不为文档研究启动训练或改环境依赖。

- 两个不相连时间窗口：采样保留两个模式，不落到二者之间。
- 时间单调性与 codec round-trip；`T_min≥T_max` 显式失败；拟合后再验收。
- 有解析解的平移障碍：核对 `∂D/∂t`、`∂t/∂z` 与有限差分；避开 min 切换等不可微点。
- 空间变化、world 更新、offset 改变、margin 改变：旧走廊缓存必须失效。
- 路径节点安全但边中碰撞、高速物体穿越、初始/终端等待被扫过：必须被拒绝。
- risk mean/CVaR 随物理时间 quadrature 一致；采样点加密不应任意放大 cost。
- JointDual batch 内配对、mask、同步更新、分支 loss 归一、tau_r 的跨分支导数。
- FM flow-time 方向、端点、Euler/Heun 的实际 NFE、无 guidance 时回归与启用 guidance 时的边界保持。
- 对旧 Franka static/factorized 入口跑固定 seed smoke test；复用现有 `test_factorized_sampler.py`、`test_factorized_guidance.py`、`test_runtime_timing_contract.py` 等回归。

对连续安全没有解析证书的测试，只能声称在声明分辨率和 checker 下通过，不应把“测试没撞”写成数学保证。

### 11.4 冻结场景与公平 benchmark

沿用 `environment_count_per_category` 与 `planner_repeats` 的实验分工：前者生成不同世界，后者在同一冻结世界中更换 planner seed。每个模型消费同一环境清单、起终点、种子序列、预测/碰撞配置与限额；保存 scenario JSON、内容 hash 和模型 checkpoint hash，而不只保存随机 seed。

时间对齐模式分开报告：

- `motion_aligned`：用声明的运动启动参考事件对齐障碍相位，比较运动本身的空间/时间决策；全部模型使用相同规则与冻结参数。不能按某模型效果额外移动 crossing time。
- `absolute_world_time`：世界按共同定义的绝对时间/回放时钟推进，规划、排队与 handoff 延迟本身就是任务难度的一部分；检查机器人开始运动前的旧轨迹/保持段。

前者适合减少启动延迟对决策能力评估的混淆，后者更接近动态在线负载测试。两者都需保存 `t_snapshot,t_plan_start,t_plan_end,t_commit,t_exec`，不能混成一张成功率表而忽略时间语义。

共享 seed 不能保证不同算法内部每次随机抽样一一对应；冻结世界与输入 hash、配对任务以及同等计算预算才是公平比较的基础。环境采样器 `v3` 与模型 checkpoint 的 `v3` 也必须分字段记录，不能把两个版本概念混用。

不把 until-success 的最终成功率当作单次规划成功率：所有失败、重试次数、每次时间预算和截至成功的累计时间都应保留。对固定环境的多个 repeat，置信区间按环境/任务组 bootstrap，避免把相关重复当成独立新环境。

### 11.5 必须记录的指标

| 指标组 | 建议内容 |
|---|---|
| 先验能力 | unguided 静态/动态可行率、正确时间分支覆盖、到达窗口误差、所需 refinement 次数 |
| 任务成功 | 首次规划成功率、固定预算 success@K、最终执行成功率、timeout/无解/guard 中止分类 |
| 时间 | 网络 forward、guidance backward、走廊构建、FK/SDF、DenseCheck、序列化与通信、端到端 p50/p95/p99 |
| 执行效率 | 轨迹 duration、显式 wait、任务从请求到完成的耗时、replan/切换次数 |
| 安全 | 全身/附着物动态最小 clearance、静态/自碰撞 clearance、风险尾部、预测 horizon 越界次数 |
| 可执行性 | 速度/加速度超限量、物理 jerk、端点/bridge 残差；有逆动力学时再加 torque |
| 多样性 | 可行空间模式数、before/after 模式覆盖、时长分布、同模式内距离；不能只报 pairwise latent distance |
| 鲁棒性 | offset 扰动、预测误差、延迟峰值、感知缺失和新布局下的退化 |
| 资源 | 峰值显存、CPU/GPU 占用、真实 NFE、梯度/FK/SDF 调用数量 |

成功样本的平滑度/耗时要单独报告，并同时给出全部任务的成功率；不能删除失败后得出“更快”。对相同几何路径，全局减速天然降低速度、加速度和 jerk，因此同时报告 duration 与物理平滑度的 Pareto 曲线，不能把更慢直接解释成模型更好。

`success@K` 用实际 batch 统计；候选通常相关，不使用独立假设下的 `1-(1-p)^K` 代替观测。训练 epsilon loss、CFM loss、c loss 与 tau_r loss 的尺度不同，不直接跨目标比较数值大小。

### 11.6 最小消融矩阵

先固定路径与同一世界做 A 组，再做联合路径的 B 组，避免一开始组合爆炸。

| 实验 | 生成器/初始化 | 条件与 refinement | 主要归因 |
|---|---|---|---|
| A0 | uniform / TOPP-RA / random timing | 相同已有 risk 与优化预算 | 无学习下限与强优化基线 |
| A1 | 现有 timing diffusion，c 与 tau_r 分开 | 与 A0 相同 | 当前时间先验的增益 |
| A2 | 与 A1 相同 | 加 corridor/event 分支 | 推理 cost 的增益 |
| A3 | D1 条件 timing diffusion | 与 A2 相同 | 条件与标签真正改善 prior 的增益 |
| A4 | 同 D1 的 timing CFM | 与 A3 同条件、同后处理 | 生成目标与 sampler 的增益 |
| B0 | 当前 factorized/F3 | 相同空间 prior、旧数据 | 现有系统基线 |
| B1 | 用相同新 D2/条件重训 factorized | 统一 cost 与预算 | 排除“只是新数据更好” |
| B2 | JointDual J0/J1 | 与对应 B1 同数据/条件/预算 | 双向联合 score 的净增益 |
| B3 | JointDual-CFM | 与 B2 同数据/条件/预算 | 联合 FM 的净增益 |

每个关键实验分别运行 `w_T=0` 与 `w_T>0`，保持最大时长和 horizon 一致；再按 motion_aligned / absolute_world_time 分层。可先做小规模成对 pilot，固定配置后才扩展完整 benchmark。

门槛应在测试前预注册，例如“无新增时间契约违例、成功率置信区间不劣、p95 延迟不超过控制预算”；具体毫秒、clearance 与成功率目标由当前控制周期和风险要求确定，而不是本文虚构一个通用安全阈值。

## 12. 参考文献与代码索引

### 12.1 原始论文与官方材料

以下 URL 指向论文或作者/机构公开版本。上文 [R*] 是引用编号；方法迁移、实现顺序与实验设计是本文建议，不是这些论文已验证的共同结论。

- **[R1]** Carvalho 等，*Motion Planning Diffusion: Learning and Adapting Robot Motion Planning with Diffusion Models*，arXiv:2412.19948，本文核对 v3，尤其第 V 节的固定时长、cost 梯度与场景专门化限制。`https://arxiv.org/html/2412.19948v3`
- **[R2]** Sundaralingam 等，*cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation*，2023，arXiv:2310.17274v2。`https://arxiv.org/abs/2310.17274v2`
- **[R3]** Sundaralingam、Murali、Birchfield，*cuRoboV2: Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots*，2026，arXiv:2603.05493v2（2026-04-16）。本文重点核对第 3–6 节及限制讨论；不是仅按 v1/旧 cuRobo 推断。`https://arxiv.org/html/2603.05493v2`
- **[R4a]** Vasilopoulos 等，*RAMP: Hierarchical Reactive Motion Planning for Manipulation Tasks Using Implicit Signed Distance Functions*，IROS 2023，arXiv:2305.10534v2。`https://arxiv.org/abs/2305.10534v2`
- **[R4b]** Teshome 等，*Real-Time Adaptive Motion Planning via Point Cloud-Guided, Energy-Based Diffusion and Potential Fields*，RA-L 2025，arXiv:2507.09383v3。参见动态 refinement 的 Algorithm 2。`https://arxiv.org/html/2507.09383v3`
- **[R5]** Phillips、Likhachev，*SIPP: Safe Interval Path Planning for Dynamic Environments*，ICRA 2011，作者机构公开论文。`https://www.cs.cmu.edu/~maxim/files/sipp_icra11.pdf`
- **[R6]** Ali、Yakovlev，*Safe Interval Path Planning With Kinodynamic Constraints*，2023，arXiv:2302.00776；解释原 SIPP 的瞬时停止假设与加减速扩展。`https://arxiv.org/abs/2302.00776`
- **[R7]** Hung Pham、Quang-Cuong Pham，*A New Approach to Time-Optimal Path Parameterization based on Reachability Analysis*，2017 预印本，arXiv:1707.07239v2。`https://arxiv.org/abs/1707.07239v2`
- **[R8]** Grothe 等，`ST-RRT*: Asymptotically-Optimal Bidirectional Motion Planning through Space-Time`，ICRA 2022，arXiv:2203.02176。`https://arxiv.org/abs/2203.02176`
- **[R9]** *Flow Matching for Generative Modeling*，2022 预印本，arXiv:2210.02747。`https://arxiv.org/abs/2210.02747`
- **[R10]** *Flow Matching Guide and Code*，2024，arXiv:2412.06264。`https://arxiv.org/abs/2412.06264`
- **[R11]** *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*，2022，arXiv:2209.03003。`https://arxiv.org/abs/2209.03003`
- **[R12]** Nguyen 等，*FlowMP: Learning Motion Fields for Robot Planning with Conditional Flow Matching*，IROS 2025，arXiv:2503.06135；同时核对作者公开会议论文与官方代码说明。`https://arxiv.org/abs/2503.06135`；`https://mkhangg.com/assets/papers/nguyen2025flowmp.pdf`；`https://github.com/mkhangg/flow_mp`
- **[R13]** Dai 等，*SafeFlow: Safe Robot Motion Planning with Flow Matching via Control Barrier Functions*，arXiv:2504.08661v3（2025-11-12）；v1 标题为 *Safe Flow Matching: Robot Motion Planning with Control Barrier Functions*。`https://arxiv.org/abs/2504.08661v3`
- **[R14]** Yang 等，*UniConFlow: A Unified Constrained Flow-Matching Framework for Certified Motion Planning*，2025 首发，本文核对 arXiv:2506.02955v2（2026-01-14）。`https://arxiv.org/abs/2506.02955v2`
- **[R15]** Yu 等，*Real-Time Quadrotor Trajectory Optimization with Time-Triggered Corridor Constraints*，2022，arXiv:2208.07259。用途是通道/时间结构化优化参考，不是机械臂全身安全走廊的现成实现。`https://arxiv.org/abs/2208.07259`

### 12.2 本仓库核对索引

以下路径相对 MPD 仓库根目录，名称可直接用于代码检索：

~~~text
mpd/inference/space_time_guidance.py
mpd/inference/learned_timing.py
mpd/inference/factorized_sampler.py
mpd/inference/factorized_guidance.py
mpd/inference/dynamic_collision.py
mpd/timing_training/model.py
mpd/timing_training/trainer.py
mpd/parametric_trajectory/timing_spline.py
mpd/parametric_trajectory/normalized_timing.py
mpd/datasets/spacetime_schema.py
mpd/datasets/spacetime_timing_dataset.py
mpd/datasets/spacetime_hdf5.py
scripts/spacetime_data/README.md
scripts/spacetime_data/generate_spacetime_dataset.py
scripts/spacetime_data/validate_spacetime_dataset.py
scripts/train/TIMING_DIFFUSION.md
scripts/inference/infer_factorized.py
scripts/runtime/timing_contract.py
scripts/runtime/dynamic_runtime_engine.py
scripts/runtime/factorized_runtime_engine.py
docs/Space-Time MPD Timing 训练与数据工程实施方案.md
~~~

ROS2 侧相关目录相对 `physical_ai_runtime` 根目录：

~~~text
src/motion_planning/motion_planners/mpd_dynamic_planner_adapter/mpd_dynamic_planner_adapter/
  replan_node.py
  space_time_replan_node.py
  collision_guard.py
  candidate_selector.py
  quintic_bridge.py
  handoff_selector.py
~~~

**最终建议：先把“知道可在哪个时间窗口通过”做对，再训练“更快提出正确窗口”的时间先验；用同数据对照证明 JointDual 的双向耦合价值，最后按真实延迟瓶颈决定是否全面切换 Flow Matching。** 所有新实验使用独立配置/入口，既有 Franka 单臂行为保持不变。
