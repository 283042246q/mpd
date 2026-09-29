# Corridor A 的数学原理与当前实现（阶段 A/B/C）

本文解释仓库中已经接入 Phase5-joint 和 Factorized F1 的 **Corridor A**。
前半部分讲基础走廊和 A/B 计算实现，第 7 节补充阶段 C：C1 用 DP
到达时刻拟合 timing 初值，C2 做有预算的 K-best 窗口搜索，并说明随后
实现的备选分支救回、选择性 K 与早停。这里的“C”是优化阶段，
**不是** F1 的 `c` timing 表示：`f1_c_corridor_a` 和
`f1_tau_r_corridor_a` 都在本文范围内。代码仍保留原有的动态碰撞
mean/CVaR 代价；设计文档曾把该代价称作方案 C，那是另一个命名维度。

Corridor A 是推理时的可选后处理：**先固定一条空间路径，在相位—时间
平面寻找可达的安全窗口序列，再仅调整这条路径的时间参数**。
它不修改空间控制点，不向现有 diffusion 网络增加 corridor 条件，
也不需要重训 checkpoint。它不是末端执行器在三维空间里的几何通道。

## 1. 空间路径、时间曲线和世界时钟

给定一条候选空间路径 $P$，令相位 $s\in[0,1]$，关节位置为
$q_P(s)\in\mathbb R^7$。时间参数 $\theta$ 解码出严格递增的到达时刻

$$
t_\theta(0)=0,\qquad
t_\theta(s)=\int_0^s u_\theta(\sigma)\,d\sigma,\qquad
u_\theta(s)>0,\qquad T_\theta=t_\theta(1).
$$

Phase5 直接优化 timing spline 的控制点；F1 优化 checkpoint 输出的
标准化六维 latent，随后由同一个 timing codec 解码。二者的空间路径
$P$ 在本次 Corridor refinement 内均保持不变。Phase5 的时间密度为
$u=u_{\min}+\operatorname{softplus}(g_\theta(s))$；F1 的 `c` 与
`tau_r` 表示使用各自 codec 的解码规则，不能把六维 latent 误当成
Phase5 的物理控制点。

设世界快照时刻为 $t_{\rm snap}$，计划的轨迹执行起点为
$t_{\rm exec}$。在快照坐标中查询障碍物预测所用的时间是

$$
\Delta+t_\theta(s),\qquad
\Delta=t_{\rm exec}-t_{\rm snap}.
$$

代码通过 `plan_start_unix_ns` 保存 $t_{\rm exec}$，动态 SDF 查询会
把轨迹相对时间加上 $\Delta$；不能把“模型推理耗时”直接当作
$\Delta$。查询还必须落在世界快照的有效预测期内。

## 2. 在固定路径上建立全机械臂安全网格

取 $N_s$ 个相位 $s_i$ 和 $N_t$ 个时间 $\tau_j\in[0,T_{\max}]$。
默认相位采样数为 32、请求的时间步长为 0.2 秒；实际网格使用
`linspace(0, T_max, ceil(T_max / 0.2) + 1)`，故实际网格间距
$h=T_{\max}/(N_t-1)$ 不一定恰好是 0.2 秒。

对于路径 $q_P(s_i)$ 上的每个机器人碰撞球 $\ell$，先做正运动学得
中心 $x_\ell(q_P(s_i))$，再查询该中心在 $\tau_j$ 时刻到所有动态物体的
最小 signed distance $d_{i j\ell}$。动态世界的距离查询已经负责
物体形状与预测 inflation；这里按代码使用每个机器人球的阈值

$$
m_\ell=m_\ell^{\rm field}+m_{\rm cutoff}+m_{\rm corridor},\qquad
S_{ij}=\bigwedge_\ell[d_{ij\ell}\ge m_\ell].
$$

其中 $S_{ij}$ 是“该相位、该到达时间，整条机械臂的动态障碍检查
都通过”的布尔值。实现逐球取所有障碍物的最小距离，再对球做逻辑与，
并非只检查末端或 anchor。$m_\ell^{\rm field}$ 来自机器人碰撞场的
`collision_margins`，不要对动态 SDF 已计算的物体 inflation 再减一次。
额外 corridor clearance 默认是 0。
静态环境、自碰撞和关节位置限制不在这个动态安全网格里；
固定路径的 timing 也无法修复真正的静态空间碰撞。

相邻的安全时间格合并成相位 $s_i$ 的窗口并集

$$
\mathcal I_i=\bigcup_k[a_{ik},b_{ik}].
$$

实现对非首尾窗口边界各收缩默认 $\delta_t=0.05$ 秒，并只给收缩后
仍落在窗口内的网格格点分配窗口标签。这个收缩是离散时间裕度，
不是连续轨迹的碰撞安全证明。

## 3. 用动态规划选一条可达的窗口分支

两个窗口都安全时，不能把它们平均。例如单个相位的安全窗口是
$[1,2]\cup[4,6]$ 秒，取平均得到的 3 秒恰好可能在障碍物占用期。
因此 Corridor 先做离散选择：为每个相位选一个窗口 $k_i$，得到分支
$m=(k_0,\ldots,k_{N_s-1})$，然后在固定分支内优化连续时间曲线。

在时间网格上，DP 从 $(s_0,\tau_0=0)$ 出发，仅转移到安全格。
若相邻相位关节差是 $\Delta q_i=q_P(s_{i+1})-q_P(s_i)$，由关节速度上限
$v_j^{\max}$ 得到一个必要的最小时间格数

$$
n_i=\max\!\left(1,
\left\lceil\frac{\max_j|\Delta q_{i,j}|/v_j^{\max}}{h}\right\rceil
\right),\qquad j_{i+1}-j_i\ge n_i.
$$

没有速度上限时，代码仍要求至少前进一个时间格。最后一个相位还需
$\tau_{j_{N_s-1}}\ge T_{\min}$；时间网格本身不超过 $T_{\max}$。
此处只用端点关节差计算**必要**速度下界，没有证明相位间的真实
样条速度、加速度或无碰撞。那些约束留给物理代价和最终检查。

当前基础搜索分别使用三个时间偏好 $\beta\in\{0,-1,+1\}$ 秒。
初始 timing 在 $s_i$ 的到达时刻记为 $t_i^0$，每个偏好的 DP 目标为

$$
\min_{j_0,\ldots,j_{N_s-1}}
\frac{1}{N_s}\sum_i
\left(\tau_{j_i}-t_i^0-\beta\sin\frac{\pi i}{N_s-1}\right)^2,
$$

约束是上述安全格、单调/速度下界及总时长下界。$\beta=-1$ 倾向
中段早过，$\beta=+1$ 倾向晚过；正弦在首尾为零。DP 用前缀最优值
递推并回溯一条路径，得到各相位的窗口序列。三个偏好若得到同一
窗口序列会去重；基础版本**最多尝试三个分支**，并不穷举所有可行
时间模态。无可达分支时保留原 timing。

## 4. 固定窗口后优化连续 timing

对于选定分支 $m$，记相位 $i$ 的收缩窗口为 $[a_i,b_i]$。
当前代码实际使用的 corridor 代价是

$$
J_{\rm cor}(\theta;m)=\frac1{N_s}\sum_i
\left([a_i-t_\theta(s_i)]_+^2+
[t_\theta(s_i)-b_i]_+^2\right),\qquad [x]_+=\max(0,x).
$$

它在窗口内为零，在窗口外给出“往早/晚哪个方向移动”的梯度。
与设计文档的建议公式不同，当前实现**没有另外除以时间尺度**；
归一化来自对相位求平均。实际优化目标为

$$
J(\theta;m)=J_{\rm phys}(P,\theta;W)
+\lambda_{\rm cor}J_{\rm cor}(\theta;m),
\qquad\lambda_{\rm cor}=0.1\ \text{（默认）}.
$$

$J_{\rm phys}$ 沿用原来的物理代价：启用动态引导时，动态碰撞
penetration 的物理时间 mean/CVaR，加上关节速度/加速度超限、
总时长和 timing 平滑度。F1 还对
总时长越过上下界加入惩罚。因此 corridor 是**额外的窗口结构项**，
没有替换、删除或重新命名原 `dynamic_collision` 代价。

每个分支从原 timing 参数开始，以 Adam 做默认最多 20 次更新；
保留目标值最低且时长合法的参数，梯度按候选/分支单独裁剪。
Phase5 的不可优化 timing 控制点保持原值；F1 只更新六维 latent。
虽然 DP 回溯得到一组离散到达时刻，**基础版本并不
把它拟合成优化初值**：DP 时刻只用于选窗口分支。阶段 C1 才尝试
拟合；这也解释了
“网格上有可达路线”不一定能在有限步 Adam 内得到可用样条。

## 5. 安全边界与运行时验收

Corridor 网格是一种启发式搜索，不是最终碰撞检查：它只抽样相位和
时间，速度转移只检查端点差，也没有在 DP 中完整传播加速度状态。
初始动态 cost 接近零时，代码还会跳过该候选的 corridor refinement；
这只是省算启发式，不能解释为已证明安全。

运行时在 refinement 前检查原候选，在 refinement 后用原
DenseCheck 对每个候选的实际连续 timing 采样重新检查动态/静态环境、
自碰撞、关节及运动学限制。若某候选原来通过 DenseCheck、改过后
反而失败，就退回原 timing；最终选中的轨迹也必须通过完整
DenseCheck。它仍是有限采样验收，不应宣称数学意义上的连续时间
无碰撞证明。重规划交接、guard 和 brake 是更外层的在线执行规则，
不是 Corridor DP 的一部分。

## 6. 阶段 A/B 的计算实现

| 实现 | 数学结果与作用 | 当前定位 |
|---|---|---|
| `serial` | 逐候选、逐分支计算同一安全网格、DP 和精确物理代价 | CLI 默认参考后端 |
| `batch_exact` | 把候选 × 分支展平为批次，固定路径的 $q,q_s,q_{ss}$ 与碰撞球位置在请求内复用；分块查询网格，按行记录最优值/有效掩码并裁剪梯度 | A+B1 的精确提速后端；在线复测推荐显式选择 |
| `batch_time_table` | 在固定路径上存 $D[b,s,t,\ell]$，对实际 $t_\theta(s)$ 线性插值；分支输出再用精确物理代价重排 | B2 可选 surrogate，默认不启用 |
| `batch_event_intervals` | 匀速、固定半径/膨胀的球体用二次不等式求碰撞时间区间，再投影回原网格 | B3 可选；盒体、胶囊、时变膨胀等回退网格查询 |

`batch_exact` 改的是执行布局，不改安全阈值、三偏好搜索、原物理目标
或最终 DenseCheck。B2 的插值可能改变梯度，不能称为精确等价；
B3 只在代码明确支持的球体特例使用解析区间。这些计算后端不同于阶段 C
增加 DP 拟合初值或更多窗口序列的算法扩展。

B2 对相邻时间格 $\tau_j\le t\le\tau_{j+1}$ 使用
$\widehat D(t)=(1-\alpha)D(\tau_j)+\alpha D(\tau_{j+1})$，
$\alpha=(t-\tau_j)/(\tau_{j+1}-\tau_j)$；越出表域会拒绝，
分支结束后必须用原 SDF 精确重排。B3 的球体特例把固定机器人球中心
$x$ 与匀速障碍中心 $c(t)=c_0+vt$ 的冲突写成
$\|x-c_0-vt\|^2<(R+m)^2$，求二次不等式的时间根，再把阻塞区间
投影回原时间格。它没有把盒体、加速或时变不确定性偷换成球体。

设候选数为 $B$、相位格数为 $N_s$、时间格数为 $N_t$、碰撞球数为
$L$、障碍物数为 $O$，直接构造风险网格的粗略规模为
$O(BN_sN_tLO)$；前缀最优值使每个时间偏好的 DP 在网格形成后约为
$O(N_sN_t)$。因此先筛掉无需修复的候选、分块批量 SDF 查询和固定
路径复用，比改变 DP 数学目标更直接地影响请求耗时。

## 7. 阶段 C：有限预算内增加 timing 模态

基础走廊的三个早/晚偏好，每个偏好只从 DP 取一条最佳网格路径。
阶段 C 解决两个不同问题：**C1** 尝试让连续 timing 参数从 DP 给出的
时间表附近开始；**C2** 尝试保留不止三个窗口序列。二者均是可选的
搜索/初始化扩展，不改变固定空间路径、安全网格、原动态碰撞物理代价
或最终 DenseCheck。当前 C1/C2 要选用 `batch_exact` 等 batch 后端；
`serial` 参考实现没有执行这两个分支。

### 7.1 C1：把 DP 到达时刻投影回现有 timing 表示

对每个 DP 分支，除了窗口序列 $m$，回溯还给出离散到达时间表
$\widehat t_i$。但可执行 timing 必须由 Phase5 的控制点或 F1 的六维
latent 解码，不能直接把 $\widehat t_i$ 当成执行轨迹。代码从原参数
$\theta_0$ 出发，求一个投影近似：

$$
\min_\theta\quad
\frac1{N_s}\sum_i\left(
\frac{t_\theta(s_i)-\widehat t_i}{T_{\max}}
\right)^2
+10^{-4}\,\operatorname{mean}\bigl((\theta-\theta_0)^2\bigr).
$$

第二项避免为了拟合粗网格时间表而把参数带离原先验太远。
实现用学习率 0.08 的 Adam 做 8 步、裁剪梯度范数至 4；Phase5 的
不可优化控制点每步都恢复原值，F1 仍只操作六维 latent。
因此这是“在旧表示中近似拟合 DP 时间表”，**没有**增加 timing
控制点、改变 checkpoint 或保证精确通过所有窗口。

拟合后先检查到达时刻有限、严格递增，且总时长在
$[T_{\min},T_{\max}]$。随后用原动态 SDF 计算精确物理目标，连同
当前分支的走廊惩罚，只有

$$
J_{\rm phys}(P,\theta_{\rm fit})
+\lambda_{\rm cor}J_{\rm cor}(\theta_{\rm fit};m)
<J_{\rm phys}(P,\theta_0)
+\lambda_{\rm cor}J_{\rm cor}(\theta_0;m)
$$

才以 $\theta_{\rm fit}$ 作为后续 refinement 初值；否则回退到
$\theta_0$。这里的“精确”指没有使用 B2 插值 surrogate，**不等于**
拟合初值已经通过 DenseCheck。记录的 `dp_fit_abs_error_s_mean/max`
是拟合后实际到达时刻与 DP 时间表的绝对误差；即使拟合结果因成本
不优而被拒，该误差仍可作为表示能力诊断。六维 latent 可能无法表示
粗 DP 找到的急剧等待/赶路时间表，故 C1 不保证更快收敛或更高覆盖。

### 7.2 C2：K-best 不同窗口序列，而不是 K 条重复网格路径

单纯选代价最低的 K 条 $(s_i,\tau_{j_i})$ 路径，很可能全落在同一组
安全窗口内，无法提供“先过/后过”的新模态。当前 K-best DP 给标签
附上窗口 ID 序列 $m=(k_0,\ldots,k_i)$，在每个时间格的前驱前缀中
对相同 $m$ 只保留较低分标签，并截断为至多 $K$ 个；到终点再按
完整窗口序列去重和排序。其增量分数使用**未加早/晚偏好**的
原到达时间 $t_i^0$：

$$
\Delta J_i=\frac{(\tau_{j_i}-t_i^0)^2}{N_s}.
$$

原来的 $\beta=0,-1,+1$ 三个分支先放入候选集合；K-best 再补充
尚未出现的窗口序列，总分支数受 $K\in\{4,8\}$ 限制。
这是有限标签/分支预算，不是所有安全窗口组合的完整枚举，也不保证
精确找出全局最优的 K 种连续 timing 模态。网格标签彼此不同，经过
六维 latent 或 Phase5 spline 优化后仍可能落入**同一个**实际窗口
序列，或落在所有窗口之外；因此“DP 找到 K 条”不能当作收益指标。

### 7.3 备选分支验收与“模态收益”的真实口径

一个空间路径的多个 timing 分支优化后，按精确目标从低到高排序。
默认主分支先走原 DenseCheck。若开启备选救回且主分支失败，运行时
才按排序尝试该路径的下一条分支：每轮每条空间路径最多取一个备选，
以批处理送入**同一套 DenseCheck**，且总额外检查数受预算限制
（默认 64、单批最多 32）。相同参数的重复分支被跳过；一条备选
真正通过 DenseCheck 才记作“救回候选”。若无备选通过，原候选若
本来有效仍可按第 5 节的规则回退。

统计中的“优化后不同窗口序列”不是 DP 标签数：代码把**优化后**的
$t_\theta(s_i)$ 重新映射到每相位安全窗口，窗口外标为 `-1`，然后
按空间路径去重；含 `-1` 的序列不计入有效不同序列，另行记录窗口外
分支数。`validated_unique_window_sequences` 进一步只计
DenseCheck 接受的序列；`alternate_branch_rescued_candidates` 计主分支
失败但备选通过的候选数。这三项必须和分支数、额外 DenseCheck 次数
一起看，才能判断 C2 是否产生真实模态收益。

### 7.4 选择性扩展 K：先排除 timing 无法修复的空间问题

固定 $P$ 时，改变 $t_\theta$ 不能消除静态环境碰撞、自碰撞或关节
位置越界。启用 `selective_k` 后，运行时用原始 DenseCheck 和静态专用
clearance 标出这些路径，整条路径不进入 Corridor timing 搜索；
**动态障碍碰撞本身不在此排除条件内**。对其余路径，只有初始到达
时刻落在安全窗口外，或原始 DenseCheck 显示速度/加速度违规时，
才额外扩展 K-best；原三偏好分支仍先计算。

这只是节约预算的决策，不是“所有动态碰撞都必定扩展 K”的保证。
因为初始到达时刻可能位于粗相位安全窗口内、但在相位之间仍与动态
物体冲突，最终仍须由原物理代价和 DenseCheck 发现。

### 7.5 C1 早停：必须通过原 DenseCheck

可选早停要求先启用 C1。某分支只有同时满足下列条件才会提前停下
Adam：所有抽样相位到达时刻落入选定窗口；物理代价连续两次变化
不超过 $10^{-3}\max(|J_{\rm phys}^{\rm prev}|,1)$；迭代至少到第 4 步，
且当前步数是 4 的倍数；**该参数立即通过原 DenseCheck**。
梯度很小、窗口惩罚为零或 DP 可达都不能单独触发安全早停。
早停只是省下后续优化迭代，最终候选仍执行常规 DenseCheck。

### 7.6 当前开关和已知结果

`corridor_a_dp_init`、`corridor_a_k_best`、`corridor_a_branch_fallback`、
`corridor_a_selective_k`、`corridor_a_early_stop` 默认分别是
`False`、`0`、`False`、`False`、`False`，不会随 `corridor_a_enabled`
自动打开。备选救回、选择性 K 和早停要求 batch 后端；早停还要求
C1。正式比较时必须明确记录开关、后端、chunk size 与 DenseCheck
预算，不可把仅启用 Corridor A 与组合启用阶段 C 混为一组。

现有冻结请求与在线多场景消融记录见
[F1 Corridor 分支研究](FACTORIZED_CORRIDOR_BRANCH_STUDY.md)：增加分支数
曾显著增加耗时，但没有在这些样本中增加通过 DenseCheck 的不同
窗口序列或救回候选；DP 时间表的拟合误差也不小。因此 C1/C2 是
**已实现但默认关闭的实验功能**，不能仅凭搜索到更多离散路径宣称
规划成功率或时间模态覆盖提升。

## 8. 代码对应

- [固定路径网格、安全窗口、DP、窗口代价和 refinement](../mpd/inference/time_corridor.py)
- [动态 SDF、世界快照与执行起点时钟](../mpd/inference/dynamic_collision.py)
- [Phase5 timing spline](../mpd/parametric_trajectory/timing_spline.py)；[F1 六维 latent 解码](../mpd/inference/learned_timing.py)
- [原物理代价与 Corridor 参数默认值](../mpd/inference/space_time_guidance.py)
- [运行时 DenseCheck 与原候选回退](../scripts/runtime/space_time_runtime_engine.py)
- [阶段 A/B/C 的设计与实验边界](TIMECORRIDOR_SPEED_FIRST_IMPLEMENTATION_PLAN.md)；[实施记录](TIMECORRIDOR_SPEED_FIRST_IMPLEMENTATION_REPORT.md)
- [F1 的 C1/C2 和备选分支消融](FACTORIZED_CORRIDOR_BRANCH_STUDY.md)
