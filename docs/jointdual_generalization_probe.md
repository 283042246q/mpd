# JointDual 时间泛化诊断：固定样本、EMA、噪声与 diffusion timestep 分桶

脚本：`python -m scripts.train.eval_jointdual_generalization`。
它只读现有 checkpoint、snapshot 与缓存，输出到新的诊断目录，不启动训练。
支持 JD-KIN/JD-WORLD 和 c/tau_r；一次比较必须使用同一 snapshot、variant、表示与 diffusion schedule。

## 评估口径

- 加载每个 checkpoint 的 **EMA**，调用 `eval()`，FP32、不启用 AMP。WORLD 保留真实 KNOWN/EMPTY 条件，不随机置 NULL。
- train、validation 默认各均匀抽 4096 个不同 family，每个 family 随机取一行；两边使用相同的 family 均衡目标分布。它与训练时有放回的 family/world 采样具有相同的目标期望，但避免了重复抽中 family。`--sampling row` 改为两边均匀无放回抽行。
- seed 固定后，样本选择不随 checkpoint 变化；验证和训练 family 必须完全不重叠。
- 默认把 100 个 diffusion steps 分为 `[0,19]`、`[20,39]`、`[40,59]`、`[60,79]`、`[80,99]`。每个选中的样本都在每个桶内评估，默认每桶抽 2 次 k 和独立高斯噪声。
- k、空间噪声、时间噪声在 CPU 上预先生成，所有 checkpoint 复用。train/validation 对应抽样序号也使用相同噪声，以减少 Monte Carlo 差异；这不意味着两边任务一一匹配。
- 生成噪声不依赖 batch size 或推理 device。更改样本数、seed、桶数或 repeats 是更改评估协议；CPU/GPU 及不同 batch 的数值计算仍可能有微小差异。
- 输出原始的逐坐标均值 `P MSE`、`Z MSE`，不乘训练 loss 权重。tau_r 额外输出 tau、r，满足 `Z=(tau+5*r)/6`。这些是标准化变量的噪声预测误差，不是物理秒数或关节角度误差。
- `all` 按桶宽度加权，对应均匀 diffusion timestep 的目标分布；不等宽桶不会被简单等权平均。

## 命令

在项目根目录和已有环境中运行。下面使用 NVMe 上的新 `diagnostics` 目录保存小型分析产物；模型与缓存仍从迁移后的 data 盘读取。

```bash
conda activate mpd-splines-public
cd /home/eric/Projects/MotionPlanningDiffusion/mpd

# 先用小规模 smoke 检查环境与路径；CPU 也可运行。
python -m scripts.train.eval_jointdual_generalization \
  --snapshot data_trajectories_spacetime/jointdual-main-snapshot.json \
  --checkpoints \
    data_trained_models/jointdual/jdworld_tau_r_1/step-00061000.pt \
    data_trained_models/jointdual/jdworld_tau_r_1/step-00200000.pt \
  --samples 32 --noise-repeats 1 --batch-size 16 --device cpu \
  --output-dir /tmp/jointdual-generalization-smoke

# main：从时间最佳附近到后期，检查训练误差与验证误差是否反向变化。
python -m scripts.train.eval_jointdual_generalization \
  --snapshot data_trajectories_spacetime/jointdual-main-snapshot.json \
  --checkpoints \
    data_trained_models/jointdual/jdworld_tau_r_1/step-00061000.pt \
    data_trained_models/jointdual/jdworld_tau_r_1/step-00082000.pt \
    data_trained_models/jointdual/jdworld_tau_r_1/step-00112000.pt \
    data_trained_models/jointdual/jdworld_tau_r_1/step-00150000.pt \
    data_trained_models/jointdual/jdworld_tau_r_1/step-00200000.pt \
  --samples 4096 --sampling family --timestep-bins 5 --noise-repeats 2 \
  --seed 20261023 --batch-size 64 --device cuda:0 \
  --output-dir diagnostics/generalization/main-world-taur-seed20261023

# multiworld：检查这里是否只有较轻的时间退化。
python -m scripts.train.eval_jointdual_generalization \
  --snapshot data_trajectories_spacetime/jointdual-multiworld-snapshot.json \
  --checkpoints \
    data_trained_models/jointdual/jdworld_c_multi/step-00135000.pt \
    data_trained_models/jointdual/jdworld_c_multi/step-00153000.pt \
    data_trained_models/jointdual/jdworld_c_multi/step-00181000.pt \
    data_trained_models/jointdual/jdworld_c_multi/step-00200000.pt \
  --samples 4096 --sampling family --timestep-bins 5 --noise-repeats 2 \
  --seed 20261023 --batch-size 64 --device cuda:0 \
  --output-dir diagnostics/generalization/multi-world-c-seed20261023
```

输出目录必须不存在或为空，避免覆盖旧诊断。将 checkpoint 路径替换为 KIN 或其他表示即可；main WORLD tau_r 的完整训练在 `jdworld_tau_r_1`，不要误用仅 7000 步的 `jdworld_tau_r/last.pt`。

默认直接读已有 tensor cache，不创建 lock、不重建缓存；缺失时明确报错。可用 `--tensor-cache /path/to/existing/cache` 指定，或 `--no-tensor-cache` 从原 HDF5 读取。若需要建缓存，先单独运行 `scripts.spacetime_data.build_jointdual_tensor_cache`。

## 输出与判断

- `plan.json`：完整样本索引、family/file/group、snapshot identity、噪声协议与 SHA256。
- `report.json`：每个 checkpoint 的训练配方、文件 SHA256、分桶统计和配对差值。`complete: true` 表示全部完成。
- `summary.csv`：每行是 checkpoint × timestep 桶 × 指标；包含 `train`、`validation`、`gap=validation-train`、`delta_train`、`delta_validation` 及差值的近似 95% 区间。
- `checkpoint-000.npz` 等：train/validation 逐样本误差，形状 `[samples, bins, metrics]`，已对 noise repeats 求均值。样本、桶与指标顺序见 plan/report。

`delta_*` 始终相对命令中的**第一个 checkpoint**，对同一 split 使用相同样本、k 和噪声相减。命令顺序保持不变，不自动按 step 排序；热启动可能重置 step，分叉训练也不能简单视为连续时间线。

先看 `bucket=all, metric=z`，再看各桶和 tau/r：

| 同一评估协议下的趋势 | 更支持的解释 |
|---|---|
| train Z 下降、validation Z 上升，且多个后期点重复出现 | 时间泛化退化，与继续拟合训练集一致；仍不能单独证明逐样本记忆 |
| train/validation Z 同时上升 | 不能简单归为过拟合，还需查优化、loss 配方和共享表示变化 |
| 两者都下降，但 validation 下降较慢 | 仍在泛化改善，不能仅凭 gap 扩大判过拟合 |
| 变化集中于个别 k 桶或 tau/r 分量 | 按该噪声区间/表示分量继续诊断，不应立即整体缩小网络 |

差值的标准误按 **family 聚类**估计；row 模式中，同 family 的多个 world 不被当作独立样本。区间条件于已抽到的噪声，未覆盖训练 seed 的方差，也未做多桶、多 checkpoint 的多重比较校正。不要把单个区间排除 0 当成最终结论。

建议先用一组 seed 定位趋势，再以 `--seed 20261024`、`20261025` 和新的输出目录重复。若 multiworld 的差异很小，可将 samples 提至 8192、repeats 提至 4。每个 checkpoint 默认会评估 `2 splits × 4096 samples × 5 bins × 2 repeats = 81920` 个带噪样本；smoke 的 32 个样本只能验证流程。

该诊断不调用 test split、不训练模型，也不直接证明参数规模或梯度失衡是原因。确认泛化退化后，再以相同协议比较正则化对照，并另测物理规划成功率。
