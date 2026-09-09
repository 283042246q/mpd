# Marvin 双臂扩散网络消融

四组使用完全相同的 14 维关节轨迹、scheme-3 双槽位 EE context、扩散过程和
loss。默认组为 A，因而旧配置不增加字段时仍使用原网络。

| 组 | Context | Denoiser | 输出契约 |
|---|---|---|---|
| A | 原始 `40 -> 128` MLP | 原 14D joint `TemporalUnet` | `[B,H,14]` |
| B | 左右臂共享局部编码器，按 `left,right` 拼接后用 MLP 融合；无 attention | 原 14D joint `TemporalUnet` | `[B,H,14]` |
| C | B 的每臂 token，加 pair token，执行 2 层跨臂 Transformer，再融合到 128D | 原 14D joint `TemporalUnet` | `[B,H,14]` |
| D | C 的跨臂编码，但保留 `[left,right,pair]` 三个 token | 双流 `CoupledBimanualTemporalUnet` | `[B,H,14]` |

## Context 细节

B/C/D 将 `q_start(14)` 拆成左右两个 7D 关节向量，EE goal 使用左右两个固定
12D pose 槽位和各自 mask。关节编码器、目标编码器和局部融合器在两臂间共享
权重，另加可学习的左右身份向量。mask 为 0 时，目标 pose 特征替换为可学习的
inactive token；这避免单臂任务中未激活槽位保存的 FK pose 被误解为目标，但该臂
当前关节状态仍然参与避碰和轨迹生成。

C/D 的 token 顺序固定为 `left, right, pair`。C 把三者重新融合成 128D，因此与
原 joint U-Net 接口完全相同。D 返回扁平的 384D 张量以兼容 DataParallel 和现有
diffusion warmup，去噪器入口再还原三个 128D token。

## D 的耦合去噪器

D 将 noisy trajectory 拆成 `[B,H,7]` 左右双流。两流使用同一套 temporal
convolution 权重；每个分辨率在相同轨迹索引上执行 two-token 跨臂 attention，
瓶颈则联合 pair token 与所有 `arm x time` token 做全局 attention。FiLM 条件对
每臂使用 `diffusion_time + own_arm_token + pair_token`。最终重新拼接为 14D。

22 个原始 B-spline 控制点所对应的 17 个可学习点仍在网络内部 pad 到 24，输出
再裁回 17；四组均不要求重拟合已有 spline。

## 选择方式

YAML：

```yaml
bimanual_network_variant: D
```

命令行（命令行覆盖 YAML）：

```bash
python -m scripts.train.train_marvin_warehouse_bimanual \
  --config scripts/train/cfgs/marvin_bimanual_warehouse_independent.yaml \
  --network-variant D
```

使用 YAML 或命令行切换时，默认结果目录的 `_variant_A` 会同步替换成所选组别；
显式传入 `--results-dir` 时以用户目录为准。

在默认 `context=128, base_channels=32, dim_mults=(1,2,4,8)` 下，Context +
Denoiser 参数量约为 A 4.57M、B 4.75M、C 5.03M、D 6.86M。A/B/C 的去噪器
参数完全相同；C 相对 B 的差异来自显式 context attention。D 不是参数量匹配的
消融，它用于验证共享双流归纳偏置和跨臂/跨时间耦合的完整收益，比较时应同时
报告训练吞吐、推理延迟和显存。
