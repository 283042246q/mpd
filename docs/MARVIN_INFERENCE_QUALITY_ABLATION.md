# Marvin A5 推理质量消融

默认 runtime 已启用 A5-b32 的四项优化：pair streaming、dense validator
chunking、parent-bound broad phase 和 `foam_pika_100` guide geometry。默认 chunk 为：

- guide self/inter-arm pair chunk：1024；
- validator candidate chunk：8；
- validator time chunk：32；
- validator self-pair chunk：1024。

`benchmark_marvin_inference_quality.py` 自动生成分级任务，再在同一批任务和 diffusion seed 上一次只改变一个因素：

- 候选数：基线 32，以及 4 个独立的 32-candidate batch（总计 128）；
- DDIM steps：10、15（基线）、20、30、50；
- `t_start_guide_steps_fraction`：0.20、0.33（基线）、0.50、0.75、1.0；
- `n_guide_steps`：2、4、6（基线）、8、12；
- EE goal 后期线性增强：1x（基线）、2x、4x；
- `compute_costs_with_xrecon`：false（基线）、true；
- A/B/C/D 每个 run 中 validation loss 最低且确实存在的 1–3 个完整 checkpoint。

4×32 使用四个 seed 连续的独立推理进程，因此 GPU 峰值仍是 batch 32 的量级。聚合时，
任意一批存在有效轨迹就将该 request/seed 计为成功；候选数、四批总耗时和状态同时保留。
这测量的是“增加候选总数”的收益，包含四次模型加载开销。若以后需要测常驻模型的
纯采样耗时，应改用 runtime engine 在同一进程连续调用四次。

checkpoint 首先按 `val_losses.npy` 的 `VALIDATION total_loss` 排名；该指标只用于缩小候选
集合。最终的“最优 1–3 个”应按相同 request/seed 的规划成功率、有效轨迹数、EE goal
误差、耗时和显存确定。manifest 保存 checkpoint 文件大小、mtime 和完整训练参数，避免
checkpoint 在实验期间被替换而不被发现。

在仓库 `mpd` 目录下一键运行（环境统一为 `mpd-splines-public`）：

```bash
conda run --no-capture-output -n mpd-splines-public \
python -m scripts.inference.benchmark_marvin_inference_quality \
  --output-dir scripts/inference/logs/marvin-quality-graded --device cuda:0
```

无需准备 request.json，默认自动采样并执行。`--prepare-only` 仅生成任务和实验清单；
`--run` 保留兼容但已不必指定。子进程沿用当前 Python 环境。CUDA 不可用时在采样前
明确报错，不会将所有配置的环境错误伪装成规划失败。

默认四个层级、12 个场景，每场景 3 个新任务，共最多 36 个任务；3 个 diffusion seed，
29 个消融 case（含 ABCD 各 3 个 checkpoint）时最多启动 3456 次推理子进程。
这是一组完整实验；可先运行以下小规模检查：

```bash
conda run --no-capture-output -n mpd-splines-public \
python -m scripts.inference.benchmark_marvin_inference_quality \
  --output-dir scripts/inference/logs/marvin-quality-smoke \
  --tiers easy medium --tasks-per-scenario 1 \
  --skip-checkpoints --cases baseline candidates-4x32 --seeds 12345
```

可先只跑 D checkpoint：先执行一次带 `--prepare-only` 的命令，从 manifest 复制实际 case 名，
然后传给 `--cases`。也可以用 `--checkpoint-runs` 显式加入 D warm-start 或新数据训练 run；
脚本会拒绝把不同 `dataset_subdir` 的模型混入同一组对比。

`--cases` 属于不可变实验清单，改变 case 集合需要新的 output-dir。

| 层级 | 场景 | 历史依据 |
|---|---|---|
| easy | 双臂同侧桌面 → 同侧桌面 | 桌面有效端点较充足，作为相对容易的基线 |
| medium | 桌面 → 左/右书架下层核心区，另一臂仍在同侧桌面 | 使用采样后收缩的核心区，左右分别统计 |
| hard | 左/右上层核心区；左/右跨桌 x 边缘；双臂同时到下层 | 上层、跨区及双臂目标组合增加约束 |
| extreme | 左/右跨桌 y 边缘；左/右上层 y 外缘 | 右跨 y 仅 6/200 无碰撞目标；上层 y 左0/200、右5/200 |

区域来自生产采样 YAML 的实际左右臂区域；上层 y 外缘来自
`benchmark_marvin_workspace_edges.edge_regions()`，历史依据见
[边缘加密统计](MARVIN_WORKSPACE_EDGE_RESULTS.md)。这些是区域难度假设，不能保证每条
easy 轨迹都简单或每条 hard 轨迹都困难。历史边缘 RRT 测试与这里的双臂任务不同，
不能把历史成功率直接当作本次成功率。

新 output-dir 使用系统熵随机采样，与 diffusion seed 无关。每个场景最多 30 次完整采样，
每次截止时间默认 15 秒（在数值求解边界检查，单次求解可能略超时）；IK 使用已有
region_ik 的随机目标/初值流程，每次 `_target_state` 默认最多 30 次提案，
并非同一 TCP 固定目标重启 30 次。所有端点经过完整整机碰撞检查，两臂的关节变化
均至少 0.08 rad。不会预先筛选 RRT 或 MPD 成功，失败不会换到另一种场景补额。
极难区可能没有任务，其缺额单独报告，不计为 MPD 失败。

`tasks/settings.json` 保存区域完整 XYZ/姿态配置和采样预算；`tasks/tasks.json` 保存
每个场景尝试次数、缺额、实际 TCP 起终点、14D 关节状态及内部请求。
重新运行相同命令会复用已生成任务和完成的推理，保证不同 checkpoint 比较同一批目标。
重新随机生成请更换 output-dir。任务采样在独立 CPU 子进程结束后才加载 GPU 模型。

总结果持续写入 `summary.json`，分层结果写入 `summary-by-tier.json`，每次推理的耗时、
显存、OOM/timeout 等状态保留在 `runs/**/benchmark-result.json`。成功率的分母是实际
生成的有效任务×seed；必须同时查看采样缺额，避免把难区缺样误解为高成功率。
仍可通过可选 `--requests` 复现外部任务（不能带 checkpoint_hash 或绝对 deadline）。

验证记录：`mpd-splines-public` 下 6 个质量消融测试通过，覆盖采样预算/缺额、恢复、
分层分母以及默认执行 4×32。实际对 12 个场景执行每场景最多 2 次、每次 3 秒的
采样冒烟测试，生成 8 个有效任务，4 个场景记录缺额。
另一次真实 CUDA A5-b32 自动桌面任务测试约 13.2 秒完成，无 OOM，结果为
`no_valid_trajectory`，已正确进入 easy 层失败统计；这验证流程，不代表规划成功。
沙箱内 CUDA 不可用，沙箱外相同 conda 环境 CUDA 正常。尚未运行全部分级消融。
