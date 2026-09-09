# Marvin 双臂 MPD Phase 2：Isaac Lab 检测与回放

Phase 2 保留 MPD 与 Isaac Lab 两个独立 Python/Conda 进程。MPD 使用
`mpd-splines-public`，Isaac Lab 使用 `env_isaaclab`。`plan()` Python API、
`--backend mpd` 默认值及 Franka 入口均不变；与 Franka 一致，物理后端由
`--sim-backend isaaclab` 选择。

## 资产约定

运行时使用仓库内的规范文件：

- `marvin_pika_bimanual_mpd.urdf`；
- 与 URDF 相邻的本地 Marvin/Pika STL；
- `pika_assets.lock.yaml` 中锁定的来源 commit、URDF SHA256 和整体资产 SHA256。

不下载远程机器人或 marker USD。首次启动 Isaac Lab 时，规范 URDF 被转换到
`.cache/isaaclab/marvin_bimanual/`；该目录已被 git 忽略。每份检测/回放结果同时记录
URDF、顶层 USD 和包含 payload 的完整 USD bundle hash。

Isaac 的 14 个 native joint 按左右臂交错排列，loader 会按名称重排成 MPD 的
`Joint1_L..Joint7_L, Joint1_R..Joint7_R`，因此不会向夹爪或 mimic joint 写值。
两个 `*_pika_gripper_tcp` 是 massless fixed Xform，并非 Isaac rigid body；其 pose 由
`*_gripper_base_link` 和经 URDF hash 门禁验证的 `[0, 0, 0.21] m` 固定变换计算。

## 回放已有 inference artifact

先对全部 Top-K 做物理检测：

```bash
conda run -n env_isaaclab python \
  scripts/isaaclab/evaluate_marvin_bimanual_trajectories.py \
  --artifact /absolute/path/to/artifact \
  --output /absolute/path/to/artifact/isaaclab-evaluation.json \
  --device cuda:0 \
  --viz none \
  --graceful-shutdown
```

再回放指定的 Top-K 轨迹。`--evaluation` 用于验证 artifact hash，并在左右 TCP 路径、
双目标 frame 之外标出第一个接触 waypoint：

```bash
conda run -n env_isaaclab python \
  scripts/isaaclab/replay_marvin_bimanual_trajectory.py \
  --artifact /absolute/path/to/artifact \
  --evaluation /absolute/path/to/artifact/isaaclab-evaluation.json \
  --trajectory-index 0 \
  --output-video /absolute/path/to/artifact/isaaclab-replay.mp4 \
  --screenshot /absolute/path/to/artifact/isaaclab-replay.png \
  --output-json /absolute/path/to/artifact/isaaclab-replay.json \
  --device cuda:0 \
  --viz none \
  --graceful-shutdown
```

相机输出依赖 Isaac RTX renderer 和可用的 NVIDIA 驱动。无渲染 GPU 时省略
`--output-video`、`--screenshot`，仍会执行相同的双臂物理回放并写 `--output-json`，
适合 CI/服务器烟测。

## MPD 推理后直接检测并回放

该入口先完整、原子地写出 Phase 1 artifact，然后释放 MPD CUDA cache，再顺序启动
Isaac evaluator 和 replay 子进程：

```bash
conda run -n mpd-splines-public python \
  scripts/inference/inference_marvin_bimanual.py \
  --request /absolute/path/to/request.json \
  --output-dir /absolute/path/to/run \
  --backend mpd \
  --device cuda:0 \
  --sim-backend isaaclab \
  --isaaclab-conda-env env_isaaclab \
  --isaaclab-device cuda:0 \
  --isaaclab-trajectory-index 0
```

默认生成：

- `result.json`、`trajectory.npz`、`scene.json`；
- `isaaclab-evaluation.json` 与日志；
- `isaaclab-replay.mp4`、`isaaclab-replay.png`、`isaaclab-replay.json`；
- 汇总两个子阶段的 `isaaclab-run.json`。

无 GPU 的物理烟测可增加 `--isaaclab-device cpu --no-isaaclab-capture`。只做检测则增加
`--no-isaaclab-replay`。若 Isaac 子阶段失败，已经成功写出的 MPD artifact 不会被覆盖，
错误会写入 `isaaclab-run.json`，进程返回码为 6。检测完成但发现 MPD→Isaac safety
false-negative 时仍会生成回放，汇总状态为 `safety_validation_failed`，入口返回码为 7。

`contract_stub` 只用于链路/格式测试，不表示 MPD 安全，因此其 evaluation 不产生
MPD-vs-Isaac 安全 confusion 结论。Phase 2 的最终放行应使用真实 checkpoint 生成的
Warehouse golden artifacts，并要求 `safety_false_negative=false` 且 MPD/Isaac 双 TCP
FK 误差通过报告中的门限。
