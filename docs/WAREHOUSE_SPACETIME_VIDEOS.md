# Warehouse 静态与动态环境 IsaacLab 视频演示

2026-10-11。独立演示入口，展示实际 `EnvWarehouse` 桌面、货架、单臂 Panda 和冻结的移动障碍预测。包含机械臂轨迹回放；无需 ROS 服务或训练 checkpoint。

## 运行与现成结果

已生成的 30 段视频位于：

```text
benchmark_results/warehouse_spacetime_videos/20261011/videos/
```

查看 [30 段视频索引](../benchmark_results/warehouse_spacetime_videos/20261011/videos/INDEX.md)、[总览图](../benchmark_results/warehouse_spacetime_videos/20261011/videos/contact-sheet.jpg)、[验收报告](../benchmark_results/warehouse_spacetime_videos/20261011/videos/report.json)。这些生成产物由 git ignore 管理，不提交大文件。

从头生成另一组：

```bash
cd /home/eric/Projects/MotionPlanningDiffusion/mpd
conda activate mpd-splines-public

python -m scripts.isaaclab.run_warehouse_spacetime_videos \
  --output-root benchmark_results/warehouse_spacetime_videos/my-suite \
  --variants 3 \
  --backend torch-forest \
  --planning-device cuda:0 \
  --fps 24 --width 1280 --height 720
```

默认 10 个模式，每种三个不同任务/world；每个场景包含 sphere、box、z-capsule 三个物体，**三个物体都使用该场景对应的运动模式**，方向、尺寸、事件时刻与位置仍不同。每种模式的三个 variant 使用不同 seed/任务；程序检查轨迹与 world signature 不重复。

| 模式 | 动态物体运动 |
| --- | --- |
| linear | 匀速穿越 |
| accelerating | 平滑加减速，保持前进 |
| speed_variation | 周期速度变化 |
| curved | 前进叠加正弦侧摆 |
| zigzag | 前进叠加三角波侧摆 |
| shuttle | 往返横扫 |
| orbit | 环绕运动 |
| vertical | 纵向穿越并轻微侧摆 |
| stop_go | 在交互点停留后继续运动 |
| helix | 前进叠加螺旋侧向运动 |

仅生成某几种：

```bash
python -m scripts.isaaclab.run_warehouse_spacetime_videos \
  --output-root benchmark_results/warehouse_spacetime_videos/orbit-demo \
  --modes orbit helix --variants 3
```

纯 CPU 原版 OMPL ST-RRT* **规划阶段**可使用：

```bash
python -m scripts.isaaclab.run_warehouse_spacetime_videos \
  --output-root benchmark_results/warehouse_spacetime_videos/ompl-suite \
  --backend ompl-strrtstar --planning-device cpu --cpu-workers 4 \
  --variants 3
```

IsaacLab RTX 视频录制仍使用 GPU，运行在 `env_isaaclab`；planning-device 不决定渲染设备。默认生成后端 `torch-forest` 是本仓库 GPU 时空 RRT-Connect teacher，受 ST-pRRTC 的目标时间森林/根回收启发；完整原版 ST-RRT* 对应 `ompl-strrtstar`。

## 分阶段与恢复

只生成数据与 portable payload：

```bash
python -m scripts.isaaclab.run_warehouse_spacetime_videos \
  --output-root benchmark_results/warehouse_spacetime_videos/my-suite \
  --prepare-only
```

只录制已经准备完整的 suite，并复用完成的视频：

```bash
python -m scripts.isaaclab.run_warehouse_spacetime_videos \
  --output-root benchmark_results/warehouse_spacetime_videos/my-suite \
  --render-only --resume
```

完整流水线恢复时使用原生成参数加 `--resume`。planning 参数变更受原生成器的冻结配置检查保护；更换 seed/算法应使用新 output root。未准备完整的 suite 不允许进入正式录制阶段。

渲染会核对 payload/static/software hash、尺寸、fps、相机与资产地址；只有这些条件一致且视频大小与完成记录一致才复用视频。更换 renderer 或设置后重新录制对应视频。视频先写临时文件，完成后原子替换；随后转 H.264、逐帧读取并更新验收报告。

直接调用 IsaacLab renderer，可限制数量或调相机：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  CONDA_PREFIX=/home/eric/anaconda3/envs/env_isaaclab \
  /home/eric/IsaacLab/isaaclab.sh -p \
  scripts/isaaclab/replay_warehouse_spacetime.py \
  --suite benchmark_results/warehouse_spacetime_videos/my-suite/suite.json \
  --output_dir benchmark_results/warehouse_spacetime_videos/my-suite/videos \
  --camera_eye -2.2 -2.2 2.15 --camera_target 0.35 0 0.45 \
  --limit 1 --headless --enable_cameras --resume
```

直接 renderer 先写 MP4V；完整流水线会自动调用 `finalize_warehouse_spacetime_videos.py` 转为 H.264。最终转码工具在 `env_isaaclab` 中使用已有 `imageio_ffmpeg` 的 ffmpeg，不依赖系统 ffmpeg。

## 世界与时间语义

- 静态场景由现有 `export_isaaclab_scene_payload(..., include_boxes=True)` 导出，完整包含 12 个 Warehouse primitive；导出有 unsupported 几何时拒绝录制。
- 动态 obstacle 使用保存的 `world/times/positions/shape/size`，按真实秒数做分段线性插值，不从运动模式名重新计算另一条曲线。box size 为 half extents；capsule 的圆柱段长度为两倍 half-length，另加两端球帽。
- 世界使用 `execution_start=0`，规划耗时不推进障碍。每个物体必须在实际机械臂执行期间进入工作区；事件不发生在轨迹结束以后。
- 静态/动态显示 primitive 不启用物理碰撞响应；障碍与货架/桌面以及其他障碍可以穿模，规定的路径不改变。
- 机械臂使用保存的执行 q(t)。每帧写入当前关节位置，零速度刷新 PhysX/Fabric 并重新施加记录姿态，避免控制跟踪误差改变展示轨迹。检查 Isaac 手部位置与 MPD FK 的误差，超过 1 cm 拒绝输出合格视频。
- 这是**运动学回放**。轨迹通过球/primitive 模型的静态、自碰撞、动态及速度/加速度检查；视频不等于额外的 mesh/contact/动力学执行验收。
- 终点帧使用精确终点时间；因为首尾帧都编码，MP4 时长比执行时长多不足 `2/fps` 秒，没有延长 task 的碰撞 horizon 或增加 terminal hold。

视频 HUD 显示模式、variant、当前秒数/执行时长和各物体的工作区事件时刻。蓝线为机械臂手部路径；红/青/黄分别对应 sphere/box/capsule 的近期与未来短轨迹线。截图选择中间事件附近的帧；某个物体在该帧被静态物体遮挡不表示没有生成，其完整运动在视频中回放。

默认 Franka USD 使用当前有效的 `Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd`；可通过 runner 的 `--robot-usd` 或 renderer 的 `--robot_usd` 指定本地资产。runner 会清理跨 conda 环境的库路径，使用 `/home/eric/IsaacLab/isaaclab.sh` 启动；不修改 IsaacLab 的全局配置。

## 产物和验收

```text
OUTPUT/
  static_scene.json             与生成环境一致的静态 primitive
  suite.json                    10 modes × 3 variants 的完整清单
  datasets/MODE/                原生成器的 HDF5 shards/config/index/report
  payloads/MODE-NN.npz           q/dq/time、动态 world、FK 手部路径、来源/质量
  logs/                         规划和 IsaacLab 日志
  videos/MODE-NN.mp4             最终 H.264 MP4
  videos/MODE-NN.png             事件时刻截图
  videos/MODE-NN.json            fps/帧数/参数/hash/FK 误差/质量/验收范围
  videos/report.json            完成数量、每模式数量、逐帧解码结果
  videos/INDEX.md                视频与截图链接
  videos/contact-sheet.jpg       每行一个模式、每列一个 variant
```

2026-10-11 验收：10 种模式各 3 段、共 30 段，1280×720 / 24 fps；30 条磁盘轨迹以更细时间步长独立复检全部通过。视频共 4992 帧，执行时长约 5.26–13.08 s；最大渲染手部误差约 0.0221 mm。视频转 H.264 后逐帧解码、分辨率/fps/帧数和非空画面检查通过；没有重复视频文件。

此次独立复检结果保存在该 run 根目录的 `validation.json`。常规 runner 保留原生成器的密集轨迹验收；需要单独进行 5 ms 磁盘复检时，对每个 `datasets/MODE` 使用既有 `validate_warehouse_spacetime` 入口。

数据/时间/helper 与旧 batch RRT 回归测试共 20 项通过。脚本入口：[runner](../scripts/isaaclab/run_warehouse_spacetime_videos.py)、[renderer](../scripts/isaaclab/replay_warehouse_spacetime.py)、[payload](../scripts/isaaclab/warehouse_spacetime_payload.py)、[视频终检](../scripts/isaaclab/finalize_warehouse_spacetime_videos.py)。
