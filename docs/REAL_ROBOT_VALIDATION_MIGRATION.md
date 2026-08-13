# Real Robot Validation Migration Guide

本文档说明当前 MPD + IsaacLab 项目迁移到实机验证时应该怎么拆流程、补哪些接口，以及每个阶段的安全检查标准。这里的“实机验证”默认指 Panda/Franka 一类 7 自由度机械臂的关节空间轨迹执行；如果换成其他机器人，需要先替换 URDF、关节名、关节顺序、限位和控制器接口。

## 总体原则

当前项目已经形成清晰分工：

- MPD 环境负责数据、训练、推理、轨迹优化，输出候选关节轨迹。
- IsaacLab 环境负责把 MPD 生成的 Panda joint-space 轨迹放进仿真里批量验证，并用 replay 导出视频和截图。
- 实机环境不应该直接替代 MPD 或 IsaacLab，而应该作为第三层：只接收已经通过离线检查和仿真验证的关节轨迹，再通过机器人控制器低速执行。

实机迁移的推荐链路：

```text
MPD inference
  -> results_single_plan-XXX.pt
  -> isaaclab-trajectories-XXX.pt
  -> IsaacLab statistics + replay mp4/png
  -> real robot trajectory export
  -> offline safety check
  -> robot-side controller adapter
  -> low-speed real robot execution
  -> execution log + camera/robot-state record
```

不要把 `q_trajs_pos_best` 或 `q_trajs_pos_valid` 直接发给实机控制器。MPD 输出的是规划轨迹，不是已经满足实机控制周期、速度、加速度、jerk、碰撞场景和急停策略的控制命令。

## 当前已有输出

推理和 IsaacLab 验证后，当前项目主要会产生这些文件：

- `scripts/inference/logs/<RUN_ID>/results_single_plan-XXX.pt`：MPD 单次规划结果，包含 best/valid trajectories、metrics、sim_statistics 等。
- `scripts/inference/logs/<RUN_ID>/isaaclab-trajectories-XXX.pt`：IsaacLab evaluator/replay 共用 payload。
- `scripts/inference/logs/<RUN_ID>/isaaclab-statistics-XXX.json`：IsaacLab 批量接触/碰撞统计。
- `scripts/inference/logs/<RUN_ID>/isaaclab-replay-XXX.mp4`：单条轨迹视觉 replay 视频。
- `scripts/inference/logs/<RUN_ID>/isaaclab-replay-XXX.png`：单条轨迹最终截图。
- `scripts/inference/logs/<RUN_ID>/isaaclab-replay-XXX.json`：replay 元数据。

`isaaclab-trajectories-XXX.pt` 当前关键字段：

```python
{
    "q_trajs_pos": ...,   # [H, B, 7]，H 是 waypoint 数，B 是候选轨迹数量
    "q_pos_starts": ...,  # [B, 7]
    "q_pos_goal": ...,    # [7] 或目标相关张量
    "robot_name": "panda",
    "env_name": "...",
    "dt": ...,
    "scene": {...},       # sphere/box obstacle payload
}
```

实机迁移应该优先从这个 payload 中挑选已经通过 IsaacLab 统计和 replay 视觉确认的单条轨迹，而不是重新从训练脚本或模型内部取中间张量。

## 实机轨迹契约

建议新增一个稳定的实机轨迹文件格式，例如 `real_robot_trajectory.json` 或 `real_robot_trajectory.pt`。第一版建议用 JSON，方便人工检查和版本记录。

建议字段：

```json
{
  "schema": "mpd_real_robot_trajectory",
  "schema_version": 1,
  "robot_name": "panda",
  "joint_names": [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7"
  ],
  "q_traj": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
  "dt": 0.05,
  "timestamps": [0.0],
  "q_start": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "q_goal": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "limits": {
    "max_velocity_scale": 0.1,
    "max_acceleration_scale": 0.1,
    "max_joint_step_rad": 0.03
  },
  "source": {
    "mpd_run_dir": "scripts/inference/logs/<RUN_ID>",
    "isaaclab_payload": "isaaclab-trajectories-000.pt",
    "trajectory_index": 0,
    "isaaclab_statistics": "isaaclab-statistics-000.json",
    "isaaclab_replay": "isaaclab-replay-000.mp4"
  },
  "validation": {
    "isaaclab_collision_free": true,
    "offline_limit_check": true,
    "operator_approved": false
  }
}
```

实机 adapter 只接收这个稳定文件，不直接读取 MPD 模型、训练日志或 cfg。这样后续换 MoveIt、ROS2 controller、libfranka 或别的机器人时，MPD 主流程都不需要再改。

## 推荐实现路线

优先走 ROS2/MoveIt 的 `FollowJointTrajectory` 或等价控制接口，再考虑直接接 libfranka/franka_ros2。

原因：

- MoveIt/ROS controller 路径更适合第一阶段实机验证：有标准 joint trajectory message、控制器状态、速度缩放和仿真/假执行路径。
- libfranka 直连适合后期低延迟控制，但对控制周期、轨迹平滑、异常恢复和安全策略要求更高，不适合作为第一条实机通路。
- 当前 MPD 输出是 waypoint 轨迹，不是 1 kHz torque/position command；直接接底层接口容易把规划误差变成控制风险。

建议新增目录：

```text
scripts/real_robot/
  export_trajectory.py
  check_trajectory.py
  send_follow_joint_trajectory.py
  configs/
    panda_real.yaml
```

这些脚本当前还没有实现。建议按下面阶段逐步补。

## 阶段 0：离线导出和人工确认

目标：不连接实机，只把当前 MPD/IsaacLab 的输出整理成实机轨迹契约。

先跑 MPD + IsaacLab：

```bash
cd /home/eric/MotionPlanningDiffusion/mpd-splines-public
source set_env_variables.sh
conda activate mpd-splines-public

cd scripts/inference
python inference.py \
  --cfg_inference_path ./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml \
  --sim_backend isaaclab \
  --device cuda:0 \
  --isaaclab_root /home/eric/IsaacLab_ori \
  --isaaclab_conda_env env_isaaclab_ori \
  --isaaclab_device cuda:0 \
  --isaaclab_timeout_s 900 \
  --results_dir /home/eric/MotionPlanningDiffusion/mpd-splines-public/scripts/inference/logs/ \
  --isaaclab_headless True
```

再导出 replay 视频和截图：

```bash
cd /home/eric/MotionPlanningDiffusion/mpd-splines-public
/home/eric/IsaacLab_ori/isaaclab.sh -p scripts/isaaclab/replay_mpd_trajectory.py \
  --input scripts/inference/logs/<RUN_ID>/isaaclab-trajectories-000.pt \
  --trajectory_index 0 \
  --output_video scripts/inference/logs/<RUN_ID>/isaaclab-replay-000.mp4 \
  --screenshot_path scripts/inference/logs/<RUN_ID>/isaaclab-replay-000.png \
  --output_json scripts/inference/logs/<RUN_ID>/isaaclab-replay-000.json \
  --device cuda:0 \
  --headless \
  --enable_cameras
```

待实现的导出命令建议长这样：

```bash
python scripts/real_robot/export_trajectory.py \
  --input scripts/inference/logs/<RUN_ID>/isaaclab-trajectories-000.pt \
  --statistics scripts/inference/logs/<RUN_ID>/isaaclab-statistics-000.json \
  --trajectory_index 0 \
  --output scripts/inference/logs/<RUN_ID>/real_robot_trajectory-000.json
```

这一阶段的验收标准：

- `isaaclab-statistics-000.json` 显示目标轨迹无碰撞。
- `isaaclab-replay-000.mp4` 和 `isaaclab-replay-000.png` 视觉上没有穿模、撞障碍物或明显跳变。
- 导出的 `real_robot_trajectory-000.json` 中 `joint_names`、`q_traj`、`dt`、`source` 字段完整。
- 不连接实机，不发送任何控制命令。

## 阶段 1：离线安全检查

目标：对导出的实机轨迹做严格离线检查，仍然不连接实机。

待实现命令建议：

```bash
python scripts/real_robot/check_trajectory.py \
  --input scripts/inference/logs/<RUN_ID>/real_robot_trajectory-000.json \
  --config scripts/real_robot/configs/panda_real.yaml \
  --output scripts/inference/logs/<RUN_ID>/real_robot_check-000.json
```

`check_trajectory.py` 至少检查：

- 关节数量必须是 7，关节名必须严格匹配 `panda_joint1` 到 `panda_joint7`。
- 每个 waypoint 必须在实机 joint limit 内，并保留安全 margin。
- 相邻 waypoint 的最大关节变化不能超过阈值，例如 `max_joint_step_rad`。
- 根据 `dt` 计算速度、加速度，必须小于配置文件里的实机限制乘以安全缩放。
- `q_start` 必须接近实机准备起始位姿；第一阶段建议要求误差小于 `0.02 rad`。
- 轨迹必须已经有 IsaacLab 无碰撞统计和 replay 文件引用。
- 文件必须带 `operator_approved: false`，由人工检查后再改成 true 或通过单独审批文件记录。

`panda_real.yaml` 建议包含：

```yaml
robot_name: panda
joint_names:
  - panda_joint1
  - panda_joint2
  - panda_joint3
  - panda_joint4
  - panda_joint5
  - panda_joint6
  - panda_joint7
max_velocity_scale: 0.1
max_acceleration_scale: 0.1
max_joint_step_rad: 0.03
start_tolerance_rad: 0.02
workspace:
  require_empty_workspace_for_first_run: true
```

## 阶段 2：控制器假执行

目标：先把实机轨迹送到 fake controller 或仿真 controller，验证 ROS/MoveIt 接口、关节顺序和时间戳，不驱动真实硬件。

待实现命令建议：

```bash
python scripts/real_robot/send_follow_joint_trajectory.py \
  --input scripts/inference/logs/<RUN_ID>/real_robot_trajectory-000.json \
  --controller /joint_trajectory_controller/follow_joint_trajectory \
  --dry_run True
```

这一阶段检查：

- controller 接收的 joint order 和 JSON 中一致。
- RViz/MoveIt 显示的轨迹方向和 IsaacLab replay 一致。
- 起点、终点、末端运动方向没有镜像、反向或关节错位。
- 发送脚本默认 `dry_run=True`，没有显式 `--execute True` 时不能真正下发。

## 阶段 3：实机只读联调

目标：连接机器人，但只读取状态，不发送轨迹。

检查项：

- 急停按钮、示教器、限位保护和机器人控制柜状态正常。
- 机器人周围清空，第一轮不放任何障碍物。
- 读取实机当前关节角，确认关节顺序和单位是 rad。
- 把实机当前关节角和 `real_robot_trajectory-000.json` 的 `q_start` 对比。
- 如果起点误差超过配置阈值，不允许执行轨迹，先用官方工具或手动方式移动到准备位姿。

建议记录：

```text
real_robot_logs/<RUN_ID>/
  robot_state_before.json
  operator_checklist.md
```

## 阶段 4：实机低速空场执行

目标：第一次真正执行只在空工作区低速运行，不带障碍物，不追求任务成功，只验证轨迹方向、速度和停止机制。

执行前必须满足：

- `real_robot_check-000.json` 全部通过。
- `operator_approved` 已记录为 true。
- 机器人实际起点和轨迹起点误差在阈值内。
- 速度和加速度缩放不高于 `0.1`。
- 操作员手在急停附近，旁边没有人进入工作空间。
- 发送脚本支持超时、取消和异常停止。

待实现执行命令建议：

```bash
python scripts/real_robot/send_follow_joint_trajectory.py \
  --input scripts/inference/logs/<RUN_ID>/real_robot_trajectory-000.json \
  --controller /joint_trajectory_controller/follow_joint_trajectory \
  --speed_scale 0.1 \
  --execute True \
  --record real_robot_logs/<RUN_ID>/execution-000.json
```

执行后检查：

- 实机运动方向与 IsaacLab replay 一致。
- 没有控制器超限、protective stop、速度突变或明显抖动。
- 记录 actual joint states，与 planned q_traj 对齐保存。
- 如果出现任何异常，先停止并回到阶段 1，不继续加速或加障碍物。

## 阶段 5：带真实场景验证

目标：把仿真中的 scene payload 对齐到真实工作区，再逐步验证有障碍物的任务。

关键点：

- MPD/PyBullet/IsaacLab 的障碍物坐标系必须和真实机器人 base 坐标系一致。
- sphere/box 的尺寸、位置要来自真实测量或标定，不要只凭视觉估计。
- 第一轮真实障碍物要留更大的安全 margin，不能贴近机器人或末端。
- 每次改变障碍物位置，都要重新生成或重新验证 scene payload、IsaacLab statistics 和 replay。
- 如果真实场景不能稳定测量，先不要做自动规划到实机闭环，只做固定轨迹低速复现。

## 需要新增的代码接口

第一批建议只做三个脚本，避免直接把实机逻辑塞进 `inference.py`：

1. `scripts/real_robot/export_trajectory.py`

输入 `isaaclab-trajectories-XXX.pt` 和可选 `isaaclab-statistics-XXX.json`，输出 `real_robot_trajectory-XXX.json`。这个脚本负责选 `trajectory_index`、写 joint names、写 source 信息，不负责控制机器人。

2. `scripts/real_robot/check_trajectory.py`

读取 `real_robot_trajectory-XXX.json` 和 `panda_real.yaml`，做 joint limits、速度、加速度、waypoint step、起点、IsaacLab 验证引用检查。失败时返回非零退出码。

3. `scripts/real_robot/send_follow_joint_trajectory.py`

读取通过检查的 JSON，默认 dry-run。只有显式 `--execute True` 才发送到真实控制器。发送前再次读取当前机器人状态，确认起点误差。

后续可选接口：

- `scripts/real_robot/record_robot_state.py`：只读记录机器人状态。
- `scripts/real_robot/compare_execution.py`：比较 planned 和 actual trajectory。
- `scripts/real_robot/calibrate_scene.py`：把真实障碍物坐标写成 MPD/IsaacLab 可复用的 scene payload。

## 不建议现在做的事

- 不建议把实机控制直接写进 `scripts/inference/inference.py`。
- 不建议让训练或推理脚本直接连接机器人。
- 不建议跳过 IsaacLab replay，只凭 `results_single_plan-XXX.pt` 上实机。
- 不建议在第一版直接接底层 torque control。
- 不建议复用仿真里的 `dt` 作为唯一时间参数；实机前必须重新检查速度、加速度和控制器采样要求。
- 不建议用 Warehouse 或复杂障碍物作为第一轮实机验证场景。先用 Panda + EnvSpheres3D 或空场景把链路跑通。

## 最小验收清单

实机第一次执行前，至少满足：

- MPD 推理完成，目标轨迹存在于 `isaaclab-trajectories-XXX.pt`。
- IsaacLab evaluator 输出无碰撞统计。
- IsaacLab replay 视频和截图人工确认通过。
- `real_robot_trajectory-XXX.json` 导出完成，字段完整。
- `check_trajectory.py` 全部通过。
- fake controller 或 dry-run 通过。
- 实机只读状态检查通过，当前关节角接近轨迹起点。
- 工作区清空，急停可用，速度缩放不超过 `0.1`。
- 第一轮执行后保存 planned/actual 对比日志。

## 推荐优先级

1. 先实现 `export_trajectory.py`，把现有 `isaaclab-trajectories-XXX.pt` 稳定转成实机轨迹文件。
2. 再实现 `check_trajectory.py`，把所有实机前置安全检查自动化。
3. 接 fake controller 或 MoveIt dry-run，验证 joint order 和时间戳。
4. 做实机只读状态检查脚本。
5. 最后才实现 `--execute True` 的真实发送路径，并且默认低速空场执行。
