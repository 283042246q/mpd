# 两个约束环境的 IsaacLab 使用说明

`EnvOpenDrawerShelf` 和 `EnvThreePillarsPassage` 已接入 inference 的 IsaacLab
批量评估与 replay 链路。迁移保持 MPD 环境中的碰撞几何不变：

- `EnvOpenDrawerShelf` 导出 20 个 box：柜体 8 个、打开的底层抽屉 5 个、相邻书架 7 个。
- `EnvThreePillarsPassage` 导出 3 个落地长柱 box。
- box 的中心、尺寸和姿态由 `MultiBoxField` 直接写入 scene payload；IsaacLab
  evaluator/replay 使用 `sim_utils.CuboidCfg` 重建相同的静态碰撞体。
- Panda 使用 IsaacLab 的 `FRANKA_PANDA_HIGH_PD_CFG`，MPD 的 7 个手臂关节按
  `panda_joint.*` 映射，两个手指关节保持在 `0.04 m`。

## 从 inference 自动评估

以下命令均从仓库的 `scripts/inference` 目录执行。只做批量接触统计时使用
`--isaaclab_replay False`；删除该参数或设为 `True` 会继续导出视频、截图和 replay JSON。

抽屉外移动到底层抽屉一角：

```bash
conda run -n mpd-splines-public-cu128 python inference.py \
  --cfg_inference_path ./cfgs/config_EnvOpenDrawerShelf-RobotPanda-regions-to-drawer.yaml \
  --sim_backend isaaclab \
  --device cpu \
  --isaaclab_root /home/eric/IsaacLab_ori \
  --isaaclab_conda_env env_isaaclab_ori \
  --isaaclab_device cuda:0 \
  --isaaclab_headless True \
  --isaaclab_replay False
```

从底层抽屉一角移动到相邻书架：

```bash
conda run -n mpd-splines-public-cu128 python inference.py \
  --cfg_inference_path ./cfgs/config_EnvOpenDrawerShelf-RobotPanda-regions-drawer-to-shelf.yaml \
  --sim_backend isaaclab \
  --device cpu \
  --isaaclab_root /home/eric/IsaacLab_ori \
  --isaaclab_conda_env env_isaaclab_ori \
  --isaaclab_device cuda:0 \
  --isaaclab_headless True \
  --isaaclab_replay False
```

穿过三根长柱：

```bash
conda run -n mpd-splines-public-cu128 python inference.py \
  --cfg_inference_path ./cfgs/config_EnvThreePillarsPassage-RobotPanda-regions.yaml \
  --sim_backend isaaclab \
  --device cpu \
  --isaaclab_root /home/eric/IsaacLab_ori \
  --isaaclab_conda_env env_isaaclab_ori \
  --isaaclab_device cuda:0 \
  --isaaclab_headless True \
  --isaaclab_replay False
```

inference 会在每个结果目录中生成：

- `isaaclab-trajectories-XXX.pt`：轨迹、环境名和完整 scene payload。
- `isaaclab-statistics-XXX.json`：接触率、首次接触 waypoint、障碍物数量和类型。
- `isaaclab-evaluator-XXX.log`：IsaacLab 子进程日志。
- 启用 replay 时还会生成 `isaaclab-replay-XXX.mp4/.png/.json`。

## 自动检查

不启动 Omniverse 的 payload 回归测试：

```bash
conda run -n mpd-splines-public-cu128 \
  python -m pytest -q tests/test_isaaclab_constrained_scene_payloads.py
```

测试会检查两个环境已进入 IsaacLab 白名单、所有 box 均被导出、中心和尺寸与
MPD 环境常量逐项一致、四元数采用 IsaacLab 的 `wxyz` 顺序且不存在 unsupported
obstacle。若环境没有安装 `pytest`，可先运行 Python 编译检查；不要为了测试修改
记录环境或令牌的文件。

## 当前机器的运行状态

2026-07-30 的静态导出和跨 conda payload 加载检查均已通过。真实 IsaacLab
headless 冒烟测试在创建 `SimulationContext` 前被本机 CUDA 驱动错误 804
（forward compatibility was attempted on non supported hardware）阻断，因此当前结果
不能视为 PhysX 接触验证通过。修复 NVIDIA 驱动后，直接重新运行上面的 inference
命令即可继续，不需要重新生成环境代码。
