# Marvin 双 Pika：MPD 资产与一致性门禁

## 1. 本次落地范围

`RobotMarvinBimanual()` 现在默认加载带双 Pika 的完整几何，规划维数仍为 14：

```text
[Joint1_L ... Joint7_L, Joint1_R ... Joint7_R]
```

左右各包含 Marvin→Pika adaptor、gripper body、两根 FinRay 手指及 TCP。
视觉与 URDF mesh collision 来自当前 ROS 组合 Xacro。四个 prismatic joint
（包含两个 mimic joint）在导出时固定，默认单指行程为 `0.045 m`，不是总开口宽度。
正确的固定方式是将 `R_origin × axis × q` 烘焙进 joint origin，再移除
axis/limit/mimic；不是只把 joint type 改成 fixed。

MPD 的可微碰撞模型为原机械臂碰撞球加双 Pika 保守包络。手指包络包含
从闭合到全开的**连续完整行程**，与固定的视觉姿态不同。它不要求把夹爪开度
加入 diffusion 训练，也不跟随每次夹爪状态变化。

默认末端改为 `left_pika_gripper_tcp`、`right_pika_gripper_tcp`。
要复现原 arm-only 模型，请显式使用：

```python
robot = RobotMarvinBimanual(with_pika=False, tensor_args=tensor_args)
# 末端仍为 flange_L / flange_R，原 URDF、meshes、碰撞配置未覆盖。
```

Franka/Panda 资产、单臂训练/推理入口未改动。碰撞 Jacobian 优化器只为实现了
关节顺序适配接口的模型启用转换，Panda 原路径不变。

## 2. 文件放在哪里

以下路径均相对于本 MPD 仓库；`D` 表示 `mpd/torch_robotics/torch_robotics/data`。

| 路径 | 用途 |
| --- | --- |
| `D/urdf/robots/marvin/marvin_pika_bimanual_mpd.urdf` | 14 维、关闭硬件标签、本地 mesh 路径的运行模型 |
| `D/urdf/robots/marvin/meshes/marvin_description/` | 新组合模型引用的 Marvin mesh 副本 |
| `D/urdf/robots/marvin/meshes/pika_gripper_description/` | 实际引用的 Pika body、FinRay、Marvin adaptor mesh |
| `D/configs/marvin/pika/collision_spheres.yaml` | 双 Pika + 原机械臂细碰撞球和自碰撞关系 |
| `D/configs/marvin/pika/collision_parent_bounds.yaml` | 保守粗筛包围球，包含所有细球 |
| `D/configs/marvin/pika/self_collision_pairs.yaml` | 与可微模型一致的 PyBullet link pair 配置 |
| `D/configs/marvin/pika/joint_limits.yaml` | canonical 14 关节限制 |
| `D/urdf/robots/marvin/pika_assets.lock.yaml` | 来源版本、文件 SHA-256、冻结姿态、包络、校准标志 |
| `D/urdf/robots/marvin/sources/` | 三个 ROS 包的描述/配置/已有授权声明快照 |
| `D/urdf/robots/marvin/licenses/` | 授权说明、Apache 文本、Pika 上游 BSD 文本 |
| `scripts/robots/build_marvin_pika_assets.py` | 可重现生成器和只读 `--check` |
| `tests/test_marvin_pika_assets.py` | 离线与可选 ROS 实时一致性测试 |

只迁移被运行模型实际引用的 mesh，不搬 ROS interface、驱动、libmarvin 或整个
ROS 工作空间。新模型的 mesh 使用包名隔离，原 arm-only 副本保持不变。
`sources/` 是审计快照，不是可安装的 ROS 包；其中展开后的 ROS URDF 仅作数值
比对参考，其本地 mesh 路径以 `marvin/` 为基准，不能作为独立运行入口。

## 3. 碰撞策略与限制

当前共有 26 个有碰撞球的父连杆、1035 个细球，其中双 Pika 新增 642 个。
每个 Pika collision mesh 的局部 AABB 用边长不超过 `0.025 m` 的格子分割，
每格以外接球覆盖整个体积。手指 AABB 同时包含完整行程。粗筛为每父连杆一个
严格包住所有细球的球，当前共 26 个。

原 Marvin/CuRobo 球和 pair 保留。每个 Pika 部件都检查对侧机械臂、对侧 Pika、
底座和立柱，以及同侧除 `Link7` 外的机械臂连杆。
同侧夹爪内部刚性/行程包络交叠以及连接处 `Link7` 不作自碰撞对。
**没有整体屏蔽左右夹爪间碰撞。**

这是偏保守的初始模型，会封住部分真实空隙，可能拒绝窄缝抓取；细球级自碰撞
组合也增至约 37.5 万对。已验证几何覆盖和运动学，但尚未做生产规模的 GPU
吞吐/规划成功率评估。后续若精简球或改为分姿态包络，必须保持覆盖测试通过。
本次不增加被抓物碰撞体、柔性手指变形、抓取接触例外或力控策略。

数据生成的 warehouse 双臂入口已跟随机器人实例选择右 TCP 和 self-collision
pair 文件。新生成数据使用新几何；旧数据/checkpoint 不会自动转成双 Pika
有效轨迹。应重新生成/验证数据，并审核抓取 profile：`object_to_left/right`
现在必须相对于 Pika TCP，不能直接沿用未经转换的 flange 坐标变换。

## 4. 自动检查命令

### 4.1 MPD 离线完整门禁

使用已经安装本仓库依赖的 MPD 环境，不需要 ROS 运行，更不会连接硬件：

```bash
cd /home/eric/Projects/MotionPlanningDiffusion/mpd
conda run -n mpd-splines-public python -m pytest -q \
  tests/test_marvin_model_contract.py \
  tests/test_marvin_pika_assets.py \
  tests/test_parent_link_sphere_kinematics.py
```

未设置 ROS 路径时仅跳过 live ROS 测试，其余检查本地已归档源与运行资产。
测试覆盖哈希/授权文件、合法树和 14 关节、冻结变换、全行程 mesh 包络、
粗球包含细球、跨臂 pair、100 组合法配置的独立 NumPy ROS FK 与 TorchKin TCP
比对、canonical Jacobian 有限差分、部分连杆/缓存路径、Pika 障碍覆盖、
零位自碰撞、旧模型兼容与 PyBullet DIRECT 加载。

`.github/workflows/marvin-assets.yml` 提供轻量 PR/push 门禁，只需 NumPy、PyYAML、
pytest，执行纯资产部分；不会把本地路径或 ROS 工作空间作为 CI 前提。
它不代替上面的完整 MPD 测试或下面的实时源检查。

### 4.2 当前 ROS 源码与 MPD 实时比对

```bash
cd /home/eric/Projects/MotionPlanningDiffusion/mpd
MARVIN_ROS_ROOT=/home/eric/Projects/physical_ai_runtime \
  conda run -n mpd-splines-public python -m pytest -q \
  tests/test_marvin_model_contract.py \
  tests/test_marvin_pika_assets.py \
  tests/test_parent_link_sphere_kinematics.py
```

live 测试会通过 ROS 工作空间的 `pixi` 环境执行生成器 `--check`。也可单独运行：

```bash
cd /home/eric/Projects/physical_ai_runtime
pixi run bash -c 'cd /home/eric/Projects/MotionPlanningDiffusion/mpd && python -m scripts.robots.build_marvin_pika_assets --ros-root /home/eric/Projects/physical_ai_runtime --check'
```

生成器用临时 ament 索引指向指定工作空间的三个源码包，避免检查到旧 install
overlay；关闭 `ros2_control`。`--check` 在内存重新展开与生成、逐文件比较，
任何受管理文件缺失/变化都会以非零状态退出，不写 ROS/MPD 资产。

### 4.3 ROS 资产/挂载标定改变后重新导入

先审核 ROS 源码变动，再去掉 `--check` 重新生成：

```bash
cd /home/eric/Projects/physical_ai_runtime
pixi run bash -c 'cd /home/eric/Projects/MotionPlanningDiffusion/mpd && python -m scripts.robots.build_marvin_pika_assets --ros-root /home/eric/Projects/physical_ai_runtime'
```

之后再次执行 4.2 的完整门禁，检查 `git diff`，将新 URDF、mesh、配置、源快照和
lock 文件一起提交。不要只手改 MPD URDF 或只更新 lock 哈希；也不要用旧的
arm-only 导出入口生成含可动 Pika 的模型。生成器不删除旧文件；源码删去的资产
若留在磁盘上只是未引用副本，需人工审核后再决定是否清理。

默认测试锁定 `0.045 m` 张开视觉姿态；若使用 `--finger-travel` 改为其他值，应
明确更新姿态契约测试与文档。全行程碰撞包络仍覆盖源 joint limits 的完整区间。

## 5. TCP 与授权：仍需人工确认的边界

ROS 的 `gripper_tcp.yaml` 仍标记为占位值，当前 TCP 从 gripper base 沿 z 偏移
`0.21 m`；本次原样同步，lock 和机器人实例的 `tcp_calibrated` 都是 `false`。
数值一致不等于实机正确。实测挂载/TCP 后先更新 ROS，再重新导出；确认标定流程
后还需要显式更新生成器的校准元数据及对应门禁，不能仅将布尔标志改成 true。

授权来源详见
[`licenses/SOURCES.md`](../mpd/torch_robotics/torch_robotics/data/urdf/robots/marvin/licenses/SOURCES.md)。
保留了 Marvin 原 `LICENSE`、各 package.xml 和 Pika `MESH_SOURCES.md`；新增
Apache 标准文本及 Pika 上游根 BSD 文本，不修改 mesh 原始字节。

当前有两类尚未澄清的来源边界：Marvin 的 package.xml（Apache-2.0）与根
LICENSE（MIT 风格文本）不一致；Pika package-local FinRay/adaptor 没有独立明确的
逐 mesh 权利人/授权说明。Pika 上游 BSD 文本的署名是原文的 Tixiao Shan，
不能据此推断所有 Pika mesh 的所有权。迁移这些声明不等于完成对外再分发许可
审核；对外发布前应向上游确认，不能统一覆盖成 MPD 的 MIT。
