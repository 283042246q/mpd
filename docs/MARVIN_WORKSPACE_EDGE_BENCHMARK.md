# Marvin 边缘工作区加密测试

入口：`scripts/inference/benchmark_marvin_workspace_edges.py`。

根据 dense-v3 的分布，在左右书架下层 x/y 边界、上层 x/y 边界、跨区桌面 x/y 边界设置12个窄盒子。各盒子200个独立随机TCP目标；固定每个目标的位姿及另一臂参考状态，最多20次IK重启。第一个重启采用参考关节状态，后续采用关节范围内独立均匀初值。位姿容差3mm/2°，每次优化最多100次函数评估。只要达到位姿容差就记IK成功；继续尝试直到完整碰撞检查通过或重启耗尽。达到位姿容差但略超出采样盒子的FK位置仍保留并记录，避免边界筛选混淆IK求解结果。

目标位置在盒子内均匀采样，旋转沿用源regions快照中对应桌面/书架的配置。左右姿态约束沿用原配置，因此这不是严格镜像对照实验。

起点来自dense-v3对应机械臂的同侧桌面有效完整14D状态；重新检查后缓存。IK固定另一臂在抽到的参考状态。RRT的其他起点从同一已验证池抽取，求解完整14D双臂路径，因此允许另一臂协调运动。不同区域成功率条件于这个有限起点池，不能直接与旧的双臂同时进入书架任务比较。

每区最多100个不同pair。先为每个不同的有效目标分配一个起点，再轮流增加起点，每目标最多5个。记录独立起点、目标数量；有效目标不足时记录缺额。每个RRT使用10秒OMPL solve预算，沿用完整网格和球碰撞检测、0.002 validity resolution、0.025 rad最大复检步长及512点样条复检。原始路径复检和样条复检耗时另计，所以一次测试墙钟时间可超过10秒。分别记录OMPL exact、原始路径通过复检、样条通过复检。这里不执行动态限制验证或训练集写入。

```bash
conda run --no-capture-output -n mpd-splines-public \
python scripts/inference/benchmark_marvin_workspace_edges.py \
  --output-dir scripts/inference/logs/marvin-workspace-edges-2400 \
  --targets-per-cell 200 --ik-restarts 20 \
  --pairs-per-cell 100 --starts-per-goal 5 --workers 1
```

同命令恢复：已保存的目标、参考状态、完成IK结果及pairs保持不变；只运行缺失结果。每区独立进程，非正常退出最多恢复3次，返回码保存在process-status.json。未完成的区域不会算为完成。正在进行且因进程中止未落盘的随机IK/RRT尝试可在恢复时重新求解；这不是保证native随机数流逐位复现的机制。

按最新要求，RRT严格串行，`--workers`只接受1。跨进程文件锁同时约束单独的`--cell`调用。每次恢复记录到`resume-events/`，新RRT结果包含起止时间戳和slot，便于核对并行度。此前已完成结果保留，因此其历史耗时不能当作本轮串行耗时。异常断电/中止后如有空文件，确认没有活动worker后用`--recover`隔离损坏文件并补测；原文件保存在`recovery/`。JSON采用文件及目录fsync与原子替换。

实时汇总：

```bash
conda run --no-capture-output -n mpd-splines-public \
python scripts/inference/benchmark_marvin_workspace_edges.py \
  --output-dir scripts/inference/logs/marvin-workspace-edges-2400 --summarize-only
```

输出包含settings/source快照、每个目标的位姿与参考索引、逐目标IK结果、完整pair、逐pair轨迹NPZ、耗时和汇总JSON/CSV/Markdown。生成中的缺额是临时值，须结合complete读取。
