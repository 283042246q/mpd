# 已知障碍运动的单次规划 benchmark

Suite: `2edb29b6212c648092d942353d76fc2dfdbcc10978667a02ecd011a5367b97ae`

世界使用相同的冻结分段线性轨迹，无预测膨胀、重规划或 handoff。耗时包含本轮规划与验证；加载预热另存 engine.json。

仅运行部分 case 的结果属于子集实验；未运行项显示 —，不算作失败。不同设备结果应分目录保存。

| 场景 | 模型 | 已完成 | 公共验证成功 | 成功率 | 平均规划秒 | 成功轨迹平均时长秒 |
|---|---|---:|---:|---:|---:|---:|
| todrawer | phase4 | 0 | 0 | — | — | — |
| todrawer | phase5joint | 0 | 0 | — | — | — |
| todrawer | phase4_aligned | 0 | 0 | — | — | — |
| todrawer | f1c | 0 | 0 | — | — | — |
| todrawer | f1taur | 0 | 0 | — | — | — |
| todrawer | f2c | 0 | 0 | — | — | — |
| todrawer | f2taur | 0 | 0 | — | — | — |
| todrawer | f3c | 0 | 0 | — | — | — |
| todrawer | f3taur | 0 | 0 | — | — | — |
| todrawer | jdkin_c-00091000 | 0 | 0 | — | — | — |
| todrawer | jdkin_c_multi-00199000 | 0 | 0 | — | — | — |
| todrawer | jdkin_c-00123000 | 0 | 0 | — | — | — |
| todrawer | jdkin_tau_r-00089000 | 0 | 0 | — | — | — |
| todrawer | jdkin_tau_r_multi-00196000 | 0 | 0 | — | — | — |
| todrawer | jdkin_tau_r-00114000 | 0 | 0 | — | — | — |
| todrawer | jdworld_c-00065000 | 0 | 0 | — | — | — |
| todrawer | jdworld_c_multi-00153000 | 0 | 0 | — | — | — |
| todrawer | jdworld_c_multi_from135k_z010-00215000 | 0 | 0 | — | — | — |
| todrawer | jdworld_tau_r_1-00061000 | 0 | 0 | — | — | — |
| todrawer | jdworld_tau_r_multi-00132000 | 0 | 0 | — | — | — |
| todrawer | jdworld_tau_r_1-00082000 | 0 | 0 | — | — | — |
| warehouse | phase4 | 0 | 0 | — | — | — |
| warehouse | phase5joint | 0 | 0 | — | — | — |
| warehouse | phase4_aligned | 0 | 0 | — | — | — |
| warehouse | f1c | 0 | 0 | — | — | — |
| warehouse | f1taur | 0 | 0 | — | — | — |
| warehouse | f2c | 0 | 0 | — | — | — |
| warehouse | f2taur | 0 | 0 | — | — | — |
| warehouse | f3c | 0 | 0 | — | — | — |
| warehouse | f3taur | 0 | 0 | — | — | — |
| warehouse | jdkin_c-00091000 | 0 | 0 | — | — | — |
| warehouse | jdkin_c_multi-00199000 | 0 | 0 | — | — | — |
| warehouse | jdkin_c-00123000 | 0 | 0 | — | — | — |
| warehouse | jdkin_tau_r-00089000 | 0 | 0 | — | — | — |
| warehouse | jdkin_tau_r_multi-00196000 | 0 | 0 | — | — | — |
| warehouse | jdkin_tau_r-00114000 | 0 | 0 | — | — | — |
| warehouse | jdworld_c-00065000 | 0 | 0 | — | — | — |
| warehouse | jdworld_c_multi-00153000 | 0 | 0 | — | — | — |
| warehouse | jdworld_c_multi_from135k_z010-00215000 | 0 | 0 | — | — | — |
| warehouse | jdworld_tau_r_1-00061000 | 0 | 0 | — | — | — |
| warehouse | jdworld_tau_r_multi-00132000 | 0 | 0 | — | — | — |
| warehouse | jdworld_tau_r_1-00082000 | 0 | 0 | — | — | — |

各自成功样本的平均时长仅作描述，不能单独作为质量排名；完整失败分类及障碍数/到达模式分组见 report.json。
