# 已知障碍运动的单次规划 benchmark

Suite: `d7602b4f0a9d403c801ad6e832b3f211e4454258b3a78a1b501f9e312390d43e`

世界使用相同的冻结分段线性轨迹，无预测膨胀、重规划或 handoff。耗时包含本轮规划与验证；加载预热另存 engine.json。

仅运行部分 case 的结果属于子集实验；未运行项显示 —，不算作失败。不同设备结果应分目录保存。

| 场景 | 模型 | 已完成 | 公共验证成功 | 成功率 | 平均规划秒 | 成功轨迹平均时长秒 |
|---|---|---:|---:|---:|---:|---:|
| todrawer | phase4 | 1 | 0 | 0.0000 | 1.9829 | — |
| todrawer | phase5joint | 1 | 0 | 0.0000 | 2.3170 | — |
| todrawer | phase4_aligned | 1 | 0 | 0.0000 | 2.7417 | — |
| todrawer | f1c | 1 | 0 | 0.0000 | 3.1730 | — |
| todrawer | f1taur | 1 | 0 | 0.0000 | 3.1637 | — |
| todrawer | f2c | 1 | 0 | 0.0000 | 5.1010 | — |
| todrawer | f2taur | 1 | 0 | 0.0000 | 5.2344 | — |
| todrawer | f3c | 1 | 0 | 0.0000 | 3.2853 | — |
| todrawer | f3taur | 1 | 0 | 0.0000 | 3.3953 | — |
| todrawer | jdkin_c-00091000 | 1 | 0 | 0.0000 | 7.4144 | — |
| todrawer | jdkin_c_multi-00199000 | 1 | 0 | 0.0000 | 8.3506 | — |
| todrawer | jdkin_c-00123000 | 1 | 0 | 0.0000 | 8.0007 | — |
| todrawer | jdkin_tau_r-00089000 | 1 | 0 | 0.0000 | 9.2427 | — |
| todrawer | jdkin_tau_r_multi-00196000 | 1 | 0 | 0.0000 | 9.7236 | — |
| todrawer | jdkin_tau_r-00114000 | 1 | 0 | 0.0000 | 9.1781 | — |
| todrawer | jdworld_c-00065000 | 1 | 0 | 0.0000 | 7.2988 | — |
| todrawer | jdworld_c_multi-00153000 | 1 | 0 | 0.0000 | 5.8060 | — |
| todrawer | jdworld_c_multi_from135k_z010-00215000 | 1 | 0 | 0.0000 | 8.1173 | — |
| todrawer | jdworld_tau_r_1-00061000 | 1 | 0 | 0.0000 | 8.3148 | — |
| todrawer | jdworld_tau_r_multi-00132000 | 1 | 0 | 0.0000 | 9.7165 | — |
| todrawer | jdworld_tau_r_1-00082000 | 1 | 0 | 0.0000 | 5.7290 | — |
| warehouse | phase4 | 1 | 0 | 0.0000 | 2.0928 | — |
| warehouse | phase5joint | 1 | 0 | 0.0000 | 2.6714 | — |
| warehouse | phase4_aligned | 1 | 0 | 0.0000 | 2.7768 | — |
| warehouse | f1c | 1 | 0 | 0.0000 | 3.1514 | — |
| warehouse | f1taur | 1 | 0 | 0.0000 | 3.2696 | — |
| warehouse | f2c | 1 | 0 | 0.0000 | 5.2636 | — |
| warehouse | f2taur | 1 | 0 | 0.0000 | 5.1695 | — |
| warehouse | f3c | 1 | 0 | 0.0000 | 3.3525 | — |
| warehouse | f3taur | 1 | 0 | 0.0000 | 3.4585 | — |
| warehouse | jdkin_c-00091000 | 1 | 0 | 0.0000 | 7.1813 | — |
| warehouse | jdkin_c_multi-00199000 | 1 | 0 | 0.0000 | 7.4320 | — |
| warehouse | jdkin_c-00123000 | 1 | 0 | 0.0000 | 7.5331 | — |
| warehouse | jdkin_tau_r-00089000 | 1 | 0 | 0.0000 | 8.1072 | — |
| warehouse | jdkin_tau_r_multi-00196000 | 1 | 0 | 0.0000 | 7.8183 | — |
| warehouse | jdkin_tau_r-00114000 | 1 | 0 | 0.0000 | 7.8606 | — |
| warehouse | jdworld_c-00065000 | 1 | 0 | 0.0000 | 7.2772 | — |
| warehouse | jdworld_c_multi-00153000 | 1 | 0 | 0.0000 | 7.9994 | — |
| warehouse | jdworld_c_multi_from135k_z010-00215000 | 1 | 0 | 0.0000 | 7.8574 | — |
| warehouse | jdworld_tau_r_1-00061000 | 1 | 0 | 0.0000 | 7.6574 | — |
| warehouse | jdworld_tau_r_multi-00132000 | 1 | 0 | 0.0000 | 8.3214 | — |
| warehouse | jdworld_tau_r_1-00082000 | 1 | 0 | 0.0000 | 8.0114 | — |

各自成功样本的平均时长仅作描述，不能单独作为质量排名；完整失败分类及障碍数/到达模式分组见 report.json。
