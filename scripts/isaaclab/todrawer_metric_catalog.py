"""Field-level provenance and definitions for the ToDrawer benchmark report."""

from __future__ import annotations


METRIC_CATALOG: dict[str, dict[str, str]] = {}


def _add(group: str, source: str, definitions: dict[str, str]) -> None:
    for field, meaning in definitions.items():
        if field in METRIC_CATALOG:
            raise ValueError(f"duplicate ToDrawer metric definition: {field}")
        METRIC_CATALOG[field] = {"group": group, "source": source, "meaning": meaning}


_add("实验身份", "run-spec.json", {
    "scenario_id": "冻结场景 ID；同一 ID 和 repeat 是配对比较单元。",
    "environment_index": "每个场景类型内部的环境编号。",
    "environment_seed": "生成该冻结环境几何与时间参数的随机种子。",
    "category": "场景类型。",
    "difficulty": "场景类型对应的预设难度级别。",
    "repeat": "同一场景的规划重复编号，从 0 开始。",
    "planner_repeat": "规划重复编号，与 repeat 相同，保留作显式实验身份。",
    "mode": "被比较的算法或消融模式，例如 joint、strrt。",
    "phase": "pipeline 选择的实现分支，例如 phase5、strrt。",
    "timing_mode": "MPD Phase-5 的时间模式；ST-RRT* 不适用。",
    "scenario_file": "本次运行读取的冻结动态场景 JSON 路径。",
    "timing_protocol": "障碍物穿越时钟协议；absolute_world_time 在模式间共享相同场景文件。",
    "planner_seed": "配对算法共享的运行种子；ST-RRT* 在 worker 启动时固定 OMPL 种子。",
})

_add("算法配置", "run-spec.json", {
    "strrt_solve_budget_s": "ST-RRT* 单次 OMPL solve 的时间预算，单位秒。",
    "strrt_range": "ST-RRT* 空间采样树的 range 参数。",
    "strrt_edge_dt_s": "ST-RRT* 连续边碰撞复核的最大时间采样间隔，单位秒。",
    "corridor_a_enabled": "当前模式是否启用 Corridor A。",
    "corridor_a_backend": "Corridor A 使用的求解后端。",
    "corridor_a_chunk_size": "Corridor A 批处理候选的块大小。",
    "corridor_a_dp_init": "Corridor A 是否使用 DP 到达时刻初始化。",
    "corridor_a_k_best": "Corridor A 的不同窗口分支预算。",
    "corridor_a_branch_fallback": "Corridor A 是否检查备用窗口分支。",
    "corridor_a_dense_branch_budget": "Corridor A 额外 DenseCheck 分支预算。",
    "corridor_a_selective_k": "Corridor A 是否只对可修复冲突扩展 K。",
    "corridor_a_early_stop": "Corridor A 是否在精确验证后提前停止稳定分支。",
    "factorized_method": "Factorized 时间规划方法 f1/f2/f3。",
    "factorized_representation": "Factorized 时间表示 c 或 tau_r。",
    "factorized_timing_checkpoint": "Factorized 时间模型 checkpoint 路径。",
})

_add("运行与产物", "pipeline 退出码 / episode/replay-manifest.json", {
    "pipeline_returncode": "单次 pipeline 进程退出码；0 表示脚本成功结束。",
    "pipeline_completed": "退出码为 0 且存在 replay manifest。",
    "manifest_available": "是否存在可读取的 replay manifest。",
    "failure_class": "未完成运行的故障分类，例如 DDS 启动或命令连续性问题。",
    "pipeline_revalidated": "历史运行经现有 manifest 重新检查后被判定完成。",
    "attempt_count": "同一场景、repeat、模式已保存的尝试次数。",
    "infrastructure_failure_attempts": "该单元历史尝试中基础设施失败的次数。",
    "attempt_dir": "本次运行的 artifact 目录。",
    "error": "pipeline 未完成时保留的错误摘要。",
})

_add("规划状态", "planner-results/request-*/response.json 和 result.json", {
    "first_plan_completed_from_world_s": "最早写出的 planner result 时间减场景世界时钟起点；失败 result 也计入。",
    "first_plan_status": "最早写出的 planner result 的 status。",
    "planner_result_status_counts": "各 planner result status 的请求次数，由 response 诊断汇总。",
    "online_acceptance_status_counts": "各在线响应状态的请求次数，包括 OK、PLAN_FAILED、STALE。",
    "deadline_expired_after_planning_count": "规划完成但结果写盘后超过 deadline 而被拒绝的次数。",
    "server_round_trip_mean_s": "有 response 诊断的请求，从进入 worker 到响应记录的平均耗时。",
    "planner_name": "成功结果声明的规划器名称；全部失败时为空。",
    "ik_mean_s": "成功结果的请求内 IK 平均耗时；不含 worker 启动时预计算。",
    "precheck_mean_s": "成功结果在 OMPL solve 前除 IK 外的平均准备耗时。",
    "solver_mean_s": "成功结果的 OMPL solve 平均耗时；不含失败请求。",
    "postprocess_mean_s": "成功结果从原始时空路径到复核轨迹的平均后处理耗时。",
    "postprocess_invalid_count": "失败 result 的错误消息包含 postprocess_invalid 的次数。",
    "inference_total_mean_s": "成功 result 中 inference_total_sec 的均值；不含 IPC 和产物写盘。",
    "inference_total_p95_s": "成功 result 中 inference_total_sec 的第 95 百分位。",
})

_add("世界时钟与完成", "ros-replan.log / to_drawer-replan-timing.json / manifest", {
    "goal_reached": "ROS 日志出现目标到达事件；表示本次执行到达目标。",
    "goal_time_s": "目标到达日志时间减 replanner 启动日志时间；起点是节点启动，不是世界时钟。",
    "world_start_unix_s": "场景动态世界时钟的 Unix 秒时间戳。",
    "first_planning_submit_from_world_s": "首次规划提交相对世界时钟起点的秒数。",
    "first_significant_motion_from_world_s": "manifest 中关节偏离起点超过 0.01 rad 的最早世界时钟秒数。",
    "first_motion_before_crossing": "首次明显运动加 1.25 秒不晚于最早预定穿越时刻。",
    "valid_dynamic_success": "目标到达且首次运动满足穿越前条件，且没有 brake 事件。",
    "first_command_start_from_world_s": "第一条实际 JTC 命令起点相对世界时钟的秒数。",
    "first_bridge_start_from_world_s": "第一段拼接桥开始相对世界时钟的秒数。",
    "first_handoff_from_world_s": "首次新规划轨迹 handoff 相对世界时钟的秒数。",
    "initial_world_warmup_observations": "首次规划前累计的动态世界观测数。",
    "initial_world_warmup_age_s": "首次规划前动态目标 track 的年龄，单位秒。",
})

_add("场景安全边界", "scenarios/*.json 中的 objects", {
    "scheduled_crossing_time_min_s": "本场景最早物体预定穿越时刻，相对世界时钟。",
    "scheduled_crossing_time_max_s": "本场景最晚物体预定穿越时刻，相对世界时钟。",
    "minimum_static_environment_clearance_m": "动态物体轨迹对静态环境的最小几何间距，仅作场景诊断。",
    "minimum_static_interaction_clearance_m": "动态物体轨迹对静态交互区域的最小几何间距，仅作场景诊断。",
})

_add("执行安全", "ros-replan.log / episode/replay-manifest.json / to_drawer-replan-timing.json", {
    "brake_count": "manifest 中 type=brake 的安全制动事件次数。",
    "guard_dynamic_collision_rejections": "ROS 日志 JSON reason=dynamic_collision 的次数；可能包含非候选诊断。",
    "candidate_rejection_reasons": "ROS 日志中所有 JSON reason 字段的值到次数映射；可能包含非候选诊断。",
    "accepted_nonpositive_clearance_count": "生效计划的 hard_minimum_clearance_m 不大于 0 的次数。",
    "hard_minimum_clearance_m": "生效候选在 ROS 最新动态世界 guard 硬检查中的最小预测间距；不是静态/自碰撞间距。",
    "common_window_minimum_clearance_m": "候选在共同比较时间窗中的最小动态预测间距。",
    "clearance_mean_cost": "生效候选动态风险软惩罚的均值，再按本次运行的候选求均值。",
    "clearance_cvar_cost": "生效候选最危险时间采样尾部的软惩罚均值，再按候选求均值。",
    "maximum_uncovered_command_gap_s": "实际 JTC 命令之间未被任何 goal 覆盖的最长间隙。",
    "maximum_command_gap_s": "时序摘要中的最长命令间隙，保留兼容字段。",
    "guarded_terminal_hold_s": "被明确命令并经过动态 guard 检查的末端保持总时长。",
    "maximum_controller_reference_jump_rad": "切换 JTC goal 时新首点与旧控制参考的最大关节位置跳变。",
    "jtc_error_count": "ROS 日志中 JTC 动态计划错误事件次数。",
    "no_valid_trajectory_count": "ROS 日志中 NoValidTrajectoryError 字符串出现次数；与 worker 状态计数口径不同。",
})

_add("轨迹与执行", "episode/replay-manifest.json / episode/plans/*/trajectory.npz", {
    "episode_duration_s": "manifest 记录的世界 episode 时长。",
    "execution_duration_s": "所有生效计划 active_until_s - active_from_s 的总和。",
    "executed_plan_count": "同时具有 active_from_s 和 active_until_s 的计划数。",
    "plan_record_count": "manifest 中的全部计划记录数，包含未生效计划。",
    "joint_l2_path_rad": "生效命令时间片内逐采样关节增量的欧氏范数之和。",
    "joint_l1_travel_rad": "生效命令时间片内逐关节绝对增量之和。",
    "planned_duration_mean_s": "生效计划规划后缀时长的均值；共享 manifest 字段名为 mpd_suffix_s。",
})

_add("MPD DenseCheck", "成功 planner result.json", {
    "dense_environment_clearance_m": "成功 MPD 结果中环境 DenseCheck 间距的最小值；ST-RRT* 不生成此字段。",
    "dense_self_clearance_m": "成功 MPD 结果中自碰撞 DenseCheck 间距的最小值；ST-RRT* 不生成此字段。",
})

_add("Corridor A", "成功 planner result.json 的 space_time_guidance.corridor_a", {
    "corridor_a_refinement_mean_s": "匹配当前模式身份的 Corridor A 总细化耗时均值。",
    "corridor_a_grid_build_mean_s": "Corridor A 时间网格构建耗时均值。",
    "corridor_a_branch_selection_mean_s": "Corridor A 窗口分支选择耗时均值。",
    "corridor_a_refinement_forward_mean_s": "Corridor A 前向细化耗时均值。",
    "corridor_a_refinement_backward_mean_s": "Corridor A 后向细化耗时均值。",
    "corridor_a_final_validation_mean_s": "Corridor A 最终验证耗时均值。",
    "corridor_a_profiled_total_mean_s": "Corridor A 分段计时总和均值。",
    "corridor_a_changed_candidates_mean": "每次成功规划中被 Corridor A 改变的候选数均值。",
    "corridor_a_dense_fallbacks_mean": "每次成功规划中 DenseCheck 回退次数均值。",
    "corridor_a_optimized_unique_window_sequences_mean": "优化后不同窗口序列数均值。",
    "corridor_a_validated_unique_window_sequences_mean": "通过 DenseCheck 的不同窗口序列数均值。",
    "corridor_a_alternate_branch_rescued_candidates_mean": "备用分支救回的候选数均值。",
    "corridor_a_dp_fit_abs_error_s_mean": "DP 初始化时间拟合绝对误差均值，单位秒。",
    "corridor_a_alternative_branches_checked_mean": "额外检查的备用分支数均值。",
    "corridor_a_time_invariant_skipped_mean": "被判定时间调整无法修复而跳过的分支数均值。",
    "corridor_a_early_stopped_branches_mean": "满足稳定条件并提前停止的分支数均值。",
    "corridor_a_payload_match_count": "匹配 run-spec 模式身份的 Corridor A payload 数。",
    "corridor_a_payload_mismatch_count": "有 Corridor A payload 但模式身份不匹配的数量。",
})

_add("Factorized", "成功 planner result.json 的 factorized / denoiser_evaluations", {
    "factorized_timing_checkpoint_step": "成功结果中的时间模型 checkpoint step。",
    "factorized_timing_checkpoint_sha256": "成功结果中的时间模型 checkpoint SHA256。",
    "factorized_spatial_basis_adapted": "成功结果是否启用了空间基适配。",
    "factorized_space_nfe_mean": "成功 Factorized 结果的空间 denoiser 调用次数均值。",
    "factorized_timing_nfe_mean": "成功 Factorized 结果的时间 denoiser 调用次数均值。",
})

_add("Phase 5 梯度", "成功 planner result.json 的 space_time_guidance", {
    "spatial_dynamic_grad_cap": "动态空间梯度裁剪阈值的成功结果均值。",
    "spatial_clip_ratio": "空间梯度被裁剪比例的成功步骤均值。",
    "timing_clip_ratio": "时间梯度被裁剪比例的成功步骤均值。",
    "spatial_gradient_norm_mean": "空间梯度范数的成功步骤均值。",
    "static_dynamic_gradient_cosine_mean": "静态与动态非零梯度夹角余弦的成功步骤均值。",
    "static_dynamic_gradient_conflict_ratio": "上述余弦为负的步骤比例均值。",
    "static_dynamic_gradient_cosine_valid_ratio": "同时存在非零静态和动态梯度的步骤比例均值。",
})


_SOURCE_OVERRIDES = {
    "factorized_representation": "成功 result.json 的 factorized；无结果时为空",
    "pipeline_returncode": "pipeline 进程退出码",
    "pipeline_completed": "pipeline 退出码 + episode/replay-manifest.json",
    "manifest_available": "episode/replay-manifest.json",
    "failure_class": "pipeline.log / ros-replan.log / manifest 复核",
    "pipeline_revalidated": "episode/replay-manifest.json 的时序复核",
    "attempt_count": "runs/**/run-metrics.json 的同单元历史",
    "infrastructure_failure_attempts": "runs/**/run-metrics.json 的同单元历史",
    "attempt_dir": "benchmark 创建的 attempt 目录",
    "error": "pipeline/manifest 检查结果",
    "first_plan_completed_from_world_s": "首个 result.json.created_unix_time - ROS 世界时钟起点",
    "first_plan_status": "首个 planner result.json.status",
    "planner_result_status_counts": "planner-results/request-*/response.json",
    "online_acceptance_status_counts": "planner-results/request-*/response.json",
    "deadline_expired_after_planning_count": "planner-results/request-*/response.json",
    "server_round_trip_mean_s": "planner-results/request-*/response.json",
    "goal_reached": "ros-replan.log",
    "goal_time_s": "ros-replan.log 的目标与节点启动时间戳",
    "world_start_unix_s": "ros-replan.log；缺失时使用 to_drawer-replan-timing.json",
    "first_planning_submit_from_world_s": "ros-replan.log；缺失时使用 to_drawer-replan-timing.json",
    "first_significant_motion_from_world_s": "episode/replay-manifest.json 与计划轨迹",
    "first_motion_before_crossing": "首次运动时间 + scenarios/*.json 穿越时间",
    "valid_dynamic_success": "goal_reached + first_motion_before_crossing + manifest brake 事件",
    "first_command_start_from_world_s": "to_drawer-replan-timing.json",
    "first_bridge_start_from_world_s": "to_drawer-replan-timing.json",
    "first_handoff_from_world_s": "to_drawer-replan-timing.json",
    "initial_world_warmup_observations": "ros-replan.log；缺失时使用 timing JSON",
    "initial_world_warmup_age_s": "ros-replan.log；缺失时使用 timing JSON",
    "brake_count": "episode/replay-manifest.json.events",
    "guard_dynamic_collision_rejections": "ros-replan.log 中 reason=dynamic_collision",
    "candidate_rejection_reasons": "ros-replan.log 中 JSON reason 字段",
    "accepted_nonpositive_clearance_count": "manifest 生效计划的 candidate_clearance_diagnostics",
    "hard_minimum_clearance_m": "manifest 生效计划的 candidate_clearance_diagnostics",
    "common_window_minimum_clearance_m": "manifest 生效计划的 candidate_clearance_diagnostics",
    "clearance_mean_cost": "manifest 生效计划的 candidate_clearance_diagnostics",
    "clearance_cvar_cost": "manifest 生效计划的 candidate_clearance_diagnostics",
    "maximum_uncovered_command_gap_s": "to_drawer-replan-timing.json",
    "maximum_command_gap_s": "to_drawer-replan-timing.json",
    "guarded_terminal_hold_s": "to_drawer-replan-timing.json",
    "maximum_controller_reference_jump_rad": "to_drawer-replan-timing.json",
    "jtc_error_count": "ros-replan.log",
    "no_valid_trajectory_count": "ros-replan.log",
    "episode_duration_s": "episode/replay-manifest.json",
    "execution_duration_s": "episode/replay-manifest.json.plans",
    "executed_plan_count": "episode/replay-manifest.json.plans",
    "plan_record_count": "episode/replay-manifest.json.plans",
    "joint_l2_path_rad": "manifest 生效时间片 + episode/plans/*/trajectory.npz",
    "joint_l1_travel_rad": "manifest 生效时间片 + episode/plans/*/trajectory.npz",
    "planned_duration_mean_s": "episode/replay-manifest.json.plans.phase_timing",
}

for _field, _source in _SOURCE_OVERRIDES.items():
    METRIC_CATALOG[_field]["source"] = _source
