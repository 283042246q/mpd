"""Optional fixed-path Phase5/F1 time-corridor refinement.

Each candidate gets its own all-body dynamic SDF grid. A monotone, velocity-
reachable grid path selects one safe interval per phase; timing is then
optimized inside that *fixed* branch. The runtime DenseCheck remains final.
"""

from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
import math
import time

import numpy as np
import torch


class _StageProfiler:
    """Accumulate CPU/GPU stage timings without synchronizing every step."""

    def __init__(self, tensor: torch.Tensor):
        self._cuda = tensor.is_cuda
        self._cpu = {}
        self._events = {}

    @contextmanager
    def measure(self, name: str):
        started = time.perf_counter()
        event_pair = None
        if self._cuda:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            event_pair = (begin, end)
        try:
            yield
        finally:
            if event_pair is not None:
                event_pair[1].record()
                self._events.setdefault(name, []).append(event_pair)
            self._cpu[name] = self._cpu.get(name, 0.0) + time.perf_counter() - started

    def finish(self, names):
        if self._cuda and self._events:
            torch.cuda.synchronize()
        result = {}
        for name in names:
            cpu_s = self._cpu.get(name, 0.0)
            gpu_s = sum(
                begin.elapsed_time(end) / 1000.0
                for begin, end in self._events.get(name, ())
            )
            result[f"{name}_s"] = float(max(cpu_s, gpu_s))
        return result


def _safe_intervals(safe: np.ndarray, times: np.ndarray, margin: float):
    """Return shrunken intervals and the usable grid cells they contain."""
    bounds = []
    labels = np.full(len(times), -1, dtype=np.int32)
    edges = np.flatnonzero(np.diff(np.r_[False, safe, False]))
    for first, end in zip(edges[::2], edges[1::2]):
        last = end - 1
        lower = float(times[first]) + (margin if first else 0.0)
        upper = float(times[last]) - (margin if last < len(times) - 1 else 0.0)
        if lower > upper:
            continue
        label = len(bounds)
        bounds.append((lower, upper))
        labels[first:end] = np.where(
            (times[first:end] >= lower - 1e-9) & (times[first:end] <= upper + 1e-9),
            label, -1,
        )
    return bounds, labels


def _reachable_branch(safe: np.ndarray, phase_times: np.ndarray, times: np.ndarray,
                      minimum_steps: np.ndarray, duration_min: float, margin: float,
                      preference_s: float):
    """Dynamic program over time cells; preserve separate early/late intervals."""
    phase_count, time_count = safe.shape
    interval_rows = [_safe_intervals(row, times, margin) for row in safe]
    if interval_rows[0][1][0] < 0:
        return None
    score = np.full(time_count, np.inf)
    score[0] = 0.0
    parents = np.full((phase_count, time_count), -1, dtype=np.int32)
    for phase_index in range(1, phase_count):
        previous_best = np.minimum.accumulate(score)
        previous_arg = np.zeros(time_count, dtype=np.int32)
        best_index = 0
        for time_index in range(time_count):
            if score[time_index] < score[best_index]:
                best_index = time_index
            previous_arg[time_index] = best_index
        next_score = np.full(time_count, np.inf)
        target = phase_times[phase_index] + preference_s * math.sin(
            math.pi * phase_index / (phase_count - 1))
        for time_index in np.flatnonzero(interval_rows[phase_index][1] >= 0):
            predecessor_limit = time_index - int(minimum_steps[phase_index - 1])
            if predecessor_limit < 0 or not np.isfinite(previous_best[predecessor_limit]):
                continue
            next_score[time_index] = previous_best[predecessor_limit] + (
                (times[time_index] - target) ** 2 / phase_count)
            parents[phase_index, time_index] = previous_arg[predecessor_limit]
        score = next_score
    score[times < duration_min - 1e-9] = np.inf
    final = int(np.argmin(score))
    if not np.isfinite(score[final]):
        return None
    selected = np.empty(phase_count, dtype=np.int32)
    selected[-1] = final
    for phase_index in range(phase_count - 1, 0, -1):
        selected[phase_index - 1] = parents[phase_index, selected[phase_index]]
    if selected[0] != 0:
        return None
    return tuple(interval_rows[i][0][interval_rows[i][1][selected[i]]]
                 for i in range(phase_count))


def _corridor_cost(arrivals: torch.Tensor, bounds: torch.Tensor):
    lower = torch.relu(bounds[:, 0] - arrivals)
    upper = torch.relu(arrivals - bounds[:, 1])
    return (lower.square() + upper.square()).mean()


def _fixed_path_cost(guide, state, timing, *, factorized: bool):
    total, breakdown, evaluation = guide.cost_evaluator(
        trajectory_state=replace(state, timing=timing))
    if factorized:
        bounds = (torch.relu(evaluation.duration - guide.settings.duration_max).square()
                  + torch.relu(guide.settings.duration_min - evaluation.duration).square())
        total = total + 10.0 * bounds
    return total, breakdown


@torch.enable_grad()
def refine_time_corridor(guide, normalized_paths: torch.Tensor,
                         initial_parameters: torch.Tensor, *, factorized: bool):
    """Return same-shape timing parameters and per-request diagnostics.

    For F1, parameters are the checkpoint's standardized six-dimensional
    latent. For Phase5 they are physical spline controls. Spatial paths are
    fixed and never written. An empty/infeasible corridor leaves that candidate
    untouched; the runtime validates all returned candidates independently.
    """
    settings = guide.settings
    world = guide.dynamic_field.dynamic_world
    output = initial_parameters.detach().clone()
    stage_names = ("grid_build", "branch_selection",
                   "refinement_forward", "refinement_backward")
    profiler = _StageProfiler(output)
    stats = dict(enabled=True, candidates=len(output), attempted=0, changed=0,
                 no_reachable_branch=0, safe_unchanged=0, branches_tested=0,
                 world_version=int(world.world_version),
                 plan_start_unix_ns=int(world.plan_start_unix_ns))
    if not world.active_count:
        stats["safe_unchanged"] = len(output)
        stats.update(profiler.finish(stage_names))
        return output, stats
    if not factorized:
        trajectory = guide.planning_task.parametric_trajectory
        for name in ("q_vel_start", "q_vel_goal", "q_acc_start", "q_acc_goal"):
            value = getattr(trajectory, name, None)
            if value is not None and torch.any(value != 0):
                raise ValueError("fixed-path Corridor A requires zero endpoint velocity/acceleration")
    horizon = guide.timing_spline.num_phase_points
    indices = torch.linspace(0, horizon - 1, settings.corridor_a_phase_points,
                             device=output.device).round().long().unique()
    time_count = max(2, math.ceil(settings.duration_max / settings.corridor_a_time_step_s) + 1)
    grid = torch.linspace(0.0, settings.duration_max, time_count,
                          device=output.device, dtype=output.dtype)
    grid_np = grid.detach().cpu().numpy()
    margins = (guide.dynamic_field.collision_margins.to(device=output.device, dtype=output.dtype)
               + guide.dynamic_field.cutoff_margin + settings.corridor_a_clearance_m)

    for candidate in range(len(output)):
        path = normalized_paths[candidate:candidate + 1].detach()
        initial = output[candidate:candidate + 1]
        with profiler.measure("grid_build"):
            with torch.no_grad():
                if factorized:
                    state = guide.state(path, initial)
                else:
                    _, _, _, state = guide.evaluate_control_points(path, initial, return_state=True)
                _, breakdown = _fixed_path_cost(guide, state, state.timing,
                                                factorized=factorized)
                if float(breakdown["dynamic_collision"][0]) <= 1e-12:
                    stats["safe_unchanged"] += 1
                    continue
                positions = state.collision_sphere_positions[0].index_select(0, indices)
                phase_count, links = positions.shape[:2]
                query_points = positions[:, None].expand(phase_count, time_count, links, 3).contiguous()
                query_times = grid[None].expand(phase_count, -1).contiguous()
                distances = world.minimum_signed_distance(query_points, trajectory_times=query_times)
                safe = (distances >= margins[None, None, :]).all(dim=-1).cpu().numpy()
                q = state.q[0].index_select(0, indices)
                limits = guide.cost_evaluator.velocity_limits
                if limits is None:
                    minimum = np.ones(phase_count - 1, dtype=np.int32)
                else:
                    minimum_time = ((q[1:] - q[:-1]).abs() / limits).amax(dim=-1)
                    grid_step = float(grid[1] - grid[0])
                    minimum = np.maximum(1, np.ceil(minimum_time.cpu().numpy() / grid_step).astype(np.int32))
                initial_arrivals = state.timing.time_from_start[0].index_select(0, indices).cpu().numpy()
        with profiler.measure("branch_selection"):
            branches = []
            for preference in (0.0, -1.0, 1.0):
                branch = _reachable_branch(safe, initial_arrivals, grid_np, minimum,
                                           settings.duration_min,
                                           settings.corridor_a_time_margin_s, preference)
                if branch is not None and branch not in branches:
                    branches.append(branch)
        if not branches:
            stats["no_reachable_branch"] += 1
            continue
        stats["attempted"] += 1
        best_objective = float("inf")
        best_parameters = initial
        fixed_state = replace(state, collision_sphere_positions=state.collision_sphere_positions.detach(),
                              q=state.q.detach(), q_s=state.q_s.detach(), q_ss=state.q_ss.detach())
        for branch in branches:
            stats["branches_tested"] += 1
            bounds = initial.new_tensor(branch)
            parameter = initial.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([parameter], lr=settings.corridor_a_learning_rate)
            for step in range(settings.corridor_a_steps + 1):
                with profiler.measure("refinement_forward"):
                    if factorized:
                        timing, _ = guide.codec.evaluate(parameter, fixed_state.q,
                                                          fixed_state.q_s, fixed_state.q_ss)
                    else:
                        timing = guide.timing_spline.evaluate(parameter, q=fixed_state.q,
                                                              q_s=fixed_state.q_s, q_ss=fixed_state.q_ss)
                    # The dynamic world is only valid through the request's
                    # maximum duration. Do not query outside that prediction.
                    if (not torch.isfinite(timing.duration).all()
                            or torch.any(timing.duration > settings.duration_max)):
                        break
                    total, _ = _fixed_path_cost(guide, fixed_state, timing, factorized=factorized)
                    corridor = _corridor_cost(timing.time_from_start[0].index_select(0, indices), bounds)
                    objective = total.sum() + settings.corridor_a_weight * corridor
                    if torch.isfinite(objective):
                        duration = float(timing.duration[0].detach())
                        value = float(objective.detach())
                        if (settings.duration_min <= duration <= settings.duration_max
                                and value < best_objective):
                            best_objective = value
                            best_parameters = parameter.detach().clone()
                if (step == settings.corridor_a_steps or not torch.isfinite(objective)
                        or not objective.requires_grad):
                    break
                with profiler.measure("refinement_backward"):
                    optimizer.zero_grad(set_to_none=True)
                    objective.backward()
                    if not factorized:
                        parameter.grad[:, ~guide.timing_spline.optimizable_mask] = 0.0
                    torch.nn.utils.clip_grad_norm_([parameter], settings.timing_max_grad_norm)
                    optimizer.step()
                    if not factorized:
                        with torch.no_grad():
                            parameter[:, ~guide.timing_spline.optimizable_mask] = initial[:, ~guide.timing_spline.optimizable_mask]
        if not torch.equal(best_parameters, initial):
            output[candidate] = best_parameters[0]
            stats["changed"] += 1
    stats.update(profiler.finish(stage_names))
    return output, stats
