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
            result[f"{name}_host_wall_s"] = float(cpu_s)
            result[f"{name}_cuda_event_s"] = float(gpu_s) if self._cuda else None
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


def _window_signature(arrivals, interval_rows, *, tolerance=1e-5):
    """Identify the intervals reached by a continuous timing trajectory."""
    signature = []
    for arrival, (bounds, _) in zip(arrivals, interval_rows):
        label = next((index for index, (low, high) in enumerate(bounds)
                      if low - tolerance <= float(arrival) <= high + tolerance), -1)
        signature.append(label)
    return tuple(signature)


def _reachable_branch(safe: np.ndarray, phase_times: np.ndarray, times: np.ndarray,
                      minimum_steps: np.ndarray, duration_min: float, margin: float,
                      preference_s: float, *, interval_rows=None,
                      return_schedule: bool = False):
    """Dynamic program over time cells; preserve separate early/late intervals."""
    phase_count, time_count = safe.shape
    if interval_rows is None:
        interval_rows = [_safe_intervals(row, times, margin) for row in safe]
    if interval_rows[0][1][0] < 0:
        return None
    score = np.full(time_count, np.inf)
    score[0] = 0.0
    parents = np.full((phase_count, time_count), -1, dtype=np.int32)
    for phase_index in range(1, phase_count):
        previous_best = np.minimum.accumulate(score)
        improvements = score < np.r_[np.inf, previous_best[:-1]]
        previous_arg = np.maximum.accumulate(
            np.where(improvements, np.arange(time_count), 0)
        ).astype(np.int32)
        next_score = np.full(time_count, np.inf)
        target = phase_times[phase_index] + preference_s * math.sin(
            math.pi * phase_index / (phase_count - 1))
        valid_times = np.flatnonzero(interval_rows[phase_index][1] >= 0)
        predecessor_limits = valid_times - int(minimum_steps[phase_index - 1])
        reachable = predecessor_limits >= 0
        valid_times = valid_times[reachable]
        predecessor_limits = predecessor_limits[reachable]
        reachable = np.isfinite(previous_best[predecessor_limits])
        valid_times = valid_times[reachable]
        predecessor_limits = predecessor_limits[reachable]
        next_score[valid_times] = previous_best[predecessor_limits] + (
            (times[valid_times] - target) ** 2 / phase_count
        )
        parents[phase_index, valid_times] = previous_arg[predecessor_limits]
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
    branch = tuple(interval_rows[i][0][interval_rows[i][1][selected[i]]]
                   for i in range(phase_count))
    return (branch, times[selected].copy()) if return_schedule else branch


def _reachable_branches_kbest(safe: np.ndarray, phase_times: np.ndarray,
                              times: np.ndarray, minimum_steps: np.ndarray,
                              duration_min: float, margin: float, k: int,
                              *, interval_rows=None):
    """Bounded distinct-window DP; scores use the nominal preference only."""
    phase_count, time_count = safe.shape
    if interval_rows is None:
        interval_rows = [_safe_intervals(row, times, margin) for row in safe]
    first_label = int(interval_rows[0][1][0])
    if first_label < 0:
        return []
    previous = [[] for _ in range(time_count)]
    previous[0] = [(0.0, (first_label,), (0,))]
    for phase_index in range(1, phase_count):
        current = [[] for _ in range(time_count)]
        prefix = {}
        ingested = -1
        for cell in np.flatnonzero(interval_rows[phase_index][1] >= 0):
            limit = cell - int(minimum_steps[phase_index - 1])
            if limit < 0:
                continue
            while ingested < limit:
                ingested += 1
                for label in previous[ingested]:
                    old = prefix.get(label[1])
                    if old is None or label[0] < old[0]:
                        prefix[label[1]] = label
                if len(prefix) > k:
                    prefix = {
                        label[1]: label for label in
                        sorted(prefix.values(), key=lambda item: (item[0], item[2]))[:k]
                    }
            if not prefix:
                continue
            ordered = sorted(prefix.values(), key=lambda item: (item[0], item[2]))[:k]
            target = phase_times[phase_index]
            increment = (times[cell] - target) ** 2 / phase_count
            label_id = int(interval_rows[phase_index][1][cell])
            current[cell] = [(score + increment, signature + (label_id,),
                              cells + (int(cell),))
                             for score, signature, cells in ordered]
        previous = current
    finalists = {}
    for cell in np.flatnonzero(times >= duration_min - 1e-9):
        for label in previous[cell]:
            old = finalists.get(label[1])
            if old is None or label[0] < old[0]:
                finalists[label[1]] = label
    selected = sorted(finalists.values(), key=lambda item: (item[0], item[2]))[:k]
    result = []
    for _, _, cells in selected:
        branch = tuple(interval_rows[i][0][interval_rows[i][1][cells[i]]]
                       for i in range(phase_count))
        result.append((branch, times[np.asarray(cells)].copy()))
    return result


def _corridor_cost(arrivals: torch.Tensor, bounds: torch.Tensor):
    lower = torch.relu(bounds[:, 0] - arrivals)
    upper = torch.relu(arrivals - bounds[:, 1])
    return (lower.square() + upper.square()).mean()


def _fixed_path_cost(guide, state, timing, *, factorized: bool,
                     trusted_candidate_times: bool = False,
                     minimum_distance_override=None):
    total, breakdown, evaluation = guide.cost_evaluator(
        trajectory_state=replace(state, timing=timing),
        trusted_candidate_times=trusted_candidate_times,
        minimum_distance_override=minimum_distance_override,
    )
    if factorized:
        bounds = (torch.relu(evaluation.duration - guide.settings.duration_max).square()
                  + torch.relu(guide.settings.duration_min - evaluation.duration).square())
        total = total + 10.0 * bounds
    return total, breakdown


def _nonzero_endpoint_derivatives(guide) -> bool:
    trajectory = guide.planning_task.parametric_trajectory
    for name in ("q_vel_start", "q_vel_goal", "q_acc_start", "q_acc_goal"):
        value = getattr(trajectory, name, None)
        if value is not None and torch.any(value != 0):
            return True
    return False


def _indexed_fixed_state(state, rows: torch.Tensor):
    """Index only the immutable spatial tensors used by the physical cost."""
    return replace(
        state,
        q=state.q.index_select(0, rows).detach(),
        q_s=state.q_s.index_select(0, rows).detach(),
        q_ss=state.q_ss.index_select(0, rows).detach(),
        collision_sphere_positions=state.collision_sphere_positions.index_select(0, rows).detach(),
        collision_sphere_poses=None,
        collision_sphere_jacobians=None,
    )


def _time_distance_table(world, positions: torch.Tensor, grid: torch.Tensor,
                         *, candidate_chunk_size: int = 4) -> torch.Tensor:
    """Request-local [candidate, physical phase, time, link] surrogate data."""
    batches, phases, links = positions.shape[:3]
    pieces = []
    for start in range(0, batches, candidate_chunk_size):
        chunk = positions[start:start + candidate_chunk_size]
        count = len(chunk)
        points = chunk[:, :, None].expand(-1, -1, len(grid), -1, -1)
        points = points.reshape(count * phases, len(grid), links, 3).contiguous()
        times = grid[None].expand(count * phases, -1).contiguous()
        distance = world.minimum_signed_distance(points, trajectory_times=times)
        pieces.append(distance.reshape(count, phases, len(grid), links).detach())
    return torch.cat(pieces, dim=0)


def _interpolate_time_distance_table(table: torch.Tensor, grid: torch.Tensor,
                                     arrivals: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear time interpolation; reject requests outside the table."""
    if arrivals.shape != table.shape[:2]:
        raise ValueError("arrival shape must match time-distance table")
    if not bool((torch.isfinite(arrivals) & (arrivals >= grid[0])
                 & (arrivals <= grid[-1])).all()):
        raise ValueError("arrival outside time-distance table domain")
    delta = grid[1] - grid[0]
    left = torch.floor((arrivals - grid[0]) / delta).long().clamp(0, len(grid) - 2)
    right = left + 1
    batch_index = torch.arange(len(table), device=table.device)[:, None]
    phase_index = torch.arange(table.shape[1], device=table.device)[None, :]
    low = table[batch_index, phase_index, left]
    high = table[batch_index, phase_index, right]
    weight = ((arrivals - grid[left]) / (grid[right] - grid[left]))[..., None]
    return low + weight * (high - low)


def _constant_sphere_interval_grid(world, positions: torch.Tensor,
                                   margins: torch.Tensor, grid: torch.Tensor):
    """Analytic constant-velocity sphere intervals projected to the original grid.

    Returns None for boxes, capsules or time-varying inflation. Boundary cells
    are conservatively marked blocked to avoid floating-point false accepts.
    """
    count = world.active_count
    if count == 0:
        return np.ones((*positions.shape[:2], len(grid)), dtype=bool)
    if not bool((world.shape_code[:count] == 0).all()):
        return None
    if not bool((world.inflation_code[:count] == 0).all()):
        return None
    if not bool((world.horizon_inflation_rate[:count] == 0).all()):
        return None
    points = positions.detach().cpu().numpy()
    radii = world.parameters[:count, 0].detach().cpu().numpy()
    centers = world.position[:count].detach().cpu().numpy()
    velocity = world.velocity[:count].detach().cpu().numpy()
    inflation = world.base_inflation[:count].detach().cpu().numpy()
    margin = margins.detach().cpu().numpy()
    times = grid.detach().cpu().numpy()
    offset = (world.plan_start_unix_ns - world.stamp_unix_ns) * 1e-9
    centers = centers + offset * velocity
    safe = np.ones((*points.shape[:2], len(times)), dtype=bool)
    tolerance = 1e-7
    for object_index in range(count):
        speed_sq = float(np.dot(velocity[object_index], velocity[object_index]))
        delta = points - centers[object_index]
        threshold_sq = (radii[object_index] + inflation[object_index] + margin) ** 2
        if speed_sq <= 1e-16:
            safe &= (np.sum(delta * delta, axis=-1) >= threshold_sq + tolerance).all(axis=-1)[..., None]
            continue
        linear = np.sum(delta * velocity[object_index], axis=-1)
        constant = np.sum(delta * delta, axis=-1) - threshold_sq
        discriminant = linear * linear - speed_sq * constant
        half_width = np.sqrt(np.maximum(discriminant, 0.0)) / speed_sq
        center_time = linear / speed_sq
        lower = center_time - half_width
        upper = center_time + half_width
        blocked = ((discriminant >= -tolerance)[..., None]
                   & (times >= lower[..., None] - tolerance)
                   & (times <= upper[..., None] + tolerance))
        safe &= ~blocked.any(axis=-2)
    return safe


def _fit_dp_arrival_initializers(guide, fixed, initial: torch.Tensor,
                                 phase_indices: torch.Tensor, schedules: torch.Tensor,
                                 bounds: torch.Tensor, *, factorized: bool):
    """Fit the existing timing representation, then keep only exact-cost wins."""
    settings = guide.settings
    parameter = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([parameter], lr=0.08)
    diagnostics = {"abs_error_sum_s": 0.0, "abs_error_count": 0, "abs_error_max_s": 0.0}
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        if factorized:
            timing, _ = guide.codec.evaluate(parameter, fixed.q, fixed.q_s, fixed.q_ss)
        else:
            timing = guide.timing_spline.evaluate(
                parameter, q=fixed.q, q_s=fixed.q_s, q_ss=fixed.q_ss,
                require_fixed_endpoint_derivatives=False, validate_inputs=False,
            )
        residual = (timing.time_from_start.index_select(1, phase_indices) - schedules)
        objective = (residual / settings.duration_max).square().mean(dim=-1)
        objective = objective + 1e-4 * (parameter - initial).square().mean(dim=-1)
        if not bool(torch.isfinite(objective).all()):
            return initial, 0, len(initial), diagnostics
        objective.sum().backward()
        if not factorized:
            parameter.grad[:, ~guide.timing_spline.optimizable_mask] = 0.0
        torch.nn.utils.clip_grad_norm_([parameter], max_norm=4.0)
        optimizer.step()
        if not factorized:
            with torch.no_grad():
                parameter[:, ~guide.timing_spline.optimizable_mask] = initial[:, ~guide.timing_spline.optimizable_mask]
    proposed = parameter.detach()
    if not bool(torch.isfinite(proposed).all()):
        return initial, 0, len(initial), diagnostics
    with torch.no_grad():
        if factorized:
            proposed_timing, _ = guide.codec.evaluate(proposed, fixed.q, fixed.q_s, fixed.q_ss)
            original_timing, _ = guide.codec.evaluate(initial, fixed.q, fixed.q_s, fixed.q_ss)
        else:
            proposed_timing = guide.timing_spline.evaluate(
                proposed, q=fixed.q, q_s=fixed.q_s, q_ss=fixed.q_ss,
                require_fixed_endpoint_derivatives=False,
            )
            original_timing = guide.timing_spline.evaluate(
                initial, q=fixed.q, q_s=fixed.q_s, q_ss=fixed.q_ss,
                require_fixed_endpoint_derivatives=False,
            )
        fit_error = (proposed_timing.time_from_start.index_select(1, phase_indices)
                     - schedules).abs()
        finite_error = fit_error[torch.isfinite(fit_error)]
        if finite_error.numel():
            diagnostics["abs_error_sum_s"] = float(finite_error.sum().item())
            diagnostics["abs_error_count"] = int(finite_error.numel())
            diagnostics["abs_error_max_s"] = float(finite_error.max().item())
        admissible = (torch.isfinite(proposed_timing.time_from_start).all(dim=-1)
                      & (proposed_timing.duration >= settings.duration_min)
                      & (proposed_timing.duration <= settings.duration_max)
                      & (torch.diff(proposed_timing.time_from_start, dim=-1) > 0).all(dim=-1))
        if not bool(admissible.any()):
            return initial, 0, len(initial), diagnostics
        chosen = initial.clone()
        rows = torch.nonzero(admissible, as_tuple=False).flatten()
        proposed_state = _indexed_fixed_state(fixed, rows)
        proposed_time = replace(
            proposed_timing,
            phase=proposed_timing.phase,
            q=proposed_timing.q.index_select(0, rows),
            dq=proposed_timing.dq.index_select(0, rows),
            ddq=proposed_timing.ddq.index_select(0, rows),
            log_density=proposed_timing.log_density.index_select(0, rows),
            u=proposed_timing.u.index_select(0, rows),
            u_s=proposed_timing.u_s.index_select(0, rows),
            time_from_start=proposed_timing.time_from_start.index_select(0, rows),
            duration=proposed_timing.duration.index_select(0, rows),
        )
        original_cost, _ = _fixed_path_cost(
            guide, fixed, original_timing, factorized=factorized,
        )
        proposed_cost, _ = _fixed_path_cost(
            guide, proposed_state, proposed_time, factorized=factorized,
        )
        original_arrivals = original_timing.time_from_start.index_select(1, phase_indices)
        proposed_arrivals = proposed_time.time_from_start.index_select(1, phase_indices)
        original_penalty = (
            torch.relu(bounds[:, :, 0] - original_arrivals).square()
            + torch.relu(original_arrivals - bounds[:, :, 1]).square()
        ).mean(dim=-1)
        selected_bounds = bounds.index_select(0, rows)
        proposed_penalty = (
            torch.relu(selected_bounds[:, :, 0] - proposed_arrivals).square()
            + torch.relu(proposed_arrivals - selected_bounds[:, :, 1]).square()
        ).mean(dim=-1)
        better = (proposed_cost + settings.corridor_a_weight * proposed_penalty
                  < original_cost.index_select(0, rows)
                  + settings.corridor_a_weight * original_penalty.index_select(0, rows))
        if bool(better.any()):
            accepted = rows[better]
            chosen[accepted] = proposed[accepted]
        else:
            accepted = rows[:0]
    return chosen, len(accepted), len(initial) - len(accepted), diagnostics


@torch.enable_grad()
def refine_time_corridor_batch(guide, normalized_paths: torch.Tensor,
                               initial_parameters: torch.Tensor, *, factorized: bool,
                               return_branches: bool = False,
                               eligible_candidate_mask: torch.Tensor = None,
                               k_expand_mask: torch.Tensor = None,
                               validate_branch_rows=None):
    """Exact-cost, chunked candidate/branch backend; serial remains the reference.

    Only the execution layout changes: the same grid, three preferences, fixed
    paths, physical cost, number of Adam steps and per-row clipping are used.
    A failed row stops independently; all candidate outputs retain their input
    order for the final DenseCheck in the runtime.
    """
    settings = guide.settings
    world = guide.dynamic_field.dynamic_world
    output = initial_parameters.detach().clone()
    profiler = _StageProfiler(output)
    names = ("grid_build", "branch_selection", "dp_initialization", "distance_table_build",
             "exact_rerank", "refinement_forward", "refinement_backward",
             "early_dense_validation")
    backend = settings.corridor_a_backend
    use_time_table = backend == "batch_time_table"
    stats = dict(enabled=True, backend=backend, candidates=len(output), attempted=0,
                 changed=0, no_reachable_branch=0, safe_unchanged=0,
                 branches_tested=0, optimized_unique_window_sequences=0,
                 optimized_paths_with_multiple_window_sequences=0,
                 optimized_outside_window_branches=0,
                 dp_fit_abs_error_s_mean=None, dp_fit_abs_error_s_max=None,
                 time_invariant_skipped=0, k_candidates_considered=0,
                 k_candidates_expanded=0, early_stop_dense_checks=0,
                 early_stopped_branches=0, early_saved_iterations=0,
                 world_version=int(world.world_version),
                 plan_start_unix_ns=int(world.plan_start_unix_ns))
    if eligible_candidate_mask is not None and eligible_candidate_mask.shape != (len(output),):
        raise ValueError("eligible candidate mask has the wrong shape")
    if k_expand_mask is not None and k_expand_mask.shape != (len(output),):
        raise ValueError("K expansion mask has the wrong shape")
    if settings.corridor_a_early_stop and validate_branch_rows is None:
        raise ValueError("early stopping requires the exact runtime DenseCheck callback")
    branch_catalog = {}
    fit_error_sum = 0.0
    fit_error_count = 0
    fit_error_max = 0.0
    if not world.active_count:
        stats["safe_unchanged"] = len(output)
        stats.update(profiler.finish(names))
        return (output, stats, branch_catalog) if return_branches else (output, stats)
    if not factorized and _nonzero_endpoint_derivatives(guide):
        stats["skipped_nonzero_endpoint_derivatives"] = True
        stats.update(profiler.finish(names))
        return (output, stats, branch_catalog) if return_branches else (output, stats)
    indices = torch.linspace(0, guide.timing_spline.num_phase_points - 1,
                             settings.corridor_a_phase_points, device=output.device).round().long().unique()
    time_count = max(2, math.ceil(settings.duration_max / settings.corridor_a_time_step_s) + 1)
    grid = torch.linspace(0.0, settings.duration_max, time_count,
                          device=output.device, dtype=output.dtype)
    grid_np = grid.detach().cpu().numpy()
    margins = (guide.dynamic_field.collision_margins.to(device=output.device, dtype=output.dtype)
               + guide.dynamic_field.cutoff_margin + settings.corridor_a_clearance_m)
    chunk_size = settings.corridor_a_chunk_size
    for chunk_start in range(0, len(output), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(output))
        paths = normalized_paths[chunk_start:chunk_end].detach()
        originals = output[chunk_start:chunk_end]
        with profiler.measure("grid_build"):
            with torch.no_grad():
                if factorized:
                    state = guide.state(paths, originals)
                    _, breakdown = _fixed_path_cost(guide, state, state.timing, factorized=True)
                else:
                    _, breakdown, _, state = guide.evaluate_control_points(
                        paths, originals, return_state=True
                    )
                active = breakdown["dynamic_collision"] > 1e-12
                if eligible_candidate_mask is not None and settings.corridor_a_selective_k:
                    eligible = eligible_candidate_mask[chunk_start:chunk_end]
                    stats["time_invariant_skipped"] += int((active & ~eligible).sum().item())
                    active = active & eligible
                active_indices = torch.nonzero(active, as_tuple=False).flatten()
                stats["safe_unchanged"] += len(originals) - len(active_indices)
                if not len(active_indices):
                    continue
                active_state = _indexed_fixed_state(state, active_indices)
                positions = active_state.collision_sphere_positions.index_select(1, indices)
                active_count, phases, links = positions.shape[:3]
                query_points = positions[:, :, None].expand(-1, -1, time_count, -1, -1)
                query_points = query_points.reshape(active_count * phases, time_count, links, 3).contiguous()
                query_times = grid[None].expand(active_count * phases, -1).contiguous()
                analytic_safe = None
                if backend == "batch_event_intervals":
                    analytic_safe = _constant_sphere_interval_grid(
                        world, positions, margins, grid,
                    )
                if analytic_safe is None:
                    distances = world.minimum_signed_distance(query_points, trajectory_times=query_times)
                    safe = (distances.reshape(active_count, phases, time_count, links)
                            >= margins[None, None, None, :]).all(dim=-1).cpu().numpy()
                    if backend == "batch_event_intervals":
                        stats["analytic_fallback_chunks"] = stats.get("analytic_fallback_chunks", 0) + 1
                else:
                    safe = analytic_safe
                    stats["analytic_supported_chunks"] = stats.get("analytic_supported_chunks", 0) + 1
                q = active_state.q.index_select(1, indices)
                limits = guide.cost_evaluator.velocity_limits
                if limits is None:
                    minimum = np.ones((active_count, phases - 1), dtype=np.int32)
                else:
                    delta = ((q[:, 1:] - q[:, :-1]).abs() / limits).amax(dim=-1)
                    minimum = np.maximum(1, np.ceil((delta / (grid[1] - grid[0])).cpu().numpy()).astype(np.int32))
                arrivals = state.timing.time_from_start.index_select(0, active_indices)
                arrivals = arrivals.index_select(1, indices).cpu().numpy()
        with profiler.measure("branch_selection"):
            row_candidates = []
            row_bounds = []
            row_schedules = []
            intervals_by_candidate = {}
            for local_active, local_candidate in enumerate(active_indices.tolist()):
                branches = []
                interval_rows = [
                    _safe_intervals(row, grid_np, settings.corridor_a_time_margin_s)
                    for row in safe[local_active]
                ]
                intervals_by_candidate[local_candidate] = interval_rows
                for preference in (0.0, -1.0, 1.0):
                    found = _reachable_branch(safe[local_active], arrivals[local_active], grid_np,
                                              minimum[local_active], settings.duration_min,
                                              settings.corridor_a_time_margin_s, preference,
                                              interval_rows=interval_rows, return_schedule=True)
                    if found is not None and all(found[0] != item[0] for item in branches):
                        branches.append(found)
                if settings.corridor_a_k_best:
                    stats["k_candidates_considered"] += 1
                    expansion_needed = True
                    if settings.corridor_a_selective_k:
                        window_conflict = -1 in _window_signature(arrivals[local_active], interval_rows)
                        timing_invalid = (k_expand_mask is not None and bool(
                            k_expand_mask[chunk_start + local_candidate].item()))
                        expansion_needed = window_conflict or timing_invalid
                    if expansion_needed:
                        stats["k_candidates_expanded"] += 1
                        extras = _reachable_branches_kbest(
                            safe[local_active], arrivals[local_active], grid_np,
                            minimum[local_active], settings.duration_min,
                            settings.corridor_a_time_margin_s, settings.corridor_a_k_best,
                            interval_rows=interval_rows,
                        )
                        for found in extras:
                            if len(branches) >= settings.corridor_a_k_best:
                                break
                            if all(found[0] != item[0] for item in branches):
                                branches.append(found)
                if not branches:
                    stats["no_reachable_branch"] += 1
                    continue
                stats["attempted"] += 1
                for branch, schedule in branches:
                    row_candidates.append(local_candidate)
                    row_bounds.append(branch)
                    row_schedules.append(schedule)
        if not row_candidates:
            continue
        stats["branches_tested"] += len(row_candidates)
        row_indices = torch.as_tensor(row_candidates, device=output.device, dtype=torch.long)
        fixed = _indexed_fixed_state(state, row_indices)
        table_rows = None
        if use_time_table:
            unique_candidates = list(dict.fromkeys(row_candidates))
            unique_indices = torch.as_tensor(unique_candidates, device=output.device, dtype=torch.long)
            with profiler.measure("distance_table_build"):
                unique_state = _indexed_fixed_state(state, unique_indices)
                unique_table = _time_distance_table(
                    world, unique_state.collision_sphere_positions, grid,
                )
                lookup = {candidate: index for index, candidate in enumerate(unique_candidates)}
                table_rows = unique_table.index_select(
                    0, torch.as_tensor([lookup[candidate] for candidate in row_candidates],
                                       device=output.device, dtype=torch.long),
                )
            stats["distance_table_candidates"] = stats.get("distance_table_candidates", 0) + len(unique_candidates)
        initial_rows = originals.index_select(0, row_indices).detach()
        bounds = initial_rows.new_tensor(np.asarray(row_bounds))
        if settings.corridor_a_dp_init:
            with profiler.measure("dp_initialization"):
                fitted, accepted, rejected, fit_diagnostics = _fit_dp_arrival_initializers(
                    guide, fixed, initial_rows, indices,
                    initial_rows.new_tensor(np.asarray(row_schedules)), bounds,
                    factorized=factorized,
                )
                initial_rows = fitted
            fit_error_sum += fit_diagnostics["abs_error_sum_s"]
            fit_error_count += fit_diagnostics["abs_error_count"]
            fit_error_max = max(fit_error_max, fit_diagnostics["abs_error_max_s"])
            stats["dp_initializers_accepted"] = stats.get("dp_initializers_accepted", 0) + accepted
            stats["dp_initializers_rejected"] = stats.get("dp_initializers_rejected", 0) + rejected
        parameter = initial_rows.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([parameter], lr=settings.corridor_a_learning_rate)
        alive = torch.ones(len(row_candidates), dtype=torch.bool, device=output.device)
        best_values = torch.full((len(row_candidates),), float("inf"), device=output.device,
                                 dtype=output.dtype)
        best_parameters = initial_rows.clone()
        if settings.corridor_a_early_stop:
            previous_physical = torch.full_like(best_values, float("nan"))
            stable_steps = torch.zeros(len(row_candidates), dtype=torch.int32, device=output.device)
        for step in range(settings.corridor_a_steps + 1):
            active_rows = torch.nonzero(alive, as_tuple=False).flatten()
            if not len(active_rows):
                break
            with profiler.measure("refinement_forward"):
                selected = parameter.index_select(0, active_rows)
                if not factorized:
                    finite_parameters = torch.isfinite(selected).all(dim=-1)
                    alive[active_rows[~finite_parameters]] = False
                    active_rows = active_rows[finite_parameters]
                    if not len(active_rows):
                        continue
                    selected = parameter.index_select(0, active_rows)
                selected_state = _indexed_fixed_state(fixed, active_rows)
                if factorized:
                    timing, _ = guide.codec.evaluate(selected, selected_state.q,
                                                      selected_state.q_s, selected_state.q_ss)
                else:
                    timing = guide.timing_spline.evaluate(selected, q=selected_state.q,
                                                          q_s=selected_state.q_s, q_ss=selected_state.q_ss,
                                                          require_fixed_endpoint_derivatives=False,
                                                          validate_inputs=False)
                admissible = (torch.isfinite(timing.duration)
                              & (timing.duration <= settings.duration_max)
                              & torch.isfinite(timing.time_from_start).all(dim=-1)
                              & (torch.diff(timing.time_from_start, dim=-1) > 0).all(dim=-1))
                good_rows = torch.nonzero(admissible, as_tuple=False).flatten()
                alive[active_rows[~admissible]] = False
                if not len(good_rows):
                    continue
                if len(good_rows) == len(active_rows):
                    valid_global = active_rows
                    valid_state = selected_state
                    valid_parameters = selected
                    valid_timing = timing
                else:
                    valid_global = active_rows.index_select(0, good_rows)
                    valid_state = _indexed_fixed_state(fixed, valid_global)
                    # Re-evaluate only when invalid rows must be isolated from
                    # the physical cost and its backward graph.
                    valid_parameters = parameter.index_select(0, valid_global)
                    if factorized:
                        valid_timing, _ = guide.codec.evaluate(valid_parameters, valid_state.q,
                                                                valid_state.q_s, valid_state.q_ss)
                    else:
                        valid_timing = guide.timing_spline.evaluate(valid_parameters, q=valid_state.q,
                                                                    q_s=valid_state.q_s, q_ss=valid_state.q_ss,
                                                                    require_fixed_endpoint_derivatives=False,
                                                                    validate_inputs=False)
                surrogate_distance = None
                if use_time_table:
                    selected_table = table_rows.index_select(0, valid_global)
                    surrogate_distance = _interpolate_time_distance_table(
                        selected_table, grid, valid_timing.time_from_start,
                    )
                total, _ = _fixed_path_cost(
                    guide, valid_state, valid_timing, factorized=factorized,
                    trusted_candidate_times=True,
                    minimum_distance_override=surrogate_distance,
                )
                selected_arrivals = valid_timing.time_from_start.index_select(1, indices)
                selected_bounds = bounds.index_select(0, valid_global)
                penalty = (torch.relu(selected_bounds[:, :, 0] - selected_arrivals).square()
                           + torch.relu(selected_arrivals - selected_bounds[:, :, 1]).square()).mean(dim=-1)
                objectives = total + settings.corridor_a_weight * penalty
                finite = torch.isfinite(objectives)
                alive[valid_global[~finite]] = False
                better = finite & (valid_timing.duration >= settings.duration_min) & (
                    objectives.detach() < best_values.index_select(0, valid_global))
                if better.any():
                    improved = valid_global[better]
                    best_values[improved] = objectives.detach()[better]
                    best_parameters[improved] = valid_parameters.detach()[better]
                if settings.corridor_a_early_stop:
                    previous = previous_physical.index_select(0, valid_global)
                    stable = (torch.isfinite(previous) & finite
                              & ((total.detach() - previous).abs()
                                 <= 1e-3 * previous.abs().clamp_min(1.0)))
                    old_steps = stable_steps.index_select(0, valid_global)
                    stable_steps[valid_global] = torch.where(
                        stable, old_steps + 1, torch.zeros_like(old_steps),
                    )
                    previous_physical[valid_global] = total.detach()
                    inside_window = (
                        (selected_arrivals >= selected_bounds[:, :, 0] - 1e-5)
                        & (selected_arrivals <= selected_bounds[:, :, 1] + 1e-5)
                    ).all(dim=-1)
                    ready = (finite & inside_window
                             & (stable_steps.index_select(0, valid_global) >= 2)
                             & (step >= 4) & (step % 4 == 0))
                    check_rows = torch.nonzero(ready, as_tuple=False).flatten()
                    if len(check_rows):
                        global_rows = valid_global.index_select(0, check_rows)
                        candidate_indices = row_indices.index_select(0, global_rows) + chunk_start
                        candidate_parameters = valid_parameters.index_select(0, check_rows).detach()
                        with profiler.measure("early_dense_validation"):
                            passed = validate_branch_rows(candidate_indices, candidate_parameters)
                        passed = passed.to(device=output.device, dtype=torch.bool)
                        if passed.shape != (len(check_rows),):
                            raise ValueError("early DenseCheck returned the wrong batch size")
                        stats["early_stop_dense_checks"] += len(check_rows)
                        passed_rows = global_rows[passed]
                        if len(passed_rows):
                            best_values[passed_rows] = objectives.detach()[check_rows[passed]]
                            best_parameters[passed_rows] = candidate_parameters[passed]
                            alive[passed_rows] = False
                            finite[check_rows[passed]] = False
                            stats["early_stopped_branches"] += len(passed_rows)
                            stats["early_saved_iterations"] += len(passed_rows) * (
                                settings.corridor_a_steps - step)
            if step == settings.corridor_a_steps:
                break
            gradient_rows = valid_global[finite]
            if not len(gradient_rows):
                continue
            with profiler.measure("refinement_backward"):
                optimizer.zero_grad(set_to_none=True)
                objectives[finite].sum().backward()
                gradient = parameter.grad
                if gradient is None:
                    break
                if not factorized:
                    gradient[:, ~guide.timing_spline.optimizable_mask] = 0.0
                norms = torch.linalg.vector_norm(gradient, dim=-1, keepdim=True)
                gradient.mul_((settings.timing_max_grad_norm / norms.clamp_min(1e-12)).clamp(max=1.0))
                before = parameter.detach().clone()
                optimizer.step()
                with torch.no_grad():
                    parameter[~alive] = before[~alive]
                    if not factorized:
                        parameter[:, ~guide.timing_spline.optimizable_mask] = initial_rows[:, ~guide.timing_spline.optimizable_mask]
        # Strict improvement and original branch order preserve the reference tie rule.
        values = best_values.detach().cpu().tolist()
        if use_time_table:
            finite_rows = torch.nonzero(torch.isfinite(best_values), as_tuple=False).flatten()
            if len(finite_rows):
                with profiler.measure("exact_rerank"):
                    with torch.no_grad():
                        exact_state = _indexed_fixed_state(fixed, finite_rows)
                        exact_parameters = best_parameters.index_select(0, finite_rows)
                        if factorized:
                            exact_timing, _ = guide.codec.evaluate(
                                exact_parameters, exact_state.q, exact_state.q_s, exact_state.q_ss,
                            )
                        else:
                            exact_timing = guide.timing_spline.evaluate(
                                exact_parameters, q=exact_state.q, q_s=exact_state.q_s,
                                q_ss=exact_state.q_ss,
                                require_fixed_endpoint_derivatives=False,
                            )
                        exact_cost, _ = _fixed_path_cost(
                            guide, exact_state, exact_timing, factorized=factorized,
                        )
                        exact_arrivals = exact_timing.time_from_start.index_select(1, indices)
                        exact_bounds = bounds.index_select(0, finite_rows)
                        exact_penalty = (
                            torch.relu(exact_bounds[:, :, 0] - exact_arrivals).square()
                            + torch.relu(exact_arrivals - exact_bounds[:, :, 1]).square()
                        ).mean(dim=-1)
                        exact_values = (exact_cost + settings.corridor_a_weight * exact_penalty).cpu().tolist()
                    for row, value in zip(finite_rows.tolist(), exact_values):
                        values[row] = value
                stats["exact_rerank_branches"] = stats.get("exact_rerank_branches", 0) + len(finite_rows)
        finite_rows = [row for row, value in enumerate(values) if math.isfinite(value)]
        signatures = {}
        if finite_rows:
            with torch.no_grad():
                selected_rows = torch.as_tensor(finite_rows, dtype=torch.long, device=output.device)
                selected_state = _indexed_fixed_state(fixed, selected_rows)
                selected_parameters = best_parameters.index_select(0, selected_rows)
                if factorized:
                    selected_timing, _ = guide.codec.evaluate(
                        selected_parameters, selected_state.q, selected_state.q_s, selected_state.q_ss,
                    )
                else:
                    selected_timing = guide.timing_spline.evaluate(
                        selected_parameters, q=selected_state.q, q_s=selected_state.q_s,
                        q_ss=selected_state.q_ss, require_fixed_endpoint_derivatives=False,
                    )
                optimized_arrivals = selected_timing.time_from_start.index_select(1, indices).cpu().numpy()
            for offset, row in enumerate(finite_rows):
                signatures[row] = _window_signature(
                    optimized_arrivals[offset], intervals_by_candidate[row_candidates[row]],
                )
        for local_candidate in dict.fromkeys(row_candidates):
            candidate_rows = sorted(
                (row for row, candidate in enumerate(row_candidates)
                 if candidate == local_candidate and math.isfinite(values[row])),
                key=lambda row: (values[row], row),
            )
            unique = {signatures[row] for row in candidate_rows if -1 not in signatures[row]}
            stats["optimized_unique_window_sequences"] += len(unique)
            stats["optimized_paths_with_multiple_window_sequences"] += int(len(unique) > 1)
            stats["optimized_outside_window_branches"] += sum(
                -1 in signatures[row] for row in candidate_rows
            )
            if not candidate_rows:
                continue
            winning = candidate_rows[0]
            if return_branches:
                branch_catalog[chunk_start + local_candidate] = [
                    {"parameters": best_parameters[row].detach().clone(),
                     "objective": values[row], "window_signature": signatures[row]}
                    for row in candidate_rows
                ]
            if not torch.equal(best_parameters[winning], originals[local_candidate]):
                output[chunk_start + local_candidate] = best_parameters[winning]
                stats["changed"] += 1
    if fit_error_count:
        stats["dp_fit_abs_error_s_mean"] = fit_error_sum / fit_error_count
        stats["dp_fit_abs_error_s_max"] = fit_error_max
    stats.update(profiler.finish(names))
    return (output, stats, branch_catalog) if return_branches else (output, stats)


def validate_alternative_branches(catalog, selected: torch.Tensor,
                                  selected_valid: torch.Tensor, validate,
                                  *, budget: int, batch_size: int = 32,
                                  ineligible: torch.Tensor = None):
    """Try the next lowest-cost timing only after a candidate's winner fails.

    ``validate`` receives candidate indices and matching timing rows and must
    perform the same exact DenseCheck used for the final selected trajectories.
    A bounded batch checks at most one alternative per spatial path per round.
    """
    if budget < 0 or batch_size < 1:
        raise ValueError("alternative DenseCheck budget/batch size is invalid")
    output = selected.detach().clone()
    valid = selected_valid.detach().cpu().tolist()
    excluded = (ineligible.detach().cpu().tolist() if ineligible is not None
                else [False] * len(selected))
    next_row = {candidate: 1 for candidate, options in catalog.items()
                if not valid[candidate] and not excluded[candidate] and len(options) > 1}
    checked = batches = rescued = 0
    accepted_signatures = {
        (candidate, options[0]["window_signature"])
        for candidate, options in catalog.items()
        if valid[candidate] and options and -1 not in options[0]["window_signature"]
    }
    while next_row and checked < budget:
        current = []
        for candidate in list(next_row):
            options = catalog[candidate]
            index = next_row[candidate]
            while index < len(options) and any(
                    torch.equal(options[index]["parameters"], prior["parameters"])
                    for prior in options[:index]):
                index += 1
            if index >= len(options):
                del next_row[candidate]
                continue
            next_row[candidate] = index
            current.append((candidate, index))
            if len(current) >= min(batch_size, budget - checked):
                break
        if not current:
            break
        candidate_indices = torch.as_tensor(
            [candidate for candidate, _ in current], device=selected.device, dtype=torch.long,
        )
        parameters = torch.stack([catalog[candidate][index]["parameters"]
                                  for candidate, index in current])
        accepted = validate(candidate_indices, parameters).detach().cpu().tolist()
        if len(accepted) != len(current):
            raise ValueError("alternative DenseCheck returned the wrong batch size")
        checked += len(current)
        batches += 1
        for offset, ((candidate, index), passed) in enumerate(zip(current, accepted)):
            if passed:
                output[candidate] = parameters[offset]
                valid[candidate] = True
                rescued += 1
                signature = catalog[candidate][index]["window_signature"]
                if -1 not in signature:
                    accepted_signatures.add((candidate, signature))
                del next_row[candidate]
            else:
                next_row[candidate] = index + 1
    stats = {"alternative_branches_checked": checked,
             "alternative_dense_batches": batches,
             "alternate_branch_rescued_candidates": rescued,
             "validated_unique_window_sequences": len(accepted_signatures)}
    return output, stats


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
    if not factorized and _nonzero_endpoint_derivatives(guide):
        stats["skipped_nonzero_endpoint_derivatives"] = True
        stats.update(profiler.finish(stage_names))
        return output, stats
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
