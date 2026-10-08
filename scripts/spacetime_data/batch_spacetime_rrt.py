"""GPU multi-query, directed spacetime RRT-Connect with an arrival-time forest.

Reuses the existing batched tree storage without changing its API. This is an
experimental ST-pRRTC-inspired teacher, NOT an OMPL STRRTstar implementation or
a claim of asymptotic optimality. The exact algorithm is a separate backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import torch

from scripts.generate_data.gpu_batch_rrt import _BatchedTree


@dataclass
class SpaceTimeResult:
    paths: list = field(default_factory=list)  # list of [N,dof+1], last column time
    statistics: dict = field(default_factory=dict)


def edge_collision_mask(starts, goals, query_ids, collision_fn, *,
                        joint_step=0.04, time_step=0.02, enabled=None):
    """Check complete linear q-t edges, including waits and backward growth.

    Physical forward direction / velocity is checked by the caller. Collision
    interpolation works in either direction and includes both endpoints.
    """
    shape = starts.shape[:-1]
    if starts.shape != goals.shape or query_ids.shape != shape:
        raise ValueError("edge and query shapes do not match")
    if joint_step <= 0 or time_step <= 0:
        raise ValueError("edge resolutions must be positive")
    if enabled is None:
        enabled = torch.ones(shape, dtype=torch.bool, device=starts.device)
    finite = torch.isfinite(starts).all(dim=-1) & torch.isfinite(goals).all(dim=-1)
    enabled = enabled & finite
    delta = torch.nan_to_num(goals - starts)
    steps = torch.maximum(torch.ceil(delta[..., :-1].abs().amax(dim=-1) / joint_step),
                          torch.ceil(delta[..., -1].abs() / time_step)).long().clamp_min(1)
    # Disabled lanes must not inflate allocation (e.g. unreachable NN inf).
    steps = torch.where(enabled, steps, torch.ones_like(steps))
    max_steps = int(steps.max().item())
    sample = torch.arange(max_steps + 1, device=starts.device)
    relevant = enabled[..., None] & (sample <= steps[..., None])
    alpha = (sample / steps[..., None]).clamp_max(1)
    states = starts[..., None, :] + alpha[..., None] * delta[..., None, :]
    ids = query_ids[..., None].expand_as(relevant)
    colliding = torch.zeros_like(relevant)
    if bool(relevant.any().item()):
        selected = states[relevant]
        colliding[relevant] = collision_fn(selected[:, :-1], selected[:, -1], ids[relevant])
    return ~enabled | colliding.any(dim=-1), int(relevant.sum().item())


class TorchSpaceTimeForest:
    """One shared forward tree and several backward arrival roots per query."""

    def __init__(self, collision_fn, low, high, velocity, *, edges_per_query=16,
                 roots=4, max_iterations=512, extension_range=0.4,
                 time_extension=0.65, joint_step=0.04, time_step=0.02,
                 nearest_chunk=512, recycle=True, max_solutions=3):
        self.collision_fn = collision_fn
        self.low, self.high, self.velocity = low, high, velocity
        self.edges, self.roots, self.max_iterations = int(edges_per_query), int(roots), int(max_iterations)
        self.extension_range, self.time_extension = float(extension_range), float(time_extension)
        self.joint_step, self.time_step = float(joint_step), float(time_step)
        self.nearest_chunk, self.recycle, self.max_solutions = int(nearest_chunk), bool(recycle), int(max_solutions)
        if min(self.edges, self.roots, self.max_iterations, self.nearest_chunk, self.max_solutions) < 1:
            raise ValueError("batch, roots, iterations, chunks and solutions must be positive")
        if min(self.extension_range, self.time_extension, self.joint_step, self.time_step) <= 0:
            raise ValueError("ranges and resolutions must be positive")
        if low.ndim != 1 or high.shape != low.shape or velocity.shape != low.shape:
            raise ValueError("bounds and velocity must be matching vectors")
        if not bool(((high > low) & (velocity > 0)).all().item()):
            raise ValueError("invalid bounds/velocity")

    def _travel(self, delta):
        return (delta.abs() / self.velocity).amax(dim=-1)

    def _nearest(self, tree, targets, direction, root_times=None, upper=None):
        batch, edges = targets.shape[:2]
        best = torch.full((batch, edges), torch.inf, device=targets.device)
        indices = torch.zeros((batch, edges), dtype=torch.long, device=targets.device)
        count = int(tree.counts.max().item())
        for begin in range(0, count, self.nearest_chunk):
            end = min(count, begin + self.nearest_chunk)
            nodes = tree.states[:, begin:end]
            delta_q = targets[:, :, None, :-1] - nodes[:, None, :, :-1]
            delta_t = direction * (targets[:, :, None, -1] - nodes[:, None, :, -1])
            feasible = (delta_t > 1e-5) & (delta_t + 1e-6 >= self._travel(delta_q))
            node_ids = torch.arange(begin, end, device=targets.device)
            feasible &= node_ids[None, None] < tree.counts[:, None, None]
            if root_times is not None:
                feasible &= root_times[:, None, begin:end] < upper[:, None, None] - 1e-5
            # q and seconds get explicit scales; unreachable pairs never rank.
            cost = torch.linalg.vector_norm(delta_q, dim=-1) + 0.15 * delta_t
            cost = cost.masked_fill(~feasible, torch.inf)
            values, local = cost.min(dim=-1)
            better = values < best
            indices = torch.where(better, local + begin, indices)
            best = torch.minimum(best, values)
        return indices, torch.isfinite(best)

    def _steer(self, starts, targets):
        delta = targets - starts
        spatial = torch.linalg.vector_norm(delta[..., :-1], dim=-1).clamp_min(1e-8)
        temporal = delta[..., -1].abs().clamp_min(1e-8)
        alpha = torch.minimum(self.extension_range / spatial, self.time_extension / temporal).clamp_max(1)
        return starts + alpha[..., None] * delta

    @torch.no_grad()
    def plan_batch(self, q_starts, q_goals, horizons, earliest_arrivals, *,
                   allowed_time=5.0, seed=0, reference_paths=None):
        device, dtype = self.low.device, self.low.dtype
        starts_q = torch.as_tensor(q_starts, device=device, dtype=dtype)
        goals_q = torch.as_tensor(q_goals, device=device, dtype=dtype)
        horizons = torch.as_tensor(horizons, device=device, dtype=dtype)
        earliest = torch.as_tensor(earliest_arrivals, device=device, dtype=dtype)
        if starts_q.ndim != 2 or starts_q.shape != goals_q.shape or starts_q.shape[1] != len(self.low):
            raise ValueError("q endpoints must have shape [queries,dof]")
        batch, dof = starts_q.shape
        if horizons.shape != (batch,) or earliest.shape != (batch,):
            raise ValueError("horizon and arrival bounds must be per query")
        if allowed_time <= 0 or not bool(torch.isfinite(starts_q).all() & torch.isfinite(goals_q).all()):
            raise ValueError("invalid endpoints or budget")
        earliest = torch.maximum(earliest, self._travel(goals_q - starts_q) * 1.02 + 0.05)
        eligible = earliest < horizons - 0.02
        generator = torch.Generator(device=device).manual_seed(int(seed))
        ids = torch.arange(batch, device=device)
        root = torch.cat((starts_q, torch.zeros((batch, 1), device=device)), dim=1)
        capacity = self.roots + 2 + self.max_iterations * (self.edges + self.roots)
        forward = _BatchedTree(root, capacity)
        times = earliest[:, None] + torch.linspace(0.05, 0.98, self.roots, device=device)[None] * (horizons - earliest)[:, None]
        root_states = torch.cat((goals_q[:, None].expand(-1, self.roots, -1), times[..., None]), dim=-1)
        backward = _BatchedTree(root_states[:, 0], capacity)
        root_time = torch.full((batch, capacity), torch.inf, device=device)
        root_time[:, 0] = times[:, 0]
        if self.roots > 1:
            root_index = backward.add(root_states[:, 1:],
                                      torch.full((batch, self.roots - 1), -1, device=device, dtype=torch.long),
                                      torch.ones((batch, self.roots - 1), device=device, dtype=torch.bool))
            for query in range(batch):
                root_time[query, root_index[query]] = times[query, 1:]
        root_collision = self.collision_fn(root_states[..., :-1].reshape(-1, dof),
                                           times.reshape(-1), ids[:, None].expand_as(times).reshape(-1)).reshape(batch, self.roots)
        root_time[:, :self.roots].masked_fill_(root_collision, torch.inf)
        eligible &= ~self.collision_fn(starts_q, torch.zeros(batch, device=device), ids)
        upper = horizons + 0.001
        results = [SpaceTimeResult(statistics={"backend": "torch-spacetime-forest", "checked_states": 0,
                                             "root_recycles": 0, "improvements": 0}) for _ in range(batch)]
        reference = None if reference_paths is None else torch.as_tensor(reference_paths, device=device, dtype=dtype)
        if reference is not None and (reference.ndim != 3 or reference.shape[0] != batch or reference.shape[-1] != dof):
            raise ValueError("reference_paths must be [query,phase,dof]")
        started = time.perf_counter()
        iterations, checked_total = 0, 0
        for iteration in range(self.max_iterations):
            if time.perf_counter() - started >= allowed_time or not bool(eligible.any().item()):
                break
            iterations = iteration + 1
            direction = 1 if iteration % 2 == 0 else -1
            tree, other = (forward, backward) if direction == 1 else (backward, forward)
            q = self.low + torch.rand((batch, self.edges, dof), device=device, generator=generator) * (self.high - self.low)
            if reference is not None:
                count = max(1, self.edges // 3)
                phases = torch.randint(reference.shape[1], (batch, count), device=device, generator=generator)
                q[:, :count] = reference[ids[:, None], phases]
            # Bias one lane towards the opposite boundary; time still sampled feasibly.
            q[:, -1] = goals_q if direction == 1 else starts_q
            lower_t = self._travel(q - starts_q[:, None]) + 0.001
            upper_t = upper[:, None] - self._travel(q - goals_q[:, None]) - 0.001
            random = torch.rand((batch, self.edges), device=device, generator=generator)
            target_t = lower_t + random * (upper_t - lower_t).clamp_min(0)
            targets = torch.cat((q, target_t[..., None]), dim=-1)
            # Explicit pure-wait proposals; no high-frequency cycles needed.
            wait_parent = torch.minimum((random[:, 0] * tree.counts).long(), tree.counts - 1)
            wait_state = tree.states[ids, wait_parent].clone()
            wait_state[:, -1] += direction * (0.1 + 0.5 * random[:, 1 % self.edges])
            targets[:, 0] = wait_state
            parents, reachable = self._nearest(tree, targets, direction,
                                               root_time if direction == -1 else None,
                                               upper if direction == -1 else None)
            before = tree.gather(parents)
            proposed = self._steer(before, targets)
            enabled = eligible[:, None] & reachable & (proposed[..., -1] >= 0) & (proposed[..., -1] < upper[:, None])
            lanes = ids[:, None].expand(batch, self.edges)
            blocked, checked = edge_collision_mask(before, proposed, lanes, self.collision_fn,
                                                   joint_step=self.joint_step, time_step=self.time_step, enabled=enabled)
            checked_total += checked
            accepted = enabled & ~blocked
            added = tree.add(proposed, parents, accepted)
            if direction == -1:
                query, lane = torch.nonzero(accepted, as_tuple=True)
                root_time[query, added[query, lane]] = root_time[query, parents[query, lane]]
            # A forward node bridges to a later backward node, and vice versa.
            other_parent, bridge_reachable = self._nearest(other, proposed, -direction,
                                                           root_time if direction == 1 else None,
                                                           upper if direction == 1 else None)
            other_state = other.gather(other_parent)
            bridge_enabled = accepted & bridge_reachable
            blocked, checked = edge_collision_mask(proposed, other_state, lanes, self.collision_fn,
                                                   joint_step=self.joint_step, time_step=self.time_step,
                                                   enabled=bridge_enabled)
            checked_total += checked
            connected = bridge_enabled & ~blocked
            arrival = root_time.gather(1, other_parent if direction == 1 else added.clamp_min(0))
            arrival = arrival.masked_fill(~connected, torch.inf)
            best_arrival, best_lane = arrival.min(dim=1)
            improved = best_arrival < upper - 1e-5
            for query in torch.nonzero(improved, as_tuple=False).flatten().tolist():
                lane = int(best_lane[query].item())
                first = int((added if direction == 1 else other_parent)[query, lane].item())
                last = int((other_parent if direction == 1 else added)[query, lane].item())
                path = torch.cat((forward.trace(query, first), backward.trace(query, last).flip(0)), dim=0)
                result = results[query]
                result.paths.append(path.cpu().numpy())
                result.paths.sort(key=lambda value: value[-1, -1])
                del result.paths[self.max_solutions:]
                result.statistics.setdefault("first_solution_s", time.perf_counter() - started)
                result.statistics["improvements"] += 1
                upper[query] = best_arrival[query]
            if self.recycle and bool(improved.any().item()):
                # Append fresh earlier roots. Old later subtrees remain allocated
                # but are unreachable through the root-time active mask.
                width = (upper - earliest - 0.02).clamp_min(0)
                new_times = earliest[:, None] + torch.rand((batch, self.roots), device=device, generator=generator) * width[:, None]
                new_roots = torch.cat((goals_q[:, None].expand(-1, self.roots, -1), new_times[..., None]), dim=-1)
                enabled_roots = improved[:, None] & (width[:, None] > 0.02)
                enabled_roots = enabled_roots.expand(-1, self.roots).clone()
                valid_roots = ~self.collision_fn(new_roots[..., :-1].reshape(-1, dof), new_times.reshape(-1),
                                                ids[:, None].expand_as(new_times).reshape(-1)).reshape_as(new_times)
                accepted_roots = enabled_roots & valid_roots
                new_ids = backward.add(new_roots, torch.full_like(new_times, -1, dtype=torch.long), accepted_roots)
                query, lane = torch.nonzero(accepted_roots, as_tuple=True)
                root_time[query, new_ids[query, lane]] = new_times[query, lane]
                for query in range(batch):
                    results[query].statistics["root_recycles"] += int(accepted_roots[query].sum().item())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        for query, result in enumerate(results):
            result.statistics.update(iterations=iterations, batch_wall_s=elapsed,
                                     checked_states_batch=checked_total,
                                     forward_nodes=int(forward.counts[query].item()),
                                     backward_nodes=int(backward.counts[query].item()),
                                     seed=int(seed), arrival_lower_bound_s=float(earliest[query].item()))
        return results
