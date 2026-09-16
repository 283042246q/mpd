"""GPU-batched bidirectional RRT for fixed-base joint-space planning.

The planner deliberately knows nothing about Marvin or PyBullet.  Its caller
provides a batched collision predicate over full robot states and the active
joint indices for a query.  This keeps inactive joints bitwise frozen for
single-arm requests while allowing the same implementation to plan 14-D dual
requests.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch


@dataclass
class GpuBatchRRTResult:
    path: np.ndarray | None
    iterations: int
    sampled_edges: int
    checked_states: int


class _Tree:
    def __init__(self, root, capacity):
        self.states = torch.empty(
            (capacity, root.numel()), dtype=root.dtype, device=root.device
        )
        self.states[0] = root
        self.parents = np.full(capacity, -1, dtype=np.int64)
        self.count = 1

    def add(self, states, parents):
        size = len(states)
        if self.count + size > len(self.states):
            raise RuntimeError("GPU batch RRT tree capacity exhausted")
        indices = np.arange(self.count, self.count + size, dtype=np.int64)
        self.states[self.count : self.count + size] = states
        self.parents[indices] = np.asarray(parents, dtype=np.int64)
        self.count += size
        return indices

    def trace(self, index):
        indices = []
        while index >= 0:
            indices.append(index)
            index = int(self.parents[index])
        indices.reverse()
        return self.states[torch.as_tensor(indices, device=self.states.device)]


class _BatchedTree:
    """Dense, independently-sized trees for a homogeneous query batch."""

    def __init__(self, roots, capacity):
        if roots.ndim != 2:
            raise ValueError("batched tree roots must have shape [queries, dof]")
        self.states = torch.zeros(
            (len(roots), capacity, roots.shape[1]),
            dtype=roots.dtype,
            device=roots.device,
        )
        self.states[:, 0] = roots
        self.parents = torch.full(
            (len(roots), capacity), -1, dtype=torch.long, device=roots.device
        )
        self.counts = torch.ones(len(roots), dtype=torch.long, device=roots.device)

    def nearest(self, targets, chunk_size=4096):
        if targets.ndim != 3 or targets.shape[0] != len(self.states):
            raise ValueError("targets must have shape [queries, samples, dof]")
        max_count = int(self.counts.max().item())
        best_distance = torch.full(
            targets.shape[:2], torch.inf, dtype=targets.dtype, device=targets.device
        )
        best_index = torch.zeros(targets.shape[:2], dtype=torch.long, device=targets.device)
        for begin in range(0, max_count, int(chunk_size)):
            end = min(begin + int(chunk_size), max_count)
            distance = torch.cdist(targets, self.states[:, begin:end])
            node_index = torch.arange(begin, end, device=targets.device)
            distance = distance.masked_fill(
                node_index[None, None] >= self.counts[:, None, None], torch.inf
            )
            value, index = distance.min(dim=2)
            update = value < best_distance
            best_distance = torch.where(update, value, best_distance)
            best_index = torch.where(update, index + begin, best_index)
        return best_index, best_distance

    def gather(self, indices):
        query = torch.arange(len(self.states), device=self.states.device)[:, None]
        return self.states[query, indices]

    def add(self, states, parents, accepted):
        if states.shape[:2] != parents.shape or parents.shape != accepted.shape:
            raise ValueError("batched tree insertion shapes do not match")
        additions = accepted.sum(dim=1)
        if bool((self.counts + additions > self.states.shape[1]).any().item()):
            raise RuntimeError("GPU multi-query RRT tree capacity exhausted")
        offsets = torch.cumsum(accepted.long(), dim=1) - 1
        positions = self.counts[:, None] + offsets
        node_indices = torch.full_like(parents, -1)
        query, sample = torch.nonzero(accepted, as_tuple=True)
        if len(query):
            destination = positions[query, sample]
            self.states[query, destination] = states[query, sample]
            self.parents[query, destination] = parents[query, sample]
            node_indices[query, sample] = destination
        self.counts += additions
        return node_indices

    def trace(self, query, index):
        indices = []
        while index >= 0:
            indices.append(index)
            index = int(self.parents[query, index].item())
        indices.reverse()
        return self.states[
            query, torch.as_tensor(indices, dtype=torch.long, device=self.states.device)
        ]


class GpuBatchRRTConnect:
    """Grow two trees with batched samples and batched edge collision checks.

    Each iteration expands ``batch_size`` independent nearest-neighbour edges.
    It then expands the opposite tree towards all accepted endpoints in one
    more batch.  The algorithm is intentionally conservative: every returned
    edge was checked at ``collision_step`` or finer, and the production caller
    still performs its independent dense Torch + PyBullet path audit.
    """

    def __init__(
        self,
        collision_fn,
        joint_low,
        joint_high,
        *,
        batch_size=64,
        max_iterations=512,
        extension_range=0.35,
        collision_step=0.025,
        goal_bias=0.125,
        nearest_chunk_size=4096,
        seed=0,
    ):
        if batch_size < 1 or max_iterations < 1 or nearest_chunk_size < 1:
            raise ValueError(
                "batch_size, max_iterations and nearest_chunk_size must be positive"
            )
        if extension_range <= 0 or collision_step <= 0:
            raise ValueError("extension_range and collision_step must be positive")
        if not 0 <= goal_bias <= 1:
            raise ValueError("goal_bias must lie in [0, 1]")
        self.collision_fn = collision_fn
        self.joint_low = torch.as_tensor(joint_low)
        self.joint_high = torch.as_tensor(
            joint_high, dtype=self.joint_low.dtype, device=self.joint_low.device
        )
        self.batch_size = int(batch_size)
        self.max_iterations = int(max_iterations)
        self.extension_range = float(extension_range)
        self.collision_step = float(collision_step)
        self.goal_bias = float(goal_bias)
        self.nearest_chunk_size = int(nearest_chunk_size)
        self.generator = torch.Generator(device=self.joint_low.device)
        self.generator.manual_seed(int(seed))

    def _nearest(self, tree, targets):
        nodes = tree.states[: tree.count]
        # Avoid one large temporary once a difficult query has a large tree.
        best_distance = torch.full(
            (len(targets),), torch.inf, dtype=targets.dtype, device=targets.device
        )
        best_index = torch.zeros(len(targets), dtype=torch.long, device=targets.device)
        for begin in range(0, len(nodes), self.nearest_chunk_size):
            distance = torch.cdist(
                targets, nodes[begin : begin + self.nearest_chunk_size]
            )
            value, index = distance.min(dim=1)
            update = value < best_distance
            best_distance = torch.where(update, value, best_distance)
            best_index = torch.where(update, index + begin, best_index)
        return best_index, best_distance

    def _steer(self, starts, targets, distances=None):
        if distances is None:
            distances = torch.linalg.vector_norm(targets - starts, dim=1)
        scale = torch.clamp(
            self.extension_range / distances.clamp_min(torch.finfo(starts.dtype).eps),
            max=1.0,
        )
        return starts + (targets - starts) * scale[:, None]

    def _expand_full_states(self, active_states, active_indices, frozen_state):
        full = frozen_state.expand(len(active_states), -1).clone()
        full[:, active_indices] = active_states
        return full

    def _edge_prefix(self, starts, goals, active_indices, frozen_state):
        """Return the last valid point on each edge and whether it reached goal."""
        delta = goals - starts
        steps = torch.ceil(delta.abs().amax(dim=1) / self.collision_step).long().clamp_min(1)
        max_steps = int(steps.max().item())
        sample_index = torch.arange(
            1, max_steps + 1, dtype=starts.dtype, device=starts.device
        )
        fractions = torch.minimum(
            sample_index[None] / steps[:, None],
            torch.ones(1, dtype=starts.dtype, device=starts.device),
        )
        active = starts[:, None] + fractions[..., None] * delta[:, None]
        full = self._expand_full_states(
            active.reshape(-1, active.shape[-1]), active_indices, frozen_state
        )
        collision = self.collision_fn(full).reshape(len(starts), max_steps)
        relevant = sample_index[None] <= steps[:, None]
        blocked = collision & relevant
        sentinel = torch.full_like(sample_index, max_steps)
        first_blocked = torch.where(blocked, sample_index[None] - 1, sentinel[None]).amin(dim=1)
        safe_steps = torch.minimum(first_blocked, steps)
        accepted = safe_steps > 0
        reached = accepted & (safe_steps == steps)
        safe_fraction = safe_steps.to(starts.dtype) / steps.to(starts.dtype)
        last = starts + safe_fraction[:, None] * delta
        return last, accepted, reached, int(relevant.sum().item())

    def _sample_targets(self, other_tree):
        targets = self.joint_low + torch.rand(
            (self.batch_size, len(self.joint_low)),
            dtype=self.joint_low.dtype,
            device=self.joint_low.device,
            generator=self.generator,
        ) * (self.joint_high - self.joint_low)
        biased = max(1, int(round(self.batch_size * self.goal_bias))) if self.goal_bias else 0
        if biased:
            indices = torch.randint(
                other_tree.count,
                (biased,),
                device=self.joint_low.device,
                generator=self.generator,
            )
            targets[:biased] = other_tree.states[indices]
        return targets

    def plan(self, q_start, q_goal, active_indices, allowed_time):
        device = self.joint_low.device
        dtype = self.joint_low.dtype
        q_start = torch.as_tensor(q_start, dtype=dtype, device=device)
        q_goal = torch.as_tensor(q_goal, dtype=dtype, device=device)
        active_indices = torch.as_tensor(active_indices, dtype=torch.long, device=device)
        if q_start.shape != q_goal.shape or q_start.ndim != 1:
            raise ValueError("q_start and q_goal must be matching full-state vectors")
        if len(active_indices) != len(self.joint_low):
            raise ValueError("active index count must match planner bounds")

        if bool(self.collision_fn(torch.stack((q_start, q_goal))).any().item()):
            return GpuBatchRRTResult(None, 0, 0, 2)

        # Each iteration can add at most one batch to each tree.
        capacity = 1 + self.max_iterations * self.batch_size
        start_tree = _Tree(q_start[active_indices], capacity)
        goal_tree = _Tree(q_goal[active_indices], capacity)
        sampled_edges = 0
        checked_states = 2
        deadline = time.perf_counter() + float(allowed_time)

        for iteration in range(1, self.max_iterations + 1):
            if time.perf_counter() >= deadline:
                break
            active_is_start = iteration % 2 == 1
            tree_a, tree_b = (
                (start_tree, goal_tree) if active_is_start else (goal_tree, start_tree)
            )
            targets = self._sample_targets(tree_b)
            parent_a_t, distance_a = self._nearest(tree_a, targets)
            starts_a = tree_a.states[parent_a_t]
            goals_a = self._steer(starts_a, targets, distance_a)
            last_a, accepted_a, _, checked = self._edge_prefix(
                starts_a, goals_a, active_indices, q_start
            )
            sampled_edges += len(targets)
            checked_states += checked
            keep_a = torch.nonzero(accepted_a, as_tuple=False).flatten()
            if not len(keep_a):
                continue
            states_a = last_a[keep_a]
            parents_a = parent_a_t[keep_a].detach().cpu().numpy()
            indices_a = tree_a.add(states_a, parents_a)

            parent_b_t, distance_b = self._nearest(tree_b, states_a)
            starts_b = tree_b.states[parent_b_t]
            goals_b = self._steer(starts_b, states_a, distance_b)
            last_b, accepted_b, reached_b_step, checked = self._edge_prefix(
                starts_b, goals_b, active_indices, q_start
            )
            sampled_edges += len(states_a)
            checked_states += checked
            keep_b = torch.nonzero(accepted_b, as_tuple=False).flatten()
            if not len(keep_b):
                continue
            states_b = last_b[keep_b]
            parents_b = parent_b_t[keep_b].detach().cpu().numpy()
            indices_b = tree_b.add(states_b, parents_b)

            reached = reached_b_step[keep_b] & (
                distance_b[keep_b] <= self.extension_range + 1e-6
            )
            connected = torch.nonzero(reached, as_tuple=False).flatten()
            if not len(connected):
                continue
            local = int(connected[0].item())
            original = int(keep_b[local].item())
            node_a = int(indices_a[original])
            node_b = int(indices_b[local])
            if active_is_start:
                start_node, goal_node = node_a, node_b
            else:
                start_node, goal_node = node_b, node_a
            start_path = start_tree.trace(start_node)
            goal_path = goal_tree.trace(goal_node)
            active_path = torch.cat((start_path, goal_path[:-1].flip(0)), dim=0)
            full_path = self._expand_full_states(active_path, active_indices, q_start)
            return GpuBatchRRTResult(
                full_path.detach().cpu().numpy(),
                iteration,
                sampled_edges,
                checked_states,
            )

        return GpuBatchRRTResult(
            None, iteration if "iteration" in locals() else 0, sampled_edges, checked_states
        )


class GpuMultiQueryRRTConnect(GpuBatchRRTConnect):
    """Plan several homogeneous requests in parallel on one device.

    ``query_batch_size`` and the inherited ``batch_size`` are independent:
    the former is the number of trajectories, while the latter is the number
    of candidate tree edges expanded per trajectory and iteration.  One call
    must use a common active-joint mapping, so dual, left-only and right-only
    requests are deliberately placed in separate scheduler buckets.
    """

    def _expand_full_states_batch(self, active_states, active_indices, frozen_states):
        full = frozen_states[:, None].expand(
            -1, active_states.shape[1], -1
        ).clone()
        full[:, :, active_indices] = active_states
        return full

    def _edge_prefix_batch(
        self, starts, goals, active_indices, frozen_states, enabled
    ):
        """Vectorized edge-prefix checks for ``[query, candidate, dof]``."""
        delta = goals - starts
        steps = (
            torch.ceil(delta.abs().amax(dim=2) / self.collision_step)
            .long()
            .clamp_min(1)
        )
        max_steps = int(steps.max().item())
        sample_index = torch.arange(
            1, max_steps + 1, dtype=starts.dtype, device=starts.device
        )
        fractions = torch.minimum(
            sample_index[None, None] / steps[:, :, None],
            torch.ones(1, dtype=starts.dtype, device=starts.device),
        )
        active = starts[:, :, None] + fractions[..., None] * delta[:, :, None]
        relevant = sample_index[None, None] <= steps[:, :, None]
        relevant &= enabled[:, :, None]
        # Do not run the expensive 1,035-sphere predicate for padded steps or
        # queries that already finished. Only the compact enabled state list
        # reaches the collision backend.
        query_index = torch.arange(
            len(starts), device=starts.device
        )[:, None, None].expand_as(relevant)[relevant]
        full = frozen_states[query_index].clone()
        full[:, active_indices] = active[relevant]
        collision = torch.zeros_like(relevant)
        if len(full):
            collision[relevant] = self.collision_fn(full)
        blocked = collision & relevant
        sentinel = torch.full_like(sample_index, max_steps)
        first_blocked = torch.where(
            blocked, sample_index[None, None] - 1, sentinel[None, None]
        ).amin(dim=2)
        safe_steps = torch.minimum(first_blocked, steps)
        accepted = enabled & (safe_steps > 0)
        reached = accepted & (safe_steps == steps)
        safe_fraction = safe_steps.to(starts.dtype) / steps.to(starts.dtype)
        last = starts + safe_fraction[:, :, None] * delta
        checked_per_query = relevant.sum(dim=(1, 2)).detach().cpu().numpy()
        return last, accepted, reached, checked_per_query

    def _sample_targets_batch(self, other_tree, generators):
        query_count = len(other_tree.states)
        targets = torch.empty(
            (query_count, self.batch_size, len(self.joint_low)),
            dtype=self.joint_low.dtype,
            device=self.joint_low.device,
        )
        biased = (
            max(1, int(round(self.batch_size * self.goal_bias)))
            if self.goal_bias
            else 0
        )
        for query, generator in enumerate(generators):
            targets[query] = self.joint_low + torch.rand(
                (self.batch_size, len(self.joint_low)),
                dtype=self.joint_low.dtype,
                device=self.joint_low.device,
                generator=generator,
            ) * (self.joint_high - self.joint_low)
            if biased:
                indices = torch.randint(
                    int(other_tree.counts[query].item()),
                    (biased,),
                    device=self.joint_low.device,
                    generator=generator,
                )
                targets[query, :biased] = other_tree.states[query, indices]
        return targets

    def plan_batch(
        self,
        q_starts,
        q_goals,
        active_indices,
        allowed_time,
        *,
        query_seeds=None,
    ):
        """Return one :class:`GpuBatchRRTResult` per input query."""
        device = self.joint_low.device
        dtype = self.joint_low.dtype
        q_starts = torch.as_tensor(q_starts, dtype=dtype, device=device)
        q_goals = torch.as_tensor(q_goals, dtype=dtype, device=device)
        active_indices = torch.as_tensor(
            active_indices, dtype=torch.long, device=device
        )
        if q_starts.ndim != 2 or q_starts.shape != q_goals.shape:
            raise ValueError("q_starts and q_goals must have shape [queries, full_dof]")
        if not len(q_starts):
            return []
        if len(active_indices) != len(self.joint_low):
            raise ValueError("active index count must match planner bounds")
        query_count = len(q_starts)
        if query_seeds is None:
            query_seeds = [self.generator.initial_seed() + i for i in range(query_count)]
        if len(query_seeds) != query_count:
            raise ValueError("query_seeds must contain one seed per query")
        generators = []
        for seed in query_seeds:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            generators.append(generator)

        # The inactive arm is a start-state constraint, not merely an equal
        # endpoint. Normalize it before checking goal validity or growing the
        # goal tree so arbitrary inactive values can never leak into a path.
        normalized_goals = q_starts.clone()
        normalized_goals[:, active_indices] = q_goals[:, active_indices]
        endpoints = torch.stack((q_starts, normalized_goals), dim=1)
        endpoint_collision = self.collision_fn(
            endpoints.reshape(-1, endpoints.shape[-1])
        ).reshape(query_count, 2)
        eligible = ~endpoint_collision.any(dim=1)

        capacity = 1 + self.max_iterations * self.batch_size
        start_tree = _BatchedTree(q_starts[:, active_indices], capacity)
        goal_tree = _BatchedTree(normalized_goals[:, active_indices], capacity)
        solved = torch.zeros(query_count, dtype=torch.bool, device=device)
        start_nodes = torch.full(
            (query_count,), -1, dtype=torch.long, device=device
        )
        goal_nodes = torch.full_like(start_nodes, -1)
        iterations = np.zeros(query_count, dtype=np.int64)
        sampled_edges = np.zeros(query_count, dtype=np.int64)
        checked_states = np.full(query_count, 2, dtype=np.int64)
        deadline = time.perf_counter() + float(allowed_time)

        for iteration in range(1, self.max_iterations + 1):
            active_queries = eligible & ~solved
            if not bool(active_queries.any().item()) or time.perf_counter() >= deadline:
                break
            active_np = active_queries.detach().cpu().numpy()
            iterations[active_np] = iteration
            active_is_start = iteration % 2 == 1
            tree_a, tree_b = (
                (start_tree, goal_tree)
                if active_is_start
                else (goal_tree, start_tree)
            )
            targets = self._sample_targets_batch(tree_b, generators)
            parent_a, distance_a = tree_a.nearest(
                targets, chunk_size=self.nearest_chunk_size
            )
            starts_a = tree_a.gather(parent_a)
            goals_a = self._steer(
                starts_a.reshape(-1, starts_a.shape[-1]),
                targets.reshape(-1, targets.shape[-1]),
                distance_a.reshape(-1),
            ).reshape_as(starts_a)
            enabled_a = active_queries[:, None].expand(-1, self.batch_size)
            last_a, accepted_a, _, checked = self._edge_prefix_batch(
                starts_a,
                goals_a,
                active_indices,
                q_starts,
                enabled_a,
            )
            sampled_edges[active_np] += self.batch_size
            checked_states += checked.astype(np.int64, copy=False)
            indices_a = tree_a.add(last_a, parent_a, accepted_a)

            parent_b, distance_b = tree_b.nearest(
                last_a, chunk_size=self.nearest_chunk_size
            )
            starts_b = tree_b.gather(parent_b)
            goals_b = self._steer(
                starts_b.reshape(-1, starts_b.shape[-1]),
                last_a.reshape(-1, last_a.shape[-1]),
                distance_b.reshape(-1),
            ).reshape_as(starts_b)
            last_b, accepted_b, reached_b, checked = self._edge_prefix_batch(
                starts_b,
                goals_b,
                active_indices,
                q_starts,
                accepted_a,
            )
            sampled_edges += accepted_a.sum(dim=1).detach().cpu().numpy()
            checked_states += checked.astype(np.int64, copy=False)
            indices_b = tree_b.add(last_b, parent_b, accepted_b)
            connected = (
                reached_b
                & accepted_b
                & (distance_b <= self.extension_range + 1e-6)
            )
            for query in torch.nonzero(
                connected.any(dim=1) & ~solved, as_tuple=False
            ).flatten().tolist():
                candidate = int(
                    torch.nonzero(connected[query], as_tuple=False)[0].item()
                )
                node_a = indices_a[query, candidate]
                node_b = indices_b[query, candidate]
                if active_is_start:
                    start_nodes[query], goal_nodes[query] = node_a, node_b
                else:
                    start_nodes[query], goal_nodes[query] = node_b, node_a
                solved[query] = True

        results = []
        for query in range(query_count):
            path = None
            if bool(solved[query].item()):
                start_path = start_tree.trace(query, int(start_nodes[query].item()))
                goal_path = goal_tree.trace(query, int(goal_nodes[query].item()))
                active_path = torch.cat(
                    (start_path, goal_path[:-1].flip(0)), dim=0
                )
                full_path = q_starts[query].expand(len(active_path), -1).clone()
                full_path[:, active_indices] = active_path
                path = full_path.detach().cpu().numpy()
            results.append(
                GpuBatchRRTResult(
                    path,
                    int(iterations[query]),
                    int(sampled_edges[query]),
                    int(checked_states[query]),
                )
            )
        return results
