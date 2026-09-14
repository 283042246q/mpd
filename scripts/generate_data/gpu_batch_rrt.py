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
        seed=0,
    ):
        if batch_size < 1 or max_iterations < 1:
            raise ValueError("batch_size and max_iterations must be positive")
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
        self.generator = torch.Generator(device=self.joint_low.device)
        self.generator.manual_seed(int(seed))

    def _nearest(self, tree, targets):
        nodes = tree.states[: tree.count]
        # Avoid one large temporary once a difficult query has a large tree.
        best_distance = torch.full(
            (len(targets),), torch.inf, dtype=targets.dtype, device=targets.device
        )
        best_index = torch.zeros(len(targets), dtype=torch.long, device=targets.device)
        for begin in range(0, len(nodes), 4096):
            distance = torch.cdist(targets, nodes[begin : begin + 4096])
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
