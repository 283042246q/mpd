"""Event-driven, cross-shard coordinator for Marvin GPU generation."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import time

import numpy as np

from scripts.generate_data.marvin_gpu_scheduler import (
    ActiveShardWindow,
    CandidateStage,
    ShardState,
    TaskState,
    TaskStatus,
)
from scripts.generate_data.marvin_gpu_task_contract import stable_seed


MODE_BATCH_CONFIG = {
    "dual_independent": "gpu_query_batch_size_dual",
    "left_only": "gpu_query_batch_size_left",
    "right_only": "gpu_query_batch_size_right",
}


def factor_dual_candidates(proposals, mode, maximum):
    if mode != "dual_independent":
        return list(proposals)
    values = sorted(proposals, key=lambda item: item["candidate_index"])
    result = []
    for left in values:
        for right in values:
            result.append(
                {
                    **left,
                    "q_start": np.r_[left["q_start"][:7], right["q_start"][7:]],
                    "q_goal": np.r_[left["q_goal"][:7], right["q_goal"][7:]],
                    "ee_goal_pose": np.stack(
                        (left["ee_goal_pose"][0], right["ee_goal_pose"][1])
                    ).astype(np.float32, copy=False),
                }
            )
            if len(result) >= int(maximum):
                return result
    return result


def _metadata(task, item):
    path = np.asarray(item["path"])
    return path, {
        "task_id": task.task_id,
        "task_mode": task.mode,
        "direction": task.direction,
        "q_start": np.asarray(item["q_start"]),
        "q_goal": np.asarray(item["q_goal"]),
        "planning_time": float(item["batch_seconds"]),
        "bspline": item["spline"],
        "ee_goal_pose": np.asarray(item["ee_goal_pose"], dtype=np.float32),
        "joint_path_length": float(
            np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        ),
        "source_region_left": task.source["left"],
        "source_region_right": task.source["right"],
        "goal_region_left": task.goal["left"],
        "goal_region_right": task.goal["right"],
    }


class StreamingCoordinator:
    """Keep endpoint, CUDA and mesh actors busy over several atomic shards."""

    def __init__(
        self,
        config,
        specs,
        contract,
        endpoint_actors,
        gpu_actor,
        bullet_actor,
        publish,
        spool=None,
    ):
        self.config = dict(config)
        self.contract = contract
        self.endpoint_actors = list(endpoint_actors)
        self.gpu_actor = gpu_actor
        self.bullet_actor = bullet_actor
        self.publish = publish
        self.spool = spool
        self.base_seed = int(config["seed"])
        self.window = ActiveShardWindow(
            specs,
            int(config.get("gpu_pipeline_active_window_tasks", 80)),
            int(config.get("gpu_pipeline_max_open_shards", 8)),
        )
        self.endpoint_requests = {}
        self.gpu_request = None
        self.bullet_request = None
        self.completed_paths = []
        self.total_accepted = 0
        self.published_since_recycle = 0
        self.global_stats = Counter()
        self.idle_sleep = float(config.get("gpu_pipeline_coordinator_idle_sleep", 0.001))
        self.plan_wait = float(config.get("gpu_pipeline_plan_batch_max_wait_seconds", 1.0))

    def _open_shard(self, start, size, path):
        tasks = {
            task_id: TaskState(self.contract.build(task_id), start)
            for task_id in range(start, start + size)
        }
        shard = ShardState(start, size, path, tasks)
        if self.spool is not None:
            restored = self.spool.restore(shard)
            self.total_accepted += len(restored)
        return shard

    def _task(self, task_id):
        for shard in self.window.open.values():
            if task_id in shard.tasks:
                return shard.tasks[task_id]
        raise KeyError(task_id)

    def _shard(self, task_id):
        return self.window.open[self._task(task_id).shard_start]

    def _count(self, task_id, key, value=1):
        self._shard(task_id).stats[key] += value
        self.global_stats[key] += value

    def _record_busy(self, items, key, seconds):
        if not items:
            return
        groups = Counter(self._task(item["task_id"]).shard_start for item in items)
        total = len(items)
        for start, count in groups.items():
            self.window.open[start].stats[key] += round(1000 * seconds * count / total)
        self.global_stats[key] += seconds

    def _endpoint_eligible(self):
        active = [
            task
            for task in self.window.active_tasks()
            if task.status in {TaskStatus.ACTIVE, TaskStatus.DEFERRED}
            and not task.endpoint_inflight
            and not task.live_candidates()
        ]
        return sorted(
            active,
            key=lambda task: (
                task.status == TaskStatus.DEFERRED,
                task.attempt,
                task.task_id,
            ),
        )

    def _dispatch_endpoints(self):
        progressed = False
        eligible = iter(self._endpoint_eligible())
        hard_limit = int(self.config.get("gpu_pipeline_max_attempts_per_task", 100))
        count = int(self.config.get("gpu_pipeline_endpoint_candidates_per_attempt", 8))
        chunk_size = int(self.config.get("gpu_pipeline_endpoint_candidate_chunk_size", 2))
        for index, actor in enumerate(self.endpoint_actors):
            if index in self.endpoint_requests:
                continue
            for task in eligible:
                reserved = task.begin_endpoint_chunk(hard_limit, count, chunk_size)
                if reserved is None:
                    continue
                attempt, indices = reserved
                if task.endpoint_candidate_cursor == len(indices):
                    self._count(task.task_id, "task_attempts")
                job = {
                    "task": asdict(task.definition),
                    "attempt": attempt,
                    "candidate_indices": indices,
                    "candidate_seeds": [
                        stable_seed(
                            self.base_seed,
                            task.task_id,
                            attempt,
                            f"endpoint-{candidate}",
                        )
                        for candidate in indices
                    ],
                }
                message = {"op": "propose", "jobs": [job]}
                actor.send(message)
                self.endpoint_requests[index] = {
                    "task_id": task.task_id,
                    "started": time.perf_counter(),
                }
                progressed = True
                break
        return progressed

    def _poll_endpoints(self):
        progressed = False
        for index in tuple(self.endpoint_requests):
            ready, result = self.endpoint_actors[index].poll()
            if not ready:
                continue
            request = self.endpoint_requests.pop(index)
            task = self._task(request["task_id"])
            task.finish_endpoint_work()
            elapsed = time.perf_counter() - request["started"]
            self._record_busy(
                [{"task_id": task.task_id}], "endpoint_actor_busy_milliseconds", elapsed
            )
            proposals = []
            for item in result:
                if item.get("proposal") is None:
                    self._count(task.task_id, "endpoint_proposal_failures")
                else:
                    proposals.append(item)
            proposals = factor_dual_candidates(
                proposals,
                task.definition.mode,
                int(self.config.get("gpu_pipeline_dual_cross_candidates", 64)),
            )
            self._count(
                task.task_id, "endpoint_candidates_after_dual_cross", len(proposals)
            )
            for item in proposals:
                item = dict(item)
                item["candidate_index"] = task.candidate_serial
                task.add_candidate(item)
            progressed = True
        return progressed

    def _candidates(self, stage):
        result = []
        for task in self.window.active_tasks():
            if task.accepted:
                continue
            result.extend(
                candidate
                for candidate in task.candidates.values()
                if candidate.stage == stage
            )
        return sorted(result, key=lambda item: (item.queued_at, item.task_id, item.candidate_id))

    def _task_has_trajectory_inflight(self, task_id):
        task = self._task(task_id)
        if any(
            candidate.stage == CandidateStage.PYBULLET_TRAJECTORY
            for candidate in task.candidates.values()
        ):
            return True
        if self.bullet_request and self.bullet_request["op"] == "trajectories":
            if any(item["task_id"] == task_id for item in self.bullet_request["items"]):
                return True
        return False

    def _plan_batch(self):
        by_mode = defaultdict(list)
        for candidate in self._candidates(CandidateStage.PLAN):
            if self._task_has_trajectory_inflight(candidate.task_id):
                continue
            if by_mode[self._task(candidate.task_id).definition.mode] and any(
                item.task_id == candidate.task_id
                for item in by_mode[self._task(candidate.task_id).definition.mode]
            ):
                continue
            by_mode[self._task(candidate.task_id).definition.mode].append(candidate)
        choices = []
        upstream = bool(self.endpoint_requests) or bool(
            self._candidates(CandidateStage.GPU_ENDPOINT)
            or self._candidates(CandidateStage.PYBULLET_ENDPOINT)
        )
        now = time.perf_counter()
        for order, mode in enumerate(MODE_BATCH_CONFIG):
            values = by_mode[mode]
            if not values:
                continue
            size = int(self.config.get(MODE_BATCH_CONFIG[mode], 1))
            oldest = now - values[0].queued_at
            if len(values) >= size or oldest >= self.plan_wait or not upstream:
                choices.append((min(len(values), size) / size, oldest, -order, mode))
        if not choices:
            return None, []
        mode = max(choices)[-1]
        size = int(self.config.get(MODE_BATCH_CONFIG[mode], 1))
        return mode, by_mode[mode][:size]

    def _dispatch_gpu(self):
        if self.gpu_request is not None:
            return False
        endpoints = self._candidates(CandidateStage.GPU_ENDPOINT)
        if endpoints:
            size = int(self.config.get("gpu_pipeline_gpu_endpoint_batch_size", 128))
            selected = endpoints[:size]
            items = [candidate.payload for candidate in selected]
            self.gpu_actor.send({"op": "endpoints", "items": items})
            self.gpu_request = {
                "op": "endpoints",
                "candidates": selected,
                "items": items,
                "started": time.perf_counter(),
            }
            return True
        mode, selected = self._plan_batch()
        if not selected:
            return False
        items = [candidate.payload for candidate in selected]
        query_seeds = [
            stable_seed(
                self.base_seed,
                candidate.task_id,
                candidate.attempt,
                f"rrt-{candidate.candidate_id}",
            )
            for candidate in selected
        ]
        self.gpu_actor.send(
            {"op": "plan", "mode": mode, "items": items, "query_seeds": query_seeds}
        )
        self.gpu_request = {
            "op": "plan",
            "mode": mode,
            "candidates": selected,
            "items": items,
            "started": time.perf_counter(),
        }
        for candidate in selected:
            candidate.queued_at = float("inf")
        return True

    def _poll_gpu(self):
        if self.gpu_request is None:
            return False
        ready, result = self.gpu_actor.poll()
        if not ready:
            return False
        request = self.gpu_request
        self.gpu_request = None
        elapsed = time.perf_counter() - request["started"]
        key = (
            "gpu_endpoint_audit_busy_milliseconds"
            if request["op"] == "endpoints"
            else "gpu_plan_audit_busy_milliseconds"
        )
        self._record_busy(request["items"], key, elapsed)
        if request["op"] == "endpoints":
            for candidate, valid in zip(request["candidates"], result):
                if self._task(candidate.task_id).accepted:
                    candidate.stale()
                elif valid:
                    candidate.advance(CandidateStage.PYBULLET_ENDPOINT)
                else:
                    candidate.reject("gpu_endpoint")
                    self._count(candidate.task_id, "endpoint_gpu_rejected")
            return True
        mode = request["mode"]
        self.global_stats[f"gpu_plan_batches/{mode}"] += 1
        self.global_stats[f"gpu_plan_queries/{mode}"] += len(result)
        for candidate, plan in zip(request["candidates"], result):
            self._count(candidate.task_id, f"gpu_plan_queries/{mode}")
            if self._task(candidate.task_id).accepted:
                candidate.stale()
            elif plan["path"] is None:
                reason = plan["rejection_reason"]
                candidate.reject(f"gpu_{reason}")
                self._count(candidate.task_id, f"gpu_rejected/{reason}")
            else:
                candidate.payload.update(plan)
                candidate.advance(CandidateStage.PYBULLET_TRAJECTORY)
        return True

    def _dispatch_bullet(self):
        if self.bullet_request is not None:
            return False
        endpoints = self._candidates(CandidateStage.PYBULLET_ENDPOINT)
        if endpoints:
            size = int(self.config.get("gpu_pipeline_pybullet_endpoint_batch_size", 64))
            selected = endpoints[:size]
            items = [candidate.payload for candidate in selected]
            self.bullet_actor.send({"op": "endpoints", "items": items})
            self.bullet_request = {
                "op": "endpoints",
                "candidates": selected,
                "items": items,
                "started": time.perf_counter(),
            }
            return True
        trajectories = self._candidates(CandidateStage.PYBULLET_TRAJECTORY)
        if not trajectories:
            return False
        size = int(self.config.get("gpu_pipeline_pybullet_trajectory_batch_size", 4))
        selected = trajectories[:size]
        items = [candidate.payload for candidate in selected]
        self.bullet_actor.send({"op": "trajectories", "items": items})
        self.bullet_request = {
            "op": "trajectories",
            "candidates": selected,
            "items": items,
            "started": time.perf_counter(),
        }
        for candidate in selected:
            candidate.queued_at = float("inf")
        return True

    def _poll_bullet(self):
        if self.bullet_request is None:
            return False
        ready, result = self.bullet_actor.poll()
        if not ready:
            return False
        request = self.bullet_request
        self.bullet_request = None
        elapsed = time.perf_counter() - request["started"]
        key = (
            "pybullet_endpoint_audit_busy_milliseconds"
            if request["op"] == "endpoints"
            else "pybullet_trajectory_audit_busy_milliseconds"
        )
        self._record_busy(request["items"], key, elapsed)
        for candidate, valid in zip(request["candidates"], result):
            task = self._task(candidate.task_id)
            if task.accepted:
                candidate.stale()
                self._count(task.task_id, "stale_candidates_discarded")
            elif not valid:
                reason = (
                    "pybullet_endpoint"
                    if request["op"] == "endpoints"
                    else "pybullet_trajectory"
                )
                candidate.reject(reason)
                self._count(task.task_id, f"rejected/{reason}")
            elif request["op"] == "endpoints":
                candidate.advance(CandidateStage.PLAN)
            else:
                path, metadata = _metadata(task.definition, candidate.payload)
                if task.accept(path, metadata):
                    shard = self._shard(task.task_id)
                    shard.stats["accepted"] += 1
                    shard.stats[f"accepted/{task.definition.mode}"] += 1
                    shard.stats["rrt_iterations"] += int(
                        candidate.payload["rrt_iterations"]
                    )
                    shard.stats["rrt_sampled_edges"] += int(
                        candidate.payload["rrt_sampled_edges"]
                    )
                    shard.stats["rrt_checked_states"] += int(
                        candidate.payload["rrt_checked_states"]
                    )
                    if self.spool is not None:
                        self.spool.save_task(task, shard.size)
                        self.spool.save_stats(shard)
                    self.total_accepted += 1
                    print(
                        f"[gpu pipeline] accepted total={self.total_accepted} "
                        f"task={task.task_id} mode={task.definition.mode} "
                        f"attempt={task.attempt}",
                        flush=True,
                    )
        return True

    def _publish_complete(self):
        progressed = False
        recycle = int(
            self.config.get("gpu_pipeline_pybullet_recycle_trajectories", 50)
        )
        for start, shard in list(sorted(self.window.open.items())):
            if not shard.complete:
                continue
            paths, metadata = shard.ordered_results()
            shard.stats["checkpoint_wall_milliseconds"] = round(
                1000 * (time.perf_counter() - shard.opened_at)
            )
            shard.stats["endpoint_actor_restarts"] = sum(
                actor.restart_count for actor in self.endpoint_actors
            )
            shard.stats["gpu_actor_restarts"] = self.gpu_actor.restart_count
            shard.stats["pybullet_actor_restarts"] = self.bullet_actor.restart_count
            self.publish(
                shard.path, self.config, shard.start, paths, metadata, shard.stats
            )
            if self.spool is not None:
                self.spool.clear_shard(shard.start)
            self.completed_paths.append(shard.path)
            self.published_since_recycle += shard.size
            self.window.remove(start)
            progressed = True
        if (
            recycle > 0
            and self.published_since_recycle >= recycle
            and self.bullet_request is None
        ):
            self.bullet_actor.recycle()
            self.published_since_recycle %= recycle
        return progressed

    def _hard_failures(self):
        return [
            task.task_id
            for task in self.window.active_tasks()
            if task.status == TaskStatus.HARD_FAILED
        ]

    def run(self):
        self.window.fill(self._open_shard)
        while not self.window.finished:
            progressed = False
            progressed |= self._poll_endpoints()
            progressed |= self._poll_gpu()
            progressed |= self._poll_bullet()
            progressed |= self._publish_complete()
            self.window.fill(self._open_shard)
            progressed |= self._dispatch_endpoints()
            progressed |= self._dispatch_gpu()
            progressed |= self._dispatch_bullet()
            failures = self._hard_failures()
            productive = any(
                task.status in {TaskStatus.ACTIVE, TaskStatus.DEFERRED}
                or task.live_candidates()
                or task.endpoint_inflight
                for task in self.window.active_tasks()
            )
            if failures and not productive:
                raise RuntimeError(f"GPU pipeline hard-failed task(s): {failures}")
            if not progressed:
                time.sleep(self.idle_sleep)
        return self.completed_paths, self.global_stats
