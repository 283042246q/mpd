"""Pure-Python scheduling state for the streaming Marvin GPU pipeline."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time

from scripts.generate_data.marvin_gpu_task_contract import TaskDefinition


class TaskStatus(str, Enum):
    ACTIVE = "active"
    DEFERRED = "deferred"
    ACCEPTED = "accepted"
    HARD_FAILED = "hard_failed"


class CandidateStage(str, Enum):
    GPU_ENDPOINT = "gpu_endpoint"
    PYBULLET_ENDPOINT = "pybullet_endpoint"
    PLAN = "plan"
    PYBULLET_TRAJECTORY = "pybullet_trajectory"
    REJECTED = "rejected"
    STALE = "stale"


@dataclass
class CandidateState:
    candidate_id: int
    task_id: int
    attempt: int
    payload: dict
    stage: CandidateStage = CandidateStage.GPU_ENDPOINT
    queued_at: float = field(default_factory=time.perf_counter)
    rejection_reason: str | None = None

    def advance(self, stage):
        if self.stage in {CandidateStage.REJECTED, CandidateStage.STALE}:
            raise RuntimeError(f"terminal candidate {self.candidate_id} cannot advance")
        self.stage = CandidateStage(stage)
        self.queued_at = time.perf_counter()

    def reject(self, reason):
        self.stage = CandidateStage.REJECTED
        self.rejection_reason = str(reason)

    def stale(self):
        self.stage = CandidateStage.STALE


@dataclass
class TaskState:
    definition: TaskDefinition
    shard_start: int
    status: TaskStatus = TaskStatus.ACTIVE
    attempt: int = 0
    endpoint_inflight: bool = False
    endpoint_candidate_cursor: int = 0
    candidate_serial: int = 0
    candidates: dict[int, CandidateState] = field(default_factory=dict)
    accepted_path: object | None = None
    accepted_metadata: dict | None = None
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def task_id(self):
        return self.definition.task_id

    @property
    def accepted(self):
        return self.status == TaskStatus.ACCEPTED

    def begin_attempt(self, hard_limit):
        if self.status not in {TaskStatus.ACTIVE, TaskStatus.DEFERRED}:
            raise RuntimeError(f"task {self.task_id} is not eligible for another attempt")
        if self.endpoint_inflight:
            raise RuntimeError(f"task {self.task_id} already has endpoint work in flight")
        if self.attempt >= int(hard_limit):
            self.status = TaskStatus.HARD_FAILED
            return None
        self.attempt += 1
        self.endpoint_candidate_cursor = 0
        self.endpoint_inflight = True
        return self.attempt

    def begin_endpoint_chunk(self, hard_limit, candidates_per_attempt, chunk_size):
        """Reserve deterministic candidate indices for one small actor job."""
        if self.endpoint_inflight or self.live_candidates():
            return None
        if self.attempt == 0 or self.endpoint_candidate_cursor >= candidates_per_attempt:
            attempt = self.begin_attempt(hard_limit)
            if attempt is None:
                return None
        else:
            self.endpoint_inflight = True
        begin = self.endpoint_candidate_cursor
        end = min(begin + int(chunk_size), int(candidates_per_attempt))
        self.endpoint_candidate_cursor = end
        return self.attempt, list(range(begin, end))

    def finish_endpoint_work(self):
        if not self.endpoint_inflight:
            raise RuntimeError(f"task {self.task_id} has no endpoint work in flight")
        self.endpoint_inflight = False

    def add_candidate(self, payload):
        candidate = CandidateState(
            candidate_id=self.candidate_serial,
            task_id=self.task_id,
            attempt=self.attempt,
            payload=payload,
        )
        self.candidates[candidate.candidate_id] = candidate
        self.candidate_serial += 1
        return candidate

    def live_candidates(self):
        terminal = {CandidateStage.REJECTED, CandidateStage.STALE}
        return [item for item in self.candidates.values() if item.stage not in terminal]

    def accept(self, path, metadata):
        if self.status == TaskStatus.ACCEPTED:
            return False
        self.status = TaskStatus.ACCEPTED
        self.accepted_path = path
        self.accepted_metadata = metadata
        self.endpoint_inflight = False
        for candidate in self.live_candidates():
            candidate.stale()
        return True

    def defer(self):
        if self.status == TaskStatus.ACTIVE:
            self.status = TaskStatus.DEFERRED


@dataclass
class ShardState:
    start: int
    size: int
    path: Path
    tasks: dict[int, TaskState]
    stats: Counter = field(default_factory=Counter)
    opened_at: float = field(default_factory=time.perf_counter)

    @property
    def complete(self):
        return len(self.tasks) == self.size and all(
            task.status == TaskStatus.ACCEPTED for task in self.tasks.values()
        )

    @property
    def hard_failed(self):
        return [
            task.task_id
            for task in self.tasks.values()
            if task.status == TaskStatus.HARD_FAILED
        ]

    def ordered_results(self):
        if not self.complete:
            raise RuntimeError(f"shard {self.start} is incomplete")
        ordered = [self.tasks[index] for index in range(self.start, self.start + self.size)]
        return (
            [task.accepted_path for task in ordered],
            [task.accepted_metadata for task in ordered],
        )


class ActiveShardWindow:
    """Bounded collection of shards refilled by unfinished-task capacity."""

    def __init__(self, specs, max_tasks, max_shards):
        self.pending = deque(specs)
        self.max_tasks = int(max_tasks)
        self.max_shards = int(max_shards)
        if self.max_tasks < 1 or self.max_shards < 1:
            raise ValueError("active window limits must be positive")
        self.open = {}

    @property
    def task_count(self):
        return sum(shard.size for shard in self.open.values())

    @property
    def unfinished_task_count(self):
        return sum(
            task.status != TaskStatus.ACCEPTED
            for shard in self.open.values()
            for task in shard.tasks.values()
        )

    def fill(self, factory):
        opened = []
        while self.pending and len(self.open) < self.max_shards:
            spec = self.pending[0]
            size = int(spec[1])
            if self.open and self.unfinished_task_count + size > self.max_tasks:
                break
            if not self.open and size > self.max_tasks:
                raise ValueError("one shard is larger than the active task window")
            self.pending.popleft()
            shard = factory(*spec)
            if shard.start in self.open:
                raise ValueError(f"duplicate shard start {shard.start}")
            self.open[shard.start] = shard
            opened.append(shard)
        return opened

    def remove(self, start):
        return self.open.pop(int(start))

    @property
    def finished(self):
        return not self.pending and not self.open

    def active_tasks(self):
        for start in sorted(self.open):
            for task_id in sorted(self.open[start].tasks):
                yield self.open[start].tasks[task_id]
