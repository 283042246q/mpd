"""Bounded asynchronous planning with ordered, resumable attempt outcomes.

Preparation and validation run in the caller, while submitted queries run in
the process pool. No HDF5 handles or CUDA collision objects cross threads.
"""
from dataclasses import dataclass
import time


@dataclass
class AttemptOutcome:
    attempt_id: int
    record: object = None
    reason: str = None


class StreamingSpaceTime:
    def __init__(self, prepare, submit, validate, *, start_attempt, max_attempts,
                 target, capacity, timeout):
        if capacity < 1 or timeout <= 0 or target < 0:
            raise ValueError("invalid streaming queue limits")
        self.prepare, self.submit, self.validate = prepare, submit, validate
        self.start_attempt, self.max_attempts = start_attempt, max_attempts
        self.target, self.capacity, self.timeout = target, capacity, timeout
        self.statistics = dict(queue_capacity=capacity, prepared=0, submitted=0,
                               completed=0, accepted=0, max_in_flight=0,
                               max_uncommitted_attempts=0, preparation_wall_s=0.,
                               postprocess_wall_s=0., poll_wait_wall_s=0.,
                               planner_wall_s_total=0.)

    def __iter__(self):
        next_attempt = next_emit = self.start_attempt
        pending, outcomes = {}, {}
        stats = self.statistics
        # A slow earliest attempt must not cause unbounded reorder buffers.
        attempt_window = max(32, 4 * self.capacity)
        while pending or outcomes or (next_attempt < self.max_attempts and stats["accepted"] < self.target):
            progressed = False
            # Harvest a ready query regardless of submission order. Validation
            # failures release a target slot so preparation can replace them.
            for attempt_id, (job, future, submitted_at) in list(pending.items()):
                if future.ready():
                    result = future.get()  # exceptions reach the pool cleanup
                    del pending[attempt_id]
                    started = time.perf_counter()
                    record, reason = self.validate(job, result)
                    stats["postprocess_wall_s"] += time.perf_counter() - started
                    stats["completed"] += 1
                    stats["planner_wall_s_total"] += result.statistics.get("wall_s", 0.)
                    stats["accepted"] += int(record is not None)
                    outcomes[attempt_id] = AttemptOutcome(attempt_id, record, reason)
                    progressed = True
                    break
                if time.perf_counter() - submitted_at > self.timeout:
                    raise TimeoutError("STRRT* worker did not return attempt %d within %.1f seconds" %
                                       (attempt_id, self.timeout))

            # Queue capacity counts VALID jobs, not attempted candidates. One
            # candidate per turn lets ready results and commits stay responsive.
            if (len(pending) < self.capacity and
                len(pending) + stats["accepted"] < self.target and
                next_attempt < self.max_attempts and
                next_attempt - next_emit < attempt_window):
                attempt_id = next_attempt
                next_attempt += 1
                started = time.perf_counter()
                job, reason = self.prepare(attempt_id)
                stats["preparation_wall_s"] += time.perf_counter() - started
                stats["prepared"] += 1
                if job is None:
                    outcomes[attempt_id] = AttemptOutcome(attempt_id, reason=reason)
                else:
                    future = self.submit(job)
                    pending[attempt_id] = (job, future, time.perf_counter())
                    stats["submitted"] += 1
                    stats["max_in_flight"] = max(stats["max_in_flight"], len(pending))
                stats["max_uncommitted_attempts"] = max(
                    stats["max_uncommitted_attempts"], next_attempt - next_emit)
                progressed = True

            # Only persistence is ordered: later queries can finish, validate
            # and release queue slots before an earlier query has completed.
            if next_emit in outcomes:
                outcome = outcomes.pop(next_emit)
                next_emit += 1
                yield outcome
                progressed = True
            if not progressed:
                started = time.perf_counter()
                time.sleep(0.005)
                stats["poll_wait_wall_s"] += time.perf_counter() - started
