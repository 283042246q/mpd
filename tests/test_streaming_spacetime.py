from multiprocessing.pool import ThreadPool
from threading import Event
from types import SimpleNamespace
import time

import pytest

from scripts.spacetime_data.streaming_spacetime import StreamingSpaceTime


def result(value):
    return SimpleNamespace(value=value, statistics={"wall_s": 0.01})


def test_replenishes_valid_jobs_and_handles_results_before_slow_first_query():
    started, release = Event(), Event()
    validated, prepared = [], []
    pool = ThreadPool(2)

    def solve(job):
        if job == 1:
            started.set()
            assert release.wait(2)
        if job == 4:
            release.set()
        return result(job)

    def prepare(attempt):
        prepared.append(attempt)
        if attempt == 0:
            return None, "invalid_source"
        if attempt == 2:
            assert started.wait(1)
        if attempt == 3:
            assert started.is_set() and not release.is_set()
        return attempt, None

    def validate(job, solved):
        validated.append(job)
        return (None, "fit_rejected") if job == 2 else (solved.value, None)

    try:
        stream = StreamingSpaceTime(prepare, lambda job: pool.apply_async(solve, (job,)), validate,
                                   start_attempt=0, max_attempts=20, target=3, capacity=2, timeout=2)
        outcomes = list(stream)
        assert [o.attempt_id for o in outcomes] == list(range(5))
        assert [o.record for o in outcomes if o.record is not None] == [1, 3, 4]
        assert [o.reason for o in outcomes if o.record is None] == ["invalid_source", "fit_rejected"]
        assert validated.index(2) < validated.index(1)
        assert validated.index(3) < validated.index(1)
        assert prepared == list(range(5))  # no excess target reservations
        assert stream.statistics["max_in_flight"] == 2
        assert stream.statistics["submitted"] == stream.statistics["completed"] == 4
    finally:
        release.set()
        pool.close()
        pool.join()


def test_attempt_budget_and_resume_start_are_preserved():
    stream = StreamingSpaceTime(lambda attempt: (None, "invalid_source"),
                               lambda job: pytest.fail("invalid job submitted"),
                               lambda job, solved: pytest.fail("invalid job validated"),
                               start_attempt=17, max_attempts=24, target=2, capacity=4, timeout=1)
    outcomes = list(stream)
    assert [o.attempt_id for o in outcomes] == list(range(17, 24))
    assert stream.statistics["submitted"] == stream.statistics["accepted"] == 0


def test_slow_first_query_does_not_create_an_unbounded_reorder_buffer():
    pool = ThreadPool(2)

    def solve(job):
        if job == 0:
            time.sleep(0.1)
        return result(job)

    try:
        stream = StreamingSpaceTime(lambda attempt: (attempt, None),
                                   lambda job: pool.apply_async(solve, (job,)),
                                   lambda job, solved: (None, "no_solution"),
                                   start_attempt=0, max_attempts=100, target=5, capacity=2, timeout=2)
        assert [o.attempt_id for o in stream] == list(range(100))
        assert stream.statistics["max_uncommitted_attempts"] <= 32
        assert stream.statistics["max_in_flight"] <= 2
    finally:
        pool.close()
        pool.join()


def test_worker_exception_and_missing_result_are_not_silent_failures():
    class BrokenFuture:
        def ready(self):
            return True

        def get(self):
            raise RuntimeError("worker initialization failed")

    def make_stream(future):
        return StreamingSpaceTime(lambda attempt: (attempt, None), lambda job: future,
                                  lambda job, solved: pytest.fail("unexpected validation"),
                                  start_attempt=0, max_attempts=1, target=1, capacity=1, timeout=0.01)

    with pytest.raises(RuntimeError, match="initialization"):
        list(make_stream(BrokenFuture()))

    class MissingFuture:
        def ready(self):
            return False

    with pytest.raises(TimeoutError, match="attempt 0"):
        list(make_stream(MissingFuture()))
