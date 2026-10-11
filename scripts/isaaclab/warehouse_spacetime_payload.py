"""Portable frozen Warehouse replay data; importable without MPD or Isaac Sim."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def interpolate(times, values, instant):
    """Interpolate arrays at a relative execution time [s], without extrapolation."""
    if not np.isfinite(instant) or instant < times[0] - 1e-8 or instant > times[-1] + 1e-8:
        raise ValueError("replay time outside frozen prediction")
    upper = int(np.clip(np.searchsorted(times, instant, side="right"), 1, len(times) - 1))
    lower = upper - 1
    alpha = float(np.clip((instant - times[lower]) / (times[upper] - times[lower]), 0, 1))
    return values[lower] + alpha * (values[upper] - values[lower])


def load_case(path):
    """Load a safe NPZ payload, with positions [m], joints [rad] and time [s]."""
    with np.load(Path(path), allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    data["metadata"] = json.loads(str(data.pop("metadata_json").item()))
    for name in ("execution_time", "forecast_time"):
        times = data[name]
        if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("invalid time grid: " + name)
        if abs(times[0]) > 1e-8:
            raise ValueError("time origin must be execution_start=0")
    q, positions = data["q"], data["obstacle_positions"]
    objects = len(data["shape"])
    if q.shape != (len(data["execution_time"]), 7) or not np.isfinite(q).all():
        raise ValueError("invalid Panda trajectory")
    if positions.shape != (len(data["forecast_time"]), objects, 3) or not np.isfinite(positions).all():
        raise ValueError("invalid obstacle positions")
    if data["size"].shape != (objects, 3) or not np.isfinite(data["size"]).all():
        raise ValueError("invalid obstacle size")
    if not np.isin(data["shape"], [0, 1, 2]).all() or np.any(data["size"][:, 0] <= 0):
        raise ValueError("invalid obstacle shape/radius")
    if np.any(data["size"][data["shape"] == 1] <= 0) or np.any(data["size"][data["shape"] == 2, 1] < 0):
        raise ValueError("invalid box/capsule extent")
    duration = float(data["execution_time"][-1])
    events = data["event_times"]
    if duration > data["forecast_time"][-1] or events.shape != (objects,):
        raise ValueError("prediction does not cover execution")
    if not np.isfinite(events).all() or np.any(events <= .05 * duration) or np.any(events >= .95 * duration):
        raise ValueError("object events must occur during arm execution")
    return data


def frame_times(duration, fps):
    """Uniform video timestamps [s], including the exact terminal pose."""
    if not np.isfinite(duration) or not np.isfinite(fps) or duration <= 0 or fps <= 0:
        raise ValueError("duration/fps must be positive and finite")
    # Last frame is clamped to the endpoint; encoded length exceeds T by <2/fps.
    return np.minimum(np.arange(int(np.ceil(duration * fps)) + 1) / fps, duration)


def validate_suite(suite):
    """Reject partial suites or duplicate variants before starting Isaac Sim."""
    if suite.get("schema") != "warehouse_spacetime_video_suite_v1":
        raise ValueError("unsupported suite")
    modes, count, cases = suite["modes"], suite["variants_per_mode"], suite["cases"]
    if not modes or len(set(modes)) != len(modes) or count < 1:
        raise ValueError("invalid mode/variant counts")
    if suite["expected_cases"] != len(modes) * count or len(cases) != suite["expected_cases"]:
        raise ValueError("suite is incomplete")
    ids = [case["case_id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case IDs")
    for mode in modes:
        selected = [case for case in cases if case["mode"] == mode]
        if sorted(case["variant"] for case in selected) != list(range(1, count + 1)):
            raise ValueError("missing variants: " + mode)
        if len({case["signature"] for case in selected}) != count:
            raise ValueError("duplicate worlds/trajectories: " + mode)
        if any(case["case_id"] != "%s-%02d" % (mode, case["variant"]) for case in selected):
            raise ValueError("case ID does not match mode/variant")
