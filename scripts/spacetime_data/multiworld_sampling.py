"""World sampler for the independent family-based torch-forest entry point.

Keep the original sampler untouched so existing generation runs remain resumable.
Anchor phase and event time intentionally share one fraction; no reference retime.
"""
from __future__ import annotations

import numpy as np

from scripts.spacetime_data.warehouse_world import (
    Forecast, MOTION_MODES, WORKSPACE_LOW, WORKSPACE_HIGH,
)


def make_multiworld_forecast(reference_centers, nominal_duration, horizon, rng, *,
                             object_count=3, modes=MOTION_MODES, forecast_dt=0.02):
    """Independent sampler: widen phase/events only for counts >= 3.

    The motion formulas deliberately match warehouse_world.make_forecast. This
    separate implementation preserves the original generator software identity
    and resume contracts. Tests compare the unchanged low-count distributions.
    """
    if not 0 < nominal_duration <= horizon or object_count < 0 or forecast_dt <= 0:
        raise ValueError("invalid duration, object_count or forecast_dt")
    if object_count == 0:
        return Forecast(np.array([0., horizon]), np.empty((2, 0, 3)),
                        np.empty(0, dtype=np.int64), np.empty((0, 3)),
                        np.empty(0), [], np.empty((0, 3)))
    centers = np.asarray(reference_centers, dtype=float)
    if centers.ndim != 3 or centers.shape[-1] != 3 or len(centers) < 3:
        raise ValueError("reference_centers must be [phase,sphere,3]")
    if not modes or any(mode not in MOTION_MODES for mode in modes):
        raise ValueError("unknown motion mode")
    # Independent draws avoid coupling shape/event time to count or object
    # index. Events can cluster or occur in any object order; do not sort them.
    fraction_range = (0.25, 0.80) if object_count >= 3 else (0.25, 0.65)
    fractions = rng.uniform(*fraction_range, size=object_count)
    sampled_shapes = rng.integers(0, 3, size=object_count)
    sampled_modes = rng.integers(0, len(modes), size=object_count)
    events = fractions * nominal_duration
    grid = np.linspace(0, horizon, int(np.ceil(horizon / forecast_dt)) + 1)
    times = np.unique(np.r_[grid, events])  # exact anchor knots survive serialization
    positions, shapes, sizes, anchors, chosen = [], [], [], [], []
    for index, (fraction, event) in enumerate(zip(fractions, events)):
        phase = min(len(centers) - 1, int(round(fraction * (len(centers) - 1))))
        candidates = centers[phase, max(0, centers.shape[1] // 3):]
        inside = ((candidates >= WORKSPACE_LOW) & (candidates <= WORKSPACE_HIGH)).all(axis=1)
        if not inside.any():
            raise ValueError("reference has no moving-link anchor in warehouse workspace")
        anchor = candidates[inside][int(rng.integers(inside.sum()))].copy()
        direction = rng.normal(size=3)
        direction[2] *= 0.35
        direction /= np.linalg.norm(direction)
        lateral = np.cross(direction, [0., 0., 1.])
        lateral /= np.linalg.norm(lateral)
        vertical = np.cross(direction, lateral)
        mode = modes[int(sampled_modes[index])]
        speed = float(rng.uniform(0.22, 0.42))
        dt = times - event
        along, side, up = speed * dt, np.zeros_like(dt), np.zeros_like(dt)
        if mode == "accelerating":
            # Positive derivative throughout; no accidental reversal far from event.
            along += 0.16 * speed * (np.sqrt(dt * dt + 1) - 1)
        elif mode == "speed_variation":
            along += 0.35 * speed * np.sin(1.6 * dt) / 1.6
        elif mode == "curved":
            side = 0.16 * np.sin(1.4 * dt)
        elif mode == "zigzag":
            side = 0.16 * (2 / np.pi) * np.arcsin(np.sin(2.1 * dt))
        elif mode == "shuttle":
            along = 0.38 * np.sin(speed / 0.38 * dt)
        elif mode == "orbit":
            along = 0.32 * np.sin(speed / 0.32 * dt)
            side = 0.32 * (np.cos(speed / 0.32 * dt) - 1)
        elif mode == "vertical":
            direction = np.array([0., 0., 1.])
            lateral = np.array([1., 0., 0.])
            side = 0.08 * np.sin(1.8 * dt)
        elif mode == "stop_go":
            # Exact prescribed obstacle dwell around its interaction anchor.
            along = speed * np.sign(dt) * np.maximum(np.abs(dt) - 0.28, 0)
        elif mode == "helix":
            side = 0.12 * np.sin(2 * dt)
            up = 0.12 * (np.cos(2 * dt) - 1)
        positions.append(anchor + along[:, None] * direction + side[:, None] * lateral + up[:, None] * vertical)
        shape = int(sampled_shapes[index])
        radius = float(rng.uniform(0.04, 0.075))
        shapes.append(shape)
        sizes.append([radius, 0., 0.] if shape == 0 else
                     ([radius, radius * 1.3, radius * 1.1] if shape == 1 else [radius, 0.09, 0.]))
        anchors.append(anchor)
        chosen.append(mode)
    return Forecast(times, np.stack(positions, axis=1), np.asarray(shapes, dtype=np.int64),
                    np.asarray(sizes), events, chosen, np.asarray(anchors))
