"""Method-independent prescribed worlds anchored to a static RRT trajectory."""
from __future__ import annotations

import numpy as np


MOTION_MODES = ("linear", "accelerating", "speed_variation", "curved", "zigzag",
                "shuttle", "orbit", "vertical", "stop_go", "helix")


def balanced_choices(total, values, weights, rng):
    """Largest-remainder quotas, shuffled once; proportions do not drift by model."""
    w = np.asarray(weights, dtype=float)
    if total < 1 or len(w) != len(values) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("invalid weighted quota")
    exact = total * w / w.sum()
    counts = np.floor(exact).astype(int)
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[:total - counts.sum()]] += 1
    result = [v for v, n in zip(values, counts) for _ in range(n)]
    rng.shuffle(result)
    return result


def arrival_phases(count, mode, rng, phase_range=(.25, .8), gap=.06):
    lo, hi = phase_range
    if not 0 < lo < hi < 1 or gap < 0 or count < 0:
        raise ValueError("invalid arrival phase parameters")
    if count == 0:
        return np.empty(0)
    if count == 1 or mode == "simultaneous":
        return np.full(count, rng.uniform(lo, hi))
    groups = count if mode == "sequential" else min(2, count) if mode == "grouped" else None
    if groups is None or gap * (groups - 1) > hi - lo:
        raise ValueError("unknown or infeasible arrival mode")
    events = np.sort(rng.uniform(lo, hi - gap * (groups - 1), groups)) + gap * np.arange(groups)
    phases = events[np.arange(count) % groups]
    rng.shuffle(phases)  # object index never implies event order
    return phases


def sample_world(reference_centers, moving_indices, count, arrival_mode, rng, config):
    phases = arrival_phases(count, arrival_mode, rng, config["phase_range"], config["sequential_phase_gap"])
    events = phases * config["nominal_duration_s"]
    horizon = config["horizon_s"]
    times = np.unique(np.r_[np.linspace(0., horizon, int(np.ceil(horizon / config["forecast_dt_s"])) + 1), events])
    positions, shapes, sizes, anchors, modes, parameters = [], [], [], [], [], []
    centers = np.asarray(reference_centers)
    shape_weights = np.asarray(config["shape_weights"], dtype=float)
    shape_weights /= shape_weights.sum()
    for phase, event in zip(phases, events):
        # Same phase determines both reference position and event time, as in
        # JointDual data generation. Interpolation avoids rounding the phase.
        at = phase * (len(centers) - 1)
        lower = int(np.floor(at)); upper = min(lower + 1, len(centers) - 1)
        sphere = int(rng.choice(moving_indices))
        anchor = (1 - at + lower) * centers[lower, sphere] + (at - lower) * centers[upper, sphere]
        direction = rng.normal(size=3); direction[2] *= .35
        direction /= np.linalg.norm(direction)
        lateral = np.cross(direction, [0., 0., 1.]); lateral /= np.linalg.norm(lateral)
        vertical = np.cross(direction, lateral)
        mode = str(rng.choice(config["motion_modes"]))
        if mode not in MOTION_MODES:
            raise ValueError("unknown motion mode")
        speed = float(rng.uniform(*config["speed_range_m_s"]))
        dt = times - event
        along, side, up = speed * dt, np.zeros_like(dt), np.zeros_like(dt)
        if mode == "accelerating":
            along += .16 * speed * (np.sqrt(dt * dt + 1) - 1)
        elif mode == "speed_variation":
            along += .35 * speed * np.sin(1.6 * dt) / 1.6
        elif mode == "curved":
            side = .16 * np.sin(1.4 * dt)
        elif mode == "zigzag":
            side = .16 * (2 / np.pi) * np.arcsin(np.sin(2.1 * dt))
        elif mode == "shuttle":
            along = .38 * np.sin(speed / .38 * dt)
        elif mode == "orbit":
            along = .32 * np.sin(speed / .32 * dt)
            side = .32 * (np.cos(speed / .32 * dt) - 1)
        elif mode == "vertical":
            direction = np.array([0., 0., 1.]); lateral = np.array([1., 0., 0.])
            side = .08 * np.sin(1.8 * dt)
        elif mode == "stop_go":
            along = speed * np.sign(dt) * np.maximum(np.abs(dt) - .28, 0)
        elif mode == "helix":
            side = .12 * np.sin(2 * dt); up = .12 * (np.cos(2 * dt) - 1)
        positions.append(anchor + along[:, None] * direction + side[:, None] * lateral + up[:, None] * vertical)
        shape = int(rng.choice(3, p=shape_weights))
        radius = float(rng.uniform(*config["radius_range_m"]))
        sizes.append([radius, 0., 0.] if shape == 0 else
                     [radius, 1.3 * radius, 1.1 * radius] if shape == 1 else [radius, .09, 0.])
        shapes.append(shape); anchors.append(anchor); modes.append(mode)
        parameters.append(dict(speed=speed, direction=direction.tolist(), lateral=lateral.tolist(),
                               vertical=vertical.tolist(), reference_sphere=sphere))
    arrays = dict(times=times, positions=np.stack(positions, axis=1) if count else np.empty((len(times), 0, 3)),
                  shapes=np.asarray(shapes, dtype=np.int64), sizes=np.asarray(sizes).reshape(count, 3))
    metadata = dict(object_count=count, arrival_mode="empty" if not count else "single" if count == 1 else arrival_mode,
                    phases=phases.tolist(), event_times=events.tolist(), anchors=np.asarray(anchors).reshape(count, 3).tolist(),
                    motion_modes=modes, motion_parameters=parameters,
                    obstacle_static_collision="allowed", obstacle_obstacle_collision="allowed",
                    interpolation="piecewise_linear", inflation="none", time_origin="execution_start")
    return arrays, metadata


def base_sweep_clear(arrays, base_low, base_high, margin):
    """Conservative continuous sweep test against the robot-base AABB.

    Every obstacle fits in its bounding sphere. A segment outside the AABB
    expanded by that radius cannot intersect the base. No temporal subsampling.
    """
    for i, (shape, size) in enumerate(zip(arrays["shapes"], arrays["sizes"])):
        radius = size[0] if shape == 0 else np.linalg.norm(size) if shape == 1 else size[0] + size[1]
        low, high = np.asarray(base_low) - radius - margin, np.asarray(base_high) + radius + margin
        p = arrays["positions"][:, i]
        start, direction = p[:-1], np.diff(p, axis=0)
        parallel = np.abs(direction) < 1e-14
        safe_direction = np.where(parallel, 1., direction)
        a, b = (low - start) / safe_direction, (high - start) / safe_direction
        near, far = np.minimum(a, b), np.maximum(a, b)
        near = np.where(parallel, -np.inf, near)
        far = np.where(parallel, np.inf, far)
        outside_parallel = (parallel & ((start < low) | (start > high))).any(axis=1)
        enters = np.maximum(near.max(axis=1), 0.)
        leaves = np.minimum(far.min(axis=1), 1.)
        if ((enters <= leaves) & ~outside_parallel).any():
            return False
    return True
