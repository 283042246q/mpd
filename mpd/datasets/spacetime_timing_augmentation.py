"""Add duration/shape-decoupled timing labels to canonical Space-Time shards."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Dict, Mapping, Optional

import h5py
import numpy as np
from scipy import interpolate

from mpd.datasets.spacetime_legacy import open_uniform_knots
from mpd.datasets.spacetime_schema import SCHEMA_VERSION
from mpd.parametric_trajectory.normalized_timing import (
    NORMALIZED_TIMING_REPRESENTATION,
    NormalizedTimingSplineNumpy,
    duration_to_logit,
    minimum_duration_for_path,
)
from mpd.parametric_trajectory.timing_fitting import TimingSplineNumpy


NORMALIZED_TIMING_EXTENSION_VERSION = "spacetime_normalized_timing_v1"

NORMALIZED_TIMING_FIELDS = {
    "timing/shape_control_points": (np.dtype("float32"), (5,)),
    "timing/t_min": (np.dtype("float32"), ()),
    "timing/t_min_velocity": (np.dtype("float32"), ()),
    "timing/t_min_acceleration": (np.dtype("float32"), ()),
    "timing/t_max": (np.dtype("float32"), ()),
    "timing/duration_fraction": (np.dtype("float32"), ()),
    "timing/tau": (np.dtype("float32"), ()),
    "quality/normalized_timing_fit_rmse": (np.dtype("float32"), ()),
    "quality/normalized_timing_density_clip_fraction": (np.dtype("float32"), ()),
    "quality/duration_logit_clipped": (np.dtype("bool"), ()),
    "quality/duration_bounds_valid": (np.dtype("bool"), ()),
}


def augmentation_config(
    *,
    duration_max: float,
    duration_floor: float,
    density_floor: float,
    logit_clip: float,
    t_min_safety_factor: float,
    bounds_tolerance: float,
    max_fit_rmse: float,
    shape_source: str,
    timing_degree: int = 3,
    num_phase_points: int = 128,
) -> Dict[str, object]:
    if duration_max <= 0.0:
        raise ValueError("duration_max must be positive")
    if duration_floor < 0.0 or duration_max <= duration_floor:
        raise ValueError("duration_floor must be non-negative and below duration_max")
    if not 0.0 <= density_floor < 1.0:
        raise ValueError("density_floor must be in [0, 1)")
    if not 0.0 < logit_clip < 0.5:
        raise ValueError("logit_clip must be in (0, 0.5)")
    if t_min_safety_factor < 1.0:
        raise ValueError("t_min_safety_factor must be at least one")
    if bounds_tolerance < 0.0:
        raise ValueError("bounds_tolerance must be non-negative")
    if max_fit_rmse < 0.0:
        raise ValueError("max_fit_rmse must be non-negative")
    if timing_degree < 1 or num_phase_points < 2:
        raise ValueError("timing_degree and num_phase_points are invalid")
    if shape_source not in ("control_points", "reference_time"):
        raise ValueError("shape_source must be control_points or reference_time")
    return {
        "extension_version": NORMALIZED_TIMING_EXTENSION_VERSION,
        "representation": NORMALIZED_TIMING_REPRESENTATION,
        "num_shape_control_points": 5,
        "degree": int(timing_degree),
        "num_phase_points": int(num_phase_points),
        "duration_max_s": float(duration_max),
        "duration_minimum": "sampled_velocity_acceleration_bound",
        "duration_floor_s": float(duration_floor),
        "density_floor": float(density_floor),
        "logit_clip": float(logit_clip),
        "t_min_safety_factor": float(t_min_safety_factor),
        "bounds_tolerance_s": float(bounds_tolerance),
        "max_normalized_fit_rmse": float(max_fit_rmse),
        "shape_source": shape_source,
        "invalid_sample_policy": "retain_row_and_set_quality/duration_bounds_valid_false",
    }


def _create_or_replace_dataset(
    shard: h5py.File, name: str, values: np.ndarray, *, compression: Optional[str]
) -> None:
    if name in shard:
        del shard[name]
    parent, _ = name.rsplit("/", 1)
    shard.require_group(parent)
    row_chunks = min(max(1, values.shape[0]), 256)
    shard.create_dataset(
        name,
        data=values,
        maxshape=(None,) + values.shape[1:],
        chunks=(row_chunks,) + values.shape[1:],
        compression=compression,
        dtype=values.dtype,
    )


def _derive_fields(
    shard: h5py.File,
    *,
    spatial_degree: int,
    timing_spline: TimingSplineNumpy,
    normalized_spline: NormalizedTimingSplineNumpy,
    dq_max: np.ndarray,
    ddq_max: np.ndarray,
    config: Mapping[str, object],
) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
    required = (
        "spatial/control_points",
        "timing/control_points",
        "timing/duration",
        "timing/reference_time",
    )
    missing = [name for name in required if name not in shard]
    if missing:
        raise ValueError(f"shard is missing required fields: {missing}")
    rows = int(shard["timing/duration"].shape[0])
    if any(shard[name].shape[0] != rows for name in required):
        raise ValueError("required fields have inconsistent row counts")
    spatial_control_points = shard["spatial/control_points"]
    timing_control_points = shard["timing/control_points"]
    stored_duration = shard["timing/duration"]
    reference_time = shard["timing/reference_time"]
    knots = open_uniform_knots(spatial_control_points.shape[1], spatial_degree)
    result = {
        name: np.empty((rows,) + shape, dtype=dtype)
        for name, (dtype, shape) in NORMALIZED_TIMING_FIELDS.items()
    }
    duration_max = float(config["duration_max_s"])
    duration_floor = float(config["duration_floor_s"])
    safety_factor = float(config["t_min_safety_factor"])
    logit_clip = float(config["logit_clip"])
    tolerance = float(config["bounds_tolerance_s"])
    max_fit_rmse = float(config["max_normalized_fit_rmse"])
    shape_source = str(config["shape_source"])

    for row in range(rows):
        old_timing = timing_spline.evaluate(
            np.asarray(timing_control_points[row], dtype=np.float64)
        )
        duration = float(stored_duration[row])
        if not np.isclose(duration, old_timing.duration, rtol=1e-5, atol=1e-6):
            raise ValueError(
                f"row {row}: stored duration {duration} does not match TimingSpline "
                f"duration {old_timing.duration}"
            )
        if shape_source == "control_points":
            shape_target = old_timing.time_from_start / old_timing.duration
        else:
            reference = np.asarray(reference_time[row], dtype=np.float64)
            reference = reference - reference[0]
            shape_target = reference / reference[-1]
        fit = normalized_spline.fit_normalized_time(shape_target)
        normalized = normalized_spline.evaluate(fit.shape_control_points)
        spatial = interpolate.BSpline(
            knots,
            np.asarray(spatial_control_points[row], dtype=np.float64),
            spatial_degree,
            axis=0,
        )
        minimum = minimum_duration_for_path(
            spatial,
            normalized,
            dq_max=dq_max,
            ddq_max=ddq_max,
            duration_floor=duration_floor,
            safety_factor=safety_factor,
        )
        tau, fraction, clipped = duration_to_logit(
            duration, minimum.value, duration_max, clip=logit_clip
        )
        bounds_valid = bool(
            np.isfinite(tau)
            and fit.rmse <= max_fit_rmse
            and duration_max > minimum.value
            and duration >= minimum.value - tolerance
            and duration <= duration_max + tolerance
        )
        result["timing/shape_control_points"][row] = fit.shape_control_points
        result["timing/t_min"][row] = minimum.value
        result["timing/t_min_velocity"][row] = minimum.velocity
        result["timing/t_min_acceleration"][row] = minimum.acceleration
        result["timing/t_max"][row] = duration_max
        result["timing/duration_fraction"][row] = fraction
        result["timing/tau"][row] = tau
        result["quality/normalized_timing_fit_rmse"][row] = fit.rmse
        result["quality/normalized_timing_density_clip_fraction"][row] = (
            fit.relative_density_clip_fraction
        )
        result["quality/duration_logit_clipped"][row] = clipped
        result["quality/duration_bounds_valid"][row] = bounds_valid

    valid = result["quality/duration_bounds_valid"]
    finite_t_min = result["timing/t_min"][np.isfinite(result["timing/t_min"])]
    report = {
        "rows": rows,
        "valid_rows": int(np.sum(valid)),
        "invalid_rows": int(rows - np.sum(valid)),
        "logit_clipped_rows": int(np.sum(result["quality/duration_logit_clipped"])),
        "fit_rmse_max": float(np.max(result["quality/normalized_timing_fit_rmse"]))
        if rows
        else float("nan"),
        "t_min_min": float(np.min(finite_t_min)) if finite_t_min.size else float("nan"),
        "t_min_max": float(np.max(finite_t_min)) if finite_t_min.size else float("nan"),
    }
    return result, report


def augment_shard_atomic(
    shard_path: Path,
    *,
    spatial_degree: int,
    timing_spline: TimingSplineNumpy,
    normalized_spline: NormalizedTimingSplineNumpy,
    dq_max: np.ndarray,
    ddq_max: np.ndarray,
    config: Mapping[str, object],
) -> Dict[str, float]:
    """Rebuild one shard beside the original, then atomically replace it."""

    shard_path = Path(shard_path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{shard_path.name}.", suffix=".augmenting", dir=str(shard_path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(str(shard_path), str(temporary_path))
        with h5py.File(str(temporary_path), "r+") as shard:
            if shard.attrs.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported schema version in {shard_path}: "
                    f"{shard.attrs.get('schema_version')}"
                )
            fields, report = _derive_fields(
                shard,
                spatial_degree=spatial_degree,
                timing_spline=timing_spline,
                normalized_spline=normalized_spline,
                dq_max=dq_max,
                ddq_max=ddq_max,
                config=config,
            )
            compression = shard["timing/control_points"].compression
            for name, values in fields.items():
                _create_or_replace_dataset(shard, name, values, compression=compression)
            shard.attrs["normalized_timing_extension_version"] = (
                NORMALIZED_TIMING_EXTENSION_VERSION
            )
            shard.attrs["normalized_timing_config_json"] = json.dumps(
                dict(config), sort_keys=True
            )
            shard.attrs["normalized_timing_valid_rows"] = report["valid_rows"]
            shard["timing"].attrs["normalized_representation"] = (
                NORMALIZED_TIMING_REPRESENTATION
            )
            shard["timing"].attrs["task_duration_max_s"] = config["duration_max_s"]
            shard.flush()
        os.replace(str(temporary_path), str(shard_path))
        return report
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
