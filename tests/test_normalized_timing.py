import h5py
import numpy as np
from scipy import interpolate

from mpd.datasets.spacetime_legacy import open_uniform_knots
from mpd.datasets.spacetime_schema import SCHEMA_VERSION
from mpd.datasets.spacetime_timing_augmentation import (
    NORMALIZED_TIMING_FIELDS,
    augment_shard_atomic,
    augmentation_config,
)
from mpd.parametric_trajectory.normalized_timing import (
    NormalizedTimingSplineNumpy,
    decode_shape_latent,
    duration_to_logit,
    minimum_duration_for_path,
)
from mpd.parametric_trajectory.timing_fitting import TimingSplineNumpy


def test_normalized_timing_shape_integrates_to_one_and_fits_roundtrip():
    spline = NormalizedTimingSplineNumpy()
    shape = np.asarray([0.4, -0.2, 0.7, 0.1, 0.5])

    evaluation = spline.evaluate(shape)
    fitted = spline.fit_normalized_time(evaluation.normalized_time)

    assert np.isclose(evaluation.normalized_time[-1], 1.0, atol=1e-12)
    assert np.all(evaluation.density > 0.0)
    assert fitted.rmse < 5e-5
    np.testing.assert_allclose(
        spline.evaluate(fitted.shape_control_points).normalized_time,
        evaluation.normalized_time,
        atol=5e-5,
    )
    np.testing.assert_array_equal(
        decode_shape_latent(np.zeros(5)), np.zeros(8)
    )


def test_minimum_duration_makes_sampled_velocity_and_acceleration_feasible():
    control_points = np.zeros((12, 2), dtype=np.float64)
    control_points[3:-3, 0] = np.linspace(0.1, 0.9, 6)
    control_points[-3:, 0] = 1.0
    spatial = interpolate.BSpline(
        open_uniform_knots(12, 5), control_points, 5, axis=0
    )
    normalized = NormalizedTimingSplineNumpy().evaluate(
        np.asarray([0.4, -0.2, 0.7, 0.1, 0.5])
    )
    dq_max = np.asarray([0.5, 0.5])
    ddq_max = np.asarray([1.0, 1.0])

    minimum = minimum_duration_for_path(
        spatial, normalized, dq_max=dq_max, ddq_max=ddq_max
    )
    q_s = spatial.derivative(1)(normalized.phase)
    q_ss = spatial.derivative(2)(normalized.phase)
    duration = minimum.value
    dq = q_s / (duration * normalized.density[:, None])
    ddq = q_ss / np.square(duration * normalized.density[:, None]) - (
        q_s
        * normalized.density_s[:, None]
        / (duration**2 * np.power(normalized.density[:, None], 3))
    )

    assert np.max(np.abs(dq) / dq_max[None, :]) <= 1.0 + 1e-10
    assert np.max(np.abs(ddq) / ddq_max[None, :]) <= 1.0 + 1e-10


def test_duration_logit_roundtrip_and_clipping():
    tau, fraction, clipped = duration_to_logit(8.0, 2.0, 14.0)
    reconstructed = 2.0 + 12.0 / (1.0 + np.exp(-tau))

    assert fraction == 0.5
    assert not clipped
    assert np.isclose(reconstructed, 8.0)
    _, fraction, clipped = duration_to_logit(15.0, 2.0, 14.0)
    assert fraction > 1.0
    assert clipped


def test_augment_shard_adds_normalized_fields_atomically(tmp_path):
    path = tmp_path / "part-0000000.hdf5"
    timing_spline = TimingSplineNumpy(num_phase_points=32)
    old_control_points = np.full(8, 2.0, dtype=np.float64)
    old_timing = timing_spline.evaluate(old_control_points)
    spatial_control_points = np.zeros((12, 2), dtype=np.float64)
    spatial_control_points[3:-3, 0] = np.linspace(0.01, 0.09, 6)
    spatial_control_points[-3:, 0] = 0.1
    with h5py.File(path, "w") as shard:
        shard.attrs["schema_version"] = SCHEMA_VERSION
        shard.create_dataset(
            "spatial/control_points", data=spatial_control_points[None].astype(np.float32)
        )
        shard.create_dataset(
            "timing/control_points", data=old_control_points[None].astype(np.float32)
        )
        shard.create_dataset(
            "timing/duration", data=np.asarray([old_timing.duration], dtype=np.float32)
        )
        shard.create_dataset(
            "timing/reference_time",
            data=old_timing.time_from_start[None].astype(np.float32),
        )
    normalized_spline = NormalizedTimingSplineNumpy(num_phase_points=32)
    config = augmentation_config(
        duration_max=14.0,
        duration_floor=0.0,
        density_floor=1e-3,
        logit_clip=1e-3,
        t_min_safety_factor=1.0,
        bounds_tolerance=1e-3,
        max_fit_rmse=0.05,
        shape_source="control_points",
        timing_degree=3,
        num_phase_points=32,
    )

    report = augment_shard_atomic(
        path,
        spatial_degree=5,
        timing_spline=timing_spline,
        normalized_spline=normalized_spline,
        dq_max=np.asarray([10.0, 10.0]),
        ddq_max=np.asarray([10.0, 10.0]),
        config=config,
    )

    assert report["rows"] == 1
    assert report["valid_rows"] == 1
    with h5py.File(path, "r") as shard:
        assert not list(tmp_path.glob("*.augmenting"))
        for name, (dtype, shape) in NORMALIZED_TIMING_FIELDS.items():
            assert shard[name].shape == (1,) + shape
            assert shard[name].dtype == dtype
        assert shard["timing/t_max"][0] == 14.0
        assert bool(shard["quality/duration_bounds_valid"][0])
