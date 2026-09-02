import h5py
import numpy as np

from mpd.datasets.spacetime_legacy import LegacyWarehouseReader, fit_canonical_spatial_path


def _smooth_path(num_points=64):
    phase = np.linspace(0.0, 1.0, num_points)
    return np.stack(
        [phase, phase**2, np.sin(phase), np.zeros_like(phase), phase**3, -phase, phase * 0.2],
        axis=1,
    )


def test_fit_canonical_spatial_path_has_rest_boundary_control_points():
    path = _smooth_path()
    fitted = fit_canonical_spatial_path(path, num_control_points=29, degree=5)

    assert fitted.control_points.shape == (29, 7)
    np.testing.assert_allclose(fitted.control_points[:3], np.repeat(path[0][None], 3, axis=0), atol=0.0)
    np.testing.assert_allclose(fitted.control_points[-3:], np.repeat(path[-1][None], 3, axis=0), atol=0.0)
    assert fitted.scipy_spline().derivative(1)(0.0).max() == 0.0
    assert fitted.scipy_spline().derivative(1)(1.0).max() == 0.0
    assert np.isfinite(fitted.fit_rmse)


def test_legacy_warehouse_reader_streams_root_fields(tmp_path):
    source = tmp_path / "legacy.hdf5"
    paths = np.stack([_smooth_path(), _smooth_path()[::-1]])
    with h5py.File(source, "w") as output:
        output.create_dataset("sol_path", data=paths)
        output.create_dataset("task_id", data=np.asarray([41, 42], dtype=np.int64))

    with LegacyWarehouseReader(source, dof=7) as reader:
        assert len(reader) == 2
        fitted, task_id = reader.read_canonical_spatial_path(
            1, num_control_points=29, degree=5
        )

    assert task_id == 42
    np.testing.assert_allclose(fitted.q_start, paths[1, 0])
    np.testing.assert_allclose(fitted.q_goal, paths[1, -1])
