"""Read legacy MPD spatial paths without invoking the old training loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
from scipy import interpolate


def open_uniform_knots(num_control_points: int, degree: int) -> np.ndarray:
    if degree < 1 or num_control_points <= degree:
        raise ValueError("expected num_control_points > degree >= 1")
    interior_grid_size = num_control_points + degree + 1 - 2 * degree
    return np.pad(np.linspace(0.0, 1.0, interior_grid_size), degree, mode="edge")


@dataclass(frozen=True)
class CanonicalSpatialPath:
    control_points: np.ndarray
    knots: np.ndarray
    degree: int
    q_start: np.ndarray
    q_goal: np.ndarray
    fit_rmse: float

    def scipy_spline(self) -> interpolate.BSpline:
        return interpolate.BSpline(self.knots, self.control_points, self.degree, axis=0)


def _canonicalize_legacy_coefficients(coefficients: np.ndarray, dof: int) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 2:
        raise ValueError(f"legacy B-spline coefficients must be rank 2, got {coefficients.shape}")
    possible = []
    if coefficients.shape[1] == dof:
        possible.append(coefficients)
    if coefficients.shape[0] == dof:
        possible.append(coefficients.T)
    if len(possible) != 1:
        raise ValueError(
            "legacy B-spline coefficient orientation is ambiguous or incompatible: "
            f"shape={coefficients.shape}, dof={dof}"
        )
    return np.ascontiguousarray(possible[0])


def fit_canonical_spatial_path(
    path: np.ndarray,
    *,
    num_control_points: int,
    degree: int,
) -> CanonicalSpatialPath:
    """Fit the same clamped B-spline representation used by spatial MPD."""

    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] <= degree:
        raise ValueError(f"spatial path must be [waypoints, dof], got {path.shape}")
    if not np.all(np.isfinite(path)):
        raise ValueError("spatial path contains NaN or Inf")
    knots = open_uniform_knots(num_control_points, degree)
    tck, phase = interpolate.splprep(path.T, k=degree, t=knots, task=-1, quiet=1)
    fitted_knots, coefficients_raw, fitted_degree = tck
    coefficients = _canonicalize_legacy_coefficients(coefficients_raw, path.shape[1])
    if coefficients.shape != (num_control_points, path.shape[1]):
        raise ValueError(
            f"fitted spatial control points have shape {coefficients.shape}, expected "
            f"{(num_control_points, path.shape[1])}"
        )

    # Match ParametricTrajectoryBspline's rest boundary contract exactly.
    coefficients[0:3] = path[0]
    coefficients[-3:] = path[-1]
    spline = interpolate.BSpline(fitted_knots, coefficients, fitted_degree, axis=0)
    residual = spline(phase) - path
    fit_rmse = float(np.sqrt(np.mean(np.square(residual))))
    return CanonicalSpatialPath(
        control_points=np.asarray(coefficients, dtype=np.float64),
        knots=np.asarray(fitted_knots, dtype=np.float64),
        degree=int(fitted_degree),
        q_start=path[0].copy(),
        q_goal=path[-1].copy(),
        fit_rmse=fit_rmse,
    )


class LegacyWarehouseReader:
    """Streaming reader for root-level MPD trajectory HDF5 files."""

    def __init__(self, path: Path, *, dof: int) -> None:
        self.path = Path(path)
        self.dof = int(dof)
        self._file: Optional[h5py.File] = None

    def __enter__(self) -> "LegacyWarehouseReader":
        self._file = h5py.File(str(self.path), "r")
        if "sol_path" not in self._file or "task_id" not in self._file:
            raise ValueError("legacy dataset must contain sol_path and task_id")
        shape = self._file["sol_path"].shape
        if len(shape) != 3 or shape[-1] != self.dof:
            raise ValueError(f"sol_path must have shape [N,W,{self.dof}], got {shape}")
        if self._file["task_id"].shape != (shape[0],):
            raise ValueError("task_id length does not match sol_path")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise RuntimeError("LegacyWarehouseReader must be used as a context manager")
        return self._file

    def __len__(self) -> int:
        return int(self.file["sol_path"].shape[0])

    def read_path(self, index: int) -> Tuple[np.ndarray, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        path = np.asarray(self.file["sol_path"][index], dtype=np.float64)
        task_id = int(self.file["task_id"][index])
        return path, task_id

    def read_canonical_spatial_path(
        self,
        index: int,
        *,
        num_control_points: int,
        degree: int,
    ) -> Tuple[CanonicalSpatialPath, int]:
        path, task_id = self.read_path(index)
        has_stored_spline = all(
            key in self.file for key in ("bspline_params_tt", "bspline_params_cc", "bspline_params_k")
        )
        if not has_stored_spline:
            return (
                fit_canonical_spatial_path(
                    path,
                    num_control_points=num_control_points,
                    degree=degree,
                ),
                task_id,
            )

        knots = np.asarray(self.file["bspline_params_tt"][index], dtype=np.float64)
        coefficients = _canonicalize_legacy_coefficients(
            self.file["bspline_params_cc"][index], self.dof
        )
        stored_degree = int(self.file["bspline_params_k"][index])
        if stored_degree != degree or coefficients.shape[0] != num_control_points:
            raise ValueError(
                "stored B-spline contract does not match requested output: "
                f"degree={stored_degree}, control_points={coefficients.shape[0]}"
            )
        spline = interpolate.BSpline(knots, coefficients, stored_degree, axis=0)
        phase = np.linspace(0.0, 1.0, path.shape[0])
        fit_rmse = float(np.sqrt(np.mean(np.square(spline(phase) - path))))
        return (
            CanonicalSpatialPath(
                control_points=coefficients,
                knots=knots,
                degree=stored_degree,
                q_start=coefficients[0].copy(),
                q_goal=coefficients[-1].copy(),
                fit_rmse=fit_rmse,
            ),
            task_id,
        )
