import subprocess
import sys

import numpy as np
import torch

from scripts.generate_data.marvin_gpu_planning_backend import (
    batched_paths_collision_free,
    fit_marvin_spline,
    resample_path,
    shortcut_paths,
)


def test_gpu_backend_module_does_not_import_ompl_or_pybullet():
    command = (
        "import sys; "
        "import scripts.generate_data.marvin_gpu_planning_backend; "
        "assert 'ompl' not in sys.modules; "
        "assert 'pybullet' not in sys.modules; "
        "assert 'pb_ompl.pb_ompl' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_batched_path_audit_handles_variable_lengths():
    def collision(states):
        states = torch.as_tensor(states)
        return states[:, 0].abs() < 0.1

    paths = [
        np.array([[-0.8, 0.0], [-0.4, 0.0]]),
        np.array([[-0.8, 0.0], [0.8, 0.0], [0.9, 0.0]]),
    ]
    assert batched_paths_collision_free(paths, collision, 0.05) == [True, False]


def test_shortcut_preserves_endpoints_and_removes_detour():
    def collision(states):
        return torch.zeros(len(states), dtype=torch.bool)

    path = np.array(
        [[-0.8, 0.0], [-0.5, 0.5], [0.0, 0.8], [0.5, 0.5], [0.8, 0.0]]
    )
    result = shortcut_paths([path], collision, 0.05, attempts=100, seeds=[7])[0]
    np.testing.assert_array_equal(result[[0, -1]], path[[0, -1]])
    assert len(result) == 2


def test_resample_and_spline_preserve_marvin_endpoints():
    parameter = np.linspace(0.0, 1.0, 32)
    path = np.stack(
        [parameter * (joint + 1) * 0.01 for joint in range(14)], axis=1
    )
    resampled = resample_path(path, 128)
    spline, evaluated = fit_marvin_spline(resampled, 5, 22, 512)

    assert resampled.shape == (128, 14)
    assert evaluated.shape == (512, 14)
    assert spline[1].shape == (14, 22)
    np.testing.assert_allclose(resampled[[0, -1]], path[[0, -1]])
    np.testing.assert_allclose(evaluated[[0, -1]], path[[0, -1]], atol=1e-12)
