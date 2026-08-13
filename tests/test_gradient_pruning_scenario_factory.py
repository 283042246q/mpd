import tempfile
import unittest
from pathlib import Path

import torch

from torch_robotics.environments import EnvGradientPruning2DTest


class GradientPruningScenarioFactoryTest(unittest.TestCase):
    def test_narrow_gap_matches_manifest(self):
        scenario = """
id: narrow_test
type: narrow_2d
gap_width: 0.2
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.yaml"
            path.write_text(scenario, encoding="utf-8")
            env = EnvGradientPruning2DTest(
                path,
                tensor_args={"device": "cpu", "dtype": torch.float64},
            )
        boxes = env.obj_fixed_list[0].fields[0]
        lower_inner_edge = boxes.centers[0, 1] + boxes.sizes[0, 1] / 2
        upper_inner_edge = boxes.centers[1, 1] - boxes.sizes[1, 1] / 2
        self.assertAlmostEqual((upper_inner_edge - lower_inner_edge).item(), 0.2)

    def test_same_manifest_generates_identical_geometry(self):
        scenario = """
id: circle_test
type: simple_2d
obstacles:
  - shape: circle
    center: [0.1, -0.2]
    radius: 0.03
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.yaml"
            path.write_text(scenario, encoding="utf-8")
            kwargs = {"tensor_args": {"device": "cpu", "dtype": torch.float64}}
            first = EnvGradientPruning2DTest(path, **kwargs)
            second = EnvGradientPruning2DTest(path, **kwargs)
        torch.testing.assert_close(first.obj_fixed_list[0].fields[0].centers, second.obj_fixed_list[0].fields[0].centers)
        torch.testing.assert_close(first.obj_fixed_list[0].fields[0].radii, second.obj_fixed_list[0].fields[0].radii)


if __name__ == "__main__":
    unittest.main()
