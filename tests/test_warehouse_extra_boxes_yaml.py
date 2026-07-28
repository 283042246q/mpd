import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

np.int = int
np.float = float
np.bool = bool

from torch_robotics.environments import EnvWarehouseExtraBoxes, EnvWarehouseExtraObjectsV00
from torch_robotics.environments.env_warehouse_extra_boxes import (
    load_warehouse_extra_boxes_yaml,
)


TENSOR_ARGS_CPU = {"device": "cpu", "dtype": torch.float32}


class WarehouseExtraBoxesYamlTest(unittest.TestCase):
    def test_v00_yaml_matches_original_hardcoded_box(self):
        env = EnvWarehouseExtraObjectsV00(
            precompute_sdf_obj_fixed=False,
            precompute_sdf_obj_extra=False,
            tensor_args=TENSOR_ARGS_CPU,
        )

        self.assertEqual(env.name, "EnvWarehouseExtraObjectsV00")
        self.assertEqual(len(env.obj_extra_list), 1)
        boxes = env.obj_extra_list[0].fields[0]
        np.testing.assert_allclose(boxes.centers.cpu().numpy(), [[0.50, 0.14, 0.14]])
        np.testing.assert_allclose(boxes.sizes.cpu().numpy(), [[0.05, 0.28, 0.28]])
        self.assertTrue(env.extra_boxes_yaml.endswith("warehouse_extra_boxes_v00.yaml"))

    def test_parameterized_simple_scene_has_no_extra_objects(self):
        env = EnvWarehouseExtraBoxes(
            extra_boxes_yaml="warehouse/extra_boxes/warehouse_extra_boxes_simple.yaml",
            precompute_sdf_obj_fixed=False,
            precompute_sdf_obj_extra=False,
            tensor_args=TENSOR_ARGS_CPU,
        )

        self.assertEqual(env.name, "EnvWarehouseExtraBoxes")
        self.assertEqual(env.obj_extra_list, [])
        self.assertEqual(env.extra_boxes_config["boxes"], [])

    def test_narrow_scenario_has_requested_gap(self):
        config = load_warehouse_extra_boxes_yaml("warehouse/extra_boxes/warehouse_extra_boxes_narrow_020.yaml")
        boxes = config["boxes"]
        negative_y_inner_edge = boxes[0]["center"][1] + boxes[0]["size"][1] / 2
        positive_y_inner_edge = boxes[1]["center"][1] - boxes[1]["size"][1] / 2

        self.assertAlmostEqual(positive_y_inner_edge - negative_y_inner_edge, 0.20)

    def test_invalid_nonpositive_size_is_rejected(self):
        invalid_yaml = """
schema: mpd_warehouse_extra_boxes
schema_version: 1
reference_frame: panda_link0_unrotated
boxes:
  - name: invalid
    center: [0.0, 0.0, 0.0]
    size: [0.1, 0.0, 0.1]
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.yaml"
            path.write_text(invalid_yaml, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must all be positive"):
                load_warehouse_extra_boxes_yaml(path)

    def test_duplicate_box_names_are_rejected(self):
        invalid_yaml = """
schema: mpd_warehouse_extra_boxes
schema_version: 1
reference_frame: panda_link0_unrotated
boxes:
  - name: duplicate
    center: [0.0, 0.0, 0.0]
    size: [0.1, 0.1, 0.1]
  - name: duplicate
    center: [0.2, 0.0, 0.0]
    size: [0.1, 0.1, 0.1]
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.yaml"
            path.write_text(invalid_yaml, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_warehouse_extra_boxes_yaml(path)


if __name__ == "__main__":
    unittest.main()
