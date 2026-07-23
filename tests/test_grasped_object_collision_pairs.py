import unittest

import numpy as np
import torch

np.int = int
np.float = float
np.bool = bool

from torch_robotics.environments import GraspedObjectBox
from torch_robotics.robots import RobotPanda


class GraspedObjectCollisionPairsTest(unittest.TestCase):
    def test_robot_spheres_precede_object_spheres_and_pairs_exclude_hand(self):
        tensor_args = {"device": "cpu", "dtype": torch.float32}
        grasped_object = GraspedObjectBox(
            attached_to_frame=RobotPanda.link_name_ee,
            object_collision_margin=0.02,
            tensor_args=tensor_args,
        )
        robot = RobotPanda(
            gripper=True,
            grasped_object=grasped_object,
            tensor_args=tensor_args,
        )

        object_indices = set(robot.grasped_object_collision_sphere_indices)
        self.assertTrue(object_indices)
        self.assertEqual(min(object_indices), robot.n_robot_collision_spheres)

        parent_link_by_child = {joint.child: joint.parent for joint in robot.robot_urdf.joints}
        sphere_parent_links = [parent_link_by_child[name] for name in robot.link_collision_spheres_names]
        object_pairs = [
            pair for pair in robot.link_self_collision_tuples if pair[0] in object_indices or pair[1] in object_indices
        ]

        checked_robot_indices = {pair[1] if pair[0] in object_indices else pair[0] for pair in object_pairs}
        expected_robot_indices = {
            idx
            for idx in range(robot.n_robot_collision_spheres)
            if sphere_parent_links[idx] not in grasped_object.allowed_self_collision_links
        }
        self.assertEqual(checked_robot_indices, expected_robot_indices)
        self.assertTrue(all(sphere_parent_links[idx] != RobotPanda.link_name_ee for idx in checked_robot_indices))

        robot_parent_pairs = {
            (sphere_parent_links[pair[0]], sphere_parent_links[pair[1]])
            for pair in robot.link_self_collision_tuples
            if pair[0] not in object_indices and pair[1] not in object_indices
        }
        expected_parent_pairs = {
            ("panda_link1", "panda_link7"),
            ("panda_link2", "panda_link7"),
            ("panda_link3", "panda_link6"),
            ("panda_link3", "panda_hand"),
            ("panda_link4", "panda_link7"),
            ("panda_link4", "panda_hand"),
        }
        self.assertTrue(expected_parent_pairs.issubset(robot_parent_pairs))


if __name__ == "__main__":
    unittest.main()
