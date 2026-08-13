"""Compatibility hooks for the TorchKin revision pinned by this repository."""

from __future__ import annotations


def ensure_torchkin_pose_cache_api() -> bool:
    """Install the pose-cache factory required by MPD's parent-link scan path."""

    import torchkin
    from torchkin.forward_kinematics import ForwardKinematicsFactory

    if hasattr(torchkin, "get_forward_kinematics_pose_cache_fns"):
        return False

    def get_forward_kinematics_pose_cache_fns(robot, link_names=None):
        (
            _,
            _,
            _,
            backward_helper,
            link_ids,
            _,
            related_link_ids,
        ) = ForwardKinematicsFactory(robot, link_names)

        links = robot.get_links()
        related_link_names = [links[link_id].name for link_id in related_link_ids]
        RelatedForwardKinematics, *_ = ForwardKinematicsFactory(
            robot,
            related_link_names,
        )

        def forward_kinematics_pose_cache(angles):
            output = RelatedForwardKinematics.apply(angles)
            return output[:-1]

        def spatial_jacobian_from_pose_cache(related_poses):
            if len(related_poses) != len(related_link_ids):
                raise ValueError("Pose cache length does not match the related TorchKin links.")
            poses = [None] * robot.num_links
            for link_id, pose in zip(related_link_ids, related_poses):
                poses[link_id] = pose
            jacobian_poses = backward_helper(poses)
            selected_poses = tuple(poses[link_id] for link_id in link_ids)
            jacobians = []
            for link_id in link_ids:
                jacobian = jacobian_poses.new_zeros(
                    jacobian_poses.shape[0],
                    6,
                    robot.dof,
                )
                active_joints = robot.links[link_id].ancestor_non_fixed_joint_ids
                jacobian[:, :, active_joints] = jacobian_poses[:, :, active_joints]
                jacobians.append(jacobian)
            return jacobians, selected_poses

        return (
            forward_kinematics_pose_cache,
            spatial_jacobian_from_pose_cache,
            related_link_ids,
            link_ids,
        )

    torchkin.get_forward_kinematics_pose_cache_fns = get_forward_kinematics_pose_cache_fns
    return True
