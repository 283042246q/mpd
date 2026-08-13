import torchkin

from torch_robotics.robots.torchkin_compat import ensure_torchkin_pose_cache_api


def test_pose_cache_compatibility_hook_is_idempotent(monkeypatch):
    monkeypatch.delattr(
        torchkin,
        "get_forward_kinematics_pose_cache_fns",
        raising=False,
    )

    assert ensure_torchkin_pose_cache_api() is True
    installed = torchkin.get_forward_kinematics_pose_cache_fns
    assert callable(installed)
    assert ensure_torchkin_pose_cache_api() is False
    assert torchkin.get_forward_kinematics_pose_cache_fns is installed
