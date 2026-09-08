import torch
import torch.nn as nn

from mpd.models.layers.layers import MLP


class ContextModelEEPoseGoal(nn.Module):

    def __init__(self, out_dim=64, n_layers=2, act="relu", **kwargs):
        super().__init__()
        # 9d representation of rotation
        # 3d representation of position
        self.in_dim = 9 + 3
        self.out_dim = out_dim

        # self.net = nn.Identity()
        self.net = MLP(self.in_dim, out_dim, hidden_dim=out_dim, n_layers=n_layers, act=act)

    def forward(self, ee_goal_orientation_normalized, ee_goal_position_normalized, **kwargs):
        pose_repr = torch.cat((ee_goal_orientation_normalized, ee_goal_position_normalized), dim=-1)
        emb = self.net(pose_repr)
        return emb


class ContextModelMarvinDualEE(nn.Module):
    """Encode the exact 40-D Marvin scheme-3 conditioning vector."""

    def __init__(self, out_dim=128, n_layers=2, act="relu", **kwargs):
        super().__init__()
        self.in_dim = 14 + 12 + 12 + 2
        self.out_dim = out_dim
        self.net = MLP(self.in_dim, out_dim, hidden_dim=out_dim, n_layers=n_layers, act=act)

    def build_raw_context(
        self,
        qs_normalized,
        ee_goal_orientation_normalized,
        ee_goal_position_normalized,
        active_ee_mask,
    ):
        if qs_normalized.shape[-1:] != (14,):
            raise ValueError("Marvin scheme-3 q_start must have shape (..., 14)")
        if ee_goal_orientation_normalized.shape[-2:] != (2, 9):
            raise ValueError("Marvin scheme-3 rotations must have shape (..., 2, 9)")
        if ee_goal_position_normalized.shape[-2:] != (2, 3):
            raise ValueError("Marvin scheme-3 positions must have shape (..., 2, 3)")
        if active_ee_mask is None or active_ee_mask.shape[-1:] != (2,):
            raise ValueError("Marvin scheme-3 requires active_ee_mask (..., 2)")
        return torch.cat(
            (
                qs_normalized,
                ee_goal_orientation_normalized[..., 0, :],
                ee_goal_position_normalized[..., 0, :],
                ee_goal_orientation_normalized[..., 1, :],
                ee_goal_position_normalized[..., 1, :],
                active_ee_mask.to(dtype=qs_normalized.dtype),
            ),
            dim=-1,
        )

    def forward(
        self,
        qs_normalized=None,
        ee_goal_orientation_normalized=None,
        ee_goal_position_normalized=None,
        active_ee_mask=None,
        **kwargs,
    ):
        raw_context = self.build_raw_context(
            qs_normalized,
            ee_goal_orientation_normalized,
            ee_goal_position_normalized,
            active_ee_mask,
        )
        if raw_context.shape[-1] != self.in_dim:
            raise RuntimeError(f"expected a 40-D context, got {raw_context.shape[-1]}")
        return self.net(raw_context)


class ContextModelQs(nn.Module):

    def __init__(self, in_dim, out_dim=64, n_layers=2, act="relu", **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # self.net = nn.Identity()
        self.net = MLP(self.in_dim, out_dim, hidden_dim=out_dim, n_layers=n_layers, act=act)

    def forward(self, qs_normalized=None, **kwargs):
        emb = self.net(qs_normalized)
        return emb


class ContextModelCombined(nn.Module):

    def __init__(
        self, context_model_qs=None, context_model_ee_pose_goal=None, out_dim=64, n_layers=1, act="relu", **kwargs
    ):
        assert not (context_model_qs is None and context_model_ee_pose_goal is None)
        super().__init__()

        self.context_model_qs = context_model_qs
        self.context_model_ee_pose_goal = context_model_ee_pose_goal

        self.in_dim = 0
        if self.context_model_qs is not None:
            self.in_dim += self.context_model_qs.out_dim
        if self.context_model_ee_pose_goal is not None:
            self.in_dim += self.context_model_ee_pose_goal.out_dim

        self.out_dim = out_dim
        self.net = MLP(self.in_dim, self.out_dim, hidden_dim=out_dim, n_layers=n_layers, act=act)

    def forward(
        self, qs_normalized=None, ee_goal_orientation_normalized=None, ee_goal_position_normalized=None, **kwargs
    ):
        emb_q = None
        if self.context_model_qs is not None:
            emb_q = self.context_model_qs(qs_normalized)

        emb_ee_goal_pose = None
        if self.context_model_ee_pose_goal is not None:
            emb_ee_goal_pose = self.context_model_ee_pose_goal(
                ee_goal_orientation_normalized,
                ee_goal_position_normalized,
                active_ee_mask=kwargs.get("active_ee_mask"),
            )

        if emb_q is not None and emb_ee_goal_pose is not None:
            emb = torch.cat((emb_q, emb_ee_goal_pose), dim=-1)
        elif emb_q is not None:
            emb = emb_q
        elif emb_ee_goal_pose is not None:
            emb = emb_ee_goal_pose

        context_emb = self.net(emb)
        return context_emb
