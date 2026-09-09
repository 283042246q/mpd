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


class _MarvinPerArmEEEncoder(nn.Module):
    """Shared left/right encoder for Marvin's two fixed EE context slots."""

    def __init__(self, token_dim=128, n_layers=2, act="relu"):
        super().__init__()
        self.token_dim = token_dim
        feature_dim = token_dim // 2
        if feature_dim < 1:
            raise ValueError("Marvin arm token dimension must be positive")
        self.q_encoder = MLP(7, feature_dim, hidden_dim=token_dim, n_layers=n_layers, act=act)
        self.goal_encoder = MLP(12, feature_dim, hidden_dim=token_dim, n_layers=n_layers, act=act)
        self.inactive_goal = nn.Parameter(torch.zeros(feature_dim))
        self.arm_fusion = MLP(
            feature_dim * 2 + 1,
            token_dim,
            hidden_dim=token_dim,
            n_layers=n_layers,
            act=act,
        )
        # The encoders are shared, while this embedding preserves left/right identity.
        self.arm_identity = nn.Parameter(torch.empty(2, token_dim))
        nn.init.normal_(self.arm_identity, std=0.02)
        self.norm = nn.LayerNorm(token_dim)

    @staticmethod
    def _validate(qs, rotations, positions, mask):
        if qs is None or qs.shape[-1:] != (14,):
            raise ValueError("Marvin scheme-3 q_start must have shape (..., 14)")
        if rotations is None or rotations.shape[-2:] != (2, 9):
            raise ValueError("Marvin scheme-3 rotations must have shape (..., 2, 9)")
        if positions is None or positions.shape[-2:] != (2, 3):
            raise ValueError("Marvin scheme-3 positions must have shape (..., 2, 3)")
        if mask is None or mask.shape[-1:] != (2,):
            raise ValueError("Marvin scheme-3 requires active_ee_mask (..., 2)")

    def forward(self, qs, rotations, positions, mask):
        self._validate(qs, rotations, positions, mask)
        q_per_arm = qs.reshape(*qs.shape[:-1], 2, 7)
        pose_per_arm = torch.cat((rotations, positions), dim=-1)
        mask = mask.to(dtype=qs.dtype)

        q_features = self.q_encoder(q_per_arm)
        goal_features = self.goal_encoder(pose_per_arm)
        active = mask.unsqueeze(-1)
        inactive = self.inactive_goal.view(*([1] * (goal_features.ndim - 1)), -1)
        goal_features = active * goal_features + (1.0 - active) * inactive
        tokens = self.arm_fusion(torch.cat((q_features, goal_features, active), dim=-1))
        identity = self.arm_identity.view(*([1] * (tokens.ndim - 2)), 2, self.token_dim)
        return self.norm(tokens + identity)


class ContextModelMarvinStructuredEE(nn.Module):
    """Variant B: shared per-arm encoders followed by ordered MLP fusion."""

    def __init__(self, out_dim=128, n_layers=2, act="relu", **kwargs):
        super().__init__()
        self.in_dim = 40
        self.out_dim = out_dim
        self.arm_encoder = _MarvinPerArmEEEncoder(out_dim, n_layers=n_layers, act=act)
        self.fusion = MLP(out_dim * 2, out_dim, hidden_dim=out_dim, n_layers=n_layers, act=act)

    def forward(
        self,
        qs_normalized=None,
        ee_goal_orientation_normalized=None,
        ee_goal_position_normalized=None,
        active_ee_mask=None,
        **kwargs,
    ):
        arm_tokens = self.arm_encoder(
            qs_normalized,
            ee_goal_orientation_normalized,
            ee_goal_position_normalized,
            active_ee_mask,
        )
        return self.fusion(arm_tokens.flatten(start_dim=-2))


class _MarvinCrossArmContextBase(nn.Module):
    """Common two-arm token encoder and explicit cross-arm Transformer."""

    def __init__(
        self,
        token_dim=128,
        n_layers=2,
        act="relu",
        attention_heads=4,
        attention_layers=2,
        ff_multiplier=2,
        dropout=0.0,
    ):
        super().__init__()
        if token_dim % attention_heads:
            raise ValueError("context token dimension must be divisible by attention heads")
        self.in_dim = 40
        self.token_dim = token_dim
        self.arm_encoder = _MarvinPerArmEEEncoder(token_dim, n_layers=n_layers, act=act)
        self.pair_token = nn.Parameter(torch.empty(1, 1, token_dim))
        nn.init.normal_(self.pair_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=attention_heads,
            dim_feedforward=token_dim * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_arm_attention = nn.TransformerEncoder(
            layer,
            num_layers=attention_layers,
            norm=nn.LayerNorm(token_dim),
        )

    def encode_tokens(self, qs, rotations, positions, mask):
        arm_tokens = self.arm_encoder(qs, rotations, positions, mask)
        batch_shape = arm_tokens.shape[:-2]
        if len(batch_shape) != 1:
            raise ValueError("Marvin cross-arm context currently expects a batched [B, ...] input")
        pair = self.pair_token.expand(arm_tokens.shape[0], -1, -1)
        encoded = self.cross_arm_attention(torch.cat((pair, arm_tokens), dim=1))
        # Stable public ordering: left, right, pair.
        return torch.cat((encoded[:, 1:3], encoded[:, 0:1]), dim=1)


class ContextModelMarvinCrossArmEE(_MarvinCrossArmContextBase):
    """Variant C: two-layer cross-arm attention, then a flat joint context."""

    def __init__(self, out_dim=128, **kwargs):
        super().__init__(token_dim=out_dim, **kwargs)
        self.out_dim = out_dim
        n_layers = kwargs.get("n_layers", 2)
        act = kwargs.get("act", "relu")
        self.fusion = MLP(out_dim * 3, out_dim, hidden_dim=out_dim, n_layers=n_layers, act=act)

    def forward(
        self,
        qs_normalized=None,
        ee_goal_orientation_normalized=None,
        ee_goal_position_normalized=None,
        active_ee_mask=None,
        **kwargs,
    ):
        tokens = self.encode_tokens(
            qs_normalized,
            ee_goal_orientation_normalized,
            ee_goal_position_normalized,
            active_ee_mask,
        )
        return self.fusion(tokens.flatten(start_dim=1))


class ContextModelMarvinCrossArmTokens(_MarvinCrossArmContextBase):
    """Variant D context: return flattened [left, right, pair] tokens."""

    def __init__(self, out_dim=128, **kwargs):
        super().__init__(token_dim=out_dim, **kwargs)
        # A flat tensor remains compatible with DataParallel and diffusion warmup.
        self.out_dim = out_dim * 3

    def forward(
        self,
        qs_normalized=None,
        ee_goal_orientation_normalized=None,
        ee_goal_position_normalized=None,
        active_ee_mask=None,
        **kwargs,
    ):
        tokens = self.encode_tokens(
            qs_normalized,
            ee_goal_orientation_normalized,
            ee_goal_position_normalized,
            active_ee_mask,
        )
        return tokens.flatten(start_dim=1)


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
