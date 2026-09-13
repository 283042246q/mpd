"""Opt-in learned F1/F2/F3 runtime; old Phase-4/5 entry points are unchanged."""
from pathlib import Path

import numpy as np
import torch

from mpd.datasets.spacetime_schema import load_panda_robot_bundle, sha256_file
from mpd.inference.factorized_guidance import FactorizedCostGuide
from mpd.inference.factorized_sampler import FactorizedSettings, FactorizedSampler, FactorizedModelAdapter
from mpd.inference.learned_timing import LearnedTimingCodec, load_timing_checkpoint
from scripts.runtime.space_time_runtime_engine import SpaceTimeMpdRuntimeEngine


class FactorizedMpdRuntimeEngine(SpaceTimeMpdRuntimeEngine):
    def __init__(self, *args, timing_checkpoint, factorized_settings=None, adapt_spatial_basis=False, **kwargs):
        self.factorized_settings = FactorizedSettings(**(factorized_settings or {}))
        self.timing_checkpoint = Path(timing_checkpoint).expanduser().resolve()
        self.adapt_spatial_basis = bool(adapt_spatial_basis)
        # Fail early on known corrupted timing checkpoints, before loading the
        # spatial scene/model and running its warmup.
        timing_model, payload = load_timing_checkpoint(self.timing_checkpoint)
        kwargs["timing_mode"] = "phase5_joint"  # internal validator compatibility
        settings = dict(kwargs.pop("space_time_settings", {}) or {})
        settings.setdefault("duration_min", 2.)
        settings.setdefault("duration_max", 14.)
        kwargs["space_time_settings"] = settings
        super().__init__(*args, **kwargs)
        if self.args_inference.planner_alg != "mpd":
            raise ValueError("factorized inference requires planner_alg: mpd")
        self._validate_contract(payload)
        if self.spatial_basis_adapted:
            from scipy.interpolate import BSpline
            from mpd.datasets.spacetime_legacy import open_uniform_knots
            encoder = timing_model.denoiser.path_encoder
            count = int(self.planning_task.parametric_trajectory.bspline.n_pts)
            basis = BSpline(open_uniform_knots(count, encoder.degree), np.eye(count), encoder.degree)
            phase = np.linspace(0., 1., encoder.num_phase_points)
            for name, order in (("basis", 0), ("basis_d1", 1), ("basis_d2", 2)):
                setattr(encoder, name, torch.tensor(basis.derivative(order)(phase), dtype=torch.float32))
            encoder.num_control_points = count
        timing_model = timing_model.to(self.device)
        codec = LearnedTimingCodec(payload["normalization"], num_phase_points=int(self.args_inference.num_T_pts),
            duration_min=self.space_time_settings.duration_min, duration_max=self.space_time_settings.duration_max,
            u_min=self.space_time_settings.u_min, velocity_limits=self.planning_task.robot.dq_max,
            acceleration_limits=self.planning_task.robot.ddq_max).to(**self.tensor_args)
        old = self.space_time_guide
        self.space_time_guide = FactorizedCostGuide(old.spatial_guide, self.planning_task, self.planner.dataset,
            self.dynamic_field, self.space_time_settings, self.tensor_args,
            codec=codec, factorized_settings=self.factorized_settings)
        self.planner.cost_guide = self.space_time_guide
        self.factorized_sampler = FactorizedSampler(self.planner.model, timing_model,
            self.space_time_guide, self.factorized_settings)
        self.planner.model = FactorizedModelAdapter(self.factorized_sampler)
        self.timing_checkpoint_step = int(payload["step"])
        self.timing_environment = payload["environment"]
        self.timing_representation = payload["representation"]
        self.timing_checkpoint_sha256 = sha256_file(self.timing_checkpoint)

    def _validate_contract(self, payload):
        contract = payload["dataset_identity"]["contract"]
        trajectory = self.planning_task.parametric_trajectory
        expected = (int(trajectory.bspline.n_pts), int(trajectory.bspline.d), self.planning_task.robot.q_dim)
        actual = (contract["spatial_num_control_points"], contract["spatial_degree"], contract["dof"])
        self.spatial_basis_adapted = expected[0] != actual[0]
        if expected[1:] != actual[1:] or (self.spatial_basis_adapted and not self.adapt_spatial_basis):
            raise ValueError(f"timing/spatial spline contract mismatch: runtime={expected}, checkpoint={actual}")
        model = payload["model_config"]
        if (model["num_spatial_control_points"], model["spatial_degree"], model["dof"]) != actual:
            raise ValueError("timing model dimensions disagree with its dataset contract")
        if payload["normalization"]["representation"] != payload["representation"]:
            raise ValueError("timing normalization representation mismatch")
        if (contract["timing_num_control_points"], contract["timing_degree"]) != (8, 3):
            raise ValueError("learned timing requires the 8-control-point cubic contract")
        if (self.space_time_settings.num_timing_control_points, self.space_time_settings.timing_degree) != (8, 3):
            raise ValueError("runtime timing spline must use 8 control points and degree 3")
        bundle = load_panda_robot_bundle(Path(__file__).resolve().parents[2])
        keys = ("urdf_sha256", "joint_limits_sha256", "collision_spheres_sha256", "collision_parent_bounds_sha256")
        if tuple(contract["robot_hashes"]) != tuple(bundle.hashes[k] for k in keys):
            raise ValueError("timing checkpoint robot fingerprint differs from runtime Panda assets")
        robot = self.planning_task.robot
        for name, value in (("dq_max", bundle.dq_max), ("ddq_max", bundle.ddq_max)):
            if not np.allclose(getattr(robot, name).detach().cpu().numpy(), value):
                raise ValueError(f"runtime {name} differs from canonical timing limits")

    def health(self):
        response = super().health()
        if hasattr(self, "timing_representation"):
            response["factorized"] = dict(settings=self.factorized_settings.__dict__,
                representation=self.timing_representation, timing_checkpoint=str(self.timing_checkpoint),
                timing_checkpoint_step=self.timing_checkpoint_step, timing_environment=self.timing_environment,
                timing_checkpoint_sha256=self.timing_checkpoint_sha256,
                spatial_basis_adapted=self.spatial_basis_adapted,
                rest_to_rest_only=True, f3_rollout_finetuning_verified=False)
        return response

    def plan(self, raw_request):
        for name in ("q_vel_start", "q_vel_goal", "q_acc_start", "q_acc_goal"):
            if np.any(np.asarray(raw_request.get(name, [0.] * 7)) != 0.):
                raise ValueError("factorized learned timing currently supports rest-to-rest requests only")
        artifacts = super().plan(raw_request)
        arrays = artifacts.trajectory_arrays
        # Never label tau/r (or normalized latents) as physical c control points.
        latent = arrays.pop("timing_control_points")
        codec = self.space_time_guide.codec
        raw = latent * codec.target_std.detach().cpu().numpy() + codec.target_mean.detach().cpu().numpy()
        arrays["timing_latents"] = raw
        arrays["timing_representation"] = np.asarray(self.timing_representation)
        if self.timing_representation == "c":
            arrays["timing_control_points"] = raw[:, [0, 0, 1, 2, 3, 4, 5, 5]]
        else:
            arrays["timing_tau"] = raw[:, 0]
            arrays["timing_shape_control_points"] = raw[:, 1:]
        metadata = self.health()["factorized"]
        metadata["steps"] = self.factorized_sampler.statistics
        metadata["denoiser_evaluations"] = {
            branch: sum(s["branch"] == branch for s in self.factorized_sampler.statistics)
            for branch in ("space", "timing")}
        artifacts.result_payload["factorized"] = metadata
        artifacts.result_payload["trajectory"]["timing_mode"] = self.factorized_settings.method
        artifacts.result_payload["trajectory"]["timing_representation"] = self.timing_representation
        return artifacts
