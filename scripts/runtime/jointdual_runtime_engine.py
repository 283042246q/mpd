"""Independent JointDual engine using the existing service and schema-v3 contracts.

No legacy model bootstrap or training HDF5 is needed. Static geometry, dynamic
forecast queries, Phase5 physical costs, dense validator and ranking are reused.
"""
import json
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import numpy as np
import torch
import yaml

from mpd.datasets.spacetime_schema import load_panda_robot_bundle, sha256_file
from mpd.inference.dynamic_collision import FixedCapacityDynamicWorld, StaticDynamicCollisionField, DynamicWorldError
from mpd.inference.dense_trajectory_validator import DenseTrajectoryValidator
from mpd.inference.jointdual_guidance import JointDualCostGuide
from mpd.inference.jointdual_sampler import JointDualSampler, JointDualSettings
from mpd.inference.jointdual_world import predicted_world_features
from mpd.inference.inference import _compute_candidate_ranking
from mpd.joint_training.trainer import load_joint_checkpoint
from mpd.parametric_trajectory.joint_timing_codec import JointTimingCodec
from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline
from scripts.spacetime_data.warehouse_world import WarehouseCollision
from scripts.runtime.runtime_engine import PlanArtifacts
from scripts.runtime.timing_contract import attach_candidate_timing
from scripts.runtime.space_time_runtime_engine import _phase5_selection_score
from scripts.isaaclab.scene_payload import export_isaaclab_scene_payload
from scripts.runtime.infer_once import (validate_request, _pose_xyzw_to_transform, _jsonable,
    NoValidTrajectoryError, ConfigurationError, RequestValidationError, _canonical_json_sha256)
from torch_robotics.tasks.tasks import PlanningTask

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-jointdual.yaml"


class JointDualMpdRuntimeEngine:
    def __init__(self, *, joint_checkpoint, config_path=DEFAULT_CONFIG, runtime_output_root=None,
                 device_text="cpu", state_callback=None, sampler_settings=None):
        started=time.perf_counter()
        notify=state_callback or (lambda _:None);notify("LOADING")
        self.instance_id=str(uuid.uuid4());self.device=torch.device(device_text)
        if self.device.type=="cpu":torch.set_num_threads(4)
        self.tensor_args=dict(device=self.device,dtype=torch.float32)
        self.config_path=Path(config_path).resolve();self.config=yaml.safe_load(self.config_path.read_text())
        for key,default in (("candidates",64),("top_k",8),("output_points",128),("cost_points",128)):
            if not isinstance(self.config.get(key,default),int) or self.config.get(key,default)<(3 if "points" in key else 1):
                raise ConfigurationError("invalid JointDual runtime count: "+key)
        for key,default in (("ee_position_tolerance",.015),("ee_orientation_tolerance_deg",1.5),("collision_margin",.005)):
            value=float(self.config.get(key,default))
            if not np.isfinite(value) or value<0 or (key!="collision_margin" and value==0):
                raise ConfigurationError("invalid JointDual runtime tolerance: "+key)
        self.runtime_output_root=Path(runtime_output_root or "/tmp/jointdual")
        self.model,self.payload=load_joint_checkpoint(joint_checkpoint,self.device)
        self.checkpoint_path=Path(joint_checkpoint).resolve();self.checkpoint_sha256=sha256_file(self.checkpoint_path)
        self.bundle=load_panda_robot_bundle(REPO)
        if self.payload["robot_hashes"]!=self.bundle.hashes:
            raise ConfigurationError("checkpoint robot identity differs from runtime")
        for name,value in self.payload["static_geometry_hashes"].items():
            if sha256_file(REPO/"mpd/torch_robotics/torch_robotics/environments"/name)!=value:
                raise ConfigurationError("checkpoint static geometry differs from runtime")
        if self.config.get("scene_id","EnvWarehouse")!="EnvWarehouse":
            raise ConfigurationError("JointDual v1 is trained for EnvWarehouse static geometry")
        self.collision=WarehouseCollision(device_text,margin=float(self.config.get("collision_margin",.005)))
        self.robot=self.collision.robot
        self.codec=JointTimingCodec(self.payload["normalization"],self.payload["representation"],
                                   self.robot.dq_max,self.robot.ddq_max).to(self.device)
        self.settings=JointDualSettings(**dict(self.config.get("sampler",{}),**(sampler_settings or {})))
        self.sampler=JointDualSampler(self.model,self.settings)
        trajectory=ParametricTrajectoryBspline(n_control_points=21,degree=5,remove_outer_control_points=True,
            keep_last_control_point=True,num_T_pts=int(self.config.get("output_points",128)),tensor_args=self.tensor_args)
        self.planning_task=PlanningTask(env=self.collision.environment,robot=self.robot,
            parametric_trajectory=trajectory,margin_for_dense_collision_checking=self.collision.margin,
            tensor_args=self.tensor_args)
        self.static_field=self.planning_task.df_collision_objects
        self.dynamic_world=FixedCapacityDynamicWorld(int(self.config.get("max_dynamic_objects",16)),
            trajectory_duration_s=14.,tensor_args=self.tensor_args,
            capacity_buckets_enabled=True,shape_grouping_enabled=True,fused_reduction_enabled=True)
        self.dense_validator=DenseTrajectoryValidator(self.planning_task,config={"chunking":{"enabled":True,
                                                       "time_chunk_size":256}})
        self.actual_scene_id="EnvWarehouse"
        self.scene_payload=export_isaaclab_scene_payload(self.collision.environment,include_boxes=True)
        self.scene_sha256=_canonical_json_sha256(self.scene_payload)
        notify("WARMING")
        with torch.no_grad():
            p=torch.zeros(1,16,7,**self.tensor_args);z=torch.zeros(1,6,**self.tensor_args)
            task=torch.zeros(1,19,**self.tensor_args)
            self.model.denoiser(p,z,torch.zeros(1,dtype=torch.long,device=self.device),task,
                torch.zeros(1,1,13,**self.tensor_args),torch.zeros(1,1,dtype=torch.bool,device=self.device),
                torch.ones(1,dtype=torch.long,device=self.device))
            self.robot.get_EE_pose(((self.robot.q_pos_min+self.robot.q_pos_max)/2)[None])
        if self.device.type=="cuda":torch.cuda.synchronize(self.device)
        self.warmup_elapsed_sec=time.perf_counter()-started

    def health(self):
        return dict(instance_id=self.instance_id,device=str(self.device),scene_id=self.actual_scene_id,
            scene_sha256=self.scene_sha256,checkpoint_sha256=self.checkpoint_sha256,
            warmup_elapsed_sec=self.warmup_elapsed_sec,
            space_time=dict(enabled=True,mode=self.payload["variant"],representation=self.payload["representation"],
                candidate_specific_time=True,trajectory_schema_version=3,timing_schema_version=1,
                rest_to_rest_only=True,codec_contract=self.payload["codec_contract"]),
            dynamic_world=dict(world_version=self.dynamic_world.world_version),
            dense_validation=dict(enabled=True,fully_warmed=False))

    def update_world(self,snapshot):
        if snapshot.get("frame_id") not in ("fr3_link0","panda_link0"):
            raise ValueError("world must be expressed in the configured robot base frame")
        version=self.dynamic_world.update(snapshot)
        self.world_snapshot_hash=_canonical_json_sha256(snapshot)
        return version

    def _condition(self,qs,goal):
        norm=self.payload["normalization"]
        raw=torch.cat((goal[:3,3],goal[:3,:3].reshape(9)))
        return torch.cat(((qs-torch.tensor(norm["q_mean"],**self.tensor_args))/torch.tensor(norm["q_std"],**self.tensor_args),
            (raw-torch.tensor(norm["ee"]["mean"],**self.tensor_args))/torch.tensor(norm["ee"]["std"],**self.tensor_args)))

    @torch.no_grad()
    def _validate(self,context,goal,qs):
        duration=context["duration"].item()
        if not 2-1e-8<=duration<=14+1e-8 or not np.isfinite(duration):return False,{"reason":"duration"}
        # Independent dense phase and <=10ms physical-time samples share one codec.
        times=torch.linspace(0,duration,int(np.ceil(duration/.01))+1,device=self.device,dtype=torch.float64)
        checked=[]
        # Keep grids separate: near-duplicate union times can collapse in FP32.
        for phases in (self.codec.phase,self.codec.phase_at_times(context,times)[0]):
            evaluation=self.codec.evaluate(context,phases)
            values={name:getattr(evaluation,name).to(**self.tensor_args) for name in ("q","dq","ddq","time_from_start")}
            if not all(torch.isfinite(v).all() for v in values.values()):return False,{"reason":"nonfinite"}
            dense=self.dense_validator.validate(control_points=None,num_points=len(phases),
                q_position=values["q"],q_velocity=values["dq"],q_acceleration=values["ddq"],
                trajectory_times=values["time_from_start"],check_environment=True,check_self_collision=True,
                check_joint_position=True,check_joint_velocity=True,check_joint_acceleration=True)
            checked.append(dense)
        pose=self.robot.get_EE_pose(values["q"][:,-1])[0]
        position=torch.linalg.vector_norm(pose[:3,3]-goal[:3,3]).item()
        cosine=((goal[:3,:3].T@pose[:3,:3]).trace()-1)/2
        angle=torch.acos(cosine.clamp(-1,1)).item()*180/np.pi
        boundary=max((evaluation.q[0,0]-qs).abs().max().item(),evaluation.dq[0,0].abs().max().item(),
                     evaluation.ddq[0,0].abs().max().item(),evaluation.dq[0,-1].abs().max().item(),
                     evaluation.ddq[0,-1].abs().max().item())
        valid=all(bool(d.trajectory_valid_mask.item()) for d in checked) and position<=float(self.config.get("ee_position_tolerance",.015))
        valid=valid and angle<=float(self.config.get("ee_orientation_tolerance_deg",1.5)) and boundary<=1e-5
        valid=valid and all(float(d.minimum_environment_clearance[0])>self.collision.margin
                           and float(d.minimum_self_clearance[0])>self.collision.margin for d in checked)
        return valid,dict(ee_position_error=position,ee_orientation_error_deg=angle,boundary_error=boundary,
            environment_clearance=min(float(d.minimum_environment_clearance[0]) for d in checked),
            self_clearance=min(float(d.minimum_self_clearance[0]) for d in checked))

    def plan(self,raw_request,*,oracle_world=None):
        started=time.perf_counter();request=validate_request(raw_request)
        if any(raw_request.get(k) is not None for k in ("initial_x","initial_noise_timestep","initial_state")):
            raise RequestValidationError("JointDual v1 does not support warm-start")
        if request["scene_id"]!=self.actual_scene_id:raise ConfigurationError("scene_id must be EnvWarehouse")
        for k in ("q_vel_start","q_vel_goal","q_acc_start","q_acc_goal"):
            if np.any(request[k]!=0):raise RequestValidationError("JointDual v1 requires rest-to-rest boundaries")
        qs=torch.as_tensor(request["q_pos_start"],**self.tensor_args)
        if ((qs<self.robot.q_pos_min)|(qs>self.robot.q_pos_max)).any():raise RequestValidationError("start outside joint limits")
        if request["goal_type"]=="joint":
            qgoal=torch.as_tensor(request["q_pos_goal"],**self.tensor_args)
            if ((qgoal<self.robot.q_pos_min)|(qgoal>self.robot.q_pos_max)).any():raise RequestValidationError("goal outside joint limits")
            goal=self.robot.get_EE_pose(qgoal)[0]
        else:
            goal=torch.tensor(_pose_xyzw_to_transform(request["ee_pose_goal"])[:3,:4],**self.tensor_args)
        if oracle_world is None:
            if any(isinstance(raw_request.get(k),bool) or not isinstance(raw_request.get(k),int)
                   for k in ("_dynamic_world_version","_trajectory_start_unix_ns")):
                raise RequestValidationError("dynamic requests require integer world version and handoff")
            try:
                self.dynamic_world.set_plan_start(raw_request.get("_trajectory_start_unix_ns"),
                                                  world_version=raw_request.get("_dynamic_world_version"))
            except DynamicWorldError as exc:
                raise RequestValidationError(str(exc)) from exc
            world=self.dynamic_world
            if self.payload["variant"]=="jd_world":
                try:
                    features,mask,state=predicted_world_features(world)
                except ValueError as exc:
                    raise RequestValidationError(str(exc)) from exc
            else:
                features,mask,state=np.zeros((1,13),np.float32),np.zeros(1,bool),2
        else:
            world=oracle_world;features,mask,state=world.features()
        if oracle_world is not None:
            world_hash=_canonical_json_sha256({k:getattr(world,k).cpu().tolist()
                for k in ("times","positions","shapes","sizes")})
        else:
            world_hash=self.world_snapshot_hash
        self.planning_task.df_collision_objects=StaticDynamicCollisionField(self.static_field,world)
        count=int(self.config.get("candidates",64))
        task=self._condition(qs,goal)[None].expand(count,-1)
        generator=torch.Generator(device=self.device).manual_seed(request["seed"])
        guide=JointDualCostGuide(self.codec,self.collision,world,qs,goal,
                                cost_points=int(self.config.get("cost_points",128)))
        feature_tensor=torch.tensor(features,**self.tensor_args)[None]
        mask_tensor=torch.tensor(mask,device=self.device)[None]
        encoding_started=time.perf_counter()
        with torch.no_grad():
            encoded=self.model.denoiser.encode_world(feature_tensor,mask_tensor,
                torch.tensor([state],device=self.device))
            if encoded is not None:encoded=tuple(v.expand(count,*v.shape[1:]) for v in encoded)
        if self.device.type=="cuda":torch.cuda.synchronize(self.device)
        encoding_s=time.perf_counter()-encoding_started
        sample=self.sampler.sample(task,world_features=feature_tensor.expand(count,-1,-1),
            world_mask=mask_tensor.expand(count,-1),encoded_world=encoded,
            world_state=torch.full((count,),state,device=self.device,dtype=torch.long),
            guide=guide if self.config.get("guidance",True) else None,generator=generator)
        generated=time.perf_counter()
        valid_ids=[];diagnostics=[]
        for i in range(count):
            with torch.no_grad():
                context=self.codec.prepare(self.codec.full_path(sample.p[i:i+1],qs),sample.z[i:i+1])
                valid,details=self._validate(context,goal,qs)
            diagnostics.append(dict(candidate=i,valid=valid,**details))
            if valid:valid_ids.append(i)
        if not valid_ids:raise NoValidTrajectoryError(f"no valid JointDual candidate among {count}; {diagnostics}")
        with torch.no_grad():
            ids=torch.tensor(valid_ids,device=self.device)
            p,z=sample.p[ids],sample.z[ids]
            _,parts,context,state=guide.evaluate(p,z)
            score,_=_compute_candidate_ranking("weighted_metrics",self.config,
                SimpleNamespace(context_ee_goal_pose=True),self.planning_task,None,p,
                state.q.float(),state.timing.dq.float(),state.timing.ddq.float(),goal)
            score,_=_phase5_selection_score(score,parts["dynamic_collision"],context["duration"],
                parts["timing_smoothness"],duration_min=2.,duration_max=14.)
            order=score.argsort()[:int(self.config.get("top_k",8))]
            selected=ids[order];p,z=sample.p[selected],sample.z[selected]
            context=self.codec.prepare(self.codec.full_path(p,qs),z)
            output_times=context["duration"][:,None]*torch.linspace(0,1,int(self.config.get("output_points",128)),
                device=self.device,dtype=torch.float64)[None]
            evaluation=self.codec.evaluate(context,self.codec.phase_at_times(context,output_times))
            spheres=guide.centers(evaluation.q.reshape(-1,7)).reshape(len(p),evaluation.q.shape[1],-1,3)
        arrays={"topk_positions":evaluation.q.cpu().numpy(),"topk_velocities":evaluation.dq.cpu().numpy(),
            "topk_accelerations":evaluation.ddq.cpu().numpy(),"topk_scores":score[order].cpu().numpy(),
            "topk_source_candidate_indices":selected.cpu().numpy(),"joint_names":np.asarray(request["joint_names"]),
            "topk_collision_sphere_positions":spheres.cpu().numpy(),
            "collision_sphere_radii":self.robot.link_collision_spheres_radii.cpu().numpy(),
            "timing_representation":np.asarray(self.payload["representation"])}
        for name,value in self.codec.export(z).items():arrays[name]=value.cpu().numpy()
        payload=dict(schema_version=1,status="success",request_id=request["request_id"],
            joint_names=request["joint_names"],trajectory_file="trajectory.npz",
            request=dict(seed=request["seed"],sha256=_canonical_json_sha256(raw_request),
                         robot_model=request["robot_model"],planning_frame=request["planning_frame"]),
            goal=dict(input_type=request["goal_type"],ee_pose_goal=goal,q_pos_goal_condition=None,
                      end_effector_frame=self.bundle.ee_link),
            scene=dict(scene_id=self.actual_scene_id,mpd_scene_sha256=self.scene_sha256),
            model=dict(checkpoint_path=str(self.checkpoint_path),checkpoint_sha256=self.checkpoint_sha256),
            trajectory=dict(timing_representation=self.payload["representation"]),
            trajectory_artifact=dict(best_trajectory_deduplicated=True),
            candidates=dict(generated=count,
                dense_checked=sum("reason" not in d for d in diagnostics),
                bounds_rejected=sum(d.get("reason")=="duration" for d in diagnostics),
                evaluated=count,dense_complete=True,valid=len(valid_ids),returned_top_k=len(p)),
            jointdual=dict(variant=self.payload["variant"],representation=self.payload["representation"],
                sampler_steps=sample.statistics,guide_statistics=guide.statistics,
                spatial_cost_recipe="legacy_sdf_hinge_se3_phase_mean_v1",
                codec_contract=self.payload["codec_contract"],diagnostics=diagnostics),
            dynamic_world=dict(world_version=raw_request.get("_dynamic_world_version"),
                trajectory_start_unix_ns=raw_request.get("_trajectory_start_unix_ns"),fixed_timing=False,
                world_sha256=world_hash,frame_id="panda_link0" if oracle_world is not None else world.frame_id,
                snapshot_stamp_unix_ns=None if oracle_world is not None else world.stamp_unix_ns,
                valid_until_unix_ns=None if oracle_world is not None else world.valid_until_unix_ns,
                prediction="oracle" if oracle_world is not None else "constant_velocity"),
            timing=dict(generator_sec=generated-started,dense_validation_and_ranking_sec=time.perf_counter()-generated,
                world_encoding_sec=encoding_s+sum(s.get("elapsed_s",0.) for s in sample.statistics),
                denoiser_sec=sum(s.get("denoiser_s",0.) for s in sample.statistics),
                guide_sec=sum(s.get("guide_s",0.) for s in sample.statistics),
                request_total_sec=time.perf_counter()-started))
        artifacts=PlanArtifacts(_jsonable(payload),arrays)
        return attach_candidate_timing(artifacts,topk_time_from_start=evaluation.time_from_start.cpu().numpy(),
            timing_mode=self.payload["variant"],duration_min=2.,duration_max=14.)
