import json
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import pytest
import torch
import yaml

from mpd.inference.jointdual_sampler import JointDualSampler,JointDualSettings,JointSampleResult
from mpd.inference.jointdual_world import OracleForecastWorld,predicted_world_features
from mpd.models.diffusion_models.jointdual import JointDualDenoiser,JointDiffusion
from scripts.runtime import infer_jointdual_server
from scripts.runtime.jointdual_runtime_engine import JointDualMpdRuntimeEngine
from scripts.runtime.infer_once import EXPECTED_JOINT_NAMES,NoValidTrajectoryError,RequestValidationError
from scripts.spacetime_data.warehouse_world import Forecast,ForecastBatch
from test_jointdual_training import SMALL


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("variant",["jd_kin","jd_world"])
@pytest.mark.parametrize("architecture",["v1","v2"])
def test_joint_sampler_cfg_and_budgets(variant,architecture):
    model=JointDiffusion(JointDualDenoiser(variant=variant,architecture_version=architecture,**SMALL),num_diffusion_steps=10)
    if variant == "jd_world" and architecture == "v2":
        with torch.no_grad():
            model.denoiser.timing_attention.gates[-1].bias.fill_(.3)
            model.denoiser.spatial_attention.gates[-1].bias.fill_(.3)
    sampler=JointDualSampler(model,JointDualSettings(steps=4,refinement_steps=2,cfg_scale=1.5))
    class Guide:
        calls=0
        def refine(self,p,z,settings):
            self.calls+=1
            return p*.99,z*.99
    guide=Guide()
    task=torch.randn(2,19)
    args=dict(world_features=torch.randn(2,1,13),world_mask=torch.zeros(2,1,dtype=torch.bool),
              world_state=torch.tensor([1,2]),guide=guide)
    one=sampler.sample(task,generator=torch.Generator().manual_seed(1),**args)
    two=sampler.sample(task,generator=torch.Generator().manual_seed(1),**args)
    torch.testing.assert_close(one.p,two.p,atol=0,rtol=0)
    torch.testing.assert_close(one.z,two.z,atol=0,rtol=0)
    assert one.p.shape==(2,16,7) and one.z.shape==(2,6)
    assert guide.calls==8  # two late steps + two final steps per call
    assert sum(s["denoiser_evaluations"] for s in one.statistics)==(8 if variant=="jd_world" else 4)
    with pytest.raises(ValueError,match="warm-start"):
        sampler.sample(task,initial_state=(one.p,one.z),**args)


def test_oracle_sdf_and_time_gradient_match_generator():
    times=np.array([0.,7.,14.]);positions=np.array([[[0.,0.,0.],[2.,0.,0.],[3.,0.,0.]],
        [[1.,0.,0.],[2.,1.,0.],[3.,0.,1.]],[[0.,0.,0.],[2.,0.,0.],[3.,0.,0.]]])
    shapes=np.array([0,1,2]);sizes=np.array([[.2,0,0],[.1,.2,.3],[.1,.3,0]])
    world=OracleForecastWorld(times,positions,shapes,sizes)
    forecast=Forecast(times,positions,shapes,sizes,np.array([3.,4.,5.]),["linear"]*3,np.zeros((3,3)))
    reference=ForecastBatch([forecast],dtype=torch.float64)
    query_times=torch.tensor([[1.,3.,5.]],dtype=torch.float64,requires_grad=True)
    points=torch.tensor([[[[.6,.2,.1]],[[.6,.2,.1]],[[.6,.2,.1]]]],dtype=torch.float64,requires_grad=True)
    distances=world.minimum_signed_distance(points,query_times)
    expected=reference.clearance(points[0],torch.zeros(1,dtype=torch.float64),query_times[0],torch.zeros(3,dtype=torch.long))
    torch.testing.assert_close(distances[0,:,0],expected)
    gp,gt=torch.autograd.grad(distances.sum(),(points,query_times))
    assert torch.isfinite(gp).all() and torch.isfinite(gt).all() and gt.norm()>0


def test_prediction_tokens_use_handoff_offset():
    from mpd.inference.dynamic_collision import FixedCapacityDynamicWorld,DynamicWorldError
    world=FixedCapacityDynamicWorld(2,trajectory_duration_s=14.,tensor_args=dict(device="cpu",dtype=torch.float64))
    world.update(dict(world_version=1,frame_id="fr3_link0",stamp_unix_ns=1_000_000_000,
        valid_until_unix_ns=30_000_000_000,objects=[dict(id="one",local_sdf=dict(type="sphere",radius=.1),
            pose=dict(position=[0.,0.,0.],orientation_xyzw=[0.,0.,0.,1.]),linear_velocity=[1.,0.,0.])]))
    world.set_plan_start(2_000_000_000,world_version=1)
    features,mask,state=predicted_world_features(world)
    assert features[0,0]==pytest.approx(1.) and features[-1,0]==pytest.approx(15.)
    world.set_plan_start(3_000_000_000,world_version=1)
    assert predicted_world_features(world)[0][0,0]==pytest.approx(2.)
    with pytest.raises(DynamicWorldError):world.set_plan_start(20_000_000_000,world_version=1)


def test_new_server_is_independent(tmp_path,monkeypatch):
    captured={}
    class Engine:
        def __init__(self,**kwargs):captured.update(kwargs)
    class Service:
        def __init__(self,socket,root,factory,**kwargs):factory(lambda _:None)
        def serve_forever(self):pass
    monkeypatch.setattr(infer_jointdual_server,"JointDualMpdRuntimeEngine",Engine)
    monkeypatch.setattr(infer_jointdual_server,"DynamicResidentPlannerService",Service)
    assert infer_jointdual_server.main(["--joint-checkpoint",str(tmp_path/"j.pt"),"--socket",str(tmp_path/"s.sock"),
        "--output-root",str(tmp_path/"out"),"--device","cpu"])==0
    assert captured["joint_checkpoint"]==tmp_path/"j.pt"


@pytest.mark.parametrize("representation",["c","tau_r"])
def test_real_engine_ground_truth_candidate_and_guidance(representation,tmp_path,monkeypatch):
    # Optional local integration fixture created by the real P21 raw-refit smoke.
    root=Path("/tmp/jointdual_p21_eegoal_smoke_final")
    checkpoint=Path(f"/tmp/jointdual_four_smoke/jd_world_{representation}/last.pt")
    if not root.exists() or not checkpoint.exists():pytest.skip("local real-data smoke artifacts unavailable")
    config=dict(scene_id="EnvWarehouse",candidates=1,top_k=1,output_points=128,guidance=True,
                sampler=dict(steps=4,refinement_steps=1))
    cfg=tmp_path/"config.yaml";cfg.write_text(yaml.safe_dump(config))
    engine=JointDualMpdRuntimeEngine(joint_checkpoint=checkpoint,config_path=cfg,device_text="cpu")
    e=json.loads((root/"index.json").read_text())["splits"]["train"][0]
    with h5py.File(root/e["file"]) as f:
        g=f[e["group"]]
        qs=g["condition/q_start"][:];qg=g["condition/q_goal"][:]
        p=torch.tensor(g["spatial/learnable_control_points"][:],dtype=torch.float32)[None]
        p=(p-engine.codec.q_mean.float())/engine.codec.q_std.float()
        raw=g["timing/tau_r"][:] if representation=="tau_r" else g["timing/control_points"][:][[0,2,3,4,5,7]]
        z=(torch.tensor(raw,dtype=torch.float32)[None]-engine.codec.target_mean.float())/engine.codec.target_std.float()
        world=OracleForecastWorld(g["world/times"][:],g["world/positions"][:],g["world/shape"][:],g["world/size"][:])
    request=dict(scene_id="EnvWarehouse",joint_names=list(EXPECTED_JOINT_NAMES),goal_type="joint",
                 q_pos_start=qs.tolist(),q_pos_goal=qg.tolist())
    from mpd.inference.jointdual_guidance import JointDualCostGuide
    goal=engine.robot.get_EE_pose(torch.tensor(qg,dtype=torch.float32))[0]
    guide=JointDualCostGuide(engine.codec,engine.collision,world,torch.tensor(qs,dtype=torch.float32),goal,cost_points=32)
    changed=guide.refine(p,z,engine.settings)
    assert all(torch.isfinite(x).all() for x in changed)
    monkeypatch.setattr(engine.sampler,"sample",lambda *a,**kw:JointSampleResult(p,z,[]))
    artifacts=engine.plan(request,oracle_world=world)
    a=artifacts.trajectory_arrays
    assert a["artifact_schema_version"]==3 and a["timing_schema_version"]==1
    assert a["topk_positions"].shape==(1,128,7)
    assert np.all(np.diff(a["topk_time_from_start"],axis=1)>0)
    assert artifacts.result_payload["trajectory"]["timing_representation"]==representation
    # A different EE goal cannot be accepted merely because the path is collision-free.
    bad=dict(request,q_pos_goal=(qg+np.array([.2,0,0,0,0,0,0])).tolist())
    with pytest.raises(NoValidTrajectoryError):engine.plan(bad,oracle_world=world)
    moving=dict(request,q_vel_start=[.1]+[0.]*6)
    with pytest.raises(RequestValidationError,match="rest-to-rest"):engine.plan(moving,oracle_world=world)
    # The unchanged dynamic service enforces request sequence/world/deadline and
    # writes the new engine's candidate-specific NPZ through its existing path.
    from scripts.runtime.infer_dynamic_server import DynamicResidentPlannerService
    service=DynamicResidentPlannerService(tmp_path/"worker.sock",tmp_path/"service",lambda _:engine,
                                          trajectory_compression=False)
    service._engine=engine;service.set_state("READY")
    response=service.dispatch(dict(schema_version=1,op="update_world",world=dict(world_version=1,
        frame_id="fr3_link0",stamp_unix_ns=1,valid_until_unix_ns=2**63-1,objects=[])))
    assert response["status"]=="OK"
    response=service.dispatch(dict(schema_version=1,op="plan",request_seq=1,world_version=1,
                                   trajectory_start_unix_ns=1,request=request))
    assert response["status"]=="OK",response
    with np.load(response["trajectory_path"],allow_pickle=False) as saved:
        assert saved["timing_schema_version"]==1
        assert saved["topk_time_from_start"].shape==(1,128)
    rejected=service.dispatch(dict(schema_version=1,op="plan",request_seq=2,world_version=1,
                                   trajectory_start_unix_ns=1,request=moving))
    assert rejected["status"]=="PLAN_FAILED" and service.state=="READY"
