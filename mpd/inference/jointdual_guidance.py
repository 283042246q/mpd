"""Physical joint guidance reusing Phase5 time-weighted costs."""
import torch
from torchlie.functional import SE3 as SE3_Func

from mpd.inference.space_time_guidance import (
    SpaceTimeCostEvaluator, SpaceTimeTrajectoryState, SpaceTimeGuidanceSettings, _clip_per_candidate,
)
from mpd.parametric_trajectory.timing_spline import TimingSplineEvaluation


class _EmptyWorld:
    def minimum_signed_distance(self,points,**kwargs):
        return torch.full(points.shape[:-1],float("inf"),device=points.device,dtype=points.dtype)


class JointDualCostGuide:
    def __init__(self,codec,collision,world,q_start,ee_goal,*,cost_points=128,settings=None):
        self.codec,self.collision,self.world=codec,collision,world
        self.q_start,self.ee_goal=q_start,ee_goal
        self.cost_points=cost_points
        self.statistics=[]
        self.settings=settings or SpaceTimeGuidanceSettings(duration_min=2.,duration_max=14.)
        robot=collision.robot
        kwargs=dict(collision_margins=robot.link_collision_spheres_radii,cutoff_margin=collision.margin,
            velocity_limits=robot.dq_max,acceleration_limits=robot.ddq_max,settings=self.settings)
        self.base_cost=SpaceTimeCostEvaluator(None,_EmptyWorld(),**kwargs)
        self.dynamic_cost=SpaceTimeCostEvaluator(None,world,**kwargs)

    def centers(self,q):
        q=q.to(**self.collision.tensor_args)
        parent=torch.stack(self.collision.robot.fk_collision_sphere_parent_links(q),dim=1)
        selected=parent[:,self.collision.robot.collision_sphere_parent_indices]
        return torch.einsum("bsij,sj->bsi",selected[...,:3,:3],
            self.collision.robot.collision_sphere_local_positions)+selected[...,:3,3]

    def state(self,p,z,phase=None):
        full=self.codec.full_path(p,self.q_start)
        context=self.codec.prepare(full,z)
        if phase is None:
            phase=torch.linspace(0,1,self.cost_points,device=p.device,dtype=torch.float64)
        timing=self.codec.evaluate(context,phase)
        positions=self.centers(timing.q.reshape(-1,7)).reshape(len(p),timing.q.shape[1],-1,3)
        return context,SpaceTimeTrajectoryState(timing.q,None,None,timing,collision_sphere_positions=positions)

    def evaluate(self,p,z):
        context,state=self.state(p,z)
        total,breakdown,_=self.base_cost(trajectory_state=state)
        feasible=torch.isfinite(state.timing.duration)&(state.timing.duration<=14+1e-8)&(state.timing.duration>=2-1e-8)
        risk=total*0
        ids=torch.nonzero(feasible,as_tuple=False).flatten()
        if len(ids):
            timing=TimingSplineEvaluation(*[getattr(state.timing,k).to(state.collision_sphere_positions).index_select(0,ids)
                for k in ("phase","log_density","u","u_s","time_from_start","duration","q","dq","ddq")])
            subset=SpaceTimeTrajectoryState(timing.q,None,None,timing,
                                           collision_sphere_positions=state.collision_sphere_positions[ids])
            _,parts,_=self.dynamic_cost(trajectory_state=subset)
            risk=risk.index_copy(0,ids,parts["dynamic_collision"].to(risk))
            if self.settings.dynamic_guidance_enabled:
                total=total+self.settings.dynamic_collision_weight*risk
        robot=self.collision.robot
        positions=state.collision_sphere_positions
        # Some Warehouse primitives expose SDF values but no analytic gradient
        # method. Autograd through their SDF preserves the same hinge objective.
        static=self.collision.field.object_signed_distances(positions).amin(-2)
        margins=robot.link_collision_spheres_radii.to(static)+self.collision.margin
        static_penalty=torch.relu(margins-static).sum(-1).mean(-1)
        # Reuse the self field's stable analytic gradient without retaining every
        # sphere pair in the joint graph.
        with torch.no_grad():
            self_cost,self_grad=robot.df_collision_self.compute_distance_field_cost_and_gradient(
                positions.detach(),pair_chunk_size=4096)
        delta=positions-positions.detach()
        self_penalty=(self_cost+(delta*self_grad).sum((-1,-2))).mean(-1)
        achieved=robot.get_EE_pose(state.q[:,-1].to(**self.collision.tensor_args))
        goal=self.ee_goal.to(achieved)
        pose_error=SE3_Func.log(SE3_Func.compose(goal.expand_as(achieved),SE3_Func.inv(achieved)))
        position_error=.5*pose_error[...,:3].square().sum(-1)
        orientation_error=.5*pose_error[...,3:].square().sum(-1)
        deadline=torch.relu(context["duration"]-14).square()+torch.relu(2-context["duration"]).square()
        total=total+10*static_penalty+self_penalty+position_error+orientation_error+10*deadline
        return total,dict(breakdown,dynamic_collision=risk,ee_position=position_error,
                          ee_orientation=orientation_error),context,state

    @torch.enable_grad()
    def refine(self,p,z,settings):
        pp=p.detach().requires_grad_();zz=z.detach().requires_grad_()
        total,_,_,_=self.evaluate(pp,zz)
        gp,gz=torch.autograd.grad(total.sum(),(pp,zz))
        finite=torch.isfinite(gp).flatten(1).all(-1)&torch.isfinite(gz).all(-1)
        gp=torch.where(finite[:,None,None],gp,torch.zeros_like(gp))
        gz=torch.where(finite[:,None],gz,torch.zeros_like(gz))
        gp,p_norm,p_clipped=_clip_per_candidate(gp,settings.space_clip)
        gz,z_norm,z_clipped=_clip_per_candidate(gz,settings.timing_clip)
        self.statistics.append(dict(p_gradient_norm=p_norm.detach().cpu().tolist(),
            z_gradient_norm=z_norm.detach().cpu().tolist(),p_clipped=p_clipped.detach().cpu().tolist(),
            z_clipped=z_clipped.detach().cpu().tolist(),finite=finite.detach().cpu().tolist()))
        proposed_p=(p-settings.space_lr*gp).clamp(-1,1)
        proposed_z=z-settings.timing_lr*gz
        with torch.no_grad():
            def violation(a,b):
                t=self.codec.prepare(self.codec.full_path(a,self.q_start),b)["duration"]
                return torch.relu(t-14)+torch.relu(2-t)
            baseline=violation(p,z)
            for _ in range(8):
                v=violation(proposed_p,proposed_z)
                invalid=~torch.isfinite(v)|(v>baseline+1e-6)
                if not invalid.any(): break
                proposed_p=torch.where(invalid[:,None,None],.5*(p+proposed_p),proposed_p)
                proposed_z=torch.where(invalid[:,None],.5*(z+proposed_z),proposed_z)
            v=violation(proposed_p,proposed_z);invalid=~torch.isfinite(v)|(v>baseline+1e-6)
            return torch.where(invalid[:,None,None],p,proposed_p).detach(),torch.where(invalid[:,None],z,proposed_z).detach()
