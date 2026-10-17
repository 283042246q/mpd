"""Offline oracle world implementing the same per-sphere SDF semantics as runtime."""
import numpy as np
import torch

from mpd.datasets.warehouse_joint_dataset import world_features


class OracleForecastWorld:
    def __init__(self,times,positions,shapes,sizes,*,device="cpu"):
        world_features(times,positions,shapes,sizes)  # validate dimensions and coverage
        self.times=torch.as_tensor(times,dtype=torch.float64,device=device)
        self.positions=torch.as_tensor(positions,dtype=torch.float64,device=device)
        self.shapes=torch.as_tensor(shapes,dtype=torch.long,device=device)
        self.sizes=torch.as_tensor(sizes,dtype=torch.float64,device=device)

    def centers(self,times):
        grid=self.times.to(times)
        upper=torch.searchsorted(grid,times.contiguous(),right=True).clamp(1,len(grid)-1)
        lower=upper-1
        fraction=(times-grid[lower])/(grid[upper]-grid[lower])
        return self.positions.to(times)[lower]+fraction[...,None,None]*(
            self.positions.to(times)[upper]-self.positions.to(times)[lower])

    def signed_distances(self,points,trajectory_times=None,*,validate_times=True):
        times=trajectory_times.to(points)
        if validate_times and (not torch.isfinite(times).all() or (times<0).any() or (times>14+1e-6).any()):
            raise ValueError("oracle query outside [0,14]")
        delta=points[:,:,None,:,:]-self.centers(times)[:,:,:,None,:]
        size=self.sizes.to(points)[None,None,:,None,:]
        sphere=torch.linalg.vector_norm(delta,dim=-1)-size[...,0]
        box_delta=delta.abs()-size
        box=torch.linalg.vector_norm(box_delta.clamp_min(0),dim=-1)+box_delta.amax(-1).clamp_max(0)
        capsule=torch.cat((delta[...,:2],(delta[...,2]-delta[...,2].clamp(-size[...,1],size[...,1]))[...,None]),-1)
        capsule=torch.linalg.vector_norm(capsule,dim=-1)-size[...,0]
        return torch.where(self.shapes[None,None,:,None]==0,sphere,
             torch.where(self.shapes[None,None,:,None]==1,box,capsule))

    def minimum_signed_distance(self,points,trajectory_times=None,*,validate_times=True):
        if not len(self.shapes):
            if validate_times and (not torch.isfinite(trajectory_times).all()
                or (trajectory_times<0).any() or (trajectory_times>14+1e-6).any()):
                raise ValueError("oracle query outside [0,14]")
            return torch.full(points.shape[:-1],float("inf"),dtype=points.dtype,device=points.device)
        return self.signed_distances(points,trajectory_times,validate_times=validate_times).amin(-2)

    def features(self):
        return world_features(self.times.cpu().numpy(),self.positions.cpu().numpy(),
                              self.shapes.cpu().numpy(),self.sizes.cpu().numpy())


def predicted_world_features(world):
    """Align a resident CV forecast to its already validated handoff start."""
    n=world.active_count
    shape=world.shape_code[:n]
    if n:
        rotations=world.rotation[:n][shape!=0]
        if len(rotations) and not torch.allclose(rotations,torch.eye(3,device=rotations.device).expand_as(rotations),atol=1e-5):
            raise ValueError("WORLD v1 tokens require axis-aligned boxes/z-capsules")
    times=np.linspace(0.,14.,32)
    relative=torch.tensor(times,**world.tensor_args)+(world.plan_start_unix_ns-world.stamp_unix_ns)*1e-9
    positions=world.position[:n][None]+relative[:,None,None]*world.velocity[:n][None]
    return world_features(times,positions.cpu().numpy(),shape.cpu().numpy(),world.parameters[:n].cpu().numpy())
