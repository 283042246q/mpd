from abc import ABC, abstractmethod

import einops
import numpy as np
import torch

from torch_robotics.torch_kinematics_tree.geometrics.utils import SE3_distance
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.trajectory.utils import interpolate_points_v1


class DistanceField(ABC):
    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS):
        self.tensor_args = tensor_args

    def distances(self):
        pass

    def compute_collision(self):
        pass

    @abstractmethod
    def compute_distance(self, *args, **kwargs):
        pass

    def compute_cost(self, q_pos, link_pos, *args, **kwargs):
        q_orig_shape = q_pos.shape
        link_orig_shape = link_pos.shape
        if len(link_orig_shape) == 2:
            h = 1
            b, d = link_orig_shape
            link_pos = einops.rearrange(link_pos, "... d -> ... 1 d")  # add dimension of task space link
        elif len(link_orig_shape) == 3:
            h = 1
            b, t, d = link_orig_shape
        elif len(link_orig_shape) == 4:  # batch, horizon, num_links, 3  # position tensor
            b, h, t, d = link_orig_shape
            link_pos = einops.rearrange(link_pos, "... t d -> (...) t d")
        elif len(link_orig_shape) == 5:  # batch, horizon, num_links, 4, 4  # homogeneous transform tensor
            b, h, t, d, d = link_orig_shape
            link_pos = einops.rearrange(link_pos, "... t d d -> (...) t d d")
        else:
            raise NotImplementedError

        # link_tensor_pos
        # position: (batch horizon) x num_links x 3
        cost = self.compute_costs_impl(q_pos, link_pos, *args, **kwargs)

        if cost.ndim == 1:
            cost = einops.rearrange(cost, "(b h) -> b h", b=b, h=h)

        # if len(link_orig_shape) == 4 or len(link_orig_shape) == 5:
        #     cost = einops.rearrange(cost, "(b h) -> b h", b=b, h=h)

        return cost

    @abstractmethod
    def compute_costs_impl(self, *args, **kwargs):
        pass

    @abstractmethod
    def zero_grad(self):
        pass


class EmbodimentDistanceFieldBase(DistanceField):

    def __init__(
        self,
        robot,
        num_interpolated_points=30,
        collision_margins=0.0,
        cutoff_margin=0.001,
        field_type="sdf",
        clamp_sdf=True,
        interpolate_link_pos=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert robot is not None, "You need to pass a robot instance to the embodiment distance fields"
        self.robot = robot
        self.num_interpolated_points = num_interpolated_points
        self.collision_margins = collision_margins
        self.cutoff_margin = cutoff_margin
        self.field_type = field_type
        self.clamp_sdf = clamp_sdf
        self.interpolate_link_pos = interpolate_link_pos

    def compute_embodiment_cost(self, q_pos, link_pos, field_type=None, **kwargs):  # position tensor
        if field_type is None:
            field_type = self.field_type
        if field_type == "rbf":
            return self.compute_embodiment_rbf_distances(link_pos, **kwargs).sum((-1, -2))
        elif field_type == "sdf":  # this computes the negative cost from the DISTANCE FUNCTION
            margin = self.collision_margins + self.cutoff_margin
            # returns all distances from each link to the environment
            margin_minus_sdf = -(self.compute_embodiment_signed_distances(q_pos, link_pos, **kwargs) - margin)
            if self.clamp_sdf:
                clamped_sdf = torch.relu(margin_minus_sdf)
            else:
                clamped_sdf = margin_minus_sdf
            if len(clamped_sdf.shape) == 3:  # cover the multiple objects case
                clamped_sdf = clamped_sdf.max(-2)[0]
            # sum over link points for gradient computation
            return clamped_sdf.sum(-1)
        elif field_type == "occupancy":
            return self.compute_embodiment_collision(q_pos, link_pos, **kwargs)
            # distances = self.self_distances(link_pos, **kwargs)  # batch_dim x (links * (links - 1) / 2)
            # return (distances < margin).sum(-1)
        else:
            raise NotImplementedError("field_type {} not implemented".format(field_type))

    def compute_costs_impl(self, q_pos, link_pos, **kwargs):
        # position link_pos tensor # batch x num_links x 3
        embodiment_cost = self.compute_embodiment_cost(q_pos, link_pos, **kwargs)
        return embodiment_cost

    def compute_distance(self, q, link_pos, **kwargs):
        raise NotImplementedError
        link_pos = interpolate_points_v1(link_pos, self.num_interpolated_points)
        self_distances = self.compute_embodiment_signed_distances(q, link_pos, **kwargs).min(-1)[0]  # batch_dim
        return self_distances

    def zero_grad(self):
        pass
        # raise NotImplementedError

    @abstractmethod
    def compute_embodiment_rbf_distances(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute_embodiment_signed_distances(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute_embodiment_collision(self, *args, **kwargs):
        raise NotImplementedError


class CollisionSelfField(EmbodimentDistanceFieldBase):

    def __init__(self, *args, link_self_collision_tuples=None, **kwargs):
        super().__init__(*args, collision_margins=0.0, cutoff_margin=0.001, **kwargs)
        assert (
            link_self_collision_tuples is not None
        ), "You need to pass the link_self_collision_tuples for self collision checking"
        self.link_self_collision_tuples = link_self_collision_tuples
        self.link_idx_1 = np.asarray([x[0] for x in self.link_self_collision_tuples], dtype=int)
        self.link_idx_2 = np.asarray([x[1] for x in self.link_self_collision_tuples], dtype=int)
        self.link_radii_1 = torch.tensor([x[2] for x in self.link_self_collision_tuples], **self.tensor_args)
        self.link_radii_2 = torch.tensor([x[3] for x in self.link_self_collision_tuples], **self.tensor_args)

    def compute_embodiment_rbf_distances(self, *args, **kwargs):  # position tensor
        raise NotImplementedError

    def compute_embodiment_signed_distances(self, q_pos, link_pos, **kwargs):  # position tensor
        # get the link positions of the links that are used for self collision checking
        link_pos_1 = link_pos[..., self.link_idx_1, :]
        link_pos_2 = link_pos[..., self.link_idx_2, :]
        distances_minus_radii = (
            torch.linalg.norm(link_pos_1 - link_pos_2, dim=-1) - self.link_radii_1 - self.link_radii_2
        )
        return distances_minus_radii

    def compute_embodiment_collision(self, q_pos, link_pos, **kwargs):  # position tensor
        distances_minus_radii = self.compute_embodiment_signed_distances(q_pos, link_pos, **kwargs)
        any_self_collision = torch.any(distances_minus_radii < 0, dim=-1)
        return any_self_collision

    def compute_distance_field_cost_and_gradient(self, link_pos, **kwargs):
        # position link_pos tensor # batch x num_links x env_dim (2D or 3D)
        link_indices = kwargs.get("link_indices")
        pair_indices = kwargs.get("self_pair_indices")
        if link_indices is None:
            local_idx_1 = torch.as_tensor(
                self.link_idx_1, dtype=torch.long, device=link_pos.device
            )
            local_idx_2 = torch.as_tensor(
                self.link_idx_2, dtype=torch.long, device=link_pos.device
            )
        else:
            link_indices = torch.as_tensor(
                link_indices, dtype=torch.long, device=link_pos.device
            )
            if pair_indices is None:
                pair_indices = torch.arange(
                    len(self.link_idx_1), dtype=torch.long, device=link_pos.device
                )
            else:
                pair_indices = torch.as_tensor(
                    pair_indices, dtype=torch.long, device=link_pos.device
                )
            if pair_indices.numel() == 0:
                return (
                    torch.zeros(link_pos.shape[:-2], dtype=link_pos.dtype, device=link_pos.device),
                    torch.zeros_like(link_pos),
                )
            original_idx_1 = torch.as_tensor(
                self.link_idx_1, dtype=torch.long, device=link_pos.device
            ).index_select(0, pair_indices)
            original_idx_2 = torch.as_tensor(
                self.link_idx_2, dtype=torch.long, device=link_pos.device
            ).index_select(0, pair_indices)
            inverse = torch.full(
                (len(self.robot.link_collision_spheres_names),),
                -1,
                dtype=torch.long,
                device=link_pos.device,
            )
            inverse[link_indices] = torch.arange(
                link_indices.numel(), dtype=torch.long, device=link_pos.device
            )
            local_idx_1 = inverse.index_select(0, original_idx_1)
            local_idx_2 = inverse.index_select(0, original_idx_2)
        link_pos_1 = link_pos[..., local_idx_1, :]
        link_pos_2 = link_pos[..., local_idx_2, :]
        difference = link_pos_1 - link_pos_2
        distance = torch.linalg.norm(difference, dim=-1)
        radii = self.link_radii_1 + self.link_radii_2
        if pair_indices is not None:
            radii = radii.index_select(0, pair_indices)
        penetration = torch.relu(radii - distance)
        cost, active_pair = torch.max(penetration, dim=-1)

        # Match torch.max/autograd semantics: one deepest pair contributes at
        # each batch/time point. A coincident pair has no unique direction, so
        # its stable subgradient is defined as zero.
        direction = difference / distance.clamp_min(torch.finfo(link_pos.dtype).eps).unsqueeze(-1)
        active_direction = torch.gather(
            direction,
            dim=-2,
            index=active_pair[..., None, None].expand(*active_pair.shape, 1, link_pos.shape[-1]),
        ).squeeze(-2)
        active_direction = torch.where((cost > 0)[..., None], active_direction, torch.zeros_like(active_direction))
        pair_1 = local_idx_1[active_pair]
        pair_2 = local_idx_2[active_pair]
        gradient = torch.zeros_like(link_pos)
        gradient.scatter_add_(
            -2,
            pair_1[..., None, None].expand(*pair_1.shape, 1, link_pos.shape[-1]),
            -active_direction.unsqueeze(-2),
        )
        gradient.scatter_add_(
            -2,
            pair_2[..., None, None].expand(*pair_2.shape, 1, link_pos.shape[-1]),
            active_direction.unsqueeze(-2),
        )
        return cost, gradient


def reshape_q(q):
    q_orig_shape = q.shape
    if len(q_orig_shape) == 2:
        h = 1
        b, d = q_orig_shape
    elif len(q_orig_shape) == 3:
        b, h, d = q_orig_shape
        q = einops.rearrange(q, "... d -> (...) d")
    else:
        raise NotImplementedError
    return q, q_orig_shape, b, h, d


class CollisionObjectBase(EmbodimentDistanceFieldBase):

    def __init__(self, *args, link_margins_for_object_collision_checking_tensor=None, **kwargs):
        super().__init__(*args, collision_margins=link_margins_for_object_collision_checking_tensor, **kwargs)

    def compute_embodiment_rbf_distances(self, link_pos, **kwargs):  # position tensor
        raise NotImplementedError
        margin = kwargs.get("margin", self.margin)
        rbf_distance = torch.exp(torch.square(self.object_signed_distances(link_pos, **kwargs)) / (-(margin**2) * 2))
        return rbf_distance

    def compute_embodiment_signed_distances(self, q_pos, link_pos, **kwargs):
        return self.object_signed_distances(link_pos, **kwargs)

    def compute_embodiment_collision(self, q, link_pos, **kwargs):
        # position tensor
        # The cutoff margin can be overridden by the margin argument in kwargs.
        # E.g., to check for collisions after planning, we can use a margin of 0 to not discard points that are not
        # in collision but might be inside the cutoff margin used to compute a cost.
        cutoff_margin = kwargs.get("margin", self.cutoff_margin)
        # collision_margins are the spheres radii around the links
        margin = self.collision_margins + cutoff_margin
        signed_distances = self.object_signed_distances(link_pos, **kwargs)
        collisions = signed_distances <= margin
        # reduce over points (dim -1) and over objects (dim -2)
        any_collision = torch.any(torch.any(collisions, dim=-1), dim=-1)
        return any_collision

    @abstractmethod
    def object_signed_distances(self, *args, **kwargs):
        raise NotImplementedError


class CollisionObjectDistanceField(CollisionObjectBase):

    def __init__(self, *args, df_obj_list_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_obj_list_fn = df_obj_list_fn

    def object_signed_distances(self, link_pos, get_gradient=False, **kwargs):
        if self.df_obj_list_fn is None:
            return torch.inf
        df_obj_list = self.df_obj_list_fn()
        link_dim = link_pos.shape[:-1]
        if not df_obj_list:
            sdf_shape = (*link_dim[:-1], 1, link_dim[-1])
            sdf_values = torch.full(sdf_shape, torch.inf, dtype=link_pos.dtype, device=link_pos.device)
            if get_gradient:
                sdf_gradient = torch.zeros(
                    (*sdf_shape, link_pos.shape[-1]),
                    dtype=link_pos.dtype,
                    device=link_pos.device,
                )
                return sdf_values, sdf_gradient
            return sdf_values
        link_pos = link_pos.reshape(-1, link_pos.shape[-1])  # flatten batch_dim and links
        dfs = []
        if get_gradient:
            dfs_gradient = []
            for df in df_obj_list:
                sdf_vals, sdf_gradient = df.compute_signed_distance(link_pos, get_gradient=get_gradient)
                dfs.append(sdf_vals.view(link_dim))  # df() returns batch_dim x links
                dfs_gradient.append(sdf_gradient.view(link_dim + (sdf_gradient.shape[-1],)))

            dfs_th = torch.stack(dfs, dim=-2)  # batch_dim x num_sdfs x links
            dfs_gradient = torch.stack(dfs_gradient, dim=-3)  # batch_dim x num_sdfs x links x 3
            return dfs_th, dfs_gradient
        else:
            for df in df_obj_list:
                sdf_vals = df.compute_signed_distance(link_pos, get_gradient=get_gradient)
                dfs.append(sdf_vals.view(link_dim))  # df() returns batch_dim x links

            dfs_th = torch.stack(dfs, dim=-2)  # batch_dim x num_sdfs x links
            return dfs_th

    def object_signed_distance_gradients(self, link_pos, **kwargs):
        """Query only task-space SDF gradients, preserving the object axis."""

        if self.df_obj_list_fn is None:
            return torch.zeros_like(link_pos).unsqueeze(-3)
        df_obj_list = self.df_obj_list_fn()
        link_dim = link_pos.shape[:-1]
        if not df_obj_list:
            return torch.zeros(
                (*link_dim[:-1], 1, link_dim[-1], link_pos.shape[-1]),
                dtype=link_pos.dtype,
                device=link_pos.device,
            )
        flat_link_pos = link_pos.reshape(-1, link_pos.shape[-1])
        gradients = []
        for df in df_obj_list:
            gradient_fn = getattr(df, "compute_signed_distance_gradient", None)
            if gradient_fn is None:
                _, sdf_gradient = df.compute_signed_distance(
                    flat_link_pos, get_gradient=True
                )
            else:
                sdf_gradient = gradient_fn(flat_link_pos)
            gradients.append(
                sdf_gradient.view(link_dim + (sdf_gradient.shape[-1],))
            )
        return torch.stack(gradients, dim=-3)

    def compute_distance_field_cost_and_gradient(self, link_pos, **kwargs):
        # position link_pos tensor # batch x num_links x env_dim (2D or 3D)
        embodiment_cost, embodiment_cost_gradient = self.compute_embodiment_taskspace_sdf_and_gradient(
            link_pos, **kwargs
        )
        return embodiment_cost, embodiment_cost_gradient

    def compute_embodiment_taskspace_sdf_and_gradient(self, link_pos, **kwargs):
        collision_margins = self.collision_margins
        link_indices = kwargs.get("link_indices")
        if link_indices is not None:
            collision_margins = collision_margins.index_select(
                0,
                torch.as_tensor(link_indices, dtype=torch.long, device=link_pos.device),
            )
        margin = collision_margins + self.cutoff_margin
        # returns all distances from each link to the environment
        precomputed_sdf_values = kwargs.pop("precomputed_sdf_values", None)
        if precomputed_sdf_values is None:
            sdf_vals, sdf_gradient = self.object_signed_distances(
                link_pos, get_gradient=True, **kwargs
            )
        else:
            sdf_vals = precomputed_sdf_values
            sdf_gradient = self.object_signed_distance_gradients(
                link_pos, **kwargs
            )
        margin_minus_sdf = -(sdf_vals - margin)
        if self.clamp_sdf:
            margin_minus_sdf_clamped = torch.relu(margin_minus_sdf)
        else:
            margin_minus_sdf_clamped = margin_minus_sdf
        if (
            margin_minus_sdf_clamped.ndim >= 3
        ):  # cover the multiple objects case ((batch, horizon, ...), objects, links)
            if (
                margin_minus_sdf_clamped.shape[-2] == 1
            ):  # if there is only one object, take this one as the maximum margin_minus_sdf
                margin_minus_sdf_clamped = margin_minus_sdf_clamped.squeeze(-2)
                sdf_gradient = sdf_gradient.squeeze(-3)
            else:
                margin_minus_sdf_clamped, idxs_max = margin_minus_sdf_clamped.max(-2)
                sdf_gradient = sdf_gradient.gather(
                    2, idxs_max.unsqueeze(2).unsqueeze(-1).expand(-1, -1, -1, -1, sdf_gradient.shape[-1])
                ).squeeze(2)

        # set sdf gradient to 0 if the point is not in collision
        idxs = torch.argwhere(margin_minus_sdf_clamped <= 0)
        sdf_gradient[idxs[:, 0], idxs[:, 1], idxs[:, 2], :] = 0.0

        # the gradient of (margin-sdf(x)) wrt to the position x is -1 * sdf_gradient(x)
        return margin_minus_sdf_clamped, -1.0 * sdf_gradient


class CollisionWorkspaceBoundariesDistanceField(CollisionObjectBase):

    def __init__(self, *args, ws_bounds_min=None, ws_bounds_max=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws_min = ws_bounds_min
        self.ws_max = ws_bounds_max

    def object_signed_distances(self, link_pos, **kwargs):
        signed_distances_bounds_min = link_pos - self.ws_min
        signed_distances_bounds_min = torch.sign(signed_distances_bounds_min) * torch.abs(signed_distances_bounds_min)
        signed_distances_bounds_max = self.ws_max - link_pos
        signed_distances_bounds_max = torch.sign(signed_distances_bounds_max) * torch.abs(signed_distances_bounds_max)
        signed_distances_bounds = torch.cat((signed_distances_bounds_min, signed_distances_bounds_max), dim=-1)
        return signed_distances_bounds.transpose(-2, -1)  # batch_dim x num_sdfs x links


if __name__ == "__main__":
    raise NotImplementedError
    import time

    mesh_file = "models/chair.obj"
    mesh = MeshDistanceField(mesh_file)
    bounds = np.array(mesh.mesh.bounds)
    print(np.linalg.norm(bounds[1] - bounds[0]))
    print(mesh.mesh.centroid)
    points = torch.rand(100, 3)
    link_tensor = torch.rand(100, 10, 4, 4)
    start = time.time()
    distances = mesh.compute_distance(points)
    costs = mesh.compute_cost(link_tensor)
    print(time.time() - start)
