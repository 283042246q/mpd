import abc
import datetime
import gc
import math
import os
import os.path
import pickle
import random
import tempfile

import einops
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from mpd.datasets.normalization import DatasetNormalizer
from mpd.datasets.utils import CPU_Unpickler
from pb_ompl.pb_ompl import fit_bspline_to_path
from torch_robotics import robots
from torch_robotics.robots import RobotPointMass2D
from torch_robotics.torch_kinematics_tree.geometrics.utils import (
    link_pos_from_link_tensor,
    link_rot_from_link_tensor,
    rmat_to_flat,
)
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import to_torch, dict_to_device, to_numpy, DEFAULT_TENSOR_ARGS


def _atomic_pickle_dump(value, target_path):
    """Write a pickle completely before exposing it under its final name."""
    target_path = os.fspath(target_path)
    target_dir = os.path.dirname(target_path) or "."
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target_dir,
            prefix=f".{os.path.basename(target_path)}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            pickle.dump(value, temporary, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target_path)
    except BaseException:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        raise


def adjust_bspline_number_control_points(
    n_control_points, context_qs, context_ee_goal_pose, zero_vel_at_start_and_goal, zero_acc_at_start_and_goal
):
    # The Unet architecture accepts horizons that are multiples of 2^depth (8 in our case if depth=3).
    # Adjust the control points, such that the number of trainable control points is a multiple of 8.
    removed_control_points = 0
    trainable_control_points = n_control_points
    if context_qs and context_ee_goal_pose:
        trainable_control_points -= 1
        removed_control_points += 1
    elif context_qs:
        trainable_control_points -= 2
        removed_control_points += 2

    if zero_vel_at_start_and_goal:
        trainable_control_points -= 2
        removed_control_points += 2
    if zero_acc_at_start_and_goal:
        trainable_control_points -= 2
        removed_control_points += 2

    # make sure the number of trainable control points is a multiple of 8 (next closest multiple of 8)
    trainable_control_points = int(math.ceil(trainable_control_points / 8) * 8)

    return trainable_control_points + removed_control_points, removed_control_points


class TrajectoryDatasetBspline(Dataset, abc.ABC):

    def __init__(
        self,
        planning_task=None,
        base_dir=None,
        dataset_file_merged="dataset_merged.hdf5",
        context_qs=True,
        context_ee_goal_pose=False,
        context_ee_goal_pose_bimanual=False,
        normalizer="SafeLimitsNormalizer",
        normalize_ee_pose_goal=True,
        reload_data=False,
        preload_data_to_device=False,
        skip_collision_statistics_for_validated_splines=False,
        n_task_samples=-1,
        tensor_args=DEFAULT_TENSOR_ARGS,
        **kwargs,
    ):

        self.planning_task = planning_task

        self.base_dir = base_dir
        self.dataset_file_merged = dataset_file_merged

        self.tensor_args = tensor_args

        ######################################################################################
        # -------------------------------- Context variables ---------------------------------
        # Use joint start (and goal) as a context variable
        self.context_qs = context_qs

        # Use end-effector pose goal as a context variable
        self.context_ee_goal_pose = context_ee_goal_pose
        self.context_ee_goal_pose_bimanual = context_ee_goal_pose_bimanual
        if self.context_ee_goal_pose_bimanual and not self.context_ee_goal_pose:
            raise ValueError("bimanual EE context requires context_ee_goal_pose=True")

        ######################################################################################
        # -------------------------------- Load trajectories ---------------------------------
        # Dataset fields
        self.field_key_control_points = "control_points"
        self.field_key_q_start = "q_start"
        self.field_key_q_goal = "q_goal"
        self.field_key_context_qs = "qs"
        self.field_key_context_ee_goal_pose = "ee_goal_pose"
        self.field_key_context_ee_goal_orientation = "ee_goal_orientation"
        self.field_key_context_ee_goal_position = "ee_goal_position"
        self.field_key_context_active_ee_mask = "active_ee_mask"
        self.fields = {}

        # ------------ load data ------------
        self.map_task_id_to_control_points_id = {}
        self.map_control_points_id_to_task_id = {}
        self.reload_data = reload_data
        self.preload_data_to_device = preload_data_to_device
        self.skip_collision_statistics_for_validated_splines = bool(
            skip_collision_statistics_for_validated_splines
        )
        self.load_data(n_task_samples=n_task_samples)
        # possibly move the dataset to the gpu
        if self.preload_data_to_device:
            self.fields = dict_to_device(self.fields, **self.tensor_args)
        torch.cuda.empty_cache()
        gc.collect()

        # dimensions
        b, h, d = self.dataset_shape = self.fields[self.field_key_control_points].shape
        self.n_trajectories = b
        self.n_learnable_control_points = h
        self.state_dim = d  # state dimension used for the generative model
        self.control_points_dim = (self.n_learnable_control_points, d)

        self.context_q_dim = self.fields[self.field_key_context_qs].shape[-1]

        # -------------------------------- Normalize data ---------------------------------
        # normalize the data
        self.normalizer = DatasetNormalizer(self.fields, normalizer=normalizer)
        self.normalizer_keys = [
            self.field_key_control_points,
            self.field_key_context_qs,
            self.field_key_context_ee_goal_orientation,
            self.field_key_context_ee_goal_position,
        ]
        self.normalize_ee_pose_goal = normalize_ee_pose_goal
        if normalize_ee_pose_goal:
            # do not normalize the orientation, only the position part of the end-effector pose
            self.normalizer_keys.append(self.field_key_context_ee_goal_position)

        self.normalize_all_data(*self.normalizer_keys)

        # Do not normalize rotation matrices
        # for compatilibity, set the normalized orientation to the unnormalized one
        self.fields[self.field_key_context_ee_goal_orientation + "_normalized"] = self.fields[
            self.field_key_context_ee_goal_orientation
        ]

        if not normalize_ee_pose_goal:
            # normalized data is the same as the original data
            self.fields[self.field_key_context_ee_goal_pose + "_normalized"] = self.fields[
                self.field_key_context_ee_goal_pose
            ]

        # move normalize class members data to devices
        self.normalizer.move_to_device(device=self.tensor_args["device"])

    def reload_data_fn(self, data_reload_pickle_path, n_task_samples=-1):
        with open(data_reload_pickle_path, "rb") as fp:
            data = CPU_Unpickler(fp).load()
        self.fields = data["fields"]
        self.map_task_id_to_control_points_id = data["map_task_id_to_control_points_id"]

        if n_task_samples != -1:
            n_task_samples = min(n_task_samples, len(self.map_task_id_to_control_points_id))
            dict_tmp = {
                key: self.map_task_id_to_control_points_id[key]
                for key in np.random.choice(
                    list(self.map_task_id_to_control_points_id.keys()), n_task_samples, replace=False
                )
            }
            self.map_task_id_to_control_points_id = dict_tmp

        self.map_control_points_id_to_task_id = data["map_control_points_id_to_task_id"]

    @staticmethod
    def _selected_serialized_rows(task_ids, n_task_samples):
        if n_task_samples == -1:
            return None
        selected_task_ids = []
        selected_set = set()
        for task_id in task_ids:
            task_id = int(task_id)
            if task_id not in selected_set:
                selected_set.add(task_id)
                selected_task_ids.append(task_id)
                if len(selected_task_ids) >= n_task_samples:
                    break
        return np.flatnonzero(np.isin(task_ids, selected_task_ids))

    def _load_serialized_bspline_rows(self, dataset_h5, n_task_samples, timer):
        """Load precomputed spline fields in HDF5-aligned row batches."""
        n_h5 = len(dataset_h5["task_id"])
        task_ids = np.asarray(dataset_h5["task_id"][:], dtype=np.int64)
        selected_rows = self._selected_serialized_rows(task_ids, n_task_samples)
        n_selected = n_h5 if selected_rows is None else len(selected_rows)
        if n_selected == 0:
            raise ValueError("n_task_samples selected no trajectories")

        field_names = ["bspline_params_cc", "task_id"]
        if self.context_ee_goal_pose_bimanual:
            field_names.extend(("ee_goal_pose", "active_ee_mask"))
        # Align reads with the largest first-axis HDF5 chunk. The generated
        # Marvin files use compatible multiples (3116/6232/12464 rows), so
        # every compressed chunk is normally decompressed only once.
        hdf5_chunk_rows = [
            dataset_h5[name].chunks[0]
            for name in field_names
            if dataset_h5[name].chunks is not None
        ]
        chunk_rows = max(hdf5_chunk_rows) if hdf5_chunk_rows else min(n_selected, 4096)
        chunk_rows = max(1, int(chunk_rows))

        inner_control_points_all = []
        q_start_all = []
        q_goal_all = []
        ee_pose_goal_all = []
        active_ee_mask_all = []
        cps_idx = 0

        print(
            f"Batch-loading {n_selected} serialized B-splines "
            f"in chunks of {chunk_rows} rows.",
            flush=True,
        )
        for output_start in range(0, n_selected, chunk_rows):
            output_stop = min(output_start + chunk_rows, n_selected)
            input_rows = (
                slice(output_start, output_stop)
                if selected_rows is None
                else selected_rows[output_start:output_stop]
            )
            coefficients = np.asarray(dataset_h5["bspline_params_cc"][input_rows])
            if isinstance(self.planning_task.robot, robots.RobotPanda) and coefficients.shape[1] > 9:
                coefficients = coefficients[:, :7, :]
            control_points = to_torch(
                coefficients,
                dtype=self.tensor_args["dtype"],
                device="cpu",
            ).transpose(1, 2)
            q_start_all.append(control_points[:, 0, :].clone())
            q_goal_all.append(control_points[:, -1, :].clone())
            inner_control_points_all.append(
                self.planning_task.parametric_trajectory.remove_control_points_fn(control_points)
            )

            if self.context_ee_goal_pose_bimanual:
                ee_pose_goal_all.append(
                    to_torch(
                        np.asarray(dataset_h5["ee_goal_pose"][input_rows]),
                        dtype=self.tensor_args["dtype"],
                        device="cpu",
                    )
                )
                active_ee_mask_all.append(
                    to_torch(
                        np.asarray(dataset_h5["active_ee_mask"][input_rows]),
                        dtype=self.tensor_args["dtype"],
                        device="cpu",
                    )
                )

            batch_task_ids = task_ids[input_rows]
            for task_id in batch_task_ids:
                task_id = int(task_id)
                self.map_control_points_id_to_task_id[cps_idx] = task_id
                self.map_task_id_to_control_points_id.setdefault(task_id, []).append(cps_idx)
                cps_idx += 1

            print(
                f"Time spent: {str(datetime.timedelta(seconds=timer.elapsed))} - "
                f"batch-loaded {output_stop}/{n_selected} "
                f"({output_stop / n_selected:.2%}) trajectories.",
                flush=True,
            )

        self.fields[self.field_key_control_points] = torch.cat(inner_control_points_all, dim=0)
        self.fields = self.build_fields_data_sample(
            self.fields,
            torch.cat(q_start_all, dim=0),
            torch.cat(q_goal_all, dim=0),
            ee_pose_goal=torch.cat(ee_pose_goal_all, dim=0) if ee_pose_goal_all else None,
            active_ee_mask=torch.cat(active_ee_mask_all, dim=0) if active_ee_mask_all else None,
            device="cpu",
        )
        return n_h5

    def _load_legacy_rows(self, dataset_h5, n_task_samples, timer):
        """Fit splines for legacy datasets that do not serialize coefficients."""
        inner_control_points_all = []
        q_start_all = []
        q_goal_all = []
        task_ids_processed = []
        cps_idx = 0
        n_discarded_trajectories = 0
        for i, path in enumerate(dataset_h5["sol_path"]):
            if n_task_samples != -1 and i > 0:
                task_ids_processed.append(dataset_h5["task_id"][i])
                task_ids_processed = list(set(task_ids_processed))
                if len(task_ids_processed) >= n_task_samples:
                    break
            try:
                _, coefficients, _ = fit_bspline_to_path(
                    path,
                    bspline_degree=self.planning_task.parametric_trajectory.bspline.d,
                    bspline_num_control_points=self.planning_task.parametric_trajectory.bspline.n_pts,
                    bspline_zero_vel_at_start_and_goal=self.planning_task.parametric_trajectory.zero_vel_at_start_and_goal,
                    bspline_zero_acc_at_start_and_goal=self.planning_task.parametric_trajectory.zero_acc_at_start_and_goal,
                    debug=False,
                )
                if np.any(coefficients.min(1) <= 2 * to_numpy(self.planning_task.robot.q_pos_min)) or np.any(
                    coefficients.max(1) >= 2 * to_numpy(self.planning_task.robot.q_pos_max)
                ):
                    n_discarded_trajectories += 1
                    raise ValueError("fitted control points exceed joint limits")
            except Exception:
                continue

            coefficients = np.asarray(coefficients)
            if isinstance(self.planning_task.robot, robots.RobotPanda) and coefficients.shape[0] > 9:
                coefficients = coefficients[:7, :]
            control_points = to_torch(
                coefficients, dtype=self.tensor_args["dtype"], device="cpu"
            ).transpose(0, 1)
            q_start_all.append(control_points[0])
            q_goal_all.append(control_points[-1])
            inner_control_points_all.append(
                self.planning_task.parametric_trajectory.remove_control_points_fn(control_points)
            )
            task_id = int(dataset_h5["task_id"][i])
            self.map_control_points_id_to_task_id[cps_idx] = task_id
            self.map_task_id_to_control_points_id.setdefault(task_id, []).append(cps_idx)
            cps_idx += 1
            if i % 20000 == 0 or i == len(dataset_h5["sol_path"]) - 1:
                print(
                    f"Time spent: {str(datetime.timedelta(seconds=timer.elapsed))} - "
                    f"loaded {i}/{len(dataset_h5['sol_path'])} "
                    f"({i / len(dataset_h5['sol_path']):.2%}) trajectories.",
                    flush=True,
                )

        print(f"Number of discarded trajectories: {n_discarded_trajectories}/{len(dataset_h5['sol_path'])}")
        self.fields[self.field_key_control_points] = torch.stack(inner_control_points_all)
        self.fields = self.build_fields_data_sample(
            self.fields,
            torch.stack(q_start_all),
            torch.stack(q_goal_all),
            device="cpu",
        )
        return len(dataset_h5["sol_path"])

    def load_data(self, n_task_samples=-1):
        # load data into CPU RAM
        with TimerCUDA() as t_load_data:
            print("Loading data ...", flush=True)

            # File name for data reload
            data_reload_prefix = f'{self.dataset_file_merged.replace(".hdf5", "")}_reload'
            data_reload_prefix += f"-ntasks_{n_task_samples}"
            data_reload_prefix += f"--bspline"
            data_reload_prefix += f"-degree_{self.planning_task.parametric_trajectory.bspline.d}"
            data_reload_prefix += f"-n_pts_{self.planning_task.parametric_trajectory.bspline.n_pts}"
            data_reload_prefix += f"-zero_vel_{self.planning_task.parametric_trajectory.zero_vel_at_start_and_goal}"
            data_reload_prefix += f"-zero_acc_{self.planning_task.parametric_trajectory.zero_acc_at_start_and_goal}"
            if self.context_ee_goal_pose_bimanual:
                # Never reuse a legacy joint/single-EE cache that lacks the
                # two stored TCP slots and their mask. Preserve Panda cache names.
                data_reload_prefix += "-marvin_dual_ee_v2_batched"
            data_reload_file_path = os.path.join(self.base_dir, f"{data_reload_prefix}.pickle")

            if os.path.exists(data_reload_file_path) and not self.reload_data:
                # load the pre-processed dataset
                self.reload_data_fn(data_reload_file_path, n_task_samples=n_task_samples)
            else:
                dataset_path = os.path.join(self.base_dir, self.dataset_file_merged)
                with h5py.File(dataset_path, "r") as dataset_h5:
                    if self.context_ee_goal_pose_bimanual:
                        missing = {"ee_goal_pose", "active_ee_mask"}.difference(dataset_h5.keys())
                        if missing:
                            raise ValueError(
                                "Marvin dual-slot EE context must be read from HDF5; "
                                f"missing fields: {sorted(missing)}"
                            )
                        n_h5 = len(dataset_h5["sol_path"])
                        if dataset_h5["ee_goal_pose"].shape != (n_h5, 2, 3, 4):
                            raise ValueError("ee_goal_pose must have shape (N, 2, 3, 4)")
                        if dataset_h5["active_ee_mask"].shape != (n_h5, 2):
                            raise ValueError("active_ee_mask must have shape (N, 2)")

                    serialized_splines = "bspline_params_cc" in dataset_h5
                    validated_splines = bool(dataset_h5.attrs.get("validated_splines", False))
                    if serialized_splines:
                        total_rows = self._load_serialized_bspline_rows(
                            dataset_h5, n_task_samples, t_load_data
                        )
                    else:
                        total_rows = self._load_legacy_rows(dataset_h5, n_task_samples, t_load_data)

                skip_statistics = getattr(
                    self, "skip_collision_statistics_for_validated_splines", False
                )
                if skip_statistics:
                    if not isinstance(self.planning_task.robot, robots.RobotMarvinBimanual):
                        raise ValueError("validated-spline collision-statistics bypass is Marvin-only")
                    if not serialized_splines or not validated_splines:
                        raise ValueError(
                            "collision statistics may only be skipped for serialized, validated splines"
                        )
                    print(
                        "Skipping duplicate collision statistics: the Marvin v3 training "
                        "entrypoint strictly validated this dataset and HDF5 declares "
                        "validated_splines=True.",
                        flush=True,
                    )
                    percentage_free_trajs_l = [1.0]
                    percentage_collision_intensity_l = [0.0]
                else:
                    percentage_free_trajs_l, percentage_collision_intensity_l = (
                        self.run_collision_statistics()
                    )

                # Save data to disk to speed up loading the next time.
                data_to_save = {
                    "cache_schema": "trajectory_bspline_reload/v2-batched-atomic",
                    "fields": self.fields,
                    "map_task_id_to_control_points_id": self.map_task_id_to_control_points_id,
                    "map_control_points_id_to_task_id": self.map_control_points_id_to_task_id,
                    "source_num_trajectories": total_rows,
                    "collision_statistics_skipped": skip_statistics,
                    "percentage_free_trajs": (np.mean(percentage_free_trajs_l), np.std(percentage_free_trajs_l)),
                    "percentage_collision_intensity": (
                        np.mean(percentage_collision_intensity_l),
                        np.std(percentage_collision_intensity_l),
                    ),
                }
                _atomic_pickle_dump(data_to_save, data_reload_file_path)

            print("... done loading data.", flush=True)
            print(f"Loading data took {t_load_data.elapsed:.2f} seconds.", flush=True)

    def build_fields_data_sample(
        self,
        fields_d,
        q_start,
        q_goal,
        ee_pose_goal=None,
        active_ee_mask=None,
        device=None,
        **kwargs,
    ):
        fields_d[self.field_key_q_start] = q_start
        fields_d[self.field_key_q_goal] = q_goal

        # task: start and goal control_points [n_trajectories, (state_dim or 2 * state_dim)]
        if self.context_qs and self.context_ee_goal_pose:
            context_qs = q_start
        else:
            context_qs = torch.cat((q_start, q_goal), dim=-1)
        fields_d[self.field_key_context_qs] = context_qs

        # Context data
        # end-effector pose goal
        if self.context_ee_goal_pose_bimanual:
            if ee_pose_goal is None or active_ee_mask is None:
                raise ValueError(
                    "dual-slot EE context requires stored ee_goal_pose and active_ee_mask; "
                    "generic robot.get_EE_pose() is intentionally disabled"
                )
            if ee_pose_goal.shape[-3:] != (2, 3, 4) or active_ee_mask.shape[-1:] != (2,):
                raise ValueError("expected ee_goal_pose (..., 2, 3, 4) and active_ee_mask (..., 2)")
        elif ee_pose_goal is None:
            ee_pose_goal = self.planning_task.robot.get_EE_pose(
                to_torch(q_goal, **self.planning_task.robot.tensor_args)
            ).to(device if device is not None else self.tensor_args["device"])
        ee_pose_goal_orientation = rmat_to_flat(link_rot_from_link_tensor(ee_pose_goal))
        ee_pose_goal_position = link_pos_from_link_tensor(ee_pose_goal)
        fields_d[self.field_key_context_ee_goal_pose] = ee_pose_goal
        fields_d[self.field_key_context_ee_goal_orientation] = ee_pose_goal_orientation
        fields_d[self.field_key_context_ee_goal_position] = ee_pose_goal_position
        if self.context_ee_goal_pose_bimanual:
            fields_d[self.field_key_context_active_ee_mask] = active_ee_mask.to(dtype=q_start.dtype)

        return fields_d

    def normalize_all_data(self, *keys):
        for key in keys:
            self.fields[f"{key}_normalized"] = self.normalizer(self.fields[f"{key}"], key)

    def __repr__(self):
        msg = (
            f"{self.__class__.__name__}\n"
            f"n_tasks: {len(self.map_task_id_to_control_points_id)}\n"
            f"n_trajs: {self.n_trajectories}\n"
            f"control_points_dim: {self.control_points_dim}\n"
        )
        return msg

    def __len__(self):
        return self.n_trajectories

    def __getitem__(self, index):
        # Generates one sample of data - one trajectory and tasks
        data = {}
        for field in self.fields:
            data[field] = self.fields[field][index]

        # build hard conditions
        hard_conds = self.get_hard_conditions(data)
        data["hard_conds"] = hard_conds

        return data

    def create_data_sample_normalized(
        self, q_start_pos, q_goal_pos, ee_pose_goal=None, active_ee_mask=None, **kwargs
    ):
        # create a data sample
        data_sample = {}
        data_sample = self.build_fields_data_sample(
            data_sample,
            q_start_pos,
            q_goal_pos,
            ee_pose_goal=ee_pose_goal,
            active_ee_mask=active_ee_mask,
        )

        for k in list(data_sample.keys()):
            v = data_sample[k]
            # do not normalize the orientation, only the position part of the end-effector goal pose
            if k == self.field_key_context_ee_goal_orientation:
                data_sample[f"{k}_normalized"] = data_sample[k]
            elif k == self.field_key_context_ee_goal_pose:
                pass
            elif k == self.field_key_context_active_ee_mask:
                data_sample[f"{k}_normalized"] = data_sample[k]
            else:
                data_sample[f"{k}_normalized"] = self.normalize(v, k)

        if not self.preload_data_to_device:
            data_sample = dict_to_device(data_sample, **self.tensor_args)

        # build hard conditions
        hard_conds = self.get_hard_conditions(data_sample)
        data_sample["hard_conds"] = hard_conds

        return data_sample

    def get_hard_conditions(self, data_d, **kwargs):
        hard_conds = {}
        if not self.context_qs:
            # start and goal joint positions
            cps_normalized = data_d[f"{self.field_key_control_points}_normalized"]
            cps_start_normalized = cps_normalized[0]
            cps_goal_normalized = cps_normalized[-1]

            horizon = self.n_learnable_control_points

            # The Initial and final velocity and acceleration should be zero
            # Set the second and second-to-last control points to be the same as the first and last control points
            # Set the third and third-to-last control points to be the same as the first and last control points
            # Do not set a hard condition on the last control point joint position if the end-effector goal is used as
            # a context variable
            hard_conds = {0: cps_start_normalized}
            if self.planning_task.parametric_trajectory.zero_vel_at_start_and_goal:
                hard_conds[1] = cps_start_normalized
            if self.planning_task.parametric_trajectory.zero_acc_at_start_and_goal:
                hard_conds[2] = cps_start_normalized

            if not self.context_ee_goal_pose:
                hard_conds[horizon - 1] = cps_goal_normalized
                if self.planning_task.parametric_trajectory.zero_vel_at_start_and_goal:
                    hard_conds[horizon - 2] = cps_goal_normalized
                if self.planning_task.plannerlf.zero_acc_at_start_and_goal:
                    hard_conds[horizon - 3] = cps_goal_normalized

        return hard_conds

    def build_context(self, data_sample):
        context_d = {}
        if self.context_qs and self.context_ee_goal_pose:
            context_d = {
                self.field_key_context_qs: data_sample[self.field_key_context_qs],
                f"{self.field_key_context_qs}_normalized": data_sample[f"{self.field_key_context_qs}_normalized"],
                self.field_key_context_ee_goal_pose: data_sample[self.field_key_context_ee_goal_pose],
                self.field_key_context_ee_goal_orientation: data_sample[self.field_key_context_ee_goal_orientation],
                f"{self.field_key_context_ee_goal_orientation}_normalized": data_sample[
                    f"{self.field_key_context_ee_goal_orientation}_normalized"
                ],
                self.field_key_context_ee_goal_position: data_sample[self.field_key_context_ee_goal_position],
                f"{self.field_key_context_ee_goal_position}_normalized": data_sample[
                    f"{self.field_key_context_ee_goal_position}_normalized"
                ],
            }
        elif self.context_qs:
            context_d = {
                self.field_key_context_qs: data_sample[self.field_key_context_qs],
                f"{self.field_key_context_qs}_normalized": data_sample[f"{self.field_key_context_qs}_normalized"],
            }
        elif self.context_ee_goal_pose:
            context_d = {
                self.field_key_context_ee_goal_pose: data_sample[self.field_key_context_ee_goal_pose],
                self.field_key_context_ee_goal_orientation: data_sample[self.field_key_context_ee_goal_orientation],
                f"{self.field_key_context_ee_goal_orientation}_normalized": data_sample[
                    f"{self.field_key_context_ee_goal_orientation}_normalized"
                ],
                self.field_key_context_ee_goal_position: data_sample[self.field_key_context_ee_goal_position],
                f"{self.field_key_context_ee_goal_position}_normalized": data_sample[
                    f"{self.field_key_context_ee_goal_position}_normalized"
                ],
            }
        if self.context_ee_goal_pose_bimanual:
            context_d[self.field_key_context_active_ee_mask] = data_sample[
                self.field_key_context_active_ee_mask
            ]
        return context_d

    def get_unnormalized(self, index):
        raise NotImplementedError

    def unnormalize(self, x, key):
        return self.normalizer.unnormalize(x, key)

    def normalize(self, x, key):
        return self.normalizer.normalize(x, key)

    def unnormalize_control_points(self, x):
        return self.unnormalize(x, self.field_key_control_points)

    def grad_unnormalized_wrt_control_points_normalized(self, x):
        return self.normalizer.grad_x_wrt_x_normalized(x, self.field_key_control_points)

    def normalize_control_points(self, x):
        return self.normalize(x, self.field_key_control_points)

    def unnormalize_tasks(self, x):
        return self.unnormalize(x, self.field_key_context_qs)

    def normalize_tasks(self, x):
        return self.normalize(x, self.field_key_context_qs)

    def normalize_ee_pose(self, x):
        if not self.normalize_ee_pose_goal:
            return x
        # normalize only the position part of the end-effector pose
        position = link_pos_from_link_tensor(x)
        position_normalized = self.normalize(position, self.field_key_context_ee_goal_position)
        x[..., :3, 3] = position_normalized
        return x

    def render(
        self,
        task_id=None,
        render_joint_trajectories=False,
        render_robot_trajectories=False,
        render_n_robot_trajectories=-1,
        **kwargs,
    ):
        # -------------------------------- Visualize ---------------------------------
        if task_id is None:
            task_id = np.random.choice(list(self.map_task_id_to_control_points_id.keys()))
        idxs = np.asarray(self.map_task_id_to_control_points_id[task_id])
        control_points = self.fields[self.field_key_control_points][idxs]
        control_points = to_torch(control_points, **self.tensor_args)

        # Get trajectories from parametric trajectory
        q_pos_start = to_torch(self.fields[self.field_key_q_start][idxs], **self.tensor_args)
        q_pos_goal = to_torch(self.fields[self.field_key_q_goal][idxs], **self.tensor_args)
        q_pos_trajs = self.planning_task.parametric_trajectory.get_q_trajectory(
            control_points, q_pos_start, q_pos_goal, get_type=["pos"]
        )["pos"]

        fig1, axs1, fig2, axs2 = [None] * 4
        if render_joint_trajectories:
            fig1, axs1 = self.planning_task.plot_joint_space_trajectories(
                q_pos_trajs=q_pos_trajs,
                q_vel_trajs=None,
                q_acc_trajs=None,
                q_pos_start=q_pos_start[0],
                q_pos_goal=q_pos_goal[0],
                # vel_start_state=torch.zeros_like(q_start), vel_goal_state=torch.zeros_like(q_goal),
                set_q_pos_limits=True,
                set_q_vel_limits=True,
                set_q_acc_limits=True,
            )

        if render_robot_trajectories:
            # render n random trajectories
            # TODO - implement parallel rendering
            if render_n_robot_trajectories != -1:
                idxs_n = np.random.choice(q_pos_trajs.shape[0], render_n_robot_trajectories)
            else:
                idxs_n = np.arange(q_pos_trajs.shape[0])
            q_pos_trajs = q_pos_trajs[idxs_n]
            ee_goal_position = self.fields[self.field_key_context_ee_goal_position][idxs[idxs_n]]
            fig2, axs2 = self.planning_task.render_robot_trajectories(
                q_pos_trajs=q_pos_trajs,
                q_pos_start=q_pos_start[0],
                q_pos_goal=q_pos_goal[0],
                ee_goal_position=ee_goal_position[0],
                **kwargs,
            )
            if isinstance(self.planning_task.robot, RobotPointMass2D):
                for cps in to_numpy(control_points[idxs_n]):
                    axs2.scatter(cps[:, 0], cps[:, 1], c="b", s=10, zorder=100)

        return fig1, axs1, fig2, axs2

    def run_collision_statistics(self, chunk_size=2000):
        # checks the collision statistics of the fitted bsplines
        if isinstance(self.planning_task.robot, robots.RobotMarvinBimanual):
            # Pika's fine model has >1000 spheres; 2000 x 128 states
            # would allocate enormous self-pair tensors before training.
            chunk_size = min(chunk_size, 2)

        # get all control points idxs
        idxs_cps_sequential = np.arange(self.fields[self.field_key_control_points].shape[0])

        # loop idxs in chunks due to memory constraints
        percentage_free_trajs_l = []
        percentage_collision_intensity_l = []
        for idxs_chunk in tqdm(
            chunker(idxs_cps_sequential, chunk_size), total=np.ceil(len(idxs_cps_sequential) / chunk_size)
        ):
            control_points = self.fields[self.field_key_control_points][idxs_chunk]
            control_points = to_torch(control_points, **self.tensor_args)

            # Get trajectories
            q_start = self.fields[self.field_key_q_start][idxs_chunk]
            q_start = to_torch(q_start, **self.tensor_args)
            q_goal = self.fields[self.field_key_q_goal][idxs_chunk]
            q_goal = to_torch(q_goal, **self.tensor_args)
            q_trajs_pos = self.planning_task.parametric_trajectory.get_q_trajectory(
                control_points, q_start, q_goal, get_type=("pos",)
            )["pos"]

            percentage_free_trajs_l.append(self.planning_task.compute_fraction_valid_trajs(q_trajs_pos))
            percentage_collision_intensity_l.append(
                self.planning_task.compute_collision_intensity_trajs(q_trajs_pos).cpu().item()
            )

        print("---------------------------------")
        print("Dataset statistics after fitting")
        print(f"percentage free trajs: ({np.mean(percentage_free_trajs_l):.4f}, {np.std(percentage_free_trajs_l):.4f})")
        print(
            f"percentage collision intensity: ({np.mean(percentage_collision_intensity_l):.4f}, {np.std(percentage_collision_intensity_l):.4f})"
        )
        print("---------------------------------\n")

        return percentage_free_trajs_l, percentage_collision_intensity_l

    def get_knn_control_points(self, q_start, q_goal, EE_pose_goal, k=2, normalized=True, **kwargs):
        assert self.context_qs, (
            "The joint positions should be used as context variables. " "TODO - implement for no context variables."
        )
        # get the k nearest neighbors context variables
        # task: start and goal control_points [n_trajectories, (state_dim or 2 * state_dim)]
        if self.context_qs and self.context_ee_goal_pose:
            # The context dataset is the concatenation of the start joint positions,
            # the end-effector goal orientation (flattened)
            # and the end-effector goal position
            # This is context has the same format as the input to the generative model
            orientation = self.fields[self.field_key_context_ee_goal_orientation]
            position = self.fields[self.field_key_context_ee_goal_position]
            if self.context_ee_goal_pose_bimanual:
                orientation = orientation.flatten(start_dim=-2)
                position = position.flatten(start_dim=-2)
            context_parts = [self.fields[self.field_key_q_start], orientation, position]
            if self.context_ee_goal_pose_bimanual:
                context_parts.append(self.fields[self.field_key_context_active_ee_mask])
            context_dataset = torch.cat(context_parts, dim=-1)
            context_dataset = to_torch(context_dataset, device=q_start.device)
            query_orientation = einops.rearrange(EE_pose_goal[..., :3, :3], "... d e -> ... (d e)")
            query_position = EE_pose_goal[..., :3, 3]
            if self.context_ee_goal_pose_bimanual:
                query_orientation = query_orientation.flatten(start_dim=-2)
                query_position = query_position.flatten(start_dim=-2)
                active_ee_mask = kwargs.get("active_ee_mask")
                if active_ee_mask is None:
                    raise ValueError("bimanual KNN query requires active_ee_mask")
                context_query = torch.cat(
                    [q_start, query_orientation, query_position, active_ee_mask.to(q_start)], dim=-1
                )[None, ...]
            else:
                context_query = torch.cat([q_start, query_orientation, query_position], dim=-1)[None, ...]
        elif self.context_qs:
            context_dataset = to_torch(self.fields[self.field_key_context_qs], device=q_start.device)
            context_query = torch.cat((q_start, q_goal), dim=-1)[None, ...]
        else:
            raise NotImplementedError

        # Compute the Euclidean distance between the query and the dataset
        dist = torch.linalg.vector_norm(context_dataset - context_query, dim=-1)
        # We evaluate our models using the training and validation sets of the dataset.
        # Therefore, it happens that for a context query value, the nearest neighbor is the query itself.
        # In a real scenario, this would not happen.
        # To prevent this, we discard samples that have a distance equal to 0.
        # We augment the number of neighbors > k + n, where n is the number of trajectories per task.
        _, cps_idxs = random.choice(list(self.map_task_id_to_control_points_id.items()))
        # Compute the neighbors and discard the ones with zero distance
        knn = dist.topk(k + len(cps_idxs) + 100, largest=False)
        idxs = knn.indices[knn.values > 0][:k]

        if normalized:
            return self.fields[f"{self.field_key_control_points}_normalized"].to(**self.tensor_args)[idxs]
        else:
            return self.fields[f"{self.field_key_control_points}"].to(**self.tensor_args)[idxs]


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))
