import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import collections
import socket
import subprocess
import time

import h5py
import numpy as np

from experiment_launcher import Launcher
from experiment_launcher.utils import is_local, get_slurm_jobs_in_queue
from mpd.utils.githash import get_git_hash_short


########################################################################################################################
# LAUNCHER

hostname = socket.gethostname()

LOCAL = is_local()
TEST = False
# USE_CUDA = True
USE_CUDA = False


N_EXPS_IN_PARALLEL = 1

N_CORES = 3
MEMORY_SINGLE_JOB = 3000
MEMORY_PER_CORE = MEMORY_SINGLE_JOB
if "logc" in hostname:
    PARTITION = None
else:
    PARTITION = "gpu" if USE_CUDA else "amd,amd2"
GRES = "gpu:1" if USE_CUDA else None
CONDA_ENV = "mpd-splines-public"
DATA_TRAJECTORIES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data_trajectories")
)


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

MAX_SLURM_JOBS_IN_QUEUE = 400

########################################################################################################################
# EXPERIMENT PARAMETERS SETUP

exp_config = collections.namedtuple(
    "config",
    "env_id robot_id"
    " num_tasks num_trajectories_per_task"
    " min_distance_robot_env"
    " parametric_trajectory planner_allowed_time"
    " bspline_num_control_points bspline_degree"
    " sample_joint_position_goals_with_same_ee_pose"
    " cfg_file",
)

PLANNER = "RRTConnect"
# PLANNER = 'AITstar'

configs_d = {
    # many trajectories per task
    "many": [
        # joint to joint
        #exp_config("EnvSimple2D", "RobotPointMass2D", 1000, 25, 0.01, PLANNER, 10.0, 30, 5, False, None),
        #exp_config("EnvNarrowPassageDense2D", "RobotPointMass2D", 1000, 25, 0.01, PLANNER, 10.0, 30, 5, False, None),
        #exp_config("EnvPlanar2Link", "RobotPlanar2Link", 1000, 25, 0.01, PLANNER, 10.0, 30, 5, False, None),
        #exp_config("EnvPlanar4Link", "RobotPlanar4Link", 10000, 10, 0.01, PLANNER, 10.0, 30, 5, False, None),
        #exp_config("EnvSpheres3D", "RobotPanda", 100000, 10, 0.02, PLANNER, 10.0, 38, 5, False, None),
        # joint or EE pose to EE pose with configuration file
        #exp_config(
        #    "EnvWarehouse",
        #    "RobotPanda",
        #    500000,
        #    1,
        #    0.02,
        #    PLANNER,
        #    10.0,
        #    29,
        #    5,
        #    False,
        #    "EnvWarehouse-RobotPanda_v01.yaml",
        #),
        exp_config(
            "EnvOpenDrawerShelf",
            "RobotPanda",
            1000,
            1,
            0.02,
            PLANNER,
            30.0,
            22,
            5,
            False,
            "EnvOpenDrawerShelf-RobotPanda-drawer-to-shelf.yaml",
        ),
    ],
    # one trajectory per task
    "one": [
        # joint to joint
        #exp_config("EnvSimple2D", "RobotPointMass2D", 10000, 1, 0.01, PLANNER, 10.0, 30, 5, False, None),
        #exp_config("EnvNarrowPassageDense2D", "RobotPointMass2D", 10000, 1, 0.01, PLANNER, 10.0, 30, 5, False, None),
        #exp_config("EnvPlanar2Link", "RobotPlanar2Link", 10000, 1, 0.01, PLANNER, 10.0, 30, 5, False, None),
        #exp_config("EnvPlanar4Link", "RobotPlanar4Link", 100000, 1, 0.01, PLANNER, 10.0, 30, 5, False, None),
        #exp_config("EnvSpheres3D", "RobotPanda", 1000000, 1, 0.02, PLANNER, 10.0, 38, 5, False, None),
        #exp_config("EnvWarehouse", "RobotPanda", 1000000, 1, 0.02, PLANNER, 10.0, 30, 5, False, None),
        #exp_config("EnvPilars3D", "RobotPanda", 500000, 1, 0.02, PLANNER, 10.0, 30, 5, False, None),
        # with configuration file
        #exp_config(
        #    "EnvWarehouse",
        #    "RobotPanda",
        #    500000,
        #    1,
        #    0.02,
        #    PLANNER,
        #    10.0,
        #    29,
        #    5,
        #    False,
        #    "EnvWarehouse-RobotPanda_v01.yaml",
        #),
        exp_config(
            "EnvOpenDrawerShelf",
            "RobotPanda",
            150000,
            1,
            0.02,
            PLANNER,
            30.0,
            22,
            5,
            False,
            "EnvOpenDrawerShelf-RobotPanda-drawer-to-shelf.yaml",
        ),
    ],
}

configs_d_keys_filter = [
    # 'many',
    "one",
]


def get_shard_results_dir(exp_name, config, selection, start_task_id):
    path_components = [
        DATA_TRAJECTORIES_DIR,
        exp_name,
        config.env_id,
        config.robot_id,
    ]
    if config.cfg_file is not None:
        path_components.append("yes")
    path_components.extend(
        [
            str(config.sample_joint_position_goals_with_same_ee_pose),
            selection,
            config.parametric_trajectory,
            str(start_task_id),
        ]
    )
    return os.path.join(*path_components)


def is_shard_complete(shard_results_dir, num_trajectories_desired):
    required_paths = [
        os.path.join(shard_results_dir, "args.yaml"),
        os.path.join(shard_results_dir, "timing_stats.pkl"),
        os.path.join(shard_results_dir, "dataset.hdf5"),
    ]
    if not all(os.path.isfile(path) for path in required_paths):
        return False

    try:
        with h5py.File(required_paths[-1], "r") as dataset:
            return (
                dataset.attrs.get("num_trajectories_desired") == num_trajectories_desired
                and "num_trajectories_generated" in dataset.attrs
            )
    except OSError:
        return False

if __name__ == "__main__":
    response = input(
        "Completed shards will be skipped; incomplete existing shards will be overwritten. "
        "Do you want to continue? (yes/no): "
    ).lower()
    if response not in ["yes", "y"]:
        raise SystemExit(1)
else:
    configs_d_keys_filter = []

for selection in configs_d_keys_filter:

    configs = configs_d[selection]

    if selection == "many":
        N_TASKS_PER_EXPERIMENT = 100
    else:
        N_TASKS_PER_EXPERIMENT = 500

    ###############################
    # Launch jobs
    n_job = 0
    for k, config in enumerate(configs):
        n_tasks_to_process_total = config.num_tasks
        start_task_id = 0

        while n_tasks_to_process_total > 0:
            if not LOCAL:  # Wait for jobs to finish in the cluster
                while get_slurm_jobs_in_queue() >= MAX_SLURM_JOBS_IN_QUEUE:
                    sleep_seconds = 30
                    print(f"Waiting for jobs to finish. Sleeping for {sleep_seconds} seconds.")
                    time.sleep(sleep_seconds)

            # Launch new jobs
            n_tasks_to_process_in_experiment = N_TASKS_PER_EXPERIMENT
            if n_tasks_to_process_in_experiment > n_tasks_to_process_total:
                n_tasks_to_process_in_experiment = n_tasks_to_process_total

            print(
                f"---------> Launched job {n_job:4d} -- {config.env_id} and {config.robot_id} "
                f"for tasks {start_task_id}-{start_task_id + n_tasks_to_process_in_experiment}"
            )

            time.sleep(2)

            exp_name = f"{config.env_id}-{config.robot_id}"
            if config.cfg_file is not None:
                if config.cfg_file == "EnvOpenDrawerShelf-RobotPanda-drawer-to-shelf.yaml":
                    exp_name += "-drawer-to-shelf-reachable-levels"
                else:
                    exp_name += "-config_file"
            if config.sample_joint_position_goals_with_same_ee_pose:
                exp_name += "-joint_ee"
            else:
                exp_name += "-joint_joint"
            exp_name += f"-{selection}"
            exp_name += f"-{config.parametric_trajectory}"

            shard_results_dir = get_shard_results_dir(
                exp_name, config, selection, start_task_id
            )
            num_trajectories_desired = (
                n_tasks_to_process_in_experiment * config.num_trajectories_per_task
            )
            if is_shard_complete(shard_results_dir, num_trajectories_desired):
                print(f"Skipping completed shard: {shard_results_dir}")
                start_task_id += n_tasks_to_process_in_experiment
                n_tasks_to_process_total -= n_tasks_to_process_in_experiment
                n_job += 1
                continue

            launcher = Launcher(
                exp_name=exp_name,
                exp_file="generate_trajectories",
                project_name="project02390",
                start_seed=start_task_id,
                n_seeds=1,
                n_exps_in_parallel=N_EXPS_IN_PARALLEL,
                n_cores=N_CORES,
                memory_per_core=MEMORY_PER_CORE,
                days=0,
                hours=23,
                minutes=59,
                seconds=0,
                partition=PARTITION,
                conda_env=CONDA_ENV,
                gres=GRES,
                use_timestamp=False,
                check_results_directories=False,
                compact_dirs=True,
                base_dir=DATA_TRAJECTORIES_DIR,
            )

            ############################################################################################################
            # RUN
            extra_cfg_file = {}
            if config.cfg_file is not None:
                extra_cfg_file["cfg_file"] = config.cfg_file
                extra_cfg_file["configs__"] = "yes"
            else:
                extra_cfg_file["cfg_file"] = "None"

            print(f"extra_cfg_file: {extra_cfg_file}")

            launcher.add_experiment(
                env_id__=config.env_id,
                robot_id__=config.robot_id,
                **extra_cfg_file,
                sample_joint_position_goals_with_same_ee_pose__=config.sample_joint_position_goals_with_same_ee_pose,
                selection__=selection,
                planner__=config.parametric_trajectory,
                start_task_id=start_task_id,
                num_tasks=n_tasks_to_process_in_experiment,
                num_trajectories_per_task=config.num_trajectories_per_task,
                min_distance_robot_env=config.min_distance_robot_env,
                min_distance_q_pos_start_goal=0.5,
                simplify_path=True,
                planner_allowed_time=config.planner_allowed_time,
                fit_bspline=False,  # do not fit a bspline during trajectory generation
                bspline_num_control_points=config.bspline_num_control_points,
                bspline_degree=config.bspline_degree,
                bspline_zero_vel_at_start_and_goal=True,
                bspline_zero_acc_at_start_and_goal=True,
                n_parallel_jobs=N_CORES,
                task_batch_size=1,
                task_timeout_seconds=300,
                device="cuda" if USE_CUDA else "cpu",
                debug=False,
            )

            launcher.run(LOCAL, TEST)

            ############################################################################################################
            # Update counters
            start_task_id += n_tasks_to_process_in_experiment
            n_tasks_to_process_total -= n_tasks_to_process_in_experiment
            n_job += 1
