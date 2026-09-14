import subprocess
import sys

import h5py
import numpy as np
import yaml

from scripts.generate_data.generate_marvin_warehouse_bimanual import DEFAULT_CONFIG
from scripts.generate_data.marvin_endpoint_proposer import MarvinEndpointProposer
from scripts.generate_data.marvin_pybullet_auditor import MarvinPyBulletAuditor


DATASET = (
    "data_public/data_trajectories/"
    "EnvWarehouse-RobotMarvinBimanual-independent-v3-res002-nosimplifier-150k/"
    "dataset_merged.hdf5"
)


def test_endpoint_proposer_import_is_ompl_pybullet_and_cuda_free():
    command = (
        "import sys; "
        "import scripts.generate_data.marvin_endpoint_proposer; "
        "assert 'ompl' not in sys.modules; "
        "assert 'pybullet' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_endpoint_proposer_preserves_single_arm_contract():
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    proposer = MarvinEndpointProposer(config)
    proposal = proposer.propose(
        "left_only",
        "random_to_random",
        {"left": "random", "right": "inactive"},
        {"left": "random", "right": "inactive"},
        seed=17,
    )
    assert proposal is not None
    q_start, q_goal = proposal
    np.testing.assert_array_equal(q_goal[7:], q_start[7:])
    assert np.linalg.norm(q_goal[:7] - q_start[:7]) >= config["min_active_joint_delta"]
    assert proposer.ee_goal_pose(q_goal).shape == (2, 3, 4)


def test_pybullet_auditor_import_does_not_load_ompl():
    command = (
        "import sys; "
        "import scripts.generate_data.marvin_pybullet_auditor; "
        "assert 'ompl' not in sys.modules; "
        "assert 'pb_ompl.pb_ompl' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_pybullet_auditor_accepts_published_endpoint_and_path():
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    auditor = MarvinPyBulletAuditor(config)
    try:
        with h5py.File(DATASET) as dataset:
            q_start = dataset["q_start"][0]
            q_goal = dataset["q_goal"][0]
            path = dataset["sol_path"][0]
            spline = (
                dataset["bspline_params_tt"][0],
                dataset["bspline_params_cc"][0],
                int(dataset["bspline_params_k"][0]),
            )
        assert auditor.endpoints_valid(q_start, q_goal)
        assert auditor.trajectory_valid(path, spline)
        invalid = q_start.copy()
        invalid[0] = 100.0
        assert not auditor.state_valid(invalid)
    finally:
        auditor.close()
