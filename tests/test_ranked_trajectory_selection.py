import torch

from mpd.inference.inference import _compute_candidate_ranking


class FakeRobot:
    dq_max = None
    ddq_max = None

    @staticmethod
    def get_position(value):
        return value


class FakeTask:
    robot = FakeRobot()


class FakeDataset:
    context_ee_goal_pose = False


def test_shortest_path_ranking_is_computed_before_validation():
    q_position = torch.tensor(
        [
            [[0.0], [2.0], [0.0]],
            [[0.0], [0.5], [1.0]],
            [[0.0], [0.1], [0.2]],
        ],
        dtype=torch.float64,
    )
    zeros = torch.zeros_like(q_position)
    control_points = torch.zeros(3, 2, 1, dtype=torch.float64)
    scores, metadata = _compute_candidate_ranking(
        "shortest_path_length",
        {},
        FakeDataset(),
        FakeTask(),
        None,
        control_points,
        q_position,
        zeros,
        zeros,
        None,
    )
    torch.testing.assert_close(scores, torch.tensor([4.0, 1.0, 0.2], dtype=torch.float64))
    assert torch.argsort(scores).tolist() == [2, 1, 0]
    assert metadata["method"] == "shortest_path_length"
