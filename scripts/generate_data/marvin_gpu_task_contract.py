"""NumPy-only task/category contract for the Marvin GPU pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from scripts.generate_data.marvin_pair_sampling import (
    mixture_distribution,
    pair_distributions,
)


MODES = ("dual_independent", "dual_independent", "dual_independent", "left_only", "right_only")
DIRECTIONS = (
    "placement_to_placement",
    "random_to_placement",
    "placement_to_random",
    "random_to_random",
)


@dataclass(frozen=True)
class TaskDefinition:
    task_id: int
    mode: str
    direction: str
    source: dict[str, str]
    goal: dict[str, str]


def stable_seed(base_seed, task_id, attempt, stage):
    payload = f"{int(base_seed)}:{int(task_id)}:{int(attempt)}:{stage}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFF_FFFF


def direction_schedule(weights):
    values = np.asarray([float(weights.get(name, 0.0)) for name in DIRECTIONS])
    blocks = np.rint(values * 20).astype(int)
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.isclose(values.sum(), 1.0)
        or not np.allclose(blocks / 20.0, values)
    ):
        raise ValueError("direction weights must be nonnegative 0.05 increments summing to one")
    remaining = blocks.copy()
    schedule = []
    for step in range(20):
        current = np.asarray([schedule.count(name) for name in DIRECTIONS])
        deficits = values * (step + 1) - current
        deficits[remaining == 0] = -np.inf
        selected = int(np.argmax(deficits))
        schedule.append(DIRECTIONS[selected])
        remaining[selected] -= 1
    return tuple(schedule)


def placement_distributions(config):
    result = {}
    weighting = config.get("placement_region_weighting", "explicit_or_uniform")
    for arm in ("left", "right"):
        values = config["arm_placement_regions"][arm]
        names = tuple(values) if not isinstance(values, dict) else tuple(values)
        if isinstance(values, dict):
            weights = np.asarray(list(values.values()), dtype=float)
        elif weighting in {"volume", "mixture"}:
            weights = np.asarray(
                [
                    np.prod(
                        [
                            sum(
                                high - low
                                for low, high in config["placement_regions"][name][
                                    "translation"
                                ][axis]
                            )
                            for axis in "xyz"
                        ]
                    )
                    for name in names
                ],
                dtype=float,
            )
        else:
            weights = np.ones(len(names), dtype=float)
        if weighting == "mixture":
            weights = mixture_distribution(config, arm, names, weights)
        else:
            boosts = config.get("placement_region_difficulty_boosts", {}).get(arm, {})
            weights *= np.asarray([float(boosts.get(name, 1.0)) for name in names])
        result[arm] = names, weights / weights.sum()
    return result


class TaskContract:
    def __init__(self, config, base_seed):
        self.config = dict(config)
        self.base_seed = int(base_seed)
        self.distributions = placement_distributions(config)
        self.pairs = pair_distributions(config, self.distributions)
        self.directions = direction_schedule(config["trajectory_direction_weights"])

    def _region(self, rng, arm, exclude=None):
        names, weights = self.distributions[arm]
        if exclude is not None and len(names) > 1:
            keep = np.asarray([name != exclude for name in names])
            names = tuple(name for name, selected in zip(names, keep) if selected)
            weights = weights[keep] / weights[keep].sum()
        return str(rng.choice(names, p=weights))

    def build(self, task_id):
        mode = MODES[int(task_id) % 5]
        direction = self.directions[(int(task_id) // 5) % 20]
        rng = np.random.default_rng(
            stable_seed(self.base_seed, task_id, 0, "task-contract")
        )
        arms = ("left", "right") if mode == "dual_independent" else (mode.split("_", 1)[0],)
        source = {"left": "inactive", "right": "inactive"}
        goal = {"left": "inactive", "right": "inactive"}
        if direction == "placement_to_placement" and self.pairs is not None:
            if mode == "dual_independent":
                index = int(rng.choice(self.pairs["dual"].size, p=self.pairs["dual"].ravel()))
                selected = dict(
                    zip(
                        ("left", "right"),
                        np.unravel_index(index, self.pairs["dual"].shape),
                    )
                )
            else:
                arm = arms[0]
                selected = {
                    arm: int(rng.choice(self.pairs[arm].size, p=self.pairs[arm].ravel()))
                }
            for arm, index in selected.items():
                names = self.distributions[arm][0]
                first, second = np.unravel_index(index, self.pairs[arm].shape)
                source[arm], goal[arm] = names[first], names[second]
        else:
            if direction in {"placement_to_placement", "placement_to_random"}:
                source.update({arm: self._region(rng, arm) for arm in arms})
            else:
                source.update({arm: "random" for arm in arms})
            if direction in {"random_to_placement", "placement_to_placement"}:
                differ = bool(
                    self.config.get(
                        "placement_goal_must_differ_from_source_region", True
                    )
                )
                goal.update(
                    {
                        arm: self._region(
                            rng,
                            arm,
                            source[arm]
                            if differ and source[arm] != "random"
                            else None,
                        )
                        for arm in arms
                    }
                )
            else:
                goal.update({arm: "random" for arm in arms})
        return TaskDefinition(int(task_id), mode, direction, source, goal)
