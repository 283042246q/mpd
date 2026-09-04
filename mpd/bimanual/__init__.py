from .runtime_contract import BimanualRequest, ContractError, JOINT_NAMES, TASK_MODES, validate_result

__all__ = [
    "BimanualRequest",
    "BimanualPlanningTask",
    "BimanualTrajectoryValidator",
    "ContractError",
    "JOINT_NAMES",
    "TASK_MODES",
    "ValidationReport",
    "validate_result",
]


def __getattr__(name):
    if name == "BimanualPlanningTask":
        from .planning_task import BimanualPlanningTask
        return BimanualPlanningTask
    if name in {"BimanualTrajectoryValidator", "ValidationReport"}:
        from .trajectory_validator import BimanualTrajectoryValidator, ValidationReport
        return {"BimanualTrajectoryValidator": BimanualTrajectoryValidator, "ValidationReport": ValidationReport}[name]
    raise AttributeError(name)
