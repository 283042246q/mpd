"""Compatibility façade for a future full MPD bimanual gradient guide."""
from .costs import closed_chain_cost, inactive_arm_hold_cost, interarm_clearance_cost, project_inactive_arm


class BimanualCostGuideManagerParametricTrajectory:
    def __init__(self, planning_task, **kwargs):
        self.planning_task = planning_task
        self.kwargs = kwargs

    def project(self, q):
        return self.planning_task.project(q)

    def compute_task_cost(self, q):
        return self.planning_task.closure_cost(q)

