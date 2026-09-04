from .cost_guide import BimanualCostGuideManagerParametricTrajectory


class BimanualDynamicCostGuide(BimanualCostGuideManagerParametricTrajectory):
    """Spatial guide with a fixed world snapshot/time table."""

    def __init__(self, planning_task, dynamic_world=None, time_table=None, **kwargs):
        super().__init__(planning_task, **kwargs)
        self.dynamic_world = dynamic_world
        self.time_table = time_table

