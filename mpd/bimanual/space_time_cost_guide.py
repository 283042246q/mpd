from .dynamic_cost_guide import BimanualDynamicCostGuide


class BimanualSpaceTimeCostGuide(BimanualDynamicCostGuide):
    """Inference-only timing extension; timing is never a training variable."""

    def cost(self, q, timing=None):
        return self.compute_task_cost(q)

