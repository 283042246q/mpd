from .dynamic_runtime_engine_marvin_bimanual import MarvinBimanualDynamicRuntimeEngine


class MarvinBimanualSpaceTimeRuntimeEngine(MarvinBimanualDynamicRuntimeEngine):
    """Timing is optimized at inference only; checkpoints remain spatial."""

