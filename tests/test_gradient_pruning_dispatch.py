import unittest

from mpd.inference.cost_guides import CostGuideManagerParametricTrajectory


class GradientPruningDispatchTest(unittest.TestCase):
    def test_disabled_uses_legacy_without_touching_pruned_path(self):
        manager = object.__new__(CostGuideManagerParametricTrajectory)
        manager.gradient_pruning_enabled = False
        manager._legacy_call = lambda value, **kwargs: ("legacy", value, kwargs)
        manager._pruned_call = lambda *args, **kwargs: self.fail("pruned path was called")
        result = manager("sample", return_cost=True)
        self.assertEqual(result[0], "legacy")
        self.assertEqual(result[1], "sample")

    def test_enabled_uses_pruned_path(self):
        manager = object.__new__(CostGuideManagerParametricTrajectory)
        manager.gradient_pruning_enabled = True
        manager._legacy_call = lambda *args, **kwargs: self.fail("legacy path was called")
        manager._pruned_call = lambda value, **kwargs: ("pruned", value, kwargs)
        result = manager("sample")
        self.assertEqual(result[0], "pruned")


if __name__ == "__main__":
    unittest.main()
