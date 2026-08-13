import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "inference" / "benchmark_gradient_pruning.py"
SPEC = importlib.util.spec_from_file_location("benchmark_gradient_pruning", SCRIPT_PATH)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class GradientPruningBenchmarkTest(unittest.TestCase):
    def test_generated_pair_differs_only_in_pruning_config(self):
        base = {"model_dir": "/tmp/model", "n_trajectory_samples": 10}
        scenario = {
            "id": "single",
            "type": "warehouse_extra_boxes",
            "extra_boxes_yaml": "warehouse/extra_boxes/warehouse_extra_boxes_v00.yaml",
        }
        baseline, pruning = BENCHMARK.paired_configs(base, scenario, seed=7, dense_points=128)
        baseline_without_switch = dict(baseline)
        pruning_without_switch = dict(pruning)
        baseline_without_switch.pop("gradient_pruning")
        pruning_without_switch.pop("gradient_pruning")
        self.assertEqual(baseline_without_switch, pruning_without_switch)
        self.assertFalse(baseline["gradient_pruning"]["enabled"])
        self.assertTrue(pruning["gradient_pruning"]["enabled"])

    def test_generate_suite_is_deterministic_except_timestamp(self):
        manifest = BENCHMARK.resolve_manifest("warehouse_panda")
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            index_a = BENCHMARK.generate_suite(manifest, first, seed=11)
            index_b = BENCHMARK.generate_suite(manifest, second, seed=11)
            self.assertEqual(
                [run["seed"] for run in index_a["runs"]],
                [run["seed"] for run in index_b["runs"]],
            )
            config_a = yaml.safe_load(Path(index_a["runs"][0]["baseline_config"]).read_text())
            config_b = yaml.safe_load(Path(index_b["runs"][0]["baseline_config"]).read_text())
            self.assertEqual(config_a, config_b)


if __name__ == "__main__":
    unittest.main()
