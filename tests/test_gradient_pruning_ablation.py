import sys
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).parents[1] / "scripts" / "inference"
sys.path.insert(0, str(SCRIPT_DIR))
import benchmark_gradient_pruning_ablation as ABLATION  # noqa: E402


class GradientPruningAblationTest(unittest.TestCase):
    def test_incremental_variants_and_timing_profile_policy(self):
        with tempfile.TemporaryDirectory() as output_dir:
            index = ABLATION.generate_ablation_suite(
                ABLATION.resolve_manifest("simple_2d"),
                output_dir,
                candidates=7,
                scenario_ids=["open_clearance"],
            )
            configs = {
                variant: yaml.safe_load(Path(path).read_text())
                for variant, path in index["runs"][0]["variant_configs"].items()
            }

            self.assertEqual(list(configs), list(ABLATION.VARIANTS))
            self.assertFalse(configs["a0_legacy"]["gradient_pruning"]["enabled"])
            self.assertFalse(
                configs["a1_pruned_full"]["gradient_pruning"]["endpoint"]["ee_only_last_point"]
            )
            self.assertFalse(
                configs["a2_endpoint"]["gradient_pruning"]["temporal"]["enabled"]
            )
            self.assertTrue(configs["a2p_parent_link"]["gradient_pruning"]["spatial"]["parent_link_kinematics"])
            self.assertFalse(
                configs["a2p_parent_link"]["gradient_pruning"]["spatial"]["dense_parent_fast_path"]
            )
            self.assertTrue(
                configs["a2p_parent_fast"]["gradient_pruning"]["spatial"]["dense_parent_fast_path"]
            )
            self.assertFalse(
                configs["a2p_parent_fast"]["gradient_pruning"]["temporal"]["enabled"]
            )
            for variant in ("a2pf_clean_x0", "a2pf_clean_x0_2steps"):
                self.assertTrue(configs[variant]["compute_costs_with_xrecon"])
                self.assertFalse(
                    configs[variant]["gradient_pruning"]["temporal"]["enabled"]
                )
                self.assertTrue(
                    configs[variant]["gradient_pruning"]["spatial"]["parent_link_kinematics"]
                )
                self.assertTrue(
                    configs[variant]["gradient_pruning"]["spatial"]["dense_parent_fast_path"]
                )
            self.assertEqual(
                configs["a2pf_clean_x0_2steps"]["ddim"]["n_guide_steps"], 2
            )
            self.assertTrue(
                configs["a3r_temporal_parent"]["gradient_pruning"]["temporal"]["enabled"]
            )
            self.assertTrue(
                configs["a3r_temporal_parent"]["gradient_pruning"]["candidate"]["enabled"]
            )
            self.assertTrue(
                configs["a3r_temporal_parent"]["gradient_pruning"]["spatial"]["parent_link_kinematics"]
            )
            self.assertTrue(
                configs["a3r_temporal_parent"]["gradient_pruning"]["spatial"]["dense_parent_fast_path"]
            )
            self.assertFalse(
                configs["a3_temporal_diagnostic"]["gradient_pruning"]["spatial"]["parent_link_kinematics"]
            )
            self.assertFalse(
                configs["a5_temporal_full_scan"]["gradient_pruning"]["temporal"]["coarse_scan"]
            )
            b_variants = {
                "b0_a2pfast_materialized": (False, False, False, False),
                "b1_candidate_sparse": (True, False, False, False),
                "b2_time_sparse": (False, True, False, False),
                "b3_link_sparse": (False, False, True, False),
                "b4_control_point_sparse": (False, False, False, True),
                "b5_candidate_time": (True, True, False, False),
                "b6_candidate_time_link": (True, True, True, False),
                "b7_all_sparse": (True, True, True, True),
            }
            for variant, expected in b_variants.items():
                pruning = configs[variant]["gradient_pruning"]
                actual = (
                    pruning["candidate"]["enabled"],
                    pruning["temporal"]["enabled"],
                    pruning["spatial"]["active_link_pruning"],
                    pruning["mapping"]["sparse_bspline_support"],
                )
                self.assertEqual(actual, expected)
                self.assertFalse(pruning["mapping"]["fused_bspline_integration"])
                self.assertTrue(pruning["spatial"]["parent_link_kinematics"])
                self.assertTrue(pruning["spatial"]["dense_parent_fast_path"])
            for variant, broad, conditional in (
                ("c1_link_broad_phase", True, False),
                ("c2_conditional_temporal", False, True),
                ("c3_broad_phase_conditional", True, True),
            ):
                pruning = configs[variant]["gradient_pruning"]
                self.assertTrue(pruning["spatial"]["active_link_pruning"])
                self.assertEqual(
                    pruning["spatial"]["link_broad_phase"]["enabled"], broad
                )
                if broad:
                    self.assertEqual(
                        pruning["spatial"]["link_broad_phase"]["scan_geometry"],
                        "fine_spheres",
                    )
                self.assertFalse(pruning["temporal"]["enabled"])
                self.assertEqual(pruning["temporal"]["conditional_enabled"], conditional)
                self.assertEqual(
                    pruning["temporal"]["conditional_active_ratio_threshold"], 0.35
                )
                self.assertEqual(
                    pruning["preselection"]["parent_bounds_scan"],
                    variant == "c2_conditional_temporal",
                )
                self.assertFalse(pruning["mapping"]["fused_bspline_integration"])
            self.assertEqual(
                ABLATION.INCREMENTAL_PARENT["c1_link_broad_phase"], "b3_link_sparse"
            )
            self.assertEqual(
                ABLATION.INCREMENTAL_PARENT["c2_conditional_temporal"], "b3_link_sparse"
            )
            span = configs["d1_span_certificate"]["gradient_pruning"]
            self.assertTrue(span["span_certificate"]["enabled"])
            self.assertTrue(span["span_certificate"]["exact_sdf_for_certificate"])
            self.assertFalse(span["temporal"]["enabled"])
            self.assertFalse(span["temporal"]["conditional_enabled"])
            self.assertFalse(span["spatial"]["link_broad_phase"]["enabled"])
            self.assertFalse(span["spatial"]["link_broad_phase"]["reuse_scan_cache"])
            self.assertEqual(
                ABLATION.INCREMENTAL_PARENT["d1_span_certificate"], "b3_link_sparse"
            )
            self.assertEqual(
                ABLATION.INCREMENTAL_PARENT["a2pf_clean_x0"], "a2p_parent_fast"
            )
            self.assertEqual(
                ABLATION.INCREMENTAL_PARENT["a3r_temporal_parent"], "a2p_parent_fast"
            )
            for config in configs.values():
                pruning = config["gradient_pruning"]
                self.assertFalse(pruning["profile"])
                self.assertTrue(pruning["record_active_statistics"])
                self.assertEqual(config["n_trajectory_samples"], 7)

            executions = ABLATION.expected_executions(index, repeats=3)
            self.assertEqual(len(executions), len(ABLATION.VARIANTS) * 3)

            focused_variants = [
                "a0_legacy",
                "a2p_parent_fast",
                "a2pf_clean_x0",
                "a3r_temporal_parent",
            ]
            focused = ABLATION.expected_executions(
                index, repeats=2, variants=focused_variants
            )
            self.assertEqual(len(focused), len(focused_variants) * 2)
            self.assertEqual(
                list(dict.fromkeys(row["variant"] for row in focused)),
                focused_variants,
            )
            with self.assertRaisesRegex(ValueError, "Unknown variant"):
                ABLATION.resolve_variants(["not_a_variant"])


if __name__ == "__main__":
    unittest.main()
