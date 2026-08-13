import unittest
from pathlib import Path

import yaml

from mpd.inference.guidance_config import (
    resolve_dense_validation_config,
    resolve_gradient_pruning_config,
)


class GradientPruningConfigTest(unittest.TestCase):
    def test_default_warehouse_inference_config_is_cacheless_b3(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-config_file_v01_00.yaml"
        )
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = resolve_gradient_pruning_config(document)

        self.assertTrue(config["enabled"])
        self.assertTrue(config["endpoint"]["ee_only_last_point"])
        self.assertTrue(config["spatial"]["parent_link_kinematics"])
        self.assertTrue(config["spatial"]["dense_parent_fast_path"])
        self.assertTrue(config["spatial"]["active_link_pruning"])
        self.assertFalse(config["candidate"]["enabled"])
        self.assertFalse(config["preselection"]["parent_bounds_scan"])
        self.assertFalse(config["span_certificate"]["enabled"])
        self.assertFalse(config["temporal"]["enabled"])
        self.assertFalse(config["temporal"]["conditional_enabled"])
        self.assertFalse(config["temporal"]["reuse_selection_within_ddim_step"])
        self.assertFalse(config["spatial"]["link_broad_phase"]["enabled"])
        self.assertFalse(config["spatial"]["link_broad_phase"]["reuse_scan_cache"])
        self.assertFalse(config["mapping"]["fused_bspline_integration"])
        self.assertFalse(config["mapping"]["sparse_bspline_support"])

    def test_missing_config_defaults_to_cacheless_b3(self):
        config = resolve_gradient_pruning_config({"costs": {}})
        self.assertTrue(config["enabled"])
        self.assertEqual(config["temporal"]["buckets"], [32, 64, 128])
        self.assertFalse(config["temporal"]["enabled"])
        self.assertFalse(config["temporal"]["conditional_enabled"])
        self.assertEqual(config["temporal"]["conditional_active_ratio_threshold"], 0.35)
        self.assertTrue(config["temporal"]["coarse_scan"])
        self.assertFalse(config["temporal"]["reuse_selection_within_ddim_step"])
        self.assertTrue(config["spatial"]["parent_link_kinematics"])
        self.assertTrue(config["spatial"]["dense_parent_fast_path"])
        self.assertFalse(config["mapping"]["fused_bspline_integration"])
        self.assertFalse(config["candidate"]["enabled"])
        self.assertTrue(config["spatial"]["active_link_pruning"])
        self.assertFalse(config["spatial"]["link_broad_phase"]["enabled"])
        self.assertFalse(config["preselection"]["parent_bounds_scan"])
        self.assertFalse(config["span_certificate"]["enabled"])
        self.assertEqual(config["span_certificate"]["max_subdivision_depth"], 3)
        self.assertTrue(config["span_certificate"]["exact_sdf_for_certificate"])
        self.assertEqual(
            config["spatial"]["link_broad_phase"]["scan_geometry"],
            "fine_spheres",
        )
        self.assertFalse(
            config["spatial"]["link_broad_phase"]["reuse_scan_cache"]
        )
        self.assertFalse(config["mapping"]["sparse_bspline_support"])

    def test_partial_config_inherits_b3_top_level_default(self):
        config = resolve_gradient_pruning_config(
            {"gradient_pruning": {"temporal": {"enabled": True, "coarse_points": 16}}}
        )
        self.assertTrue(config["enabled"])
        self.assertEqual(config["temporal"]["coarse_points"], 16)

    def test_explicit_false_is_the_legacy_opt_out(self):
        config = resolve_gradient_pruning_config(
            {"gradient_pruning": {"enabled": False}}
        )
        self.assertFalse(config["enabled"])

    def test_disabled_config_ignores_future_spatial_options(self):
        config = resolve_gradient_pruning_config(
            {
                "gradient_pruning": {
                    "enabled": False,
                    "spatial": {"parent_link_kinematics": True},
                }
            }
        )
        self.assertFalse(config["enabled"])

    def test_enabled_unsupported_spatial_option_fails_loudly(self):
        with self.assertRaisesRegex(NotImplementedError, "environment_link_broad_phase"):
            resolve_gradient_pruning_config(
                {
                    "gradient_pruning": {
                        "enabled": True,
                        "spatial": {"environment_link_broad_phase": True},
                    }
                }
            )

    def test_parent_link_kinematics_is_supported(self):
        config = resolve_gradient_pruning_config(
            {
                "gradient_pruning": {
                    "enabled": True,
                    "spatial": {"parent_link_kinematics": True},
                }
            }
        )
        self.assertTrue(config["spatial"]["parent_link_kinematics"])

    def test_span_certificate_is_independent_from_temporal_and_broad_phase(self):
        config = resolve_gradient_pruning_config(
            {
                "gradient_pruning": {
                    "enabled": True,
                    "span_certificate": {"enabled": True},
                    "temporal": {"enabled": False, "conditional_enabled": False},
                    "spatial": {
                        "parent_link_kinematics": True,
                        "link_broad_phase": {"enabled": False},
                    },
                }
            }
        )
        self.assertTrue(config["span_certificate"]["enabled"])

        with self.assertRaisesRegex(ValueError, "independent B3 temporal path"):
            resolve_gradient_pruning_config(
                {
                    "gradient_pruning": {
                        "enabled": True,
                        "span_certificate": {"enabled": True},
                        "temporal": {"enabled": True},
                    }
                }
            )

    def test_dense_validation_has_independent_switch(self):
        config = resolve_dense_validation_config(
            {
                "gradient_pruning": {"enabled": False},
                "dense_validation": {
                    "enabled": True,
                    "runtime_points": 128,
                    "benchmark_points": 128,
                },
            }
        )
        self.assertTrue(config["enabled"])
        self.assertEqual(config["runtime_points"], 128)
        self.assertFalse(config["ranked_early_exit"]["enabled"])
        self.assertEqual(
            config["ranked_early_exit"]["batch_buckets"], [8, 16, 32, 64]
        )
        self.assertTrue(config["ranked_early_exit"]["preallocate_buffers"])
        self.assertFalse(config["ranked_early_exit"]["cuda_graph"])
        self.assertEqual(config["benchmark_points"], 128)

        with self.assertRaisesRegex(ValueError, "unified at 128 points"):
            resolve_dense_validation_config(
                {"dense_validation": {"enabled": True, "runtime_points": 256}}
            )

        ranked = resolve_dense_validation_config(
            {
                "dense_validation": {
                    "enabled": True,
                    "ranked_early_exit": {
                        "enabled": True,
                        "batch_buckets": [4, 8, 32],
                    },
                }
            }
        )
        self.assertTrue(ranked["ranked_early_exit"]["enabled"])
        self.assertEqual(
            ranked["ranked_early_exit"]["batch_buckets"], [4, 8, 32]
        )

        with self.assertRaisesRegex(ValueError, "batch_buckets"):
            resolve_dense_validation_config(
                {
                    "dense_validation": {
                        "ranked_early_exit": {"batch_buckets": [8, 8, 4]}
                    }
                }
            )

        with self.assertRaisesRegex(ValueError, "requires reject_invalid"):
            resolve_dense_validation_config(
                {
                    "dense_validation": {
                        "reject_invalid": False,
                        "ranked_early_exit": {"enabled": True},
                    }
                }
            )

    def test_fused_bspline_mapping_has_explicit_fallback(self):
        config = resolve_gradient_pruning_config(
            {
                "gradient_pruning": {
                    "enabled": True,
                    "mapping": {"fused_bspline_integration": False},
                }
            }
        )
        self.assertFalse(config["mapping"]["fused_bspline_integration"])

    def test_four_sparse_axes_have_independent_switches(self):
        config = resolve_gradient_pruning_config(
            {
                "gradient_pruning": {
                    "enabled": True,
                    "candidate": {"enabled": True},
                    "temporal": {"enabled": False},
                    "spatial": {"active_link_pruning": True},
                    "mapping": {
                        "fused_bspline_integration": False,
                        "sparse_bspline_support": True,
                    },
                }
            }
        )
        self.assertTrue(config["candidate"]["enabled"])
        self.assertFalse(config["temporal"]["enabled"])
        self.assertTrue(config["spatial"]["active_link_pruning"])
        self.assertFalse(config["mapping"]["fused_bspline_integration"])
        self.assertTrue(config["mapping"]["sparse_bspline_support"])

    def test_link_broad_phase_and_conditional_temporal_are_independent(self):
        broad_only = resolve_gradient_pruning_config(
            {
                "gradient_pruning": {
                    "enabled": True,
                    "temporal": {"enabled": False, "conditional_enabled": False},
                    "spatial": {"link_broad_phase": {"enabled": True}},
                }
            }
        )
        self.assertTrue(broad_only["spatial"]["link_broad_phase"]["enabled"])
        self.assertFalse(broad_only["temporal"]["conditional_enabled"])

        conditional_only = resolve_gradient_pruning_config(
            {
                "gradient_pruning": {
                    "enabled": True,
                    "temporal": {
                        "enabled": False,
                        "conditional_enabled": True,
                        "conditional_active_ratio_threshold": 0.35,
                    },
                    "spatial": {"link_broad_phase": {"enabled": False}},
                }
            }
        )
        self.assertFalse(conditional_only["spatial"]["link_broad_phase"]["enabled"])
        self.assertTrue(conditional_only["temporal"]["conditional_enabled"])

        with self.assertRaisesRegex(ValueError, "must be in"):
            resolve_gradient_pruning_config(
                {
                    "gradient_pruning": {
                        "enabled": True,
                        "temporal": {"conditional_active_ratio_threshold": 1.1},
                    }
                }
            )

    def test_link_broad_phase_rejects_unknown_scan_geometry(self):
        with self.assertRaisesRegex(ValueError, "scan_geometry"):
            resolve_gradient_pruning_config(
                {
                    "gradient_pruning": {
                        "enabled": True,
                        "spatial": {
                            "link_broad_phase": {"scan_geometry": "capsule"}
                        },
                    }
                }
            )

    def test_parent_bound_preselection_is_independent_and_requires_parent_kinematics(self):
        config = resolve_gradient_pruning_config(
            {
                "gradient_pruning": {
                    "enabled": True,
                    "preselection": {"parent_bounds_scan": True},
                    "temporal": {"conditional_enabled": True},
                    "spatial": {"link_broad_phase": {"enabled": False}},
                }
            }
        )
        self.assertTrue(config["preselection"]["parent_bounds_scan"])
        self.assertFalse(config["spatial"]["link_broad_phase"]["enabled"])

        with self.assertRaisesRegex(ValueError, "parent_bounds_scan"):
            resolve_gradient_pruning_config(
                {
                    "gradient_pruning": {
                        "enabled": True,
                        "preselection": {"parent_bounds_scan": True},
                        "spatial": {"parent_link_kinematics": False},
                    }
                }
            )
        with self.assertRaisesRegex(ValueError, "requires parent_link_kinematics"):
            resolve_gradient_pruning_config(
                {
                    "gradient_pruning": {
                        "enabled": True,
                        "spatial": {
                            "parent_link_kinematics": False,
                            "link_broad_phase": {"enabled": True},
                        },
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()
