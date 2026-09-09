#!/usr/bin/env python3
"""Validated thin entrypoint around the existing dimension-agnostic trainer."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import yaml


def results_dir_for_variant(results_dir: str, variant: str) -> str:
    """Keep command-line ablations in separate checkpoint directories."""
    marker = "_variant_"
    if marker in results_dir and results_dir.rsplit(marker, 1)[1] in {"A", "B", "C", "D"}:
        results_dir = results_dir.rsplit(marker, 1)[0]
    return f"{results_dir}{marker}{variant}"


def validate_config(config: dict) -> dict:
    if config.get("robot_model") != "marvin_bimanual":
        raise ValueError("robot_model must be marvin_bimanual")
    if config.get("task_family") not in {"independent", "cooperative"}:
        raise ValueError("task_family must be independent or cooperative")
    if int(config.get("state_dim", 14)) != 14 or not config.get("context_qs", True):
        raise ValueError("Marvin checkpoints require state_dim=14 and context_qs=true")
    use_ee = bool(config.get("context_ee_goal_pose", False))
    use_dual_ee = bool(config.get("context_ee_goal_pose_bimanual", False))
    if use_ee != use_dual_ee:
        raise ValueError("Marvin EE conditioning must use both context_ee_goal_pose and dual-slot bimanual mode")
    expected_q_dim = 14 if use_dual_ee else 28
    if int(config.get("context_q_dim", expected_q_dim)) != expected_q_dim:
        raise ValueError(f"Marvin context_q_dim must be {expected_q_dim} for this context mode")
    if use_dual_ee and int(config.get("raw_context_dim", 40)) != 40:
        raise ValueError("dual-slot context is q_start(14) + left EE(12) + right EE(12) + mask(2) = 40")
    variant = str(config.get("bimanual_network_variant", "A")).upper()
    if variant not in {"A", "B", "C", "D"}:
        raise ValueError("bimanual_network_variant must be one of A, B, C, D")
    config["bimanual_network_variant"] = variant
    if variant != "A" and not use_dual_ee:
        raise ValueError(f"bimanual network variant {variant} requires dual-slot EE conditioning")
    heads = int(config.get("bimanual_attention_heads", 4))
    context_dim = int(config.get("context_combined_out_dim", 128))
    if variant in {"C", "D"} and (heads < 1 or context_dim % heads):
        raise ValueError("context_combined_out_dim must be divisible by bimanual_attention_heads")
    if variant in {"C", "D"} and int(config.get("bimanual_context_attention_layers", 2)) < 1:
        raise ValueError("bimanual_context_attention_layers must be positive")
    dropout = float(config.get("bimanual_attention_dropout", 0.0))
    if variant in {"C", "D"} and not 0.0 <= dropout < 1.0:
        raise ValueError("bimanual_attention_dropout must be in [0, 1)")
    if variant == "D":
        if config.get("conditioning_type", "default") != "default":
            raise ValueError("variant D requires conditioning_type=default")
        if config.get("generative_model_class", "GaussianDiffusionModel") != "GaussianDiffusionModel":
            raise ValueError("variant D currently requires GaussianDiffusionModel")
        if int(config.get("bimanual_denoiser_attention_layers", 1)) < 1:
            raise ValueError("bimanual_denoiser_attention_layers must be positive")
        unet_input_dim = int(config.get("unet_input_dim", 32))
        if unet_input_dim % heads:
            raise ValueError("unet_input_dim must be divisible by bimanual_attention_heads for variant D")
    return config


def run_training(config):
    from scripts.train.train import experiment

    # Direct calls bypass experiment_launcher's CLI default resolution. Save
    # the complete model/diffusion configuration for checkpoint reconstruction.
    resolved = {
        name: parameter.default
        for name, parameter in inspect.signature(experiment).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    resolved.update(config)
    return experiment(**resolved)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate without starting a GPU training job")
    parser.add_argument("--network-variant", type=str.upper, choices=("A", "B", "C", "D"))
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text()) or {}
    if args.network_variant is not None:
        config["bimanual_network_variant"] = args.network_variant
    if "bimanual_network_variant" in config and config.get("results_dir"):
        config["results_dir"] = results_dir_for_variant(
            config["results_dir"], str(config["bimanual_network_variant"]).upper()
        )
    config = validate_config(config)
    raw_dim = 40 if config.get("context_ee_goal_pose_bimanual") else 28
    print(
        f"validated Marvin {config['task_family']} config: state_dim=14 "
        f"raw_context_dim={raw_dim} network_variant={config['bimanual_network_variant']}"
    )
    if args.dry_run:
        return 0
    return run_training(config)


if __name__ == "__main__":
    raise SystemExit(main())
