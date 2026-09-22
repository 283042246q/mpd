#!/usr/bin/env python3
"""Run rigid Marvin transport with an independent dual-arm diffusion prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import uuid

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-cooperative-independent-prior.yaml"
)


def _parser():
    from scripts.inference import inference_marvin_bimanual

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start-goal-source", choices=("regions", "seed_file"), default=None)
    parser.add_argument("--start-goal-file", type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--request-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("mpd", "contract_stub"), default="mpd")
    parser.add_argument("--prior-mode", choices=("direct_project", "reference_residual"))
    parser.add_argument("--n-trajectory-samples", type=int)
    parser.add_argument("--ddim-sampling-timesteps", type=int)
    parser.add_argument("--guide-steps", type=int)
    projection = parser.add_mutually_exclusive_group()
    projection.add_argument(
        "--closed-chain-projection",
        dest="closed_chain_projection",
        action="store_true",
    )
    projection.add_argument(
        "--no-closed-chain-projection",
        dest="closed_chain_projection",
        action="store_false",
    )
    parser.set_defaults(closed_chain_projection=None)
    parser.add_argument("--dry-run", action="store_true")
    inference_marvin_bimanual.add_isaaclab_arguments(parser)
    return parser


def _runtime_config(args):
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cooperative = config.setdefault("cooperative_inference", {})
    if args.prior_mode is not None:
        cooperative["prior_mode"] = args.prior_mode
    if args.closed_chain_projection is not None:
        cooperative["closed_chain_projection"] = args.closed_chain_projection
    if args.n_trajectory_samples is not None:
        if args.n_trajectory_samples < 1:
            raise SystemExit("--n-trajectory-samples must be positive")
        config["n_trajectory_samples"] = args.n_trajectory_samples
        config["runtime_top_k_valid_trajectories"] = min(
            int(config["runtime_top_k_valid_trajectories"]),
            args.n_trajectory_samples,
        )
    if args.ddim_sampling_timesteps is not None:
        if args.ddim_sampling_timesteps < 1:
            raise SystemExit("--ddim-sampling-timesteps must be positive")
        config["ddim"]["ddim_sampling_timesteps"] = args.ddim_sampling_timesteps
    if args.guide_steps is not None:
        if args.guide_steps < 1:
            raise SystemExit("--guide-steps must be positive")
        config["ddim"]["n_guide_steps"] = args.guide_steps
    # A temporary override config must not change the interpretation of paths
    # that were relative to the checked-in config.
    generation = Path(config["cooperative_generation_config"])
    if not generation.is_absolute():
        config["cooperative_generation_config"] = str((config_path.parent / generation).resolve())
    return config_path, config


def main(argv=None):
    args = _parser().parse_args(argv)
    if not 0 <= args.seed <= 2**32 - 1:
        raise SystemExit("--seed must be in [0, 2^32-1]")
    if args.sample_index < -1:
        raise SystemExit("--sample-index must be -1 or non-negative")

    from mpd.bimanual.cooperative_sources import request_from_config_source
    from scripts.inference import inference_marvin_bimanual

    config_path, runtime_config = _runtime_config(args)
    request_id = args.request_id or f"marvin-cooperative-{uuid.uuid4()}"
    request = request_from_config_source(
        config_path,
        source=args.start_goal_source,
        source_path=args.start_goal_file,
        sample_index=args.sample_index,
        seed=args.seed,
        request_id=request_id,
    )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(
            yaml.safe_dump(
                {
                    "request": str(request_path),
                    "source": request["scene"]["start_goal_source"],
                    "prior_mode": runtime_config["cooperative_inference"]["prior_mode"],
                    "closed_chain_projection": runtime_config["cooperative_inference"]["closed_chain_projection"],
                    "n_trajectory_samples": runtime_config["n_trajectory_samples"],
                },
                sort_keys=False,
            )
        )
        return 0

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="marvin-cooperative-runtime-", delete=False
    ) as stream:
        yaml.safe_dump(runtime_config, stream, sort_keys=False)
        effective_config = Path(stream.name)
    try:
        delegated = [
            "--request",
            str(request_path),
            "--config",
            str(effective_config),
            "--output-dir",
            str(output),
            "--device",
            args.device,
            "--backend",
            args.backend,
        ]
        delegated.extend(inference_marvin_bimanual.isaaclab_arguments_from_namespace(args))
        return inference_marvin_bimanual.main(delegated)
    finally:
        effective_config.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
