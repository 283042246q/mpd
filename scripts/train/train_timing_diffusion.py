#!/usr/bin/env python3
"""Train conditional TimingDiffusion directly from canonical Space-Time HDF5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

import yaml

from mpd.datasets.spacetime_timing_dataset import TIMING_REPRESENTATIONS
from mpd.timing_training.trainer import train_timing_diffusion


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env", dest="environment", default=None)
    parser.add_argument("--representation", choices=TIMING_REPRESENTATIONS, default=None)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        action="append",
        default=None,
        help="repeat to combine compatible canonical datasets; overrides training_data[env]",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--allow-hash-split-fallback", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def _expand_path(value: str, *, repository_root: Path) -> Path:
    expanded = value.replace("${REPO_ROOT}", str(repository_root))
    expanded = os.path.expandvars(os.path.expanduser(expanded))
    return Path(expanded).resolve()


def resolve_dataset_roots(
    config: Mapping[str, object],
    *,
    environment: str,
    repository_root: Path,
    command_line_roots: Sequence[Path] | None,
) -> list[Path]:
    if command_line_roots:
        return [Path(path).resolve() for path in command_line_roots]
    training_data = config.get("training_data")
    if not isinstance(training_data, Mapping) or environment not in training_data:
        raise ValueError(
            f"no training_data entry for environment={environment}; pass --dataset-root"
        )
    values = training_data[environment]
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Sequence) or not values:
        raise ValueError(f"training_data[{environment}] must be a non-empty path list")
    return [_expand_path(str(value), repository_root=repository_root) for value in values]


def main() -> int:
    args = _parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    with args.config.resolve().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    environment = args.environment or config.get("environment")
    if not environment:
        raise ValueError("environment must be set in config or with --env")
    representation = args.representation or config.get("representation", "c")
    if representation not in TIMING_REPRESENTATIONS:
        raise ValueError(f"unsupported representation: {representation}")
    dataset_roots = resolve_dataset_roots(
        config,
        environment=str(environment),
        repository_root=repository_root,
        command_line_roots=args.dataset_root,
    )
    data_config = dict(config.get("data", {}))
    model_config = dict(config.get("model", {}))
    diffusion_config = dict(config.get("diffusion", {}))
    training_config = dict(config.get("training", {}))
    if args.batch_size is not None:
        data_config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_config["num_workers"] = args.num_workers
    if args.max_train_samples is not None:
        data_config["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        data_config["max_val_samples"] = args.max_val_samples
    if args.allow_hash_split_fallback:
        data_config["allow_hash_split_fallback"] = True
    if args.device is not None:
        training_config["device"] = args.device
    if args.max_steps is not None:
        training_config["max_steps"] = args.max_steps
    if args.no_amp:
        training_config["use_amp"] = False

    run_name = args.run_name or str(config.get("run_name", "default"))
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        output_root = _expand_path(
            str(
                config.get(
                    "output_root", "${REPO_ROOT}/data_trained_models/timing_diffusion"
                )
            ),
            repository_root=repository_root,
        )
        output_dir = output_root / str(environment) / str(representation) / run_name
    result = train_timing_diffusion(
        environment=str(environment),
        representation=str(representation),
        dataset_roots=dataset_roots,
        output_dir=output_dir,
        data_config=data_config,
        model_config=model_config,
        diffusion_config=diffusion_config,
        training_config=training_config,
        resume_checkpoint=args.resume,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "final_step": result.final_step,
                "train_loss": result.train_loss,
                "validation_loss": result.validation_loss,
                "elapsed_seconds": result.elapsed_seconds,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
