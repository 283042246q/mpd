#!/usr/bin/env python3
"""Length-prefixed Unix-socket service for resident Marvin bimanual MPD."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.inference_marvin_bimanual import DEFAULT_CONFIG
from scripts.runtime.infer_server import ResidentPlannerService
from scripts.runtime.runtime_engine_marvin_bimanual import MarvinBimanualRuntimeEngine


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)

    def factory(state_callback):
        return MarvinBimanualRuntimeEngine(
            config_path=args.config,
            runtime_output_root=args.output_root,
            device_text=args.device,
            state_callback=state_callback,
        )

    ResidentPlannerService(args.socket, args.output_root, factory).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
