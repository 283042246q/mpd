#!/usr/bin/env python3
"""Render an existing cooperative Marvin inference artifact in Isaac Lab.

Run this lightweight launcher from ``mpd-splines-public``.  Isaac Lab itself
is isolated in the configured Conda subprocess (``env_isaaclab`` by default).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from scripts.inference import inference_marvin_bimanual
from scripts.isaaclab.marvin_bimanual_asset import load_inference_artifact


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        help="Cooperative artifact directory, result.json, or trajectory.npz",
    )
    inference_marvin_bimanual.add_isaaclab_arguments(parser)
    parser.set_defaults(sim_backend="isaaclab", isaaclab_capture=True, isaaclab_replay=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    artifact = load_inference_artifact(args.result)
    if artifact.top_k_object_path_pose_xyzw is None:
        raise SystemExit("artifact has no cooperative payload poses; run the cooperative inference entrypoint")
    if args.sim_backend != "isaaclab":
        raise SystemExit("offline replay requires --sim-backend isaaclab")
    if args.isaaclab_capture:
        if args.isaaclab_video is None:
            args.isaaclab_video = artifact.root / "isaaclab-replay.mp4"
        if args.isaaclab_screenshot is None:
            args.isaaclab_screenshot = artifact.root / "isaaclab-replay.png"

    summary_path = artifact.root / "isaaclab-run.json"
    try:
        summary = inference_marvin_bimanual._run_isaaclab_backend(args, artifact.root)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(args.isaaclab_video if args.isaaclab_capture else summary_path)
        return 0 if summary["status"] == "completed" else 7
    except Exception as error:
        summary_path.write_text(
            json.dumps(
                {
                    "schema": "marvin_bimanual_isaaclab_run/v1",
                    "status": "fault",
                    "artifact": artifact.root.as_posix(),
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(error, file=sys.stderr)
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
