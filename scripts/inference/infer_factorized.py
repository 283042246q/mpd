#!/usr/bin/env python3
"""Independent F1/F2/F3 CLI with the existing request/world/NPZ contract."""
from pathlib import Path
import argparse
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.infer_space_time import run, _positive_integer, _nonnegative_integer
from scripts.runtime.factorized_runtime_engine import FactorizedMpdRuntimeEngine
from scripts.runtime.infer_once import _atomic_write_json


def _build_parser():
    parser = argparse.ArgumentParser(description="Learned factorized F1/F2/F3 MPD inference (rest-to-rest Panda).")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--world", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path,
        default=REPO_ROOT / "scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-factorized.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=_positive_integer, default=1)
    parser.add_argument("--seed", type=_nonnegative_integer)
    parser.add_argument("--seed-step", type=_nonnegative_integer, default=0)
    parser.add_argument("--trajectory-start-unix-ns", type=_nonnegative_integer)
    parser.add_argument("--max-dynamic-objects", type=_positive_integer, default=16)
    parser.add_argument("--covariance-sigma", type=float, default=3.)
    parser.add_argument("--process-acceleration-std", type=float, default=.01)
    parser.add_argument("--duration-min", type=float, default=2.)
    parser.add_argument("--duration-max", type=float, default=14.)
    parser.add_argument("--nominal-duration", type=float, default=10.)
    parser.add_argument("--no-static-spatial-pruning", dest="static_spatial_pruning", action="store_false", default=True)
    parser.set_defaults(timing_mode="phase5_joint", timing_control_points=8, timing_degree=3,
        timing_learning_rate=.08, u_min=.05, dynamic_space_time_pruning=False)
    parser.add_argument("--method", choices=("f1", "f2", "f3"), default="f1")
    parser.add_argument("--timing-checkpoint", type=Path, required=True)
    parser.add_argument("--adapt-spatial-basis", action="store_true",
        help="Explicitly evaluate the timing encoder on a different spatial control-point count (same degree/robot).")
    parser.add_argument("--debug", action="store_true", help="Print a full traceback on failure.")
    parser.add_argument("--space-steps", type=int, default=32)
    parser.add_argument("--timing-steps", type=int, default=100)
    parser.add_argument("--space-guide-fraction", type=float, default=.3)
    parser.add_argument("--timing-guide-fraction", type=float, default=.3)
    parser.add_argument("--space-lr", type=float, default=.01)
    parser.add_argument("--timing-lr", type=float, default=.02)
    parser.add_argument("--refinement-steps", type=int, default=5)
    parser.add_argument("--weak-dynamic-scale", type=float, default=.2)
    parser.add_argument("--eta", type=float, default=0.)
    parser.add_argument("--alternating-rounds", type=int, default=1)
    parser.add_argument("--timing-steps-per-space-step", type=int, choices=(1, 2), default=1)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    keys = ("method", "space_steps", "timing_steps", "space_guide_fraction", "timing_guide_fraction",
            "space_lr", "timing_lr", "refinement_steps", "weak_dynamic_scale", "eta",
            "alternating_rounds", "timing_steps_per_space_step")
    settings = {key: getattr(args, key) for key in keys}
    def factory(**kwargs):
        return FactorizedMpdRuntimeEngine(**kwargs, timing_checkpoint=args.timing_checkpoint,
                                         factorized_settings=settings, adapt_spatial_basis=args.adapt_spatial_basis)
    try:
        path = run(args, engine_factory=factory)
        summary = json.loads(path.read_text())
        summary.update(schema="mpd_factorized_inference", method=args.method, timing_mode=args.method,
                       timing_checkpoint=str(args.timing_checkpoint.resolve()))
        _atomic_write_json(path, summary)
        print(path)
        return 0
    except Exception as error:
        if args.debug:
            import traceback
            traceback.print_exc()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(args.output_dir / "summary.json", dict(schema="mpd_factorized_inference",
            status=getattr(error, "status", "inference_error"), method=args.method,
            error={"type": type(error).__name__, "message": str(error)}))
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return getattr(error, "exit_code", 1)


if __name__ == "__main__":
    raise SystemExit(main())
