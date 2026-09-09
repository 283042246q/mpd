#!/usr/bin/env python3
"""Strict one-shot Marvin bimanual MPD runtime.

This entry point owns request validation and artifact publication. It shares
the Phase-1 planner implementation, but does not dispatch through that CLI and
never silently selects the contract stub.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpd.bimanual.runtime_contract import BimanualRequest, ContractError
from scripts.inference.inference_marvin_bimanual import (
    DEFAULT_CONFIG,
    InferenceConfigurationError,
    NoValidTrajectoryError,
    _failure,
    _real_plan,
    _sha256_file,
    _stub_plan,
    _write_json,
    _write_npz,
)


def run_one_shot(
    request_path: Path,
    output_dir: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    device: str = "cuda:0",
    backend: str = "mpd",
    stub_points: int = 64,
    stub_duration: float = 2.0,
) -> dict:
    """Validate one request, run one plan, and atomically publish its files."""
    request_path = Path(request_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    raw_request = json.loads(request_path.read_text(encoding="utf-8"))
    request = BimanualRequest.from_dict(raw_request)
    if request.runtime_mode != "snapshot_no_time":
        raise ContractError("one-shot Phase-3 runtime requires snapshot_no_time")
    if backend == "mpd":
        result, arrays, scene = _real_plan(
            request, Path(config_path).expanduser().resolve(), device
        )
    elif backend == "contract_stub":
        result, arrays, scene = _stub_plan(request, stub_points, stub_duration)
    else:
        raise ValueError("backend must be mpd or contract_stub")

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectory.npz"
    scene_path = output_dir / "scene.json"
    result_path = output_dir / "result.json"
    trajectory_path.unlink(missing_ok=True)
    scene_path.unlink(missing_ok=True)
    _write_npz(trajectory_path, arrays)
    _write_json(scene_path, scene)
    result["artifacts"] = {
        "request_sha256": _sha256_file(request_path),
        "trajectory_sha256": _sha256_file(trajectory_path),
        "scene_file_sha256": _sha256_file(scene_path),
    }
    result["one_shot"] = {
        "entrypoint": "infer_once_marvin_bimanual.py",
        "published_unix_ns": time.time_ns(),
    }
    _write_json(result_path, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backend", choices=("mpd", "contract_stub"), default="mpd"
    )
    parser.add_argument("--stub-points", type=int, default=64)
    parser.add_argument("--stub-duration", type=float, default=2.0)
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    result_path = output_dir / "result.json"
    request_id = None
    try:
        try:
            raw = json.loads(args.request.expanduser().read_text(encoding="utf-8"))
            request_id = raw.get("request_id") if isinstance(raw, dict) else None
        except Exception:
            pass
        run_one_shot(
            args.request,
            output_dir,
            config_path=args.config,
            device=args.device,
            backend=args.backend,
            stub_points=args.stub_points,
            stub_duration=args.stub_duration,
        )
        print(result_path)
        return 0
    except (ContractError, json.JSONDecodeError, OSError) as error:
        _write_json(result_path, _failure(request_id, "invalid_request", error))
        print(error, file=sys.stderr)
        return 2
    except TimeoutError as error:
        _write_json(result_path, _failure(request_id, "deadline_exceeded", error))
        print(error, file=sys.stderr)
        return 3
    except NoValidTrajectoryError as error:
        _write_json(result_path, _failure(request_id, "no_valid_trajectory", error))
        print(error, file=sys.stderr)
        return 4
    except Exception as error:
        _write_json(result_path, _failure(request_id, "fault", error))
        print(error, file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
