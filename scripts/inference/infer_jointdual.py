"""Offline JointDual inference for request/world JSON or an oracle dataset entry."""
import argparse
import json
from pathlib import Path
import time
import h5py

from mpd.inference.jointdual_world import OracleForecastWorld
from scripts.runtime.jointdual_runtime_engine import JointDualMpdRuntimeEngine, DEFAULT_CONFIG
from scripts.runtime.infer_once import _atomic_write_json, _atomic_write_npz, EXPECTED_JOINT_NAMES


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint-checkpoint",type=Path,required=True)
    p.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument("--request",type=Path)
    source.add_argument("--dataset-root",type=Path)
    p.add_argument("--sample-index",type=int,default=0)
    p.add_argument("--split",choices=("train","validation","test"),default="test")
    p.add_argument("--world",type=Path)
    p.add_argument("--trajectory-start-unix-ns",type=int)
    args=p.parse_args(argv)
    if args.sample_index<0:p.error("sample-index must be nonnegative")
    if args.dataset_root and args.world:p.error("dataset oracle already defines the world")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name in ("trajectory.npz","result.json"):
        (args.output_dir/name).unlink(missing_ok=True)
    try:
        engine=JointDualMpdRuntimeEngine(joint_checkpoint=args.joint_checkpoint,config_path=args.config,
            runtime_output_root=args.output_dir,device_text=args.device)
        oracle=None
        if args.dataset_root:
            index=json.loads((args.dataset_root/"index.json").read_text())
            e=index["splits"][args.split][args.sample_index]
            with h5py.File(args.dataset_root/e["file"]) as f:
                g=f[e["group"]]
                oracle=OracleForecastWorld(g["world/times"][:],g["world/positions"][:],
                    g["world/shape"][:],g["world/size"][:],device=args.device)
                # Joint target is converted to EE pose; its q is never a model condition.
                request=dict(scene_id="EnvWarehouse",joint_names=list(EXPECTED_JOINT_NAMES),
                    goal_type="joint",q_pos_start=g["condition/q_start"][:].tolist(),
                    q_pos_goal=g["condition/q_goal"][:].tolist())
        else:
            request=json.loads(args.request.read_text())
            world=json.loads(args.world.read_text()) if args.world else dict(world_version=1,
                frame_id="fr3_link0",stamp_unix_ns=1,valid_until_unix_ns=2**63-1,objects=[])
            engine.update_world(world)
            request.update(_dynamic_world_version=world["world_version"],
                _trajectory_start_unix_ns=args.trajectory_start_unix_ns or world["stamp_unix_ns"])
        artifacts=engine.plan(request,oracle_world=oracle)
        _atomic_write_npz(args.output_dir/"trajectory.npz",compressed=False,**artifacts.trajectory_arrays)
        _atomic_write_json(args.output_dir/"result.json",artifacts.result_payload)
        _atomic_write_json(args.output_dir/"summary.json",dict(status="success",engine=engine.health()))
        return 0
    except Exception as exc:
        failure=dict(status=getattr(exc,"status","inference_error"),
            error=dict(type=type(exc).__name__,message=str(exc)))
        _atomic_write_json(args.output_dir/"summary.json",failure)
        _atomic_write_json(args.output_dir/"result.json",dict(failure,trajectory_file=None))
        print(f"{type(exc).__name__}: {exc}")
        return getattr(exc,"exit_code",1)


if __name__=="__main__":raise SystemExit(main())
