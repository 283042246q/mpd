"""Serve any JointDual checkpoint on its own socket; old servers are unchanged."""
import argparse
from pathlib import Path
from scripts.runtime.jointdual_runtime_engine import JointDualMpdRuntimeEngine, DEFAULT_CONFIG
from scripts.runtime.infer_dynamic_server import DynamicResidentPlannerService


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint-checkpoint",type=Path,required=True)
    p.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    p.add_argument("--socket",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args(argv)
    def factory(callback):
        return JointDualMpdRuntimeEngine(joint_checkpoint=args.joint_checkpoint,config_path=args.config,
            runtime_output_root=args.output_root,device_text=args.device,state_callback=callback)
    DynamicResidentPlannerService(args.socket,args.output_root,factory,trajectory_compression=False).serve_forever()
    return 0


if __name__=="__main__":raise SystemExit(main())
