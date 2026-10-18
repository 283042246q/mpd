"""Train jd_kin/jd_world x c/tau_r from a frozen JointDual snapshot."""
import argparse
from pathlib import Path
import yaml

from mpd.joint_training.trainer import train_jointdual


def load_config(path):
    path = Path(path)
    value = yaml.safe_load(path.read_text())
    parent = value.pop("base", None)
    if parent:
        base = load_config(path.parent / parent)
        for key, item in value.items():
            base[key] = dict(base.get(key, {}), **item) if isinstance(item, dict) else item
        return base
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--resume", type=Path)
    cache = p.add_mutually_exclusive_group()
    cache.add_argument("--tensor-cache", help="Shared cache directory, built once when absent")
    cache.add_argument("--no-tensor-cache", action="store_true", help="Read original HDF5 data directly")
    args = p.parse_args(argv)
    config = load_config(args.config)
    if args.tensor_cache is not None or args.no_tensor_cache:
        config.setdefault("training", {})["tensor_cache"] = args.tensor_cache if not args.no_tensor_cache else False
    for key in ("device", "max_steps"):
        if getattr(args, key) is not None:
            config.setdefault("training", {})[key] = getattr(args, key)
    print(train_jointdual(config, args.snapshot, args.output_dir, resume=args.resume))


if __name__ == "__main__":
    main()
