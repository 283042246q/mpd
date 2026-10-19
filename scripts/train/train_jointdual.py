"""Train jd_kin/jd_world x c/tau_r from a frozen JointDual snapshot."""
import argparse
from pathlib import Path
import yaml

from mpd.joint_training.trainer import train_jointdual, RESUME_DEFAULTS


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
    for flag in ("learning-rate", "weight-decay", "gradient-clip-norm", "ema-decay"):
        p.add_argument("--" + flag, type=float, help="Explicit training recipe override (also on resume)")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--accumulate-steps", type=int)
    amp = p.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="use_amp", action="store_true", default=None)
    amp.add_argument("--no-amp", dest="use_amp", action="store_false")
    p.add_argument("--world-dropout", type=float, help="World condition dropout probability (jd_world only)")
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path, help="Restore training state; recipe changes require explicit flags and a new output-dir")
    initialization.add_argument("--warm-start", type=Path, help="Initialize a new run from JointDual weights")
    p.add_argument("--warm-start-mode", choices=("strict", "compatible"), default="strict")
    p.add_argument("--warm-start-weights", choices=("ema", "model"), default="ema")
    cache = p.add_mutually_exclusive_group()
    cache.add_argument("--tensor-cache", help="Shared cache directory, built once when absent")
    cache.add_argument("--no-tensor-cache", action="store_true", help="Read original HDF5 data directly")
    p.add_argument("--no-tensorboard", action="store_true")
    args = p.parse_args(argv)
    config = load_config(args.config)
    if args.tensor_cache is not None or args.no_tensor_cache:
        config.setdefault("training", {})["tensor_cache"] = args.tensor_cache if not args.no_tensor_cache else False
    if args.no_tensorboard:
        config.setdefault("training", {})["tensorboard"] = False
    for key in ("device", "max_steps"):
        if getattr(args, key) is not None:
            config.setdefault("training", {})[key] = getattr(args, key)
    overrides = {key: getattr(args, key) for key in (*RESUME_DEFAULTS, "world_dropout")
                 if getattr(args, key) is not None}
    if "world_dropout" in overrides and config["variant"] != "jd_world":
        p.error("--world-dropout only applies to jd_world")
    if args.resume is None:
        for key, value in overrides.items():
            config.setdefault("model" if key == "world_dropout" else "training", {})[key] = value
    print(train_jointdual(config, args.snapshot, args.output_dir, resume=args.resume, warm_start=args.warm_start,
                          warm_start_mode=args.warm_start_mode, warm_start_weights=args.warm_start_weights,
                          resume_overrides=overrides if args.resume is not None else None))


if __name__ == "__main__":
    main()
