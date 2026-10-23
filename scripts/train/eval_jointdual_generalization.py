"""Compare fixed-noise EMA train/validation losses across JointDual checkpoints.

See docs/jointdual_generalization_probe.md for the protocol and runnable examples.
Only reads the dataset/cache/checkpoints; writes diagnostics to a new output directory.
"""
import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from mpd.datasets.spacetime_schema import sha256_file
from mpd.datasets.warehouse_joint_cache import CachedWarehouseJointDataset, default_cache_path
from mpd.datasets.warehouse_joint_dataset import WarehouseJointDataset, atomic_json, collate_joint
from mpd.joint_training.trainer import load_joint_checkpoint


def select_indices(entries, count, seed, sampling):
    """Both splits use the same target distribution, without replacement."""
    rng = np.random.default_rng(seed)
    if sampling == "row":
        if count > len(entries):
            raise ValueError("--samples exceeds the number of rows in a split")
        return rng.choice(len(entries), count, replace=False).tolist()
    groups = {}
    for i, entry in enumerate(entries):
        groups.setdefault(entry["family"], []).append(i)
    families = sorted(groups)
    if count > len(families):
        raise ValueError("--samples exceeds the number of families in a split")
    chosen = rng.choice(len(families), count, replace=False)
    return [int(rng.choice(groups[families[i]])) for i in chosen]


def make_noise_plan(count, steps, bins, repeats, seed):
    """CPU tensors are generated before batching, shared by splits and checkpoints."""
    if not 1 <= bins <= steps:
        raise ValueError("--timestep-bins must be between 1 and num_diffusion_steps")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    plan = []
    digest = hashlib.sha256()
    for bucket in np.array_split(np.arange(steps), bins):
        lo, hi = int(bucket[0]), int(bucket[-1]) + 1
        draws = []
        for _ in range(repeats):
            k = torch.randint(lo, hi, (count,), generator=generator)
            ep = torch.randn(count, 16, 7, generator=generator)
            ez = torch.randn(count, 6, generator=generator)
            draws.append((k, ep, ez))
            for value in draws[-1]:
                digest.update(value.numpy().tobytes())
        plan.append(dict(lo=lo, hi=hi, draws=draws))
    return plan, digest.hexdigest()


@torch.no_grad()
def evaluate(model, dataset, indices, noise_plan, batch_size, device, representation):
    model.eval()
    columns = 4 if representation == "tau_r" else 2
    losses = np.zeros((len(indices), len(noise_plan), columns), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        batch = collate_joint([dataset[i] for i in indices[start:end]])
        batch = {key: value.to(device) for key, value in batch.items()}
        p, z = batch["p"], batch["z"]
        encoded = model.denoiser.encode_world(batch.get("world_features"),
                                             batch.get("world_mask"), batch.get("world_state"))
        for bucket, spec in enumerate(noise_plan):
            for k, ep, ez in spec["draws"]:
                k, ep, ez = (value[start:end].to(device) for value in (k, ep, ez))
                a = model.alphas_cumprod[k]
                pk = a[:, None, None].sqrt()*p + (1-a[:, None, None]).sqrt()*ep
                zk = a[:, None].sqrt()*z + (1-a[:, None]).sqrt()*ez
                pp, pz = model.denoiser(pk, zk, k, batch["task"], encoded_world=encoded)
                spatial, timing = (pp-ep).square(), (pz-ez).square()
                values = [spatial.mean((1, 2)), timing.mean(1)]
                if representation == "tau_r":
                    values += [timing[:, 0], timing[:, 1:].mean(1)]
                values = torch.stack(values, -1)
                if not torch.isfinite(values).all():
                    raise ValueError("nonfinite evaluation loss")
                losses[start:end, bucket] += values.cpu().double().numpy()/len(spec["draws"])
    return losses


def mean_stats(values, families):
    """Approximate cluster SE; repeated worlds of a family are not independent."""
    values = np.asarray(values, dtype=np.float64)
    _, groups = np.unique(families, return_inverse=True)
    clusters = int(groups.max()) + 1
    mean = float(values.mean())
    if clusters < 2:
        return dict(mean=mean, se=None, ci95=None)
    residuals = np.bincount(groups, weights=values-mean)
    se = float(np.sqrt(clusters/(clusters-1)*np.square(residuals).sum())/len(values))
    return dict(mean=mean, se=se, ci95=[mean-1.96*se, mean+1.96*se])


def summarize(values, reference, families, plan, metrics):
    # Unequal-width buckets must be weighted by width, not averaged equally.
    widths = np.array([spec["hi"]-spec["lo"] for spec in plan], dtype=np.float64)
    buckets = [("all", None)] + [(f"{s['lo']}-{s['hi']-1}", i) for i, s in enumerate(plan)]
    result = {}
    for name, bucket in buckets:
        selected, baseline = {}, {}
        for split in ("train", "validation"):
            selected[split] = (np.average(values[split], axis=1, weights=widths)
                               if bucket is None else values[split][:, bucket])
            baseline[split] = (np.average(reference[split], axis=1, weights=widths)
                               if bucket is None else reference[split][:, bucket])
        row = {}
        for j, metric in enumerate(metrics):
            stats = {}
            for split in ("train", "validation"):
                stats[split] = mean_stats(selected[split][:, j], families[split])
                stats["delta_"+split] = mean_stats(
                    selected[split][:, j]-baseline[split][:, j], families[split])
            stats["gap"] = stats["validation"]["mean"]-stats["train"]["mean"]
            row[metric] = stats
        result[name] = row
    return result


def write_reports(output, report):
    atomic_json(output / "report.json", report)
    fields = ["checkpoint_index", "checkpoint", "step", "bucket", "metric", "train", "validation", "gap"]
    for split in ("train", "validation"):
        fields += ["delta_"+split, "delta_"+split+"_ci95_low", "delta_"+split+"_ci95_high"]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for checkpoint in report["checkpoints"]:
            for bucket, metrics in checkpoint["results"].items():
                for metric, stats in metrics.items():
                    row = dict(checkpoint_index=checkpoint["index"], checkpoint=checkpoint["path"],
                               step=checkpoint["step"], bucket=bucket, metric=metric,
                               train=stats["train"]["mean"], validation=stats["validation"]["mean"], gap=stats["gap"])
                    for split in ("train", "validation"):
                        delta = stats["delta_"+split]
                        row["delta_"+split] = delta["mean"]
                        ci = delta["ci95"] or [None, None]
                        row["delta_"+split+"_ci95_low"], row["delta_"+split+"_ci95_high"] = ci
                    writer.writerow(row)


def check_contract(payload, dataset, expected=None):
    for key in ("dataset_snapshot_hash", "normalization", "robot_hashes", "static_geometry_hashes"):
        value = dataset.snapshot["identity"] if key == "dataset_snapshot_hash" else dataset.snapshot[key]
        if payload[key] != value:
            raise ValueError("checkpoint/snapshot mismatch: " + key)
    contract = {key: payload[key] for key in ("variant", "representation", "diffusion_config")}
    if expected is not None and contract != expected:
        raise ValueError("all checkpoints must use the same variant, representation and diffusion config")
    return contract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True,
                        help="In comparison order; all deltas are relative to the first")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty directory")
    parser.add_argument("--samples", type=int, default=4096, help="Number of samples PER split")
    parser.add_argument("--sampling", choices=("family", "row"), default="family")
    parser.add_argument("--timestep-bins", type=int, default=5)
    parser.add_argument("--noise-repeats", type=int, default=2, help="Independent draws per sample per bucket")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--tensor-cache", type=Path, help="Existing cache; default is snapshot's cache")
    cache_group.add_argument("--no-tensor-cache", action="store_true", help="Read HDF5 directly")
    args = parser.parse_args(argv)
    if min(args.samples, args.batch_size, args.noise_repeats, args.timestep_bins, args.torch_threads) < 1 or args.seed < 0:
        parser.error("counts must be positive and seed must be nonnegative")
    if not args.snapshot.is_file() or any(not path.is_file() for path in args.checkpoints):
        parser.error("snapshot or checkpoint does not exist")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("output-dir must be new or empty")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    torch.set_num_threads(args.torch_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model, payload = load_joint_checkpoint(args.checkpoints[0], device, use_ema=True)
    representation = payload["representation"]
    metrics = ["p", "z"] + (["tau", "r"] if representation == "tau_r" else [])
    noise, noise_hash = make_noise_plan(args.samples, model.num_diffusion_steps,
                                        args.timestep_bins, args.noise_repeats, args.seed+917)
    schedule = model.alphas_cumprod.detach().cpu().clone()
    with ExitStack() as resources:
        datasets, indices, families = {}, {}, {}
        cache = None if args.no_tensor_cache else (args.tensor_cache or default_cache_path(args.snapshot))
        if cache is not None and not cache.is_dir():
            raise FileNotFoundError("cache missing; build it explicitly or use --no-tensor-cache: " + str(cache))
        for split, offset in (("train", 0), ("validation", 1)):
            kwargs = dict(split=split, representation=representation, include_world=payload["variant"] == "jd_world")
            dataset = (WarehouseJointDataset(args.snapshot, **kwargs) if cache is None else
                       CachedWarehouseJointDataset(args.snapshot, cache, **kwargs))
            resources.callback(dataset.close)
            datasets[split] = dataset
            indices[split] = select_indices(dataset.entries, args.samples, args.seed+offset, args.sampling)
            families[split] = [dataset.entries[i]["family"] for i in indices[split]]
        if set(e["family"] for e in datasets["train"].entries) & set(e["family"] for e in datasets["validation"].entries):
            raise ValueError("train/validation family leakage")
        contract = check_contract(payload, datasets["train"])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plan = dict(version=1, snapshot=str(args.snapshot.resolve()), snapshot_identity=datasets["train"].snapshot["identity"],
                    sampling=args.sampling, samples_per_split=args.samples, seed=args.seed, noise_seed=args.seed+917,
                    noise_repeats=args.noise_repeats, noise_sha256=noise_hash,
                    buckets=[dict(lo=s["lo"], hi_exclusive=s["hi"]) for s in noise],
                    selection={split:[dict(split_index=i, **{key:dataset.entries[i][key] for key in ("file", "group", "family")})
                                      for i in indices[split]] for split, dataset in datasets.items()})
        atomic_json(args.output_dir / "plan.json", plan)
        report = dict(version=1, complete=False, protocol=dict(weights="EMA", mode="eval", amp=False,
            world_condition="original KNOWN/EMPTY, no condition dropout", sampling=args.sampling,
            samples_per_split=args.samples, repeats_per_bucket=args.noise_repeats, metrics=metrics,
            overall_weighting="bucket width / diffusion steps", delta_reference="first checkpoint, paired by sample/noise",
            confidence_interval="approximate family-cluster normal 95%; conditional on fixed noise, no multiplicity correction",
            device=str(device), torch_version=str(torch.__version__), batch_size=args.batch_size,
            tensor_cache=str(cache.resolve()) if cache is not None else None,
            snapshot_identity=plan["snapshot_identity"], noise_sha256=noise_hash), checkpoints=[])
        reference = None
        for index, path in enumerate(args.checkpoints):
            started = time.perf_counter()
            if index:
                model, payload = load_joint_checkpoint(path, device, use_ema=True)
            check_contract(payload, datasets["train"], contract)
            if not torch.equal(model.alphas_cumprod.detach().cpu(), schedule):
                raise ValueError("diffusion alpha schedule changed")
            values = {}
            for split, dataset in datasets.items():
                values[split] = evaluate(model, dataset, indices[split], noise, args.batch_size, device, representation)
                print(json.dumps(dict(event="split_done", checkpoint=str(path), split=split)), flush=True)
            if reference is None:
                reference = values
            results = summarize(values, reference, families, noise, metrics)
            np.savez_compressed(args.output_dir / f"checkpoint-{index:03d}.npz", **values)
            report["checkpoints"].append(dict(index=index, path=str(path.resolve()), sha256=sha256_file(path),
                step=int(payload["step"]), training_config=payload["training_config"], model_config=payload["model_config"],
                results=results, elapsed_seconds=time.perf_counter()-started))
            write_reports(args.output_dir, report)
            brief = {metric:{key:stats[key]["mean"] for key in ("train", "validation", "delta_train", "delta_validation")}
                     for metric, stats in results["all"].items()}
            print(json.dumps(dict(event="checkpoint_done", index=index, step=payload["step"], all=brief)), flush=True)
            del model, payload
        report["complete"] = True
        write_reports(args.output_dir, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
