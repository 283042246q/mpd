"""Four-variant trainer with EMA, deterministic batch schedules and resume."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from mpd.datasets.warehouse_joint_dataset import WarehouseJointDataset, collate_joint, atomic_json, WORLD_CONTRACT
from mpd.datasets.warehouse_joint_cache import build_tensor_cache, CachedWarehouseJointDataset
from mpd.models.diffusion_models.jointdual import JointDualDenoiser, JointDiffusion
from mpd.parametric_trajectory.joint_timing_codec import CODEC_CONTRACT
from mpd.timing_training.trainer import _atomic_torch_save, _update_ema, _set_seed

CHECKPOINT_VERSION = "jointdual_checkpoint_v1"


def load_joint_checkpoint(path, device="cpu", *, use_ema=True, variant=None, representation=None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported JointDual checkpoint")
    if payload["variant"] not in ("jd_kin", "jd_world") or payload["representation"] not in ("c", "tau_r"):
        raise ValueError("invalid JointDual checkpoint variant/representation")
    if payload["model_config"]["variant"] != payload["variant"]:
        raise ValueError("checkpoint model variant disagrees with metadata")
    if variant is not None and variant != payload["variant"]:
        raise ValueError("checkpoint variant mismatch")
    if representation is not None and representation != payload["representation"]:
        raise ValueError("checkpoint representation mismatch")
    if payload["codec_contract"] != CODEC_CONTRACT or payload["world_contract"] != WORLD_CONTRACT:
        raise ValueError("checkpoint codec/world contract mismatch")
    model = JointDiffusion(JointDualDenoiser(**payload["model_config"]), **payload["diffusion_config"])
    state = payload["ema_model_state_dict" if use_ema else "model_state_dict"]
    if not all(torch.isfinite(x).all() for x in state.values()):
        raise ValueError("nonfinite checkpoint parameters")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False), payload


class JointBatchSampler:
    """Family -> world; each microbatch is a pure function of seed and index."""
    def __init__(self, entries, batch_size, seed, start, stop):
        groups = {}
        for i, entry in enumerate(entries):
            groups.setdefault(entry["family"], []).append(i)
        self.groups = list(groups.values())
        self.batch_size, self.seed, self.start, self.stop = batch_size, seed, start, stop

    def __iter__(self):
        for step in range(self.start, self.stop):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
            yield [int(rng.choice(self.groups[int(rng.integers(len(self.groups)))]))
                   for _ in range(self.batch_size)]

    def __len__(self):
        return self.stop - self.start


def train_jointdual(config, snapshot, output_dir, *, resume=None):
    opts = config.get("training", {})
    cache = opts.get("tensor_cache", "auto")
    with ExitStack() as resources:
        dataset_class, dataset_args = WarehouseJointDataset, {}
        if cache not in (False, None, "off"):
            path = build_tensor_cache(snapshot, None if cache in (True, "auto") else cache)
            dataset_class, dataset_args = CachedWarehouseJointDataset, dict(cache=path)
        else:
            path = None
        datasets = []
        for split in ("train", "validation"):
            dataset = dataset_class(snapshot, split=split, representation=config["representation"],
                                    include_world=config["variant"] == "jd_world", **dataset_args)
            resources.callback(dataset.close)
            datasets.append(dataset)
        return _train_jointdual(config, snapshot, output_dir, *datasets, resume=resume, cache=path,
                                resources=resources)


def _train_jointdual(config, snapshot, output_dir, data, validation, *, resume=None, cache=None, resources):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    variant, representation = config["variant"], config["representation"]
    opts = dict(config.get("training", {}))
    seed = int(opts.get("seed", 1726484688)); _set_seed(seed)
    torch.set_num_threads(int(opts.get("torch_threads", 4)))
    device_name = opts.get("device", "auto")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    if not len(data):
        raise ValueError("empty training split")
    model_config = dict(config.get("model", {}), variant=variant)
    diffusion_config = dict(config.get("diffusion", {}))
    model = JointDiffusion(JointDualDenoiser(**model_config), **diffusion_config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opts.get("learning_rate", 1e-4)),
                                 weight_decay=float(opts.get("weight_decay", 1e-4)))
    amp = bool(opts.get("use_amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    step, maximum = 0, int(opts.get("max_steps", 100000))
    batch_size, accumulate = int(opts.get("batch_size", 64)), int(opts.get("accumulate_steps", 4))
    if min(maximum, batch_size, accumulate) < 1:
        raise ValueError("training counts must be positive")
    if min(int(opts.get(k, default)) for k, default in
           (("log_every", 100), ("validate_every", 1000), ("validation_batches", 20), ("checkpoint_every", 1000))) < 1:
        raise ValueError("logging/validation/checkpoint intervals must be positive")
    contract = dict(variant=variant, representation=representation, model_config=model_config,
        diffusion_config=diffusion_config, normalization=data.normalization,
        dataset_snapshot_hash=data.snapshot["identity"], robot_hashes=data.snapshot["robot_hashes"],
        static_geometry_hashes=data.snapshot["static_geometry_hashes"],
        codec_contract=CODEC_CONTRACT, world_contract=WORLD_CONTRACT)
    if resume:
        previous = torch.load(resume, map_location="cpu", weights_only=False)
        if previous.get("checkpoint_version") != CHECKPOINT_VERSION or any(previous[k] != v for k, v in contract.items()):
            raise ValueError("resume model/data/representation contract mismatch")
        for key in ("seed", "batch_size", "accumulate_steps", "learning_rate", "weight_decay", "use_amp", "ema_decay"):
            if previous["training_config"].get(key) != opts.get(key):
                raise ValueError("resume training recipe mismatch: " + key)
        model.load_state_dict(previous["model_state_dict"], strict=True)
        ema.load_state_dict(previous["ema_model_state_dict"], strict=True)
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        scaler.load_state_dict(previous["scaler_state_dict"])
        torch.set_rng_state(previous["torch_rng_state"])
        if device.type == "cuda" and previous["cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(previous["cuda_rng_state_all"])
        step = previous["step"]
    elif (output / "last.pt").exists():
        raise FileExistsError("output already has a checkpoint; specify --resume")
    atomic_json(output / "config.json", dict(config, snapshot=str(Path(snapshot).resolve()),
                tensor_cache_path=str(cache) if cache is not None else None))
    writer = None
    if opts.get("tensorboard", True):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(output / "tensorboard"), purge_step=step+1 if resume else None)
        resources.callback(writer.close)
        writer.add_text("run/config", json.dumps(dict(variant=variant, representation=representation,
            model=model_config, training=opts, dataset_snapshot_hash=data.snapshot["identity"],
            tensor_cache_path=str(cache) if cache is not None else None), indent=2), step)
    workers = int(opts.get("num_workers", 0))
    pin_memory = bool(opts.get("pin_memory", True)) and device.type == "cuda"
    loader = DataLoader(data, batch_sampler=JointBatchSampler(data.entries, batch_size, seed,
        step*accumulate, maximum*accumulate), collate_fn=collate_joint, num_workers=workers,
        multiprocessing_context="spawn" if workers else None, persistent_workers=bool(workers),
        timeout=float(opts.get("worker_timeout", 60)) if workers else 0,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed + 113))
    iterator = iter(loader)
    def save():
        payload = dict(contract, checkpoint_version=CHECKPOINT_VERSION, step=step, training_config=opts,
            model_state_dict=model.state_dict(), ema_model_state_dict=ema.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), scaler_state_dict=scaler.state_dict(),
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_state_all=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
        _atomic_torch_save(payload, output / "last.pt")
        _atomic_torch_save(payload, output / f"step-{step:08d}.pt")
        if writer is not None:
            writer.flush()
    try:
        while step < maximum:
            started = time.perf_counter(); data_wait = 0.
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss_total = 0.; metrics_total = {}
            for _ in range(accumulate):
                waiting = time.perf_counter(); batch = next(iterator)
                data_wait += time.perf_counter() - waiting
                batch = {k: v.to(device, non_blocking=pin_memory) for k, v in batch.items()}
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss, metrics = model.training_loss(batch)
                if representation == "c":
                    metrics = {k: metrics[k] for k in ("p", "z")}
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                scaler.scale(loss / accumulate).backward(); loss_total += float(loss.detach()) / accumulate
                for key, value in metrics.items():
                    metrics_total[key] = metrics_total.get(key, 0.) + float(value) / accumulate
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(opts.get("gradient_clip_norm", .5)))
            if not torch.isfinite(norm):
                raise ValueError("nonfinite training gradient")
            scaler.step(optimizer); scaler.update()
            _update_ema(ema, model, float(opts.get("ema_decay", .999)))
            step += 1
            grad_norm = float(norm)
            seconds = time.perf_counter() - started
            record = dict(step=step, loss=loss_total, **metrics_total, grad_norm=grad_norm,
                          learning_rate=optimizer.param_groups[0]["lr"], step_seconds=seconds,
                          data_wait_seconds=data_wait, samples_per_second=batch_size*accumulate/seconds)
            if len(validation) and step % int(opts.get("validate_every", 1000)) == 0:
                gen = torch.Generator(device=device).manual_seed(seed + 917)
                val_loader = DataLoader(validation, batch_size=batch_size, collate_fn=collate_joint,
                    generator=torch.Generator().manual_seed(seed + 114))
                totals, samples = {}, 0
                with torch.no_grad():
                    for i, b in enumerate(val_loader):
                        if i >= int(opts.get("validation_batches", 20)):
                            break
                        loss, metrics = ema.training_loss({k: v.to(device) for k, v in b.items()}, generator=gen)
                        if representation == "c":
                            metrics = {k: metrics[k] for k in ("p", "z")}
                        count = len(b["p"]); samples += count
                        for key, value in dict(loss=loss, **metrics).items():
                            totals[key] = totals.get(key, 0.) + float(value)*count
                record.update({"validation_" + key: value/samples for key, value in totals.items()})
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            log_step = step == maximum or step % int(opts.get("log_every", 100)) == 0
            if writer is not None:
                if log_step:
                    for key in ("loss", "p", "z", "tau", "r", "grad_norm", "learning_rate"):
                        if key in record:
                            writer.add_scalar("train/" + key, record[key], step)
                    for key in ("step_seconds", "data_wait_seconds", "samples_per_second"):
                        writer.add_scalar("performance/" + key, record[key], step)
                for key, value in record.items():
                    if key.startswith("validation_"):
                        writer.add_scalar("validation/" + key[len("validation_"):], value, step)
            if step == maximum or step % int(opts.get("checkpoint_every", 1000)) == 0:
                save()
            if log_step:
                print(json.dumps(record), flush=True)
    finally:
        data.close(); validation.close()
    return output / "last.pt"
