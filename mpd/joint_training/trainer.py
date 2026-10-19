"""Four-variant trainer with EMA, deterministic batch schedules and resume."""
import copy
from contextlib import ExitStack
import hashlib
import inspect
import json
import math
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

# On resume, YAML defaults must not silently reset an earlier training recipe.
RESUME_DEFAULTS = dict(learning_rate=1e-4, weight_decay=1e-4, gradient_clip_norm=.5,
                       batch_size=64, accumulate_steps=4, use_amp=False, ema_decay=.999)


def _resolve_resume_config(config, previous, overrides):
    unknown = set(overrides) - (set(RESUME_DEFAULTS) | {"world_dropout"})
    if unknown:
        raise ValueError("unsupported resume overrides: " + ", ".join(sorted(unknown)))
    config = copy.deepcopy(config)
    opts = config.setdefault("training", {})
    saved = previous["training_config"]
    changes, ignored = {}, {}
    for key, default in dict(RESUME_DEFAULTS, seed=1726484688).items():
        old = saved.get(key, default)
        if key in opts and opts[key] != old:
            ignored["training." + key] = opts[key]
        opts[key] = overrides.get(key, old)
        if opts[key] != old:
            changes["training." + key] = dict(old=old, new=opts[key])
    # Historical checkpoints used the training batch size and seed for validation.
    for key, default in (("validation_batch_size", saved.get("batch_size", 64)),
                         ("validation_seed", saved.get("seed", 1726484688))):
        old = saved.get(key, default)
        if key in opts and opts[key] != old:
            ignored["training." + key] = opts[key]
        opts[key] = old
    old = _model_options(previous["model_config"])["world_dropout"]
    model = config.setdefault("model", {})
    if model.get("world_dropout", old) != old:
        ignored["model.world_dropout"] = model["world_dropout"]
    model["world_dropout"] = overrides.get("world_dropout", old)
    if model["world_dropout"] != old:
        if config["variant"] != "jd_world":
            raise ValueError("world_dropout only applies to jd_world")
        changes["model.world_dropout"] = dict(old=old, new=model["world_dropout"])
    return config, changes, ignored


def _validate_recipe(opts):
    for key in ("batch_size", "accumulate_steps", "validation_batch_size", "max_steps"):
        if type(opts[key]) is not int or opts[key] < 1:
            raise ValueError(key + " must be a positive integer")
    for key in ("learning_rate", "weight_decay", "gradient_clip_norm", "ema_decay"):
        value = opts[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(key + " must be finite")
        if value < 0 or (key == "gradient_clip_norm" and value == 0) or (key == "ema_decay" and value >= 1):
            raise ValueError("invalid " + key)
    if type(opts["use_amp"]) is not bool:
        raise ValueError("use_amp must be boolean")


def _restore_scaler(scaler, saved):
    if not scaler.is_enabled():
        return "disabled"
    if saved:
        scaler.load_state_dict(saved)
        return "restored"
    return "initialized"  # FP32 -> AMP, or a checkpoint produced on CPU.


def _check_resume_output(output, source, step, changes):
    if not output.exists() or not any(output.iterdir()):
        return
    if output.resolve() != source.parent.resolve():
        raise FileExistsError("resume target is not empty; choose a new output-dir")
    if changes:
        raise ValueError("resume recipe changed; choose a new output-dir")
    latest = output / "last.pt"
    if latest.exists() and latest.resolve() != source.resolve():
        if torch.load(latest, map_location="cpu", weights_only=False)["step"] > step:
            raise ValueError("resuming an older checkpoint requires a new output-dir")


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


def _model_options(config):
    options = inspect.signature(JointDualDenoiser).bind(**config)
    options.apply_defaults()
    result = dict(options.arguments)
    result["channels"] = list(result["channels"])
    return result


def _warm_start_model(model, checkpoint, contract, *, mode="strict", weights="ema"):
    """Initialize model weights only; keep target buffers/coordinates in compatible mode."""
    if mode not in ("strict", "compatible") or weights not in ("ema", "model"):
        raise ValueError("invalid warm-start mode or weights")
    path = Path(checkpoint).resolve()
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
        handle.seek(0)
        previous = torch.load(handle, map_location="cpu", weights_only=False)
    if previous.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported warm-start checkpoint")
    for key in ("variant", "representation", "robot_hashes", "codec_contract", "world_contract"):
        if previous.get(key) != contract[key]:
            raise ValueError("warm-start contract mismatch: " + key)
    if previous["model_config"].get("variant") != previous["variant"]:
        raise ValueError("warm-start model variant disagrees with metadata")
    old_options, new_options = _model_options(previous["model_config"]), _model_options(contract["model_config"])
    if previous["diffusion_config"].get("num_diffusion_steps", 100) != model.num_diffusion_steps:
        raise ValueError("warm-start diffusion schedule mismatch")
    if mode == "strict" and old_options != new_options:
        raise ValueError("warm-start architecture mismatch; use compatible mode for partial parameter transfer")
    state = previous["ema_model_state_dict" if weights == "ema" else "model_state_dict"]
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("nonfinite warm-start parameters")
    parameters = dict(model.named_parameters())
    if mode == "strict":
        model.load_state_dict(state, strict=True)
        loaded = sorted(parameters)
        skipped, missing = [], []
    else:
        changed_coordinates = old_options["architecture_version"] != new_options["architecture_version"]
        selected = {}
        for key, target in parameters.items():
            # A new time PE changes the world MLP input semantics, even when some
            # hidden layers have the same dimensions. Leave that encoder fresh.
            if changed_coordinates and key.startswith("denoiser.world_encoder."):
                continue
            if key in state and state[key].shape == target.shape:
                selected[key] = state[key]
        if not selected:
            raise ValueError("warm-start has no compatible parameters")
        current = model.state_dict(); current.update(selected)
        model.load_state_dict(current, strict=True)
        loaded = sorted(selected)
        skipped = sorted(set(state) - set(loaded))
        missing = sorted(set(parameters) - set(loaded))
    report = dict(checkpoint=str(path), sha256=digest.hexdigest(), source_step=int(previous["step"]),
        mode=mode, weights=weights, loaded_parameter_keys=loaded, skipped_source_keys=skipped,
        fresh_parameter_keys=missing, loaded_parameter_count=sum(parameters[key].numel() for key in loaded),
        normalization_changed=previous["normalization"] != contract["normalization"],
        dataset_changed=previous["dataset_snapshot_hash"] != contract["dataset_snapshot_hash"],
        static_geometry_changed=previous["static_geometry_hashes"] != contract["static_geometry_hashes"])
    return report


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


@torch.no_grad()
def _validate_joint(model, dataset, *, batch_size, device, representation, seed, batch_limit=None):
    """Cover the full split by default; explicit budgets use a fixed uniform subset."""
    if batch_limit is not None and (type(batch_limit) is not int or batch_limit < 1):
        raise ValueError("validation_batches must be null (full split) or a positive integer")
    total = len(dataset)
    indices = None
    if batch_limit is not None and batch_limit * batch_size < total:
        indices = torch.randperm(total, generator=torch.Generator().manual_seed(seed + 115))[
            :batch_limit * batch_size].tolist()
    loader = DataLoader(dataset, batch_size=batch_size, sampler=indices, collate_fn=collate_joint,
                        generator=torch.Generator().manual_seed(seed + 114))
    generator = torch.Generator(device=device).manual_seed(seed + 917)
    totals, samples, batches = {}, 0, 0
    for batch in loader:
        loss, metrics = model.training_loss({k: v.to(device) for k, v in batch.items()}, generator=generator)
        if representation == "c":
            metrics = {k: metrics[k] for k in ("p", "z")}
        count = len(batch["p"]); samples += count; batches += 1
        for key, value in dict(loss=loss, **metrics).items():
            totals[key] = totals.get(key, 0.) + float(value)*count
    result = {"validation_" + key: value/samples for key, value in totals.items()}
    result.update(validation_samples=samples, validation_total_samples=total,
                  validation_batches=batches, validation_coverage=samples/total if total else 0.)
    return result


def train_jointdual(config, snapshot, output_dir, *, resume=None, warm_start=None,
                    warm_start_mode="strict", warm_start_weights="ema", resume_overrides=None):
    if resume is not None and warm_start is not None:
        raise ValueError("resume and warm-start are mutually exclusive")
    if resume_overrides and resume is None:
        raise ValueError("resume_overrides requires resume")
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
                                resources=resources, warm_start=warm_start, warm_start_mode=warm_start_mode,
                                warm_start_weights=warm_start_weights, resume_overrides=resume_overrides)


def _train_jointdual(config, snapshot, output_dir, data, validation, *, resume=None, cache=None, resources,
                     warm_start=None, warm_start_mode="strict", warm_start_weights="ema", resume_overrides=None):
    output = Path(output_dir)
    previous, resume_report = None, None
    resume_history = []
    if resume is not None:
        source = Path(resume).resolve()
        with source.open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024*1024), b""):
                digest.update(chunk)
            handle.seek(0)
            previous = torch.load(handle, map_location="cpu", weights_only=False)
        if previous.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("unsupported JointDual resume checkpoint")
        config, changes, ignored = _resolve_resume_config(config, previous, resume_overrides or {})
        _check_resume_output(output, source, previous["step"], changes)
        resume_report = dict(checkpoint=str(source), sha256=digest.hexdigest(),
                             source_step=int(previous["step"]), changes=changes,
                             requested_overrides=dict(resume_overrides or {}), ignored_yaml_recipe=ignored)
        resume_history = list(previous.get("resume_history", []))
    variant, representation = config["variant"], config["representation"]
    opts = dict(RESUME_DEFAULTS, max_steps=100000)
    opts.update(config.get("training", {}))
    opts.setdefault("validation_batch_size", opts["batch_size"])
    opts.setdefault("validation_seed", opts.get("seed", 1726484688))
    _validate_recipe(opts)
    config = dict(config, training=opts)
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
           (("log_every", 100), ("validate_every", 1000), ("checkpoint_every", 1000))) < 1:
        raise ValueError("logging/validation/checkpoint intervals must be positive")
    validation_batches = opts.get("validation_batches")
    if validation_batches is not None and (type(validation_batches) is not int or validation_batches < 1):
        raise ValueError("validation_batches must be null (full split) or a positive integer")
    contract = dict(variant=variant, representation=representation, model_config=model_config,
        diffusion_config=diffusion_config, normalization=data.normalization,
        dataset_snapshot_hash=data.snapshot["identity"], robot_hashes=data.snapshot["robot_hashes"],
        static_geometry_hashes=data.snapshot["static_geometry_hashes"],
        codec_contract=CODEC_CONTRACT, world_contract=WORLD_CONTRACT)
    warm_start_report = None
    microbatches_seen, samples_seen = 0, 0
    if previous is not None:
        old_model, new_model = _model_options(previous["model_config"]), _model_options(model_config)
        old_model.pop("world_dropout"); new_model.pop("world_dropout")
        if old_model != new_model or any(previous[k] != v for k, v in contract.items() if k != "model_config"):
            raise ValueError("resume model/data/representation contract mismatch")
        if maximum <= previous["step"]:
            raise ValueError("max_steps must exceed the resumed checkpoint step (it is a cumulative target)")
        model.load_state_dict(previous["model_state_dict"], strict=True)
        ema.load_state_dict(previous["ema_model_state_dict"], strict=True)
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        # load_state_dict restores the old LR/decay as well as moments. Override last.
        for group in optimizer.param_groups:
            group["lr"] = float(opts["learning_rate"])
            group["weight_decay"] = float(opts["weight_decay"])
        resume_report["scaler_action"] = _restore_scaler(scaler, previous["scaler_state_dict"])
        torch.set_rng_state(previous["torch_rng_state"])
        if device.type == "cuda" and previous["cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(previous["cuda_rng_state_all"])
        step = previous["step"]
        old_opts = previous["training_config"]
        microbatches_seen = int(previous.get("microbatches_seen", step*old_opts.get("accumulate_steps", 4)))
        samples_seen = int(previous.get("samples_seen", microbatches_seen*old_opts.get("batch_size", 64)))
        resume_report.update(effective_batch_size=dict(
            old=old_opts.get("batch_size", 64)*old_opts.get("accumulate_steps", 4), new=batch_size*accumulate),
            microbatches_seen=microbatches_seen, samples_seen=samples_seen,
            validation_batch_size=opts["validation_batch_size"], validation_seed=opts["validation_seed"],
            amp_enabled=amp)
        resume_report["operational_changes"] = {key: dict(old=old_opts.get(key), new=value)
            for key, value in opts.items() if key not in RESUME_DEFAULTS and key not in
            ("seed", "validation_batch_size", "validation_seed") and old_opts.get(key) != value}
        resume_history.append(resume_report)
        warm_start_report = previous.get("warm_start")
    elif (output / "last.pt").exists():
        raise FileExistsError("output already has a checkpoint; specify --resume")
    elif warm_start is not None:
        warm_start_report = _warm_start_model(model, warm_start, contract, mode=warm_start_mode,
                                               weights=warm_start_weights)
        ema.load_state_dict(model.state_dict(), strict=True)
        atomic_json(output / "warm_start.json", warm_start_report)
        print(json.dumps(dict(event="warm_start", mode=warm_start_mode, weights=warm_start_weights,
            source_step=warm_start_report["source_step"],
            loaded_parameters=warm_start_report["loaded_parameter_count"],
            fresh_parameter_keys=len(warm_start_report["fresh_parameter_keys"]),
            normalization_changed=warm_start_report["normalization_changed"],
            dataset_changed=warm_start_report["dataset_changed"], report=str(output / "warm_start.json"))), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    if resume_report is not None:
        atomic_json(output / "resume.json", resume_report)
        print(json.dumps(dict(event="resume", **resume_report)), flush=True)
    atomic_json(output / "config.json", dict(config, snapshot=str(Path(snapshot).resolve()),
                tensor_cache_path=str(cache) if cache is not None else None, warm_start=warm_start_report,
                resume_history=resume_history))
    writer = None
    if opts.get("tensorboard", True):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(output / "tensorboard"), purge_step=step+1 if resume else None)
        resources.callback(writer.close)
        writer.add_text("run/config", json.dumps(dict(variant=variant, representation=representation,
            model=model_config, training=opts, dataset_snapshot_hash=data.snapshot["identity"],
            tensor_cache_path=str(cache) if cache is not None else None, warm_start=warm_start_report,
            resume=resume_report), indent=2), step)
    workers = int(opts.get("num_workers", 0))
    pin_memory = bool(opts.get("pin_memory", True)) and device.type == "cuda"
    loader = DataLoader(data, batch_sampler=JointBatchSampler(data.entries, batch_size, seed,
        microbatches_seen, microbatches_seen+(maximum-step)*accumulate), collate_fn=collate_joint, num_workers=workers,
        multiprocessing_context="spawn" if workers else None, persistent_workers=bool(workers),
        timeout=float(opts.get("worker_timeout", 60)) if workers else 0,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed + 113))
    iterator = iter(loader)
    def save():
        payload = dict(contract, checkpoint_version=CHECKPOINT_VERSION, step=step, training_config=opts,
            warm_start=warm_start_report, resume_history=resume_history,
            microbatches_seen=microbatches_seen, samples_seen=samples_seen,
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
            microbatches_seen += accumulate
            samples_seen += batch_size*accumulate
            grad_norm = float(norm)
            seconds = time.perf_counter() - started
            record = dict(step=step, loss=loss_total, **metrics_total, grad_norm=grad_norm,
                          microbatches_seen=microbatches_seen, samples_seen=samples_seen,
                          learning_rate=optimizer.param_groups[0]["lr"], step_seconds=seconds,
                          data_wait_seconds=data_wait, samples_per_second=batch_size*accumulate/seconds)
            if len(validation) and step % int(opts.get("validate_every", 1000)) == 0:
                record.update(_validate_joint(ema, validation, batch_size=opts["validation_batch_size"], device=device,
                    representation=representation, seed=opts["validation_seed"], batch_limit=validation_batches))
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
