"""Standalone training loop for path-conditioned TimingDiffusion."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from mpd.datasets.spacetime_timing_dataset import (
    SpaceTimeTimingDataset,
    TimingNormalization,
    write_normalization_json,
)
from mpd.timing_training.model import TimingDenoiser, TimingDiffusion


CHECKPOINT_VERSION = "timing_diffusion_checkpoint_v1"


@dataclass(frozen=True)
class TrainingResult:
    output_dir: Path
    final_step: int
    train_loss: float
    validation_loss: Optional[float]
    elapsed_seconds: float


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_from_config(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".saving", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(value, temporary_path)
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        ema_parameters = dict(ema_model.named_parameters())
        for name, parameter in model.named_parameters():
            ema_parameters[name].mul_(decay).add_(parameter, alpha=1.0 - decay)
        ema_buffers = dict(ema_model.named_buffers())
        for name, buffer in model.named_buffers():
            ema_buffers[name].copy_(buffer)


def _loader(
    dataset: SpaceTimeTimingDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=shuffle and len(dataset) >= batch_size,
        generator=generator,
    )


def _validate(
    model: TimingDiffusion,
    dataloader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= max_batches:
                break
            path = batch["path"].to(device=device, non_blocking=True)
            timing = batch["timing"].to(device=device, non_blocking=True)
            loss, _ = model.training_loss(path, timing)
            losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def _checkpoint_payload(
    *,
    step: int,
    model: TimingDiffusion,
    ema_model: TimingDiffusion,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    normalization: TimingNormalization,
    environment: str,
    representation: str,
    dataset_identity: Mapping[str, object],
    model_config: Mapping[str, object],
    diffusion_config: Mapping[str, object],
    training_config: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "step": int(step),
        "environment": environment,
        "representation": representation,
        "model_config": dict(model_config),
        "diffusion_config": dict(diffusion_config),
        "training_config": dict(training_config),
        "normalization": normalization.to_dict(),
        "dataset_identity": dict(dataset_identity),
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


def train_timing_diffusion(
    *,
    environment: str,
    representation: str,
    dataset_roots: Sequence[Path],
    output_dir: Path,
    data_config: Mapping[str, object],
    model_config: Mapping[str, object],
    diffusion_config: Mapping[str, object],
    training_config: Mapping[str, object],
    resume_checkpoint: Optional[Path] = None,
) -> TrainingResult:
    """Train one timing representation without invoking the original MPD trainer."""

    output_dir = Path(output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    if output_dir.exists() and any(output_dir.iterdir()) and resume_checkpoint is None:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose a new directory or resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    seed = int(training_config.get("seed", 1726484688))
    _set_seed(seed)
    device = _device_from_config(str(training_config.get("device", "auto")))
    use_amp = bool(training_config.get("use_amp", True)) and device.type == "cuda"
    allow_fallback = bool(data_config.get("allow_hash_split_fallback", False))
    split_seed = int(data_config.get("split_seed", 1726484688))
    train_dataset = SpaceTimeTimingDataset(
        dataset_roots,
        split="train",
        representation=representation,
        split_seed=split_seed,
        train_fraction=float(data_config.get("train_fraction", 0.9)),
        validation_fraction=float(data_config.get("validation_fraction", 0.05)),
        allow_hash_split_fallback=allow_fallback,
        max_samples=data_config.get("max_train_samples"),
    )
    resume_payload = None
    if resume_checkpoint is not None:
        resume_payload = torch.load(
            Path(resume_checkpoint), map_location="cpu", weights_only=False
        )
        if resume_payload.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("unsupported timing checkpoint")
        if resume_payload.get("environment") != environment:
            raise ValueError("resume checkpoint environment mismatch")
        if resume_payload.get("representation") != representation:
            raise ValueError("resume checkpoint representation mismatch")
        normalization = TimingNormalization.from_dict(resume_payload["normalization"])
    else:
        normalization = train_dataset.compute_normalization(
            chunk_rows=int(data_config.get("statistics_chunk_rows", 1024))
        )
    train_dataset.set_normalization(normalization)
    val_dataset = SpaceTimeTimingDataset(
        dataset_roots,
        split="val",
        representation=representation,
        normalization=normalization,
        split_seed=split_seed,
        train_fraction=float(data_config.get("train_fraction", 0.9)),
        validation_fraction=float(data_config.get("validation_fraction", 0.05)),
        allow_hash_split_fallback=allow_fallback,
        max_samples=data_config.get("max_val_samples"),
    )
    batch_size = int(data_config.get("batch_size", 128))
    num_workers = int(data_config.get("num_workers", 4))
    train_loader = _loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    val_loader = _loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed + 1,
        pin_memory=device.type == "cuda",
    )

    resolved_model_config = dict(model_config)
    resolved_model_config.update(
        dof=train_dataset.dof,
        num_spatial_control_points=train_dataset.num_spatial_control_points,
        spatial_degree=int(train_dataset.contract["spatial_degree"]),
    )
    denoiser = TimingDenoiser(**resolved_model_config)
    resolved_diffusion_config = dict(diffusion_config)
    model = TimingDiffusion(denoiser, **resolved_diffusion_config).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 3e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_step = 0
    if resume_payload is not None:
        if resume_payload["model_config"] != resolved_model_config:
            raise ValueError("resume checkpoint model configuration mismatch")
        if resume_payload["diffusion_config"] != resolved_diffusion_config:
            raise ValueError("resume checkpoint diffusion configuration mismatch")
        if resume_payload["dataset_identity"] != train_dataset.identity:
            raise ValueError("resume checkpoint dataset identity mismatch")
        model.load_state_dict(resume_payload["model_state_dict"])
        ema_model.load_state_dict(resume_payload["ema_model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value):
                    state[name] = value.to(device)
        scaler.load_state_dict(resume_payload["scaler_state_dict"])
        torch.set_rng_state(resume_payload["torch_rng_state"])
        if torch.cuda.is_available() and resume_payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state_all"])
        start_step = int(resume_payload["step"])

    resolved = {
        "environment": environment,
        "representation": representation,
        "dataset_roots": [str(Path(root).resolve()) for root in dataset_roots],
        "data": dict(data_config),
        "model": resolved_model_config,
        "diffusion": resolved_diffusion_config,
        "training": dict(training_config),
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    write_normalization_json(output_dir / "normalization.json", normalization)
    (output_dir / "dataset_identity.json").write_text(
        json.dumps(train_dataset.identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    max_steps = int(training_config.get("max_steps", 500000))
    log_every = int(training_config.get("log_every", 100))
    validate_every = int(training_config.get("validate_every", 1000))
    checkpoint_every = int(training_config.get("checkpoint_every", 10000))
    validation_batches = int(training_config.get("validation_batches", 20))
    gradient_clip_norm = float(training_config.get("gradient_clip_norm", 1.0))
    ema_decay = float(training_config.get("ema_decay", 0.995))
    if min(log_every, validate_every, checkpoint_every, validation_batches) <= 0:
        raise ValueError("logging, validation, and checkpoint intervals must be positive")
    if gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive")
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if max_steps <= start_step:
        raise ValueError(f"max_steps={max_steps} must exceed resume step={start_step}")
    metrics_path = output_dir / "metrics.jsonl"
    started = time.monotonic()
    step = start_step
    last_train_loss = float("nan")
    last_validation_loss: Optional[float] = None
    model.train()
    train_iterator = iter(train_loader)
    while step < max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        path = batch["path"].to(device=device, non_blocking=True)
        timing = batch["timing"].to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with amp_context:
            loss, loss_metrics = model.training_loss(path, timing)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip_norm
        )
        scaler.step(optimizer)
        scaler.update()
        _update_ema(ema_model, model, ema_decay)
        step += 1
        last_train_loss = float(loss.item())

        should_log = step == 1 or step % log_every == 0 or step == max_steps
        if should_log:
            record = {
                "step": step,
                "train_loss": last_train_loss,
                "gradient_norm": float(gradient_norm),
                "mean_timestep": float(loss_metrics["mean_timestep"]),
                "elapsed_seconds": time.monotonic() - started,
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)

        if step % validate_every == 0 or step == max_steps:
            last_validation_loss = _validate(
                ema_model,
                val_loader,
                device=device,
                max_batches=validation_batches,
            )
            record = {
                "step": step,
                "validation_loss": last_validation_loss,
                "validation_batches": validation_batches,
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)

        if step % checkpoint_every == 0 or step == max_steps:
            payload = _checkpoint_payload(
                step=step,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scaler=scaler,
                normalization=normalization,
                environment=environment,
                representation=representation,
                dataset_identity=train_dataset.identity,
                model_config=resolved_model_config,
                diffusion_config=resolved_diffusion_config,
                training_config=training_config,
            )
            _atomic_torch_save(payload, checkpoint_dir / "latest.pt")
            _atomic_torch_save(payload, checkpoint_dir / f"step-{step:08d}.pt")

    elapsed = time.monotonic() - started
    train_dataset.close()
    val_dataset.close()
    return TrainingResult(
        output_dir=output_dir,
        final_step=step,
        train_loss=last_train_loss,
        validation_loss=last_validation_loss,
        elapsed_seconds=elapsed,
    )
