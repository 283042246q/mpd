import copy
import json

import h5py
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from mpd.datasets.warehouse_joint_dataset import build_snapshot
from mpd.joint_training.trainer import train_jointdual, _warm_start_model, _validate_joint
from mpd.models.diffusion_models.jointdual import JointDiffusion, JointDualDenoiser
from scripts.spacetime_data import refit_warehouse_eegoal as refit
from test_jointdual_training import SMALL
from test_warehouse_eegoal_refit import source_fixture, args_for, CollisionFixture


@pytest.fixture
def snapshot(tmp_path):
    roots = []
    for split in ("train", "validation"):
        raw, converted = tmp_path / split, tmp_path / (split + "-converted")
        source_fixture(raw); converted.mkdir()
        refit.run(args_for(raw, converted), collision=CollisionFixture())
        if split == "validation":
            index = json.loads((converted / "index.json").read_text())
            entry = index["splits"]["train"].pop()
            index["splits"]["validation"].append(entry)
            (converted / "index.json").write_text(json.dumps(index))
            with h5py.File(converted / entry["file"], "a") as handle:
                handle[entry["group"]].attrs["split"] = "validation"
                handle[entry["group"]].attrs["family_id"] = 2
        roots.append(converted)
    path = tmp_path / "snapshot.json"; build_snapshot(roots, path)
    return path


def config(variant="jd_world", representation="tau_r", **training):
    options = dict(device="cpu", torch_threads=2, max_steps=2, batch_size=1,
                   accumulate_steps=2, checkpoint_every=1, log_every=1, validate_every=1, seed=5)
    options.update(training)
    return dict(variant=variant, representation=representation, model=dict(SMALL, architecture_version="v2"),
                diffusion=dict(num_diffusion_steps=10), training=options)


def test_tensorboard_scalars_and_resume_purges_orphan_steps(snapshot, tmp_path):
    cfg = config()
    run = tmp_path / "run"; path = train_jointdual(cfg, snapshot, run)
    events = EventAccumulator(str(run / "tensorboard")).Reload()
    tags = events.Tags()["scalars"]
    assert {"train/loss", "train/p", "train/z", "train/tau", "train/r", "train/grad_norm",
            "train/learning_rate", "validation/loss", "validation/p", "validation/z",
            "validation/tau", "validation/r", "performance/samples_per_second"} <= set(tags)
    assert {"validation/samples", "validation/total_samples", "validation/coverage", "validation/batches"} <= set(tags)
    records = [json.loads(row) for row in (run / "metrics.jsonl").read_text().splitlines()]
    for scalar, record in zip(events.Scalars("train/loss"), records):
        assert scalar.step == record["step"]
        assert scalar.value == pytest.approx(record["loss"])
        assert record["validation_coverage"] == 1.
    # Simulate events written after the latest durable checkpoint, then restart.
    with SummaryWriter(str(run / "tensorboard")) as writer:
        writer.add_scalar("train/loss", 999., 9)
    cfg["training"]["max_steps"] = 3
    cfg["training"]["tensor_cache"] = False  # backend switch must preserve exact resume contract
    train_jointdual(cfg, snapshot, run, resume=path)
    scalars = EventAccumulator(str(run / "tensorboard")).Reload().Scalars("train/loss")
    assert [event.step for event in scalars] == [1, 2, 3]
    cfg["training"]["tensorboard"] = False
    straight = train_jointdual(cfg, snapshot, tmp_path / "straight")
    resumed_state = torch.load(path, weights_only=False)["model_state_dict"]
    for key, value in torch.load(straight, weights_only=False)["model_state_dict"].items():
        torch.testing.assert_close(value, resumed_state[key], atol=0, rtol=0)


def test_accumulated_metrics_are_microbatch_averages(snapshot, tmp_path, monkeypatch):
    values = iter((1., 3.))
    def loss(self, batch, **kwargs):
        value = next(values)
        # Keep a real autograd edge without depending on model predictions.
        loss = next(self.parameters()).sum()*0 + 3*value
        return loss, {"p": torch.tensor(value), "z": torch.tensor(2*value),
                      "tau": torch.tensor(value), "r": torch.tensor(value)}
    monkeypatch.setattr(JointDiffusion, "training_loss", loss)
    cfg = config(representation="c", tensorboard=False)
    cfg["training"].update(max_steps=1, validate_every=100)
    run = tmp_path / "averages"; train_jointdual(cfg, snapshot, run)
    record = json.loads((run / "metrics.jsonl").read_text())
    assert record["loss"] == 6. and record["p"] == 2. and record["z"] == 4.
    assert "tau" not in record and "r" not in record
    assert not (run / "tensorboard").exists()


@pytest.mark.parametrize("variant", ["jd_kin", "jd_world"])
@pytest.mark.parametrize("representation", ["c", "tau_r"])
def test_warm_start_weights_only_and_resume(snapshot, tmp_path, variant, representation):
    cfg = config(variant, representation, tensorboard=False)
    source = train_jointdual(cfg, snapshot, tmp_path / "source-run")
    previous = torch.load(source, weights_only=False)
    for weights in ("ema", "model"):
        target = JointDiffusion(JointDualDenoiser(variant=variant, **cfg["model"]), num_diffusion_steps=10)
        # New data/normalizers are allowed for fine-tuning, and explicitly reported.
        contract = copy.deepcopy(previous)
        contract["normalization"]["c"]["mean"][0] += 1.
        contract["dataset_snapshot_hash"] = "new-dataset"
        report = _warm_start_model(target, source, contract, weights=weights)
        assert report["normalization_changed"] and report["dataset_changed"]
        assert not report["fresh_parameter_keys"]
        selected = previous["ema_model_state_dict" if weights == "ema" else "model_state_dict"]
        for key, value in target.state_dict().items():
            torch.testing.assert_close(value, selected[key], atol=0, rtol=0)
    # LR=0 allows observing the initialized weights after a full real training step.
    cfg["training"].update(max_steps=1, learning_rate=0., seed=23)
    run = tmp_path / "warm"; path = train_jointdual(cfg, snapshot, run, warm_start=source)
    warmed = torch.load(path, weights_only=False)
    assert warmed["step"] == 1 and warmed["warm_start"]["source_step"] == 2
    assert all(int(state["step"]) == 1 for state in warmed["optimizer_state_dict"]["state"].values())
    for key, value in warmed["model_state_dict"].items():
        torch.testing.assert_close(value, previous["ema_model_state_dict"][key], atol=0, rtol=0)
    fresh = train_jointdual(cfg, snapshot, tmp_path / "fresh")
    torch.testing.assert_close(warmed["torch_rng_state"], torch.load(fresh, weights_only=False)["torch_rng_state"],
                               atol=0, rtol=0)
    cfg["training"]["max_steps"] = 2
    train_jointdual(cfg, snapshot, run, resume=path)
    assert torch.load(path, weights_only=False)["warm_start"] == warmed["warm_start"]
    with pytest.raises(FileExistsError):
        train_jointdual(cfg, snapshot, run, warm_start=source)
    wrong = dict(previous, representation="tau_r" if representation == "c" else "c")
    with pytest.raises(ValueError, match="representation"):
        _warm_start_model(target, source, wrong)


def test_v1_to_v2_compatible_warm_start_preserves_new_coordinates(snapshot, tmp_path):
    cfg = config(tensorboard=False)
    cfg["model"].pop("architecture_version")  # historical checkpoint with implicit v1
    source = train_jointdual(cfg, snapshot, tmp_path / "v1")
    previous = torch.load(source, weights_only=False)
    target = JointDiffusion(JointDualDenoiser(variant="jd_world", architecture_version="v2", **SMALL),
                            num_diffusion_steps=10)
    contract = dict(previous, model_config=dict(cfg["model"], variant="jd_world", architecture_version="v2"))
    before = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="architecture mismatch"):
        _warm_start_model(target, source, contract)
    report = _warm_start_model(target, source, contract, mode="compatible")
    assert report["loaded_parameter_count"] > 0
    assert "denoiser.world_encoder.encoder.2.weight" in report["fresh_parameter_keys"]
    for key in ("denoiser.input.weight", "denoiser.task.0.weight", "denoiser.timing_head.0.weight"):
        torch.testing.assert_close(target.state_dict()[key], previous["ema_model_state_dict"][key], atol=0, rtol=0)
    for key in ("denoiser.position", "denoiser.spatial_phase", "denoiser.world_encoder.frequencies",
                "denoiser.world_encoder.encoder.2.weight", "denoiser.fusion.projection.weight"):
        torch.testing.assert_close(target.state_dict()[key], before[key], atol=0, rtol=0)
    assert target.denoiser.timing_attention.gates[-1].weight.count_nonzero() == 0
    assert target.denoiser.spatial_attention.gates[-1].weight.count_nonzero() == 0
    cfg["model"]["architecture_version"] = "v2"
    cfg["training"]["max_steps"] = 1
    result = train_jointdual(cfg, snapshot, tmp_path / "v2", warm_start=source, warm_start_mode="compatible")
    assert torch.load(result, weights_only=False)["step"] == 1


def test_resume_and_warm_start_are_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_jointdual({}, "unused.json", "unused", resume="old.pt", warm_start="other.pt")


class ValidationProbe:
    def __init__(self):
        self.seen = []

    def training_loss(self, batch, *, generator):
        assert not torch.is_grad_enabled()
        self.seen.extend(batch["sample_index"].tolist())
        torch.randn(len(batch["p"]), generator=generator)  # validation must not advance training RNG
        value = batch["p"].float().mean()
        return 2*value, dict(p=value, z=value, tau=value, r=value)


def validation_rows(count):
    # Sorted difficulties mimic snapshot shards: hard worlds occur after the old prefix cutoff.
    return [dict(p=torch.tensor([1. if i < 40 else 9.]), sample_index=torch.tensor(i)) for i in range(count)]


@pytest.mark.parametrize("representation", ["c", "tau_r"])
def test_full_validation_covers_late_samples_and_weights_last_batch(representation):
    model = ValidationProbe(); data = validation_rows(45)
    before = torch.get_rng_state().clone()
    result = _validate_joint(model, data, batch_size=2, device="cpu", representation=representation, seed=5)
    assert model.seen == list(range(45))  # Includes all samples beyond the former 20-batch limit.
    assert result["validation_samples"] == result["validation_total_samples"] == 45
    assert result["validation_batches"] == 23 and result["validation_coverage"] == 1.
    assert result["validation_loss"] == pytest.approx(2*(40 + 5*9)/45)
    assert ("validation_tau" in result) == (representation == "tau_r")
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)


def test_limited_validation_is_fixed_uniform_subset():
    data = validation_rows(100)
    one, two = ValidationProbe(), ValidationProbe()
    kwargs = dict(batch_size=2, device="cpu", representation="tau_r", seed=5, batch_limit=20)
    first = _validate_joint(one, data, **kwargs)
    assert _validate_joint(two, data, **kwargs) == first
    assert one.seen == two.seen and len(set(one.seen)) == 40
    assert max(one.seen) >= 40 and one.seen != list(range(40))
    assert first["validation_samples"] == 40 and first["validation_coverage"] == .4
    assert first["validation_total_samples"] == 100
    with pytest.raises(ValueError, match="validation_batches"):
        _validate_joint(one, data, **dict(kwargs, batch_limit=0))


def test_resume_overrides_preserve_state_progress_and_validation(snapshot, tmp_path, monkeypatch):
    import mpd.joint_training.trainer as trainer
    cfg = config(tensorboard=False)
    source = train_jointdual(cfg, snapshot, tmp_path / "original")
    saved = torch.load(source, weights_only=False)
    # Exercise fallback for checkpoints produced before explicit progress/validation fields.
    legacy = copy.deepcopy(saved)
    for key in ("microbatches_seen", "samples_seen", "resume_history"):
        legacy.pop(key)
    legacy["training_config"] = dict(cfg["training"])
    source = source.parent / "legacy.pt"
    torch.save(legacy, source)
    cfg["training"].update(max_steps=3, seed=999, validation_seed=999, validation_batch_size=9)
    overrides = dict(learning_rate=3e-5, weight_decay=.002, gradient_clip_norm=.2,
                     batch_size=2, accumulate_steps=3, use_amp=True, ema_decay=.98, world_dropout=.2)
    observed = {}
    original_step = torch.optim.AdamW.step
    def optimizer_step(optimizer, *args, **kwargs):
        before = optimizer.state_dict()
        observed["optimizer"] = before["param_groups"][0].copy()
        for key, state in before["state"].items():
            for name in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(state[name], saved["optimizer_state_dict"]["state"][key][name], atol=0, rtol=0)
        return original_step(optimizer, *args, **kwargs)
    original_ema = trainer._update_ema
    def update_ema(ema, model, decay):
        observed["decay"] = decay
        assert model.denoiser.world_dropout == ema.denoiser.world_dropout == .2
        for key, value in ema.state_dict().items():
            torch.testing.assert_close(value, saved["ema_model_state_dict"][key], atol=0, rtol=0)
        original_ema(ema, model, decay)
    original_clip = torch.nn.utils.clip_grad_norm_
    def clip(parameters, max_norm, *args, **kwargs):
        observed["clip"] = max_norm
        return original_clip(parameters, max_norm, *args, **kwargs)
    original_validate = trainer._validate_joint
    def validate(*args, **kwargs):
        observed["validation"] = (kwargs["batch_size"], kwargs["seed"])
        return original_validate(*args, **kwargs)
    original_iter = trainer.JointBatchSampler.__iter__
    def sampler_iter(sampler):
        observed["sampler"] = (sampler.start, sampler.stop, sampler.batch_size, sampler.seed)
        return original_iter(sampler)
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", optimizer_step)
        patch.setattr(trainer, "_update_ema", update_ema)
        patch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
        patch.setattr(trainer, "_validate_joint", validate)
        patch.setattr(trainer.JointBatchSampler, "__iter__", sampler_iter)
        target = train_jointdual(cfg, snapshot, tmp_path / "changed", resume=source, resume_overrides=overrides)
    current = torch.load(target, weights_only=False)
    assert observed["optimizer"]["lr"] == 3e-5 and observed["optimizer"]["weight_decay"] == .002
    assert observed["decay"] == .98 and observed["clip"] == .2
    assert observed["validation"] == (1, 5)
    assert observed["sampler"] == (4, 7, 2, 5)  # never recompute 2 * new accumulation=6
    assert current["step"] == 3 and current["microbatches_seen"] == 7 and current["samples_seen"] == 10
    assert all(int(state["step"]) == 3 for state in current["optimizer_state_dict"]["state"].values())
    report = json.loads((target.parent / "resume.json").read_text())
    assert report["source_step"] == 2 and len(report["changes"]) == 8
    assert report["effective_batch_size"] == {"old": 2, "new": 6}
    assert report["scaler_action"] == "disabled"  # CPU, even though AMP was requested
    assert report["sha256"] and report == current["resume_history"][-1]
    assert torch.load(source, weights_only=False)["step"] == 2
    # Re-resume using stale YAML: prior overrides persist and the new cursor is restored.
    cfg["training"]["max_steps"] = 4
    train_jointdual(cfg, snapshot, target.parent, resume=target)
    repeated = torch.load(target, weights_only=False)
    assert repeated["microbatches_seen"] == 10 and repeated["samples_seen"] == 16
    assert repeated["training_config"]["learning_rate"] == 3e-5
    assert repeated["model_config"]["world_dropout"] == .2
    assert len(repeated["resume_history"]) == 2
    direct = train_jointdual(cfg, snapshot, tmp_path / "direct", resume=source, resume_overrides=overrides)
    uninterrupted = torch.load(direct, weights_only=False)
    for name in ("model_state_dict", "ema_model_state_dict"):
        for key, value in repeated[name].items():
            torch.testing.assert_close(value, uninterrupted[name][key], atol=0, rtol=0)


def test_resume_recipe_priority_and_output_protection(snapshot, tmp_path):
    cfg = config(tensorboard=False)
    source = train_jointdual(cfg, snapshot, tmp_path / "original")
    saved = torch.load(source, weights_only=False)
    stale = copy.deepcopy(cfg)
    stale["training"].update(max_steps=3, learning_rate=.5, weight_decay=.7, gradient_clip_norm=10.,
                             batch_size=9, accumulate_steps=3, use_amp=True, ema_decay=.2, seed=999)
    stale["model"]["world_dropout"] = .7
    result = train_jointdual(stale, snapshot, tmp_path / "stale", resume=source)
    current = torch.load(result, weights_only=False)
    for key in ("learning_rate", "weight_decay", "gradient_clip_norm", "batch_size", "accumulate_steps", "use_amp", "ema_decay", "seed"):
        assert current["training_config"][key] == saved["training_config"][key]
    assert current["model_config"]["world_dropout"] == saved["model_config"]["world_dropout"]
    assert not current["resume_history"][-1]["changes"]
    cfg["training"]["max_steps"] = 3
    with pytest.raises(ValueError, match="new output-dir"):
        train_jointdual(cfg, snapshot, source.parent, resume=source, resume_overrides={"learning_rate": 3e-5})
    with pytest.raises(ValueError, match="older checkpoint"):
        train_jointdual(cfg, snapshot, source.parent, resume=source.parent / "step-00000001.pt")
    with pytest.raises(FileExistsError, match="not empty"):
        train_jointdual(cfg, snapshot, result.parent, resume=source)
    with pytest.raises(ValueError, match="unsupported resume overrides"):
        train_jointdual(cfg, snapshot, tmp_path / "bad", resume=source, resume_overrides={"seed": 1})
    cfg["training"]["max_steps"] = 2
    with pytest.raises(ValueError, match="cumulative target"):
        train_jointdual(cfg, snapshot, tmp_path / "no-progress", resume=source)
    assert torch.load(source, weights_only=False)["step"] == 2


def test_amp_scaler_transition_rules():
    from mpd.joint_training.trainer import _restore_scaler
    class Scaler:
        loaded = None
        def __init__(self, enabled): self.enabled = enabled
        def is_enabled(self): return self.enabled
        def load_state_dict(self, state): self.loaded = state
    old = {"scale": 1024.}
    disabled = Scaler(False)
    assert _restore_scaler(disabled, old) == "disabled" and disabled.loaded is None
    enabled = Scaler(True)
    assert _restore_scaler(enabled, {}) == "initialized" and enabled.loaded is None
    assert _restore_scaler(enabled, old) == "restored" and enabled.loaded == old


@pytest.mark.parametrize("key,value", [("learning_rate", float("nan")), ("weight_decay", -1.),
    ("gradient_clip_norm", 0.), ("batch_size", 0), ("accumulate_steps", 1.5), ("ema_decay", 1.), ("use_amp", "false")])
def test_invalid_resume_recipe_rejected(key, value):
    from mpd.joint_training.trainer import RESUME_DEFAULTS, _validate_recipe
    opts = dict(RESUME_DEFAULTS, validation_batch_size=64, max_steps=10)
    opts[key] = value
    with pytest.raises(ValueError, match=key):
        _validate_recipe(opts)


@pytest.mark.parametrize("amp_flag,expected", [(None, None), ("--amp", True), ("--no-amp", False)])
def test_cli_explicit_resume_flags(tmp_path, monkeypatch, amp_flag, expected):
    from scripts.train import train_jointdual as cli
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config()))
    calls = []
    monkeypatch.setattr(cli, "train_jointdual", lambda *args, **kwargs: calls.append((args, kwargs)))
    argv = ["--config", str(path), "--snapshot", "snapshot.json", "--output-dir", "new", "--resume", "old.pt",
            "--learning-rate", "0.00003", "--weight-decay", "0.002", "--gradient-clip-norm", "0.2",
            "--batch-size", "32", "--accumulate-steps", "8", "--ema-decay", "0.98", "--world-dropout", "0.2"]
    if amp_flag: argv.append(amp_flag)
    cli.main(argv)
    overrides = calls[0][1]["resume_overrides"]
    assert overrides == dict(learning_rate=3e-5, weight_decay=.002, gradient_clip_norm=.2, batch_size=32,
                            accumulate_steps=8, ema_decay=.98, world_dropout=.2,
                            **({"use_amp": expected} if expected is not None else {}))
    assert calls[0][0][0]["training"]["batch_size"] == 1  # YAML values are not mistaken for explicit overrides


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for actual AMP scaler transitions")
def test_resume_amp_transitions_on_cuda(snapshot, tmp_path):
    cfg = config(device="cuda:0", tensorboard=False, max_steps=1)
    source = train_jointdual(cfg, snapshot, tmp_path / "fp32")
    cfg["training"]["max_steps"] = 2
    target = train_jointdual(cfg, snapshot, tmp_path / "amp", resume=source, resume_overrides={"use_amp": True})
    state = torch.load(target, map_location="cpu", weights_only=False)
    assert state["resume_history"][-1]["scaler_action"] == "initialized"
    assert state["scaler_state_dict"]
    cfg["training"]["max_steps"] = 3
    train_jointdual(cfg, snapshot, target.parent, resume=target)
    assert torch.load(target, map_location="cpu", weights_only=False)["resume_history"][-1]["scaler_action"] == "restored"
    cfg["training"]["max_steps"] = 4
    target = train_jointdual(cfg, snapshot, tmp_path / "fp32-again", resume=target, resume_overrides={"use_amp": False})
    state = torch.load(target, map_location="cpu", weights_only=False)
    assert state["resume_history"][-1]["scaler_action"] == "disabled" and not state["scaler_state_dict"]
    cfg["training"]["max_steps"] = 5
    target = train_jointdual(cfg, snapshot, tmp_path / "amp-again", resume=target, resume_overrides={"use_amp": True})
    assert torch.load(target, map_location="cpu", weights_only=False)["resume_history"][-1]["scaler_action"] == "initialized"
