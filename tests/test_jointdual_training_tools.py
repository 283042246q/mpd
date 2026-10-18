import json

import h5py
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from mpd.datasets.warehouse_joint_dataset import build_snapshot
from mpd.joint_training.trainer import train_jointdual
from mpd.models.diffusion_models.jointdual import JointDiffusion
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
    return dict(variant=variant, representation=representation, model=dict(SMALL, architecture_version="v2"),
                diffusion=dict(num_diffusion_steps=10), training=dict(device="cpu", torch_threads=2,
                    max_steps=2, batch_size=1, accumulate_steps=2, checkpoint_every=1, log_every=1,
                    validate_every=1, seed=5, **training))


def test_tensorboard_scalars_and_resume_purges_orphan_steps(snapshot, tmp_path):
    cfg = config()
    run = tmp_path / "run"; path = train_jointdual(cfg, snapshot, run)
    events = EventAccumulator(str(run / "tensorboard")).Reload()
    tags = events.Tags()["scalars"]
    assert {"train/loss", "train/p", "train/z", "train/tau", "train/r", "train/grad_norm",
            "train/learning_rate", "validation/loss", "validation/p", "validation/z",
            "validation/tau", "validation/r", "performance/samples_per_second"} <= set(tags)
    records = [json.loads(row) for row in (run / "metrics.jsonl").read_text().splitlines()]
    for scalar, record in zip(events.Scalars("train/loss"), records):
        assert scalar.step == record["step"]
        assert scalar.value == pytest.approx(record["loss"])
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
