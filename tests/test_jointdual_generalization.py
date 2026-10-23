import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mpd.joint_training.trainer import train_jointdual
from mpd.models.diffusion_models.jointdual import JointDiffusion, JointDualDenoiser
from scripts.train.eval_jointdual_generalization import (
    check_contract, evaluate, main, make_noise_plan, mean_stats, select_indices, summarize,
)
from test_jointdual_training import SMALL
from test_jointdual_training_tools import config, snapshot


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_family_sampling_ignores_world_multiplicity_and_rejects_oversampling():
    entries = [{"family": family} for family, count in (("a", 1), ("b", 7), ("c", 2)) for _ in range(count)]
    indices = select_indices(entries, 3, 31, "family")
    assert {entries[i]["family"] for i in indices} == {"a", "b", "c"}
    assert indices == select_indices(entries, 3, 31, "family")
    assert len(set(select_indices(entries, 10, 31, "row"))) == 10
    with pytest.raises(ValueError, match="families"):
        select_indices(entries, 4, 31, "family")


@pytest.mark.parametrize("variant", ["jd_kin", "jd_world"])
@pytest.mark.parametrize("representation", ["c", "tau_r"])
def test_eval_is_fixed_noise_and_batch_independent(variant, representation):
    torch.manual_seed(9)
    options = dict(SMALL, world_dropout=.9)
    model = JointDiffusion(JointDualDenoiser(variant=variant, architecture_version="v2", **options), 7)
    if variant == "jd_world":
        with torch.no_grad():
            model.denoiser.timing_attention.gates[-1].bias.fill_(.2)
            model.denoiser.spatial_attention.gates[-1].bias.fill_(.2)
    rows = []
    for i in range(5):
        row = dict(p=torch.randn(16, 7), z=torch.randn(6), task=torch.randn(19))
        if variant == "jd_world":
            row.update(world_features=torch.randn(i+1, 13), world_mask=torch.ones(i+1, dtype=torch.bool),
                       world_state=torch.tensor(0))
        rows.append(row)
    plan, digest = make_noise_plan(5, 7, 3, 2, 42)
    model.train()
    one = evaluate(model, rows, list(range(5)), plan, 2, "cpu", representation)
    assert not model.training and not model.denoiser.training
    torch.manual_seed(987)
    plan2, digest2 = make_noise_plan(5, 7, 3, 2, 42)
    assert digest == digest2
    two = evaluate(model, rows, list(range(5)), plan2, 3, "cpu", representation)
    np.testing.assert_allclose(one, two, atol=2e-6, rtol=2e-6)
    if representation == "tau_r":
        np.testing.assert_allclose(one[:, :, 1], (one[:, :, 2]+5*one[:, :, 3])/6, atol=2e-6)
    else:
        assert one.shape == (5, 3, 2)
    assert all(s["lo"] <= int(k.min()) <= int(k.max()) < s["hi"] for s in plan for k, _, _ in s["draws"])


def test_bucket_weights_paired_deltas_and_cluster_uncertainty():
    plan, _ = make_noise_plan(3, 7, 3, 1, 42)
    base = np.tile(np.array([1., 2., 4.])[None, :, None], (3, 1, 2))
    reference = dict(train=base, validation=base+1)
    values = dict(train=base-.2, validation=base+1.3)
    families = dict(train=["a", "b", "c"], validation=["d", "e", "f"])
    summary = summarize(values, reference, families, plan, ["p", "z"])
    stats = summary["all"]["z"]
    assert stats["train"]["mean"] == pytest.approx(15/7-.2)
    assert stats["delta_train"]["mean"] == pytest.approx(-.2)
    assert stats["delta_validation"]["mean"] == pytest.approx(.3)
    assert stats["gap"] == pytest.approx(1.5)
    # Duplicating correlated rows does not manufacture more independent families.
    one = mean_stats([1, 2, 4], ["a", "b", "c"])
    two = mean_stats([1, 1, 2, 2, 4, 4], ["a", "a", "b", "b", "c", "c"])
    assert one == two


def test_cli_uses_ema_and_publishes_reproducible_report(snapshot, tmp_path):
    checkpoint = train_jointdual(config(tensorboard=False), snapshot, tmp_path / "train")
    original = torch.load(checkpoint, map_location="cpu", weights_only=False)
    changed = copy.deepcopy(original)
    for value in changed["model_state_dict"].values():
        value.add_(10)
    other = tmp_path / "different-raw-model.pt"
    torch.save(changed, other)
    output = tmp_path / "probe"
    assert main(["--snapshot", str(snapshot), "--checkpoints", str(checkpoint), str(other),
                 "--output-dir", str(output), "--samples", "1", "--timestep-bins", "3",
                 "--noise-repeats", "2", "--batch-size", "1", "--device", "cpu", "--torch-threads", "2"]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["complete"] and len(report["checkpoints"]) == 2
    assert report["protocol"]["weights"] == "EMA"
    for bucket in report["checkpoints"][1]["results"].values():
        for metric in bucket.values():
            assert metric["delta_train"]["mean"] == 0
            assert metric["delta_validation"]["mean"] == 0
    assert (output / "summary.csv").exists()
    with np.load(output / "checkpoint-000.npz") as saved:
        assert saved["train"].shape == (1, 3, 4)
    meta = dict(identity=original["dataset_snapshot_hash"], **{k:original[k] for k in
                ("normalization", "robot_hashes", "static_geometry_hashes")})
    meta["identity"] = "wrong-snapshot"
    with pytest.raises(ValueError, match="snapshot mismatch"):
        check_contract(original, SimpleNamespace(snapshot=meta))
