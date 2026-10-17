import json
import pickle
import h5py
import numpy as np
import pytest
import torch
from scipy.interpolate import BSpline
from torch.utils.data import DataLoader

from mpd.datasets.warehouse_joint_dataset import build_snapshot, WarehouseJointDataset, collate_joint
from mpd.datasets.spacetime_schema import load_panda_robot_bundle
from mpd.parametric_trajectory.joint_timing_codec import JointTimingCodec
from mpd.parametric_trajectory.timing_fitting import TimingSplineNumpy
from mpd.parametric_trajectory.normalized_timing import NormalizedTimingSplineNumpy, minimum_duration_for_path
from scripts.spacetime_data import refit_warehouse_eegoal as refit
from test_warehouse_eegoal_refit import source_fixture, CollisionFixture, args_for


@pytest.fixture
def snapshot(tmp_path):
    root, output = tmp_path / "source", tmp_path / "refitted"
    source_fixture(root); output.mkdir()
    refit.run(args_for(root, output), collision=CollisionFixture())
    path = tmp_path / "snapshot.json"
    build_snapshot([output], path)
    return path


def test_data_views_and_worker_collation(snapshot):
    c, r = [WarehouseJointDataset(snapshot, representation=rep) for rep in ("c", "tau_r")]
    assert c.entries == r.entries
    assert c[0]["p"].shape == (16, 7) and c[0]["task"].shape == (19,)
    assert not torch.equal(c[0]["z"], r[0]["z"]) or len(c) == 1  # single-row standardized targets are zero
    # A spawned reader receives this serialized object, never live HDF5 handles.
    copied = pickle.loads(pickle.dumps(c))
    assert not copied._handles
    for view in (c, copied):
        batch = next(iter(DataLoader(view, batch_size=1, collate_fn=collate_joint)))
        assert batch["world_features"].shape == (1, 32, 13)
        assert batch["world_mask"].all()
    c.close(); r.close(); copied.close()


@pytest.mark.parametrize("representation", ["c", "tau_r"])
def test_codec_roundtrip_and_query_grid_independence(snapshot, representation):
    dataset = WarehouseJointDataset(snapshot, representation=representation)
    bundle = load_panda_robot_bundle(refit.base.REPOSITORY)
    codec = JointTimingCodec(dataset.normalization, representation, bundle.dq_max, bundle.ddq_max)
    e = dataset.entries[0]
    with h5py.File(e["file"]) as f:
        g = f[e["group"]]
        p = torch.tensor(g["spatial/control_points"][:])[None]
        # Fixture was validated against its own limits, so use the same ones.
        codec.velocity_limits.fill_(1.); codec.acceleration_limits.fill_(3.)
        context = codec.prepare(p, dataset[0]["z"][None].double())
        short = codec.evaluate(context, torch.linspace(0, 1, 128, dtype=torch.float64))
        dense = codec.evaluate(context, codec.phase)
        np.testing.assert_allclose(dense.duration.numpy(), g["timing/reference_time"][-1], atol=1e-6)
        torch.testing.assert_close(short.duration, dense.duration, rtol=0, atol=0)
        np.testing.assert_allclose(dense.q[0].numpy(), BSpline(g["spatial/knots"][:], p[0], 5)(codec.phase), atol=1e-10)
        np.testing.assert_allclose(dense.dq[0, [0, -1]].numpy(), 0., atol=1e-9)
        if representation == "c":
            reference = TimingSplineNumpy(num_phase_points=2049).evaluate(g["timing/control_points"][:])
            np.testing.assert_allclose(dense.time_from_start[0].numpy(), reference.time_from_start, atol=1e-10)
        else:
            np.testing.assert_allclose(context["tmin"].numpy(), g["timing/T_min"][()], atol=1e-8)
        inverse = codec.phase_at_times(context, dense.time_from_start)
        torch.testing.assert_close(inverse, codec.phase[None], rtol=0, atol=1e-12)


def test_codec_gradients_and_infeasible_duration(snapshot):
    data = WarehouseJointDataset(snapshot)
    codec = JointTimingCodec(data.normalization, "tau_r", np.ones(7), np.full(7, 3.))
    p = data[0]["p"][None].double().requires_grad_()
    qs = torch.tensor([[-.5, 0, 0, 0, 0, 0, 0]], dtype=torch.float64)
    z = data[0]["z"][None].double().requires_grad_()
    # Tight limits make Tmin depend on P rather than its constant floor.
    codec.velocity_limits.fill_(.01)
    context = codec.prepare(codec.full_path(p, qs), z)
    assert context["tmin"].item() > 14
    assert context["duration"].item() > 14
    gp, gz = torch.autograd.grad(context["duration"].sum(), (p, z))
    assert torch.isfinite(gp).all() and torch.isfinite(gz).all() and gp.norm() > 0


def test_snapshot_rejects_changed_files(snapshot):
    payload = json.loads(snapshot.read_text())
    with h5py.File(payload["entries"][0]["file"], "a") as f:
        f.attrs["changed"] = True
    with pytest.raises(ValueError, match="frozen shard changed"):
        WarehouseJointDataset(snapshot)
