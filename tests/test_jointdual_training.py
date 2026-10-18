import json
import pytest
import torch
from mpd.models.diffusion_models.jointdual import JointDualDenoiser, JointDiffusion, GatedWorldAttention
from mpd.models.diffusion_models.world_encoder import WorldEncoder
from mpd.joint_training.trainer import train_jointdual, load_joint_checkpoint
from mpd.datasets.warehouse_joint_dataset import build_snapshot
from scripts.spacetime_data import refit_warehouse_eegoal as refit
from scripts.train.train_jointdual import load_config
from test_warehouse_eegoal_refit import source_fixture, args_for, CollisionFixture

SMALL = dict(channels=[16, 32, 64, 128], timing_width=32, joint_width=64, world_width=32, world_dropout=0.)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def inputs():
    return dict(p=torch.randn(2,16,7), z=torch.randn(2,6), task=torch.randn(2,19),
        world_features=torch.randn(2,32,13), world_mask=torch.ones(2,32,dtype=torch.bool),
        world_state=torch.tensor([0,1]))


@pytest.mark.parametrize("variant", ["jd_kin", "jd_world"])
@pytest.mark.parametrize("architecture", ["v1", "v2"])
def test_joint_gradients_masks_and_permutation(variant, architecture):
    net = JointDualDenoiser(variant=variant, architecture_version=architecture, **SMALL).eval()
    if variant == "jd_world" and architecture == "v2":
        # Exercise real world attention, not just the initially closed gates.
        with torch.no_grad():
            net.timing_attention.gates[-1].bias.fill_(.3)
            net.spatial_attention.gates[-1].bias.fill_(.3)
    b = inputs(); b["p"].requires_grad_(); b["z"].requires_grad_()
    step = torch.tensor([5,5])
    pp,pz = net(b["p"],b["z"],step,b["task"],b["world_features"],b["world_mask"],b["world_state"])
    assert pp.shape==(2,16,7) and pz.shape==(2,6)
    assert torch.autograd.grad(pp.square().sum(),b["z"],retain_graph=True)[0].norm()>0
    assert torch.autograd.grad(pz.square().sum(),b["p"],retain_graph=True)[0].norm()>0
    permutation=torch.randperm(32)
    changed=net(b["p"],b["z"],step,b["task"],b["world_features"][:,permutation],
                b["world_mask"][:,permutation],b["world_state"])
    torch.testing.assert_close(pp,changed[0],atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(pz,changed[1],atol=2e-5,rtol=2e-5)
    if variant == "jd_world":
        encoded = net.encode_world(b["world_features"], b["world_mask"], b["world_state"])
        cached = net(b["p"], b["z"], step, b["task"], encoded_world=encoded)
        for direct, value in zip((pp, pz), cached):
            torch.testing.assert_close(direct, value)
        shifted = b["world_features"].clone(); shifted[:, :, -1] += 1.
        shifted_outputs = net(b["p"], b["z"], step, b["task"], shifted,
                              b["world_mask"], b["world_state"])
        assert not torch.allclose(pp[0], shifted_outputs[0][0])
        assert not torch.allclose(pz[0], shifted_outputs[1][0])
        # EMPTY ignores the raw world values, whereas KNOWN sees physical time.
        torch.testing.assert_close(pp[1], shifted_outputs[0][1])
    b["world_mask"][:]=False
    assert all(torch.isfinite(x).all() for x in net(b["p"],b["z"],step,b["task"],
        b["world_features"],b["world_mask"],torch.tensor([1,2])))
    loss,_=JointDiffusion(net).training_loss(b);loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("variant", ["jd_kin", "jd_world"])
@pytest.mark.parametrize("representation", ["c", "tau_r"])
@pytest.mark.parametrize("architecture", [None, "v2"])
def test_train_save_resume_four_variants(tmp_path,variant,representation,architecture):
    root,converted=tmp_path/"source",tmp_path/"converted"
    source_fixture(root);converted.mkdir()
    refit.run(args_for(root,converted),collision=CollisionFixture())
    snapshot=tmp_path/"snapshot.json";build_snapshot([converted],snapshot)
    model_config = dict(SMALL)
    if architecture is not None:
        model_config["architecture_version"] = architecture
    config=dict(variant=variant,representation=representation,model=model_config,
        diffusion=dict(num_diffusion_steps=10),training=dict(device="cpu",torch_threads=2,
        max_steps=2,batch_size=2,accumulate_steps=1,checkpoint_every=1,seed=5))
    path=train_jointdual(config,snapshot,tmp_path/"run")
    model,payload=load_joint_checkpoint(path,variant=variant,representation=representation)
    assert payload["step"]==2 and model.denoiser.variant==variant
    assert model.denoiser.architecture_version == (architecture or "v1")
    config["training"]["max_steps"]=3
    train_jointdual(config,snapshot,tmp_path/"run",resume=path)
    assert torch.load(path,weights_only=False)["step"]==3
    with pytest.raises(ValueError,match="representation mismatch"):
        load_joint_checkpoint(path,representation="tau_r" if representation=="c" else "c")
    # Uninterrupted and resumed runs must produce identical weights and batch schedules.
    straight=train_jointdual(config,snapshot,tmp_path/"straight")
    one=torch.load(path,weights_only=False);two=torch.load(straight,weights_only=False)
    for key,value in one["model_state_dict"].items():
        torch.testing.assert_close(value,two["model_state_dict"][key],atol=0,rtol=0)
    config["model"] = dict(model_config, architecture_version="v1" if architecture == "v2" else "v2")
    with pytest.raises(ValueError, match="contract mismatch"):
        train_jointdual(config, snapshot, tmp_path/"run", resume=path)


def test_gated_attention_identity_and_branch_learning():
    torch.manual_seed(13)
    block = GatedWorldAttention(16, 12, 8, 4)
    query, world, context = torch.randn(2, 3, 16), torch.randn(2, 5, 12), torch.randn(2, 8)
    valid = torch.tensor([[True, True, False, False, False], [True]*5])
    optimizer = torch.optim.SGD(block.parameters(), lr=.1)
    output = block(query, world, valid, context)
    torch.testing.assert_close(output, query, atol=0, rtol=0)
    (output - torch.randn_like(output)).square().mean().backward()
    # Both independently gated branches can open on the first optimization step.
    for grad in block.gates[-1].weight.grad.chunk(2, dim=0):
        assert grad.norm() > 0
    optimizer.step(); optimizer.zero_grad()
    world.requires_grad_()
    block(query, world, valid, context).square().sum().backward()
    assert block.attention.out_proj.weight.grad.norm() > 0
    assert block.ffn[-1].weight.grad.norm() > 0
    assert world.grad[valid].norm() > 0
    assert world.grad[~valid].count_nonzero() == 0
    changed = world.detach().clone(); changed[~valid] = 1000.
    torch.testing.assert_close(block(query, world, valid, context), block(query, changed, valid, context))


def test_v2_coordinate_contract_and_ordered_fusion():
    from mpd.datasets.spacetime_legacy import open_uniform_knots
    net = JointDualDenoiser(architecture_version="v2", **SMALL).eval()
    knots = open_uniform_knots(21, 5)
    expected = torch.tensor([knots[i+1:i+6].mean() for i in range(3, 18)] + [1.], dtype=torch.float32)
    torch.testing.assert_close(net.spatial_phase, expected)
    assert (torch.diff(net.spatial_phase) > 0).all()
    assert torch.unique(net.position, dim=0).shape[0] == 16
    # Swapping bottleneck positions preserves mean pooling but must change fusion.
    bottleneck = torch.randn(2, SMALL["channels"][-1], 2)
    context = torch.randn(2, 3*SMALL["timing_width"])
    first = net.fusion(torch.cat((bottleneck.flatten(1), context), -1))
    second = net.fusion(torch.cat((bottleneck.flip(-1).flatten(1), context), -1))
    assert not torch.allclose(first, second)
    encoder = WorldEncoder(32, encoding_version="v2")
    seconds = torch.tensor([0., 7., 14.])[:, None]
    encoded = encoder.encode_time(seconds)
    torch.testing.assert_close(encoded[:, 0], torch.tensor([0., .5, 1.]))
    assert encoded.shape == (3, 17)
    assert not torch.allclose(encoded[0], encoded[-1])
    # Same second has the same encoding regardless of other samples in the batch.
    torch.testing.assert_close(encoder.encode_time(seconds[1:2]), encoded[1:2])
    # Highest time frequency is four cycles over 14s, below the 32-point Nyquist rate.
    assert encoder.frequencies.max() / (2*torch.pi*encoder.time_scale_seconds) < 31/28


def test_unknown_architecture_rejected():
    with pytest.raises(ValueError, match="architecture_version"):
        JointDualDenoiser(architecture_version="v3")


def test_four_cli_configurations():
    for name in ("jdkin_c","jdkin_tau_r","jdworld_c","jdworld_tau_r"):
        config=load_config(refit.base.REPOSITORY/f"scripts/train/cfgs/{name}.yaml")
        assert config["variant"] in ("jd_kin","jd_world") and config["representation"] in ("c","tau_r")
        assert config["model"]["architecture_version"] == "v2"
        assert config["training"]["validation_batches"] is None
