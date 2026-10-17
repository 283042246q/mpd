import json
import pytest
import torch
from mpd.models.diffusion_models.jointdual import JointDualDenoiser, JointDiffusion
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
def test_joint_gradients_masks_and_permutation(variant):
    net = JointDualDenoiser(variant=variant, **SMALL).eval()
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
    b["world_mask"][:]=False
    assert all(torch.isfinite(x).all() for x in net(b["p"],b["z"],step,b["task"],
        b["world_features"],b["world_mask"],torch.tensor([1,2])))
    loss,_=JointDiffusion(net).training_loss(b);loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("variant", ["jd_kin", "jd_world"])
@pytest.mark.parametrize("representation", ["c", "tau_r"])
def test_train_save_resume_four_variants(tmp_path,variant,representation):
    root,converted=tmp_path/"source",tmp_path/"converted"
    source_fixture(root);converted.mkdir()
    refit.run(args_for(root,converted),collision=CollisionFixture())
    snapshot=tmp_path/"snapshot.json";build_snapshot([converted],snapshot)
    config=dict(variant=variant,representation=representation,model=SMALL,
        diffusion=dict(num_diffusion_steps=10),training=dict(device="cpu",torch_threads=2,
        max_steps=2,batch_size=2,accumulate_steps=1,checkpoint_every=1,seed=5))
    path=train_jointdual(config,snapshot,tmp_path/"run")
    model,payload=load_joint_checkpoint(path,variant=variant,representation=representation)
    assert payload["step"]==2 and model.denoiser.variant==variant
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


def test_four_cli_configurations():
    for name in ("jdkin_c","jdkin_tau_r","jdworld_c","jdworld_tau_r"):
        config=load_config(refit.base.REPOSITORY/f"scripts/train/cfgs/{name}.yaml")
        assert config["variant"] in ("jd_kin","jd_world") and config["representation"] in ("c","tau_r")
