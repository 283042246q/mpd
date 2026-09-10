from mpd.utils.patches import numpy_monkey_patch

numpy_monkey_patch()

import os

import torch
from matplotlib import pyplot as plt

from experiment_launcher import single_experiment_yaml, run_experiment
from mpd import trainer
from mpd.models import CoupledBimanualTemporalUnet, UNET_DIM_MULTS, TemporalUnet
from mpd.models.diffusion_models.context_models import (
    ContextModelCombined,
    ContextModelEEPoseGoal,
    ContextModelMarvinCrossArmEE,
    ContextModelMarvinCrossArmTokens,
    ContextModelMarvinDualEE,
    ContextModelMarvinStructuredEE,
    ContextModelQs,
)
from mpd.trainer.trainer import get_num_epochs
from mpd.utils.loaders import get_planning_task_and_dataset, get_model, get_loss, get_summary
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


os.environ["WANDB_API_KEY"] = "999"
WANDB_MODE = "disabled"
WANDB_ENTITY = "mpd-splines"
DEBUG = True


def load_warm_start_weights(model, checkpoint_path):
    """Load only model parameters; optimizer/step/RNG state intentionally restart."""
    checkpoint_path = os.path.abspath(os.path.expanduser(os.fspath(checkpoint_path)))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Warm-start state_dict not found: {checkpoint_path}")
    if os.path.getsize(checkpoint_path) == 0:
        raise ValueError(f"Warm-start state_dict is empty: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict) or not state_dict or not all(
        isinstance(key, str) for key in state_dict
    ):
        raise TypeError(
            "--warm-start-checkpoint must point to a *_state_dict.pth file, "
            f"not a serialized full model: {checkpoint_path}"
        )
    model.load_state_dict(state_dict, strict=True)
    print(f"Warm-started model weights from: {checkpoint_path}")
    print("Optimizer, AMP scaler, EMA history, RNG state, and train-step counter start fresh.")
    return checkpoint_path


@single_experiment_yaml
def experiment(
    ########################################################################
    # Dataset
    # 数据集子目录，决定训练使用哪个环境、机器人和先验规划器生成的数据
    dataset_subdir: str = "EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect",
    # dataset_subdir: str = 'EnvWarehouse-RobotPanda-config_file_v01-joint_joint-one-RRTConnect',
    # 合并后的 hdf5 数据文件名，通常使用翻转增强后的 dataset_merged_doubled.hdf5
    dataset_file_merged: str = "dataset_merged_doubled.hdf5",
    # 是否重新从 hdf5 解析并缓存数据；False 时优先读取已有 reload pickle
    reload_data: bool = False,
    # 是否预加载数据到训练设备；大数据集/显存紧张时保持 False
    preload_data_to_device: bool = False,
    # 只抽取多少个任务用于训练/调试；-1 表示使用全部任务
    n_task_samples: int = -1,
    ########################################################################
    # Parametric trajectory
    # 轨迹参数化方式；Bspline 表示学习控制点，Waypoints 表示直接学习路径点
    parametric_trajectory_class: str = "ParametricTrajectoryBspline",
    # parametric_trajectory_class: str = 'ParametricTrajectoryWaypoints',
    # B-spline 阶数/degree，当前默认 5 阶
    bspline_degree: int = 5,
    # 期望的 B-spline 控制点数量；程序会调整，使可学习控制点数量是 8 的倍数
    bspline_num_control_points_desired: int = 22,
    # Keep the exact serialized count for prevalidated generated splines.
    bspline_num_control_points_exact: bool = False,
    # B-spline 展开后的轨迹采样点数，用于碰撞检查、可视化和后续仿真
    num_T_pts: int = 128,
    ########################################################################
    # Context model
    # condition on joint start and goal
    # 是否把关节起点/终点或起点作为条件输入模型
    context_qs: bool = True,
    # 关节条件编码 MLP 层数
    context_qs_n_layers: int = 2,
    # 关节条件编码输出维度
    context_q_out_dim: int = 128,
    # 关节条件编码激活函数
    context_qs_act: str = "relu",
    # End-effector pose conditioned model
    # 是否把末端目标位姿作为条件输入；Panda/Warehouse 通常需要开启
    context_ee_goal_pose: bool = False,
    # Two fixed Pika TCP slots plus an activity mask (Marvin only).
    context_ee_goal_pose_bimanual: bool = False,
    # 末端目标位姿条件编码 MLP 层数
    context_ee_goal_pose_n_layers: int = 2,
    # 末端目标位姿条件编码输出维度
    context_ee_goal_pose_out_dim: int = 128,
    # 末端目标位姿条件编码激活函数
    context_ee_goal_pose_act: str = "relu",
    # Combined context model
    # 多个条件编码合并后的输出维度
    context_combined_out_dim: int = 128,
    # Marvin dual-slot ablation: A=flat/joint, B=per-arm/joint,
    # C=per-arm cross-attention/joint, D=per-arm cross-attention/coupled streams.
    bimanual_network_variant: str = "A",
    # Cross-arm attention hyperparameters shared by C/D context and D denoiser.
    bimanual_attention_heads: int = 4,
    bimanual_context_attention_layers: int = 2,
    bimanual_denoiser_attention_layers: int = 1,
    bimanual_attention_ff_multiplier: int = 2,
    bimanual_attention_dropout: float = 0.0,
    ########################################################################
    # Generative prior model
    # 生成模型类型，可选扩散模型或 CVAE
    generative_model_class: str = "GaussianDiffusionModel",  # 'GaussianDiffusionModel', 'CVAEModel'
    # Diffusion Model
    # 扩散噪声调度方式
    variance_schedule: str = "cosine",
    # 训练扩散模型的总扩散步数；推理时 DDIM 采样步数可另设
    n_diffusion_steps: int = 100,
    # True 表示模型预测噪声 epsilon，False 表示模型直接预测 x0
    predict_epsilon: bool = True,
    # 条件注入 UNet 的方式
    conditioning_type: str = "default",  # 'default', 'concatenate', 'attention'
    # Unet
    # UNet 基础通道维度
    unet_input_dim: int = 32,
    # UNet 多尺度通道倍率配置索引，对应 mpd.models.UNET_DIM_MULTS
    unet_dim_mults_option: int = 1,
    # CVAE
    # CVAE 潜变量维度，仅 generative_model_class='CVAEModel' 时使用
    cvae_latent_dim: int = 32,
    # CVAE KL loss 权重
    loss_cvae_kl_weight: float = 1e-1,
    ########################################################################
    # Training parameters
    # 训练 batch size
    batch_size: int = 128,
    # AdamW 学习率
    lr: float = 3e-4,
    # 是否裁剪梯度
    clip_grad: bool = False,
    # 总训练 step 数
    num_train_steps: int = 1_000_000,
    # 仅加载模型 state_dict 进行热启动；不会恢复 optimizer/step/RNG 等训练状态
    warm_start_checkpoint: str = "",
    # 是否维护 EMA 模型；推理通常使用 EMA checkpoint 更稳定
    use_ema: bool = True,
    # 是否使用自动混合精度训练
    use_amp: bool = False,
    # Summary parameters
    # 每隔多少 step 写 summary/可视化
    steps_til_summary: int = 5000 if DEBUG else 20000,
    # summary 回调类型
    summary_class: str = "SummaryTrajectoryGeneration",
    # 每隔多少 step 保存 checkpoint
    steps_til_ckpt: int = 5000 if DEBUG else 20000,
    ########################################################################
    # 训练设备，例如 cuda:0 或 cpu
    device: str = "cuda:0",
    # debug 模式会更频繁输出/可视化
    debug: bool = DEBUG,
    ########################################################################
    # MANDATORY
    # seed: int = int(time.time()),
    # 随机种子，用于复现实验
    seed: int = 1726484688,
    # 训练日志、checkpoint、summary 输出目录
    results_dir: str = "logs",
    ########################################################################
    # WandB
    # WandB 模式："online"、"offline" 或 "disabled"
    wandb_mode: str = "disabled" if DEBUG else WANDB_MODE,
    # WandB entity
    wandb_entity: str = WANDB_ENTITY,
    # WandB project 名称
    wandb_project: str = "test_train_bspline_diffusion",
    **kwargs,
):
    print()
    print("-" * 100)
    print(f"{dataset_subdir} -- {parametric_trajectory_class}")
    print("-" * 100)
    print()

    # Set random seed for reproducibility
    fix_random_seed(seed)

    device = get_torch_device(device=device)
    tensor_args = {"device": device, "dtype": torch.float32}

    ########################################################################
    # Planning task and dataset
    planning_task, train_subset, train_dataloader, val_subset, val_dataloader = get_planning_task_and_dataset(
        parametric_trajectory_class=parametric_trajectory_class,
        dataset_subdir=dataset_subdir,
        dataset_file_merged=dataset_file_merged,
        reload_data=reload_data,
        preload_data_to_device=preload_data_to_device,
        n_task_samples=n_task_samples,
        bspline_degree=bspline_degree,
        bspline_num_control_points_desired=bspline_num_control_points_desired,
        bspline_num_control_points_exact=bspline_num_control_points_exact,
        num_T_pts=num_T_pts,
        context_qs=context_qs,
        context_ee_goal_pose=context_ee_goal_pose,
        context_ee_goal_pose_bimanual=context_ee_goal_pose_bimanual,
        batch_size=batch_size,
        results_dir=results_dir,
        save_indices=True,
        tensor_args=tensor_args,
    )

    full_dataset = train_subset.dataset

    if kwargs.get("robot_model") == "marvin_bimanual":
        expected_context_q_dim = 14 if context_ee_goal_pose_bimanual else 28
        if (planning_task.robot.q_dim, full_dataset.state_dim, full_dataset.context_q_dim) != (
            14,
            14,
            expected_context_q_dim,
        ):
            raise ValueError(
                "Marvin dimensions must be robot/state=14 and joint context="
                f"{expected_context_q_dim}"
            )
        if not context_qs:
            raise ValueError("Marvin training requires q_start context")
        if context_ee_goal_pose != context_ee_goal_pose_bimanual:
            raise ValueError("Marvin EE conditioning must use the dual-slot Pika TCP encoder")

    if debug:
        full_dataset.render(
            task_id=0,
            render_joint_trajectories=True,
            render_robot_trajectories=True if full_dataset.planning_task.env.dim == 2 else False,
            render_n_robot_trajectories=50,
        )
        plt.show()

    ########################################################################
    # Model
    bimanual_network_variant = str(bimanual_network_variant).upper()
    if bimanual_network_variant not in {"A", "B", "C", "D"}:
        raise ValueError("bimanual_network_variant must be one of A, B, C, D")

    if context_ee_goal_pose_bimanual:
        common_context_args = dict(
            out_dim=context_combined_out_dim,
            n_layers=context_ee_goal_pose_n_layers,
            act=context_ee_goal_pose_act,
        )
        if bimanual_network_variant == "A":
            # Exact legacy baseline: raw 40-D vector followed by the original MLP.
            context_model = ContextModelMarvinDualEE(**common_context_args)
        elif bimanual_network_variant == "B":
            context_model = ContextModelMarvinStructuredEE(**common_context_args)
        else:
            attention_context_args = dict(
                **common_context_args,
                attention_heads=bimanual_attention_heads,
                attention_layers=bimanual_context_attention_layers,
                ff_multiplier=bimanual_attention_ff_multiplier,
                dropout=bimanual_attention_dropout,
            )
            if bimanual_network_variant == "C":
                context_model = ContextModelMarvinCrossArmEE(**attention_context_args)
            else:
                context_model = ContextModelMarvinCrossArmTokens(**attention_context_args)
    else:
        context_model_qs = None
        if context_qs:
            context_model_qs = ContextModelQs(
                in_dim=full_dataset.context_q_dim,
                out_dim=context_q_out_dim,
                n_layers=context_qs_n_layers,
                act=context_qs_act,
            )

        context_model_ee_pose_goal = None
        if context_ee_goal_pose:
            context_model_ee_pose_goal = ContextModelEEPoseGoal(
                out_dim=context_ee_goal_pose_out_dim,
                n_layers=context_ee_goal_pose_n_layers,
                act=context_ee_goal_pose_act,
            )

        context_model = None
        if not (context_model_qs is None and context_model_ee_pose_goal is None):
            context_model = ContextModelCombined(
                context_model_qs=context_model_qs,
                context_model_ee_pose_goal=context_model_ee_pose_goal,
                out_dim=context_combined_out_dim,
            )

    diffusion_configs = dict(
        variance_schedule=variance_schedule,
        n_diffusion_steps=n_diffusion_steps,
        predict_epsilon=predict_epsilon,
    )

    cvae_configs = dict(
        cvae_latent_dim=cvae_latent_dim,
    )

    unet_configs = dict(
        state_dim=full_dataset.state_dim,
        n_support_points=full_dataset.n_learnable_control_points,
        unet_input_dim=unet_input_dim,
        dim_mults=UNET_DIM_MULTS[unet_dim_mults_option],
        conditioning_type=conditioning_type if context_model is not None else "None",
        conditioning_embed_dim=context_model.out_dim if context_model is not None else None,
    )

    denoise_model_class = TemporalUnet
    if context_ee_goal_pose_bimanual and bimanual_network_variant == "D":
        if generative_model_class != "GaussianDiffusionModel":
            raise ValueError("bimanual network variant D is implemented for GaussianDiffusionModel only")
        denoise_model_class = CoupledBimanualTemporalUnet
        unet_configs.update(
            context_token_dim=context_combined_out_dim,
            attention_num_heads=bimanual_attention_heads,
            attention_layers=bimanual_denoiser_attention_layers,
            attention_ff_multiplier=bimanual_attention_ff_multiplier,
            attention_dropout=bimanual_attention_dropout,
        )

    if context_ee_goal_pose_bimanual:
        print(
            f"Marvin network variant {bimanual_network_variant}: "
            f"context={type(context_model).__name__}, denoiser={denoise_model_class.__name__}"
        )

    model = get_model(
        model_class=generative_model_class,
        denoise_fn=denoise_model_class(**unet_configs),
        context_model=context_model,
        tensor_args=tensor_args,
        **cvae_configs,
        **diffusion_configs,
        **unet_configs,
    )
    if warm_start_checkpoint:
        load_warm_start_weights(model, warm_start_checkpoint)

    ########################################################################
    # Loss
    if generative_model_class == "GaussianDiffusionModel":
        loss_class = "GaussianDiffusionLoss"
    elif generative_model_class == "CVAEModel":
        loss_class = "CVAELoss"
    else:
        raise ValueError(f"Unknown generative_model_class: {generative_model_class}")

    loss_fn = val_loss_fn = get_loss(loss_class=loss_class, loss_cvae_kl_weight=loss_cvae_kl_weight)

    ########################################################################
    # Summary
    summary_fn = get_summary(
        summary_class=summary_class,
        debug=debug,
    )

    ########################################################################
    # Train
    trainer.train(
        model=model,
        train_dataloader=train_dataloader,
        train_subset=train_subset,
        val_dataloader=val_dataloader,
        val_subset=val_subset,
        planning_task=planning_task,
        num_train_steps=num_train_steps,
        epochs=get_num_epochs(num_train_steps, batch_size, len(train_subset)),
        model_dir=results_dir,
        summary_fn=summary_fn,
        lr=lr,
        lr_scheduler=False,
        loss_fn=loss_fn,
        val_loss_fn=val_loss_fn,
        steps_til_summary=steps_til_summary,
        steps_til_checkpoint=steps_til_ckpt,
        clip_grad=clip_grad,
        early_stopper_patience=-1,
        use_ema=use_ema,
        use_amp=use_amp,
        debug=debug,
        tensor_args=tensor_args,
    )


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(experiment)
