import os

from src import BeatGANsAutoencoderConfig, BeatGANsUNetConfig, SimpleModelConfig, VAEModelConfig
from src._types import *
from src.config import BaseConfig, TorchInstanceConfig, ClfConfig
from src.models.clf.ms_clf import MSClfNetConfig
from src.models.dae.architecture.latentnet import MLPSkipNetConfig
from src.models.dae.diffusion.base import get_named_beta_schedule
from src.models.dae.diffusion.diffusion import space_timesteps, SpacedDiffusionBeatGansConfig
from src.models.dae.diffusion.resample import UniformSampler
from variables import MS_MAIN_TYPE, DATA_DIR

"""
    Copyright (C) 2024 LeratLy - All Rights Reserved
    You may use, distribute and modify this code under the
    terms of the MIT license.
    You should have received a copy of the MIT license with
    this file.

    Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology.
    Code of VISTEC has been used and modified in this file under terms of the MIT license.
"""


def _chp64_base_conf_small() -> BaseConfig:
    """
    Create base configuration for all autoencoder models that use chP data
    This config does not depend on a model and can not directly be used for training
    :return:
    """
    conf = BaseConfig()
    conf.data = {
        "name": 'chP3D_test_small',
        "type": DataType.nii,
    }
    conf.img_size = 64  # 192
    return conf


def _chp96_base_conf() -> BaseConfig:
    """
    Create base configuration for all autoencoder models that use chP data
    This config does not depend on a model and can not directly be used for training
    :return:
    """
    conf = BaseConfig()
    conf.data = {
        "name": 'chP3D_tune_70',
        "type": DataType.nii,
    }
    conf.img_size = 96
    return conf


def _chp64_cvae_base_conf():
    conf = _chp96_base_conf()
    conf.log_interval = 50
    conf.batch_size = 1
    conf.num_epochs = 150
    conf.device = "cuda"
    conf.lr = 0.0001
    conf.model_name = ModelName.conditional_variational_autoencoder
    conf.model = "src.models.vae.architecture.autoencoder.VAEModel"
    conf.eval.eval_training_every_epoch = 10
    conf.checkpoint["save_every_epoch"] = 5
    conf.accum_batches = 8
    conf.dropout = 0
    return conf


def _chp96_diffae_base_conf():
    conf = _chp96_base_conf()
    conf.log_interval = 10
    conf.batch_size = 2
    conf.num_epochs = 450
    conf.device = "cuda"
    conf.lr = 1e-3
    conf.model_name = ModelName.beatgans_autoencoder
    conf.model = "src.models.dae.architecture.unet_autoencoder.BeatGANsAutoencoderModel"
    conf.eval.eval_training_every_epoch = 10
    conf.eval.num_samples = 2
    conf.eval.num_evals = 2
    conf.eval.num_reconstructions = 4
    conf.clip_denoised = True
    conf.accum_batches = 4
    conf.dropout = 0.1
    conf.patience = 40
    conf.preprocess_img = "crop"
    conf.classes = MS_MAIN_TYPE
    conf.ema_decay = 0.9
    conf.data_parallel = False
    conf.add_running_metrics = ["reconstruction"]
    return conf


def _chp96_diffae_model_conf(conf):
    assign_model_config(conf)
    conf.model_conf.net.resnet_two_cond = True
    conf.model_conf.net.grad_checkpoint = True
    conf.model_conf.attn_checkpoint = True
    conf.model_conf.net.ch = 32  # 64
    conf.model_conf.net.dropout = 0.1
    conf.model_conf.net.attn = (12,)
    conf.model_conf.net.ch_mult = (1, 2, 2, 4)
    conf.model_conf.net_enc.attn = (12,)
    conf.model_conf.net_enc.ch_mult = (1, 2, 2, 4, 4)
    conf.model_conf.net_enc.dropout = 0
    conf.model_conf.net_enc.pool = 'adaptivenonzero'
    return conf


def _chp64_cvae_model_conf(conf):
    assign_model_config(conf)
    conf.model_conf.in_channels = 1
    conf.model_conf.ch = 16
    conf.model_conf.num_layers = 4
    conf.model_conf.latent_size = 512
    conf.model_conf.num_classes = 2
    conf.dropout = 0
    conf.model_conf.num_target_emb_channels = 1
    return conf


def _chp96_diffae_diffusion_conf(conf):
    conf.diffusion_conf.gen_type = GenerativeType.ddim
    conf.diffusion_conf.T_eval = 20
    conf.diffusion_conf.T = 1000
    conf.diffusion_conf.beta_scheduler = "cosine"
    conf.diffusion_conf.T_sampler = "uniform"
    return conf


# ------------ latent config -------------------------
def get_final_base_config_latent():
    conf = chp96_diffae_xor_conf()
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.accum_batches = 1
    conf.num_epochs = 5000
    conf.patience = 160
    conf.use_early_stop = True
    conf.device = "cuda"
    conf.preprocess_img = None
    conf.create_checkpoint = True
    conf.eval.eval_training_every_epoch = 100
    conf.eval.eval_epoch_metrics_val_samples = False
    conf.checkpoint['dir'] = os.path.join(DATA_DIR, "final_models", "checkpoints")
    conf.logging_dir = os.path.join(DATA_DIR, "final_models", "logging")
    conf.run_dir = os.path.join(DATA_DIR, "final_models", "runs")
    conf.eval.num_samples = 2
    conf.eval.num_evals = 10
    conf.eval.num_reconstructions = 2
    conf.clip_denoised = True
    assign_model_config(conf)

    conf.scheduler = TorchInstanceConfig(
        instance_type="torch.optim.lr_scheduler.ReduceLROnPlateau",
        settings=['min', 0.5, 40, 0.0001, 'rel', 0, 1e-8]
    )
    conf = chp96_diffae_latent_conf(conf)
    conf = chp96_diffae_latent_training_conf(conf)
    conf.model_conf.net_enc.ch_mult = (1, 2, 2, 4, 4)
    conf.model_conf.net.ch_mult = (1, 2, 2, 4)
    conf.model_conf.net.ch = 32
    conf.model_conf.last_act = Activation.sigmoid
    conf.model_conf.net.attn = (12,)
    conf.model_conf.net_enc.attn = (12,)
    conf.model_conf.net.resnet_two_cond = True
    # diffusion conf is for x_T
    conf.diffusion_conf.gen_type = GenerativeType.ddim
    conf.diffusion_conf.T_eval = 20
    conf.diffusion_conf.noise_type = NoiseType.xor
    conf.diffusion_conf.loss_type = LossType.bce
    # latent_diffusion conf is for cond
    conf.latent_diffusion_conf.loss_type = LossType.l1

    conf.model_conf.net_enc.ch_mult = (1, 2, 2, 4, 4)
    conf.model_conf.net.ch_mult = (1, 2, 2, 4)
    conf.model_conf.net.ch = 32
    conf.model_conf.last_act = Activation.sigmoid
    conf.model_conf.net.attn = (12,)
    conf.model_conf.net_enc.attn = (12,)
    conf.model_conf.net.resnet_two_cond = True
    conf.model_conf.latent_net_conf.target_bias = 1
    conf.model_conf.latent_net_conf.use_target = True
    return conf


# Latent autoencoder is only used for unconditional sampling and can be found in the diffae
def chp96_diffae_latent_conf(conf):
    conf.train_mode = TrainMode.latent_diffusion

    conf.latent_diffusion_conf.noise_type = NoiseType.gaussian
    conf.latent_diffusion_conf.loss_type = LossType.l1
    conf.latent_diffusion_conf.T = 1000
    conf.latent_diffusion_conf.T_eval = 20
    conf.latent_diffusion_conf.beta_scheduler = "const0.008"
    conf.latent_diffusion_conf.num_classes = 2

    conf.model_conf.latent_net_conf.znormalize = True
    conf.model_conf.latent_net_conf.num_layers = 10
    conf.model_conf.latent_net_conf.skip_layers = list(range(1, 10))
    conf.model_conf.latent_net_conf.num_hid_channels = 2048
    conf.model_conf.latent_net_conf.use_norm = True
    conf.model_conf.latent_net_conf.condition_bias = 1
    conf.model_conf.latent_net_conf.num_classes = 2
    conf.model_conf.latent_net_conf.target_bias = 1
    conf.model_conf.latent_net_conf.use_target = True

    # diffusion conf is for x_T
    conf.diffusion_conf.gen_type = GenerativeType.ddim
    conf.diffusion_conf.T_eval = 20
    conf.diffusion_conf.noise_type = NoiseType.xor
    conf.diffusion_conf.loss_type = LossType.bce
    # latent_diffusion conf id for cond
    conf.latent_diffusion_conf.loss_type = LossType.l1
    return conf


def chp96_diffae_latent_training_conf(conf):
    conf.model_type = ModelType.latent_diffusion
    conf.model_conf.latent_net_type = LatentNetType.skip

    conf.num_epochs = 450
    conf.lr = 1e-3
    conf.accum_batches = 4
    conf.num_classes = 2
    conf.eval.eval_training_every_epoch = 10
    conf.checkpoint["save_every_epoch"] = -1
    conf.checkpoint["resume_skip_keys"] = ["model.latent_net", "ema_model.latent_net"]
    conf.checkpoint["resume_optimizer"] = False
    conf.checkpoint["resume_scheduler"] = False
    return conf


# ----------------------------------------------------


def chp96_diffae_xor_conf():
    conf = _chp96_diffae_diffusion_conf(_chp96_diffae_model_conf(_chp96_diffae_base_conf()))
    conf.diffusion_conf.noise_type = NoiseType.xor
    conf.diffusion_conf.loss_type = LossType.bce
    conf.model_conf.last_act = Activation.sigmoid
    # conf.checkpoint["name"] = \
    #     "path/to/project/bs_gen_models/tests/testing_checkpoints/xor_deep_base_20250510_201726_best"
    conf.name = "xor_deep"
    return conf


def chp96_diffae_gaussian_conf():
    conf = _chp96_diffae_diffusion_conf(_chp96_diffae_model_conf(_chp96_diffae_base_conf()))
    conf.name = "gaussian_deep"
    return conf


def chp96_cvae_bernoulli_conf():
    """
    With bernoulli decoder
    """
    conf = _chp64_cvae_model_conf(_chp64_cvae_base_conf())
    conf.loss_type = LossType.bce
    conf.model_conf.last_act = Activation.sigmoid
    return conf


def chp96_cvae_gaussian_conf():
    """
    With gaussian decoder
    """
    conf = _chp64_cvae_model_conf(_chp64_cvae_base_conf())
    conf.loss_type = LossType.bce
    conf.model_conf.last_act = Activation.tanh
    return conf


def assign_model_config(conf: BaseConfig):
    """
    Creates and assigns the model config to the config object.
    The model config contains all settings needed to create the model.
    :param conf: Base config which should be enhanced with a model config.
    """
    # Model 1: BeatGANs diffusion (ddpm with UNet only)
    if conf.model_name == ModelName.beatgans_ddpm:
        conf.diffusion_conf = SpacedDiffusionBeatGansConfig()
        conf.latent_diffusion_conf = SpacedDiffusionBeatGansConfig()

        conf.model_type = ModelType.ddpm
        conf.model_conf = BeatGANsUNetConfig()
        conf.model_conf.net.embed_channels = conf.style_ch
        conf.model_conf.dims = conf.dims
    # Model 2: BeatGANs autoencoder
    elif conf.model_name == ModelName.beatgans_autoencoder:
        conf.diffusion_conf = SpacedDiffusionBeatGansConfig()
        conf.latent_diffusion_conf = SpacedDiffusionBeatGansConfig()

        conf.model_type = ModelType.autoencoder
        conf.model_conf = BeatGANsAutoencoderConfig()
        conf.model_conf.dims = conf.dims
        assign_latent_config(conf, conf.net_latent_net_type)
    # Model 3: Simple Model (for testing)
    elif conf.model_name == ModelName.simple_model:
        conf.model_conf = SimpleModelConfig()
    # Model 4: Classifier that predicts MS class
    elif conf.model_name == ModelName.ms_clf:
        conf.model_conf = MSClfNetConfig()
        conf.model_conf.input_size = conf.img_size
        conf.model_conf.num_classes = conf.num_classes
        conf.model_conf.dropout = conf.dropout
        conf.model_conf.dims = conf.dims
    # Model 5: Conditional autoencoder
    elif conf.model_name == ModelName.conditional_variational_autoencoder:
        conf.model_conf = VAEModelConfig()
        conf.model_conf.dims = conf.dims
        conf.model_conf.dropout = conf.dropout
    else:
        raise NotImplementedError(conf.model_name)

    # Also always pass specific arguments from BaseConfig
    conf.model_conf.img_size = conf.img_size
    conf.model_conf.dropout = conf.dropout


def assign_latent_config(base_conf: BaseConfig, latent_net_type):
    """
    Set up the config for the latent net
    :param base_conf: the base config for the experiment
    :param latent_net_type: the type of latent net which is going to be used
    :return:
    """
    if latent_net_type == LatentNetType.skip:
        latent_net_conf = MLPSkipNetConfig()
        latent_net_conf.num_channels = base_conf.style_ch
        latent_net_conf.latent_type = latent_net_type
        latent_net_conf.num_classes = base_conf.num_classes
    else:
        raise NotImplementedError()
    base_conf.model_conf.latent_net_conf = latent_net_conf
    return latent_net_conf


def assign_diffusion_conf(conf: BaseConfig, T=None, latent=False):
    """
    Assigns a diffusion config to the config which can be used to create the diffusion part of the dae

    :param conf: The config of the diffusion autoencoder
    :param T: Maximum number of timesteps, default to None
    :type T: int
    :param latent: Whether latent DDIM should be used (for unconditional sampling), default to False
    :type latent: bool
    :return: configuration for the diffusion part of the model
    :rtype: :class:`.SpacedDiffusionBeatGansConfig`
    """
    # can use t < conf.t for evaluation
    # follows the guided-diffusion repo conventions
    # t's are evenly spaced
    if conf.diffusion_conf.gen_type == GenerativeType.ddpm:
        section_counts = [T]
    elif conf.diffusion_conf.gen_type == GenerativeType.ddim:
        section_counts = f'ddim{T}'
    else:
        raise NotImplementedError()
    diff_conf = conf.latent_diffusion_conf if latent else conf.diffusion_conf
    # latent's model is always ddpm
    diff_conf.model_type = ModelType.latent_diffusion if latent else conf.model_type
    # latent models share beta scheduler and full T
    diff_conf.use_timesteps = space_timesteps(num_timesteps=diff_conf.T,
                                              section_counts=section_counts)
    diff_conf.betas = get_named_beta_schedule(diff_conf.beta_scheduler, diff_conf.T)
    diff_conf.fp16 = conf.fp16
    diff_conf.device = conf.device
    diff_conf.num_classes = conf.num_classes if diff_conf.num_classes is None else diff_conf.num_classes
    return diff_conf


def make_t_sampler(conf):
    """
    Creates a t sampler, currently only the uniform sampler is implemented.
    The T_sampler specifies how different t's are going to be drawn from 0 to T
    :param conf:
    :return:
    """
    if conf.diffusion_conf.T_sampler == 'uniform':
        return UniformSampler(conf.diffusion_conf.T)
    else:
        raise NotImplementedError()
