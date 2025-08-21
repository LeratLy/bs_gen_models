
"""
    Copyright (C) 2024 LeratLy - All Rights Reserved
    You may use, distribute and modify this code under the
    terms of the MIT license.
    You should have received a copy of the MIT license with
    this file.

    Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology.
    Code of VISTEC has been used and modified in this file under terms of the MIT license.
"""
import os

from run_models.dae.tune_hyperparams import get_base_config
from run_models.dae.tune_hyperparams_latent import get_base_config_latent
from src._types import GenerativeType, LossType, ModelGradType, Activation, NoiseType
from src.models.trainer import Trainer
from variables import DATA_DIR, MS_TYPES, MODEL_DIR, ROOT_DIR


def run_latent_sample():
    conf = get_base_config_latent()
    conf.model_conf.net_enc.ch_mult = (1, 2, 2, 4, 4)
    conf.model_conf.net.ch_mult = (1, 2, 2, 4)
    conf.model_conf.net.ch = 32
    conf.model_conf.net.attn = (12,)
    conf.model_conf.net_enc.attn = (12,)
    conf.diffusion_conf.gen_type = GenerativeType.ddim
    conf.diffusion_conf.T_eval = 20
    conf.model_conf.net.resnet_two_cond = True
    conf.model_conf.latent_net_conf.num_layers = 20
    conf.model_conf.latent_net_conf.skip_layers = list(range(1, 20))
    conf.model_conf.latent_net_conf.num_hid_channels = 2048

    # diffusion conf is for x_T
    conf.diffusion_conf.gen_type = GenerativeType.ddim
    conf.diffusion_conf.T_eval = 20
    conf.diffusion_conf.noise_type = NoiseType.xor
    conf.diffusion_conf.loss_type = LossType.bce
    conf.model_conf.last_act = Activation.sigmoid
    # latent_diffusion conf id for cond
    conf.latent_diffusion_conf.loss_type = LossType.mse

    conf.checkpoint["save_every_epoch"] = -1
    conf.checkpoint["resume_skip_keys"] = ["model.latent_net", "ema_model.latent_net"]
    conf.checkpoint["resume_optimizer"] = True
    conf.checkpoint["resume_scheduler"] = True
    conf.__post_init__()
    trainer = Trainer(conf)
    trainer.conf.eval.eval_epoch_metrics_val_samples = False
    trainer.conf.eval.eval_training_every_epoch = 50
    conf.checkpoint["name"] = os.path.join(MODEL_DIR, "final_latent_5_cond_encoder_alpha_20_hidden1024_latentdiffusion_20250723_195439_best")
    trainer.get_wrapper_model().create_samples_per_class(2, 2, 1)


def run_latent_train():
    conf = get_base_config_latent()
    conf.model_conf.net_enc.ch_mult = (1, 2, 2, 4, 4)
    conf.model_conf.net.ch_mult = (1, 2, 2, 4)
    conf.model_conf.net.ch = 32
    conf.model_conf.net.attn = (12,)
    conf.model_conf.net_enc.attn = (12,)
    conf.diffusion_conf.gen_type = GenerativeType.ddim
    conf.diffusion_conf.T_eval = 20
    conf.model_conf.net.resnet_two_cond = True
    conf.model_conf.latent_net_conf.num_layers = 20
    conf.model_conf.latent_net_conf.skip_layers = list(range(1, 20))
    conf.model_conf.latent_net_conf.num_hid_channels = 2048

    # diffusion conf is for x_T
    conf.diffusion_conf.gen_type = GenerativeType.ddim
    conf.diffusion_conf.T_eval = 20
    conf.diffusion_conf.noise_type = NoiseType.xor
    conf.diffusion_conf.loss_type = LossType.bce
    conf.model_conf.last_act = Activation.sigmoid
    # latent_diffusion conf id for cond
    conf.latent_diffusion_conf.loss_type = LossType.mse

    conf.checkpoint["resume_skip_keys"] = ["model.latent_net", "ema_model.latent_net"]
    conf.checkpoint["resume_optimizer"] = False
    conf.checkpoint["resume_scheduler"] = False
    conf.create_checkpoint = False
    conf.checkpoint["name"] = os.path.join(MODEL_DIR, "tune_xor_base_20250710_215919_plus_dropout_cond_encoder_base_20250721_231432_best")

    conf.__post_init__()
    trainer = Trainer(conf)
    trainer.conf.eval.eval_epoch_metrics_val_samples = False
    trainer.conf.eval.eval_training_every_epoch = 1
    trainer.train()

def run_sample_base():
    conf = get_base_config()
    conf.batch_size = 2
    conf.diffusion_conf.beta_scheduler = "cosine"
    conf.diffusion_conf.model_grad_type = ModelGradType.eps
    # conf.randomWeightedTrainSet = True
    conf.model_conf.net_enc.ch_mult = (1, 2, 2, 4, 4)
    conf.model_conf.net.ch_mult = (1, 2, 2, 4)
    conf.model_conf.net.ch = 32
    conf.add_running_metrics = []
    conf.num_classes = 5
    conf.classes = MS_TYPES
    conf.data["name"] = "chP3D_tune_5_classes_70"

    conf.model_conf.net.grad_checkpoint = False
    conf.model_conf.attn_checkpoint = False

    conf.model_conf.last_act = Activation.sigmoid
    conf.diffusion_conf.noise_type = NoiseType.xor
    conf.diffusion_conf.loss_type = LossType.bce
    conf.checkpoint["resume_scheduler"] = False
    conf.checkpoint["resume_optimizer"] = False
    conf.ema_decay = 0.9

    conf.checkpoint["name"] = os.path.join(MODEL_DIR, "tune_xor_base_20250710_215919_plus_dropout_cond_encoder_base_20250721_231432_best")
    conf.checkpoint['dir'] = os.path.join(ROOT_DIR, "checkpoints")
    conf.logging_dir = os.path.join(ROOT_DIR, "logging")
    conf.run_dir = os.path.join(ROOT_DIR, "runs")
    conf.name= "test"
    conf.patience = 20
    conf.dropout = 0.1
    conf.num_classes = 5
    conf.__post_init__()
    conf.eval.eval_training_every_epoch = -1

    trainer = Trainer(conf)
    trainer.train()
    trainer.close()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    run_sample_base()

