import os
from functools import partial

from ray import tune

from run_models.model_templates import assign_model_config
from src._types import ModelGradType, LossType, DataType, ModelName, NoiseType, GenerativeType, Activation
from src.config import BaseConfig, TorchInstanceConfig
from src.models.trainer import Trainer
from variables import MS_MAIN_TYPE, ROOT_DIR


def main(config, model_type="gaussian"):
    result = tune.run(
        partial(train_diffae, model_type=model_type),
        config=config,
        resources_per_trial={"gpu": 1}
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['reconstruction_loss']}")
    result.dataframe().to_csv("all_trials.csv")


def train_diffae(config, model_type="gaussian"):
    conf = get_base_config()
    conf.diffusion_conf.beta_scheduler = "cosine"
    conf.diffusion_conf.model_grad_type = ModelGradType.eps
    conf.model_conf.net_enc.ch_mult = config["ch_mult"]["enc_channel_mult"]
    conf.model_conf.net.ch_mult = config["ch_mult"]["ch_mult"]
    conf.model_conf.net.ch = config["ch"]
    conf.randomWeightedTrainSet = True

    if conf.model_conf.net.ch_mult == (1, 2, 2, 4):
        conf.model_conf.net.grad_checkpoint = False
        conf.model_conf.attn_checkpoint = False

    if model_type == "xor":
        conf.name = "tune_xor"
        conf.model_conf.last_act = Activation.sigmoid
        conf.diffusion_conf.noise_type = NoiseType.xor
        conf.diffusion_conf.loss_type = LossType.bce
    elif model_type == "gaussian":
        conf.name = "tune_gaussian"
        conf.diffusion_conf.noise_type = NoiseType.gaussian
        conf.diffusion_conf.loss_type = config["loss_type"]
    conf.ema_decay = 0.9
    conf.name = f"diffae_wrts"
    conf.__post_init__()

    trainer = Trainer(conf)
    trainer.train()
    trainer.close()


def get_base_config():
    conf = BaseConfig()
    conf.log_interval = 10
    conf.data = {
        "name": 'chP3D_tune_70',
        "type": DataType.nii,
    }
    conf.img_size = 96
    conf.batch_size = 2
    conf.lr = 1e-4 # 1e-3
    conf.accum_batches = 6
    conf.num_epochs = 900
    conf.patience = 60
    conf.use_early_stop = True
    conf.device = "cuda"
    conf.preprocess_img = "crop"

    conf.create_checkpoint = True
    conf.checkpoint["save_every_epoch"] = -1
    conf.checkpoint['dir'] = os.path.join(ROOT_DIR, "checkpoints")
    conf.logging_dir = os.path.join(ROOT_DIR, "logging")
    conf.run_dir = os.path.join(ROOT_DIR, "runs")
    conf.model_name = ModelName.beatgans_autoencoder
    conf.model = "src.models.dae.architecture.unet_autoencoder.BeatGANsAutoencoderModel"
    conf.add_running_metrics = ["reconstruction"]
    conf.eval.eval_training_every_epoch = 15
    conf.eval.num_samples = 2
    conf.eval.num_evals = 2
    conf.eval.num_reconstructions = 2
    conf.classes = MS_MAIN_TYPE

    conf.clip_denoised = True
    # conf.checkpoint["save_every_epoch"] = 25
    assign_model_config(conf)
    conf.model_conf.net.resnet_two_cond = True
    conf.model_conf.net.grad_checkpoint = True
    conf.model_conf.attn_checkpoint = True
    conf.model_conf.net_enc.pool = 'adaptivenonzero'

    conf.model_conf.net.attn = (12,)
    conf.model_conf.net_enc.attn = (12,)
    conf.diffusion_conf.gen_type = GenerativeType.ddim
    conf.diffusion_conf.T_eval = 20
    conf.ema_decay = 0.9
    conf.data_parallel = False
    conf.diffusion_conf.T = 1000
    conf.diffusion_conf.loss_type_x_start = LossType.bce
    conf.diffusion_conf.T_sampler = "uniform"

    conf.scheduler = TorchInstanceConfig(
        instance_type="torch.optim.lr_scheduler.ReduceLROnPlateau",
        settings=['min', 0.5, 10, 0.0001, 'rel', 0, 1e-8]
    )
    return conf


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    config_xor = {
        "ch": tune.grid_search([32]),
        "ch_mult": tune.grid_search([
            {
                "enc_channel_mult": (1, 2, 2, 4, 4),
                "ch_mult": (1, 2, 2, 4)
            },
        ]),
    }
    config_gaussian = {
        **config_xor,
        "loss_type": tune.choice([LossType.l1, LossType.mse]),
    }
    main(config_xor, model_type="xor")
