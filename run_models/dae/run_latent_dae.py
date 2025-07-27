import os

from run_models.clf_templates import get_chP96_clf_2cond_conf
from run_models.dae.tune_hyperparams import get_base_config
from run_models.model_templates import assign_model_config, chp96_diffae_latent_training_conf, \
    chp96_diffae_xor_conf, \
    chp96_diffae_latent_conf
from src._types import LossType, ModelName, GenerativeType, NoiseType, Activation
from src.config import ClfConfig, TorchInstanceConfig
from src.models.trainer import Trainer
from variables import DATA_DIR

def main(config):
    conf = get_base_config_latent()
    conf.model_conf.latent_net_conf.num_layers = config["num_layers"]
    conf.model_conf.latent_net_conf.skip_layers = list(range(1, config["num_layers"]))
    conf.model_conf.latent_net_conf.num_hid_channels = config["num_hid_channels"]
    conf.model_conf.latent_net_conf.class_znormalize = False
    conf.model_conf.latent_net_conf.znormalize = True
    conf.model_conf.latent_net_conf.use_target = True
    conf.model_conf.latent_net_conf.shift_target = True
    conf.model_conf.latent_net_conf.scale_target_alpha = True
    conf.model_conf.latent_net_conf.num_classes = 5
    conf.num_classes = 5

    # latent_diffusion conf id for cond
    conf.latent_diffusion_conf.loss_type = config["loss_type"]
    # conf.checkpoint["name"] = os.path.join(DATA_DIR, "final_models", "checkpoints", "diffae_wrts_base_20250712_220934_best")
    conf.checkpoint["name"] = os.path.join(DATA_DIR, "final_models", "checkpoints", "tune_xor_base_20250710_215919_best")
    # conf.checkpoint["name"] = os.path.join(DATA_DIR, "final_models", "checkpoints", "diffae_cond_encoder_base_20250712_220934_best_base_20250720_093034_best")
    # conf.name = f"latent_no_wrs_class_spec_norm_layers_add_target_shift_{conf.model_conf.latent_net_conf.num_layers}_hidden{conf.model_conf.latent_net_conf.num_hid_channels}_loss{conf.latent_diffusion_conf.loss_type}"
    # conf.name = f"latent_5_class_{conf.model_conf.latent_net_conf.num_layers}_hidden{conf.model_conf.latent_net_conf.num_hid_channels}_loss{conf.latent_diffusion_conf.loss_type}"
    # conf.name = f"latent_5_shift{conf.model_conf.latent_net_conf.num_layers}_hidden{conf.model_conf.latent_net_conf.num_hid_channels}_loss{conf.latent_diffusion_conf.loss_type}"
    conf.name = f"latent_5_alpha_{conf.model_conf.latent_net_conf.num_layers}_hidden{conf.model_conf.latent_net_conf.num_hid_channels}_loss{conf.latent_diffusion_conf.loss_type}"
    conf.latent_infer_path = config["latent_infer_path"]
    conf.create_checkpoint = False
    conf.scheduler = TorchInstanceConfig(
        instance_type="torch.optim.lr_scheduler.ReduceLROnPlateau",
        settings=['min', 0.5, 10, 0.0001, 'rel', 0, 1e-8]
    )
    conf.__post_init__()
    conf.checkpoint["resume_optimizer"] = False
    conf.checkpoint["resume_scheduler"] = False
    trainer = Trainer(conf)
    trainer.conf.eval.eval_epoch_metrics_val_samples = False
    trainer.conf.eval.eval_training_every_epoch = -1
    conf.patience = 160
    conf.num_epochs = 5000
    trainer.train()
    trainer.close()


def get_base_config_latent():
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
    conf.clf_conf = ClfConfig(
        "path/to//data/final_models/checkpoints/analysis_final_ms_clf_base_20250711_101044_best",
        ModelName.ms_clf,
        get_chP96_clf_2cond_conf(),
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
    return conf


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    # 1. infer latent
    conf = get_base_config()
    conf.model_conf.net_enc.ch_mult = (1, 2, 2, 4, 4)
    conf.model_conf.net.ch_mult = (1, 2, 2, 4)
    conf.model_conf.net.ch = 32
    conf.dropout = 0.1
    # conf.checkpoint["name"] = os.path.join(DATA_DIR, "final_models", "checkpoints", "diffae_wrts_base_20250712_220934_best")
    # latent_infer_path = os.path.join(DATA_DIR, "final_models", "checkpoints", "latents_wrs.pkl")
    #
    # conf.checkpoint["name"] = os.path.join(DATA_DIR, "final_models", "checkpoints", "tune_xor_base_20250710_215919_best")
    # latent_infer_path = os.path.join(DATA_DIR, "final_models", "checkpoints", "latents_no_wrs.pkl")

    # conf.checkpoint["name"] = os.path.join(DATA_DIR, "final_models", "checkpoints", "diffae_wrts_base_20250712_220934_best")
    # conf.data["name"] = "chP3D_tune_5_classes_70"
    #
    conf.model_conf.num_classes = 5
    conf.num_classes = 5
    conf.checkpoint["name"] = os.path.join(DATA_DIR, "final_models", "checkpoints", "tune_xor_base_20250710_215919_plus_dropout_cond_encoder")
    conf.data["name"] = "chP3D_tune_5_classes_70"
    # latent_infer_path = os.path.join(DATA_DIR, "final_models", "checkpoints", "latents_class_spec_norm.pkl")
    latent_infer_path = os.path.join(DATA_DIR, "final_models", "checkpoints", "final_latents_cond_encoder_class_spec_norm.pkl")
    conf.__post_init__()
    trainer = Trainer(conf)
    trainer.infer_latents(save_path=latent_infer_path)
    trainer.close()

    # 2. tune
    config = {
        "num_layers": 10,  # 10, 20
        "num_hid_channels": 2048,  # 1024, 2048
        "loss_type":LossType.l1,
        "latent_infer_path": latent_infer_path,
    }
    main(config)
