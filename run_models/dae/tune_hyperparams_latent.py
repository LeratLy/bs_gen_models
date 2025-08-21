import os
from functools import partial

from ray import tune

from run_models.clf_templates import get_chP96_clf_2cond_conf
from run_models.model_templates import assign_model_config, chp96_diffae_latent_training_conf, chp96_diffae_xor_conf, \
    chp96_diffae_latent_conf
from src._types import LossType, ModelName, GenerativeType, NoiseType, Activation
from src.config import ClfConfig, TorchInstanceConfig
from src.models.trainer import Trainer
from variables import MODEL_DIR, ROOT_DIR


def main(config):
    result = tune.run(
        partial(train_diffae),
        config=config,
        resources_per_trial={"gpu": 1}
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    result.dataframe().to_csv("all_trials.csv")


def train_diffae(config):
    if config["num_layers"] == 10 and config["num_hid_channels"] == 1024:
        print("Skipping diffae 10 1024")
        return
    conf = get_base_config_latent()
    conf.model_conf.latent_net_conf.num_layers = config["num_layers"]
    conf.model_conf.latent_net_conf.skip_layers = list(range(1, config["num_layers"]))
    conf.model_conf.latent_net_conf.num_hid_channels = config["num_hid_channels"]
    # latent_diffusion conf id for cond
    conf.latent_diffusion_conf.loss_type = config["loss_type"]
    conf = cond_encoder_shift_scale(conf)
    conf.name = f"final_latent_5_cond_encoder_alpha_{conf.model_conf.latent_net_conf.num_layers}_hidden{conf.model_conf.latent_net_conf.num_hid_channels}"
    conf.latent_infer_path = config["latent_infer_path"]
    conf.create_checkpoint = False

    conf.model_conf.latent_net_conf.class_znormalize = True
    conf.model_conf.latent_net_conf.znormalize = False
    conf.model_conf.latent_net_conf.use_target = True

    #  For alpha scaling (+shift)
    conf.model_conf.latent_net_conf.shift_target = True
    conf.model_conf.latent_net_conf.scale_target_alpha = True

    conf.model_conf.latent_net_conf.num_classes = 5
    conf.num_classes = 5
    conf.__post_init__()
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
    conf.checkpoint['dir'] = os.path.join(ROOT_DIR, "checkpoints")
    conf.logging_dir = os.path.join(ROOT_DIR, "logging")
    conf.run_dir = os.path.join(ROOT_DIR, "runs")
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
        os.path.join(MODEL_DIR, "analysis_final_ms_clf_base_20250711_101044_best"),
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


def cond_encoder_shift_scale(conf):
    conf.model_conf.num_classes = 5
    conf.num_classes = 5
    conf.model_conf.enc_merge_time_and_cond_embedding = True
    conf.checkpoint["name"] = os.path.join(MODEL_DIR, "tune_xor_base_20250710_215919_plus_dropout_cond_encoder_base_20250721_231432_best")
    return conf


def cond_encoder_scale(conf):
    conf.model_conf.num_classes = 5
    conf.num_classes = 5
    conf.model_conf.enc_merge_time_and_cond_embedding = False
    conf.checkpoint["name"] = os.path.join(MODEL_DIR, "tune_xor_base_20250710_215919_plus_dropout_cond_encoder_scale_only_base_20250722_220005_best")
    return conf


def base_model(conf):
    conf.model_conf.num_classes = None
    conf.model_conf.enc_merge_time_and_cond_embedding = False
    conf.checkpoint["name"] = os.path.join(MODEL_DIR, "tune_xor_base_20250710_215919_plus_dropout_base_20250721_130617_best")
    return conf


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    # 1. infer latent
    # conf = get_base_config()
    # conf.model_conf.net_enc.ch_mult = (1, 2, 2, 4, 4)
    # conf.model_conf.net.ch_mult = (1, 2, 2, 4)
    # conf.model_conf.net.ch = 32
    # conf.dropout = 0.1
    # conf.checkpoint["name"] = os.path.join(DATA_DIR, "final_models", "checkpoints", "diffae_wrts_base_20250712_220934_best")
    # latent_infer_path = os.path.join(DATA_DIR, "final_models", "checkpoints", "latents_wrs.pkl")

    # conf = base_model(conf)
    # latent_infer_path = os.path.join(DATA_DIR, "final_models", "checkpoints", "latents_no_wrs.pkl")
    # latent_infer_path = os.path.join(DATA_DIR, "final_models", "checkpoints", "bdae_class_norm", "latents_class_spec_norm.pkl")

    # conf.data["name"] = "chP3D_tune_5_classes_70"
    # -- only for cond encoder --
    latent_infer_path = os.path.join(MODEL_DIR, "final_latents_cond_encoder_class_spec_norm.pkl")

    # conf = cond_encoder_scale(conf)
    # latent_infer_path = os.path.join(DATA_DIR, "final_models", "checkpoints", "cond_encoder_scale", "final_latents_cond_encoder_scale_class_spec_norm.pkl")
    # ------------------------

    # conf.data["name"] = "chP3D_tune_5_classes_70"
    # conf.__post_init__()
    # trainer = Trainer(conf)
    # trainer.infer_latents(save_path=latent_infer_path)
    # trainer.close()

    # 2. tune
    config = {
        "num_layers": tune.grid_search([10, 20]),
        "num_hid_channels": tune.grid_search([1024, 2048]),
        "loss_type": tune.grid_search([LossType.l1]),
        "latent_infer_path": latent_infer_path,
    }
    main(config)
