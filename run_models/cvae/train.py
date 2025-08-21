import os.path

from src._types import ModelName
from src.config import ClfConfig
from run_models.clf_templates import get_chP96_clf_2cond_conf
from run_models.model_templates import assign_model_config, chp96_cvae_bernoulli_conf
from src.models.trainer import Trainer
from variables import MS_MAIN_TYPE, MODEL_DIR


def train():
    conf = setup_conf()
    trainer = Trainer(conf)
    trainer.train()

def setup_conf():
    conf = chp96_cvae_bernoulli_conf()
    conf.checkpoint["save_every_epoch"] = -1
    conf.eval.eval_training_every_epoch = 1
    conf.name = "cvae_tune"
    conf.checkpoint['dir'] = "./checkpoints"
    conf.logging_dir = "./logging"
    conf.run_dir = "./runs/"
    conf.batch_size = 4
    conf.log_interval = 5
    conf.accum_batches = 8
    # conf.grad_clip = 1.0
    conf.lr = 1e-3
    conf.dropout = 0.2
    assign_model_config(conf)
    conf.model_conf.in_channels = 1
    conf.model_conf.num_classes = 3
    conf.model_conf.ch = 32
    conf.model_conf.kld_weight = 1/10
    conf.model_conf.num_layers = 0
    conf.model_conf.latent_size = 128
    conf.model_conf.num_target_emb_channels = 1
    conf.create_checkpoint = False
    conf.classes = MS_MAIN_TYPE
    conf.name = "cvae_tune"
    conf.eval.num_samples = 2
    conf.grad_clip = 1.0
    conf.dropout = 0.2
    conf.clf_conf = ClfConfig(
        os.path.join(MODEL_DIR, "analysis_final_ms_clf_base_20250711_101044_best"),
        ModelName.ms_clf,
        get_chP96_clf_2cond_conf(),
    )
    conf.__post_init__()
    return conf


if __name__ == "__main__":
    train()
