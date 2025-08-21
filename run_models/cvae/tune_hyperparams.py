import os
from functools import partial

from ray import tune

from run_models.clf_templates import get_chP96_clf_2cond_conf
from run_models.model_templates import assign_model_config, chp96_cvae_bernoulli_conf
from src._types import ModelName
from src.config import ClfConfig
from src.models.trainer import Trainer
from variables import MS_MAIN_TYPE, MODEL_DIR, ROOT_DIR


def main(config):
    result = tune.run(
        partial(train),
        config=config,
        resources_per_trial={"gpu": 1},
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    result.dataframe().to_csv("all_trials.csv")


def train(config):
    conf = setup_conf()
    assign_model_config(conf)
    conf.model_conf.ch = config["ch"]
    conf.model_conf.num_layers = config["num_layers"]
    conf.model_conf.num_target_emb_channels = config["num_target_emb_channels"]
    conf.model_conf.dropout = config["dropout"]
    conf.model_conf.latent_size = 512
    conf.model_conf.kld_weight = config["kld_weight"]
    conf.lr = 1e-4
    conf.model_conf.in_channels = 1
    conf.model_conf.num_classes = 5
    conf.name = f"final_cvae_5_classes_ch{config['ch']}_kld{config['kld_weight']}"

    conf.__post_init__()
    trainer = Trainer(conf)
    trainer.train()

def setup_conf():
    conf = chp96_cvae_bernoulli_conf()
    conf.data["name"] = "chP3D_tune_5_classes_70"
    conf.checkpoint["save_every_epoch"] = -1
    conf.create_checkpoint = False
    conf.add_running_metrics = ["reconstruction"]

    conf.eval.eval_training_every_epoch = -1
    conf.eval.num_visual_samples = 2
    conf.eval.num_reconstructions = 2
    conf.eval.num_evals = 192

    conf.classes = MS_MAIN_TYPE
    conf.min_change = 0.1
    conf.ema_decay = 0.95
    conf.patience = 80
    conf.use_early_stop = True
    conf.checkpoint['dir'] = os.path.join(ROOT_DIR, "checkpoints", "final_cvae")
    conf.logging_dir = os.path.join(ROOT_DIR, "logging", "final_cvae")
    conf.run_dir = os.path.join(ROOT_DIR, "runs", "final_cvae")
    conf.batch_size = 32
    conf.accum_batches = 1
    conf.log_interval = 50
    conf.img_size = 96
    conf.lr = 1e-3
    conf.num_epochs = 1000
    conf.warmup_epochs = 50
    conf.grad_clip = 1.0
    conf.preprocess_img = "crop"
    conf.randomWeightedTrainSet = False
    conf.clf_conf = ClfConfig(
        os.path.join(MODEL_DIR, "analysis_final_ms_clf_base_20250711_101044_best"),
        ModelName.ms_clf,
        get_chP96_clf_2cond_conf(),
    )
    return conf


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    # lr from 1-e-3 downward for 32+ ch 1e-4 downward => to 1e-4 (maybe try 1e-4 instead)
    config = {
        "ch": tune.grid_search([64]),
        "num_layers": tune.grid_search([4]),
        "num_target_emb_channels": tune.grid_search([1]),
        "min_lr": tune.grid_search([1e-8]),
        "dropout": tune.grid_search([0.1]),
        "kld_weight": tune.grid_search([1, 20, 50]),
    }
    main(config)
