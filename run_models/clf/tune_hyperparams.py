import os.path
from functools import partial

import numpy as np
from ray import tune

from run_models.clf_templates import get_chP96_clf_2cond_conf
from run_models.model_templates import assign_model_config
from src._types import LossType, DataType
from src.config import TorchInstanceConfig
from src.k_fold import run_k_fold
from ray import tune as ray_tune
from variables import MS_MAIN_TYPE, DATA_DIR, DEVICE, ROOT_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

def main(config):
    result = tune.run(
        partial(train),
        config=config,
        resources_per_trial={"gpu": 1}
    )

    best_trial = result.get_best_trial("mean_k_fold_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['mean_k_fold_loss']}")

    result.dataframe().to_csv("all_trials.csv")


def train(config):
    print(os.environ.get("OMP_NUM_THREADS"))

    conf = setup_conf()
    assign_model_config(conf)
    conf.lr = config["lr"]
    conf.preprocess_img = config["preprocess_img"]
    conf.model_conf.num_hidden_layers = config["num_hidden_layers"]
    conf.model_conf.out_num = config["out_num"]
    conf.model_conf.final_dropout = config["final_dropout"]
    conf.model_conf.hidden_ch = config["hidden_ch"]

    if config["random_sample"]:
        conf.randomWeightedTrainSet = True
        conf.loss_kwargs = {}
        conf.loss_type = LossType.cel
    else:
        conf.randomWeightedTrainSet = False
        conf.loss_kwargs = {"weights": np.array([1 - 388 / 427, 1 - 39 / 427]), "device": conf.device}
        conf.loss_type = LossType.w_cel

    # conf.loss_type_eval = LossType.w_cel
    # conf.loss_kwargs_eval = {"weights": np.array([1 - 83 / 92, 1 - 9 / 92]), "device": conf.device}

    conf.patience = 50
    conf.name = f"{conf.name}_lr{config["lr"]}_hiddenl{config["num_hidden_layers"]}_hiddench{config["hidden_ch"]}_out_num{config["out_num"]}_randomSampling{config['random_sample']}f1_maxpool"
    conf.__post_init__()
    conf.data["name"] = "chP_k_fold_0"
    mean_loss = run_k_fold(conf)
    ray_tune.report({
        "mean_k_fold_loss": mean_loss
    })

def setup_conf():
    conf = get_chP96_clf_2cond_conf()
    conf.data = {
        "name": 'chP3D_tune_70',
        "type": DataType.nii,
    }
    conf.classes = MS_MAIN_TYPE
    # conf.classes = MS_TYPES_FILTERED
    conf.num_classes = 2
    # conf.num_classes = 2
    conf.name = "clf_2_classes_min"
    conf.preprocess_img = "crop"
    conf.model_conf.num_classes = 2
    conf.img_size = 96
    conf.use_transforms = True
    # conf.model_conf.num_classes = 2
    conf.model_conf.dropout = 0.2
    conf.model_conf.final_dropout = 0.2
    conf.num_epochs = 250
    conf.use_early_stop = True
    conf.create_checkpoint = False
    conf.batch_size = 16
    conf.ema_decay = 0.95
    conf.accum_batches = 1
    conf.loss_type_eval = LossType.m_if1
    conf.loss_kwargs_eval = {"num_classes": conf.num_classes}
    conf.device = DEVICE
    conf.log_interval = 30
    # conf.data_parallel = True
    conf.checkpoint['dir'] = os.path.join(ROOT_DIR, "checkpoints")
    conf.logging_dir = os.path.join(ROOT_DIR, "logging")
    conf.run_dir = os.path.join(ROOT_DIR, "runs")
    conf.scheduler = TorchInstanceConfig(
        instance_type="torch.optim.lr_scheduler.ReduceLROnPlateau",
        settings=['min', 0.5, 10, 0.0001, 'rel', 0, 1e-8]
    )
    return conf


if __name__ == "__main__":
    config = {
        "lr": tune.grid_search([1e-3]),
        "num_hidden_layers": tune.grid_search([3]),
        "hidden_ch": tune.grid_search([64]),
        "preprocess_img": tune.grid_search(["crop"]),
        "random_sample": tune.grid_search([True]),
        "out_num": tune.grid_search([0]),
        "final_dropout": tune.grid_search([0.2])
    }
    main(config)
