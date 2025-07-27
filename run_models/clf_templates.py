import os

from run_models.model_templates import assign_model_config
from src import MSClfNetConfig
from src._types import ModelName, DataType, LossType
from src.config import BaseConfig, TorchInstanceConfig
from variables import MS_TYPES_FILTERED, MS_MAIN_TYPE, DATA_DIR


def _chP96_clf_base_conf():
    conf = BaseConfig()
    conf.name = 'ms_clf'
    conf.model_name = ModelName.ms_clf
    conf.classes = MS_MAIN_TYPE
    conf.num_classes = 2
    conf.num_epochs = 250
    conf.use_transforms = True
    conf.create_checkpoint = True
    conf.batch_size = 16
    conf.img_size = 96
    conf.log_interval = 20
    conf.eval.eval_epoch_metrics_val = True
    conf.eval.eval_epoch_metrics_val_samples = True
    conf.model = "src.models.clf.ms_clf.MSClfNet"
    conf.device = "cuda"
    conf.lr = 1e-3
    conf.eval.eval_training_every_epoch = -1
    conf.checkpoint["save_every_epoch"] = -1
    conf.accum_batches = 1
    conf.data = {
        "name": 'chP3D_tune_full',
        "type": DataType.nii,
    }
    conf.ema_decay = 0.95
    conf.loss_type = LossType.cel
    conf.loss_type_eval = LossType.m_if1
    conf.loss_kwargs_eval = {"num_classes": conf.num_classes}
    conf.randomWeightedTrainSet = True
    conf.patience = 50
    conf.preprocess_img = "crop"

    conf.checkpoint['dir'] = os.path.join(DATA_DIR, "final_models", "checkpoints")
    conf.logging_dir = os.path.join(DATA_DIR, "final_models", "logging")
    conf.run_dir = os.path.join(DATA_DIR, "final_models", "runs")

    conf.scheduler = TorchInstanceConfig(
        instance_type="torch.optim.lr_scheduler.ReduceLROnPlateau",
        settings=['min', 0.5, 10, 0.0001, 'rel', 0, 1e-8]
    )
    return conf


def get_chP96_clf_2cond_conf():
    """
    Final config for MS clf HC vs MS only (2 conditions)
    Trained with random trainset
    """
    conf = _chP96_clf_base_conf()
    assign_model_config(conf)
    conf.model_conf = chP96_clf_2cond_model_conf(conf.model_conf)
    return conf

def get_chP96_clf_5cond_conf():
    """
    Final config for MS clf HC vs MS only (5 conditions)
    Trained with random trainset
    """
    conf = _chP96_clf_base_conf()
    conf.data["name"] = "chP3D_tune_full_5_classes"
    assign_model_config(conf)
    conf.model_conf = chP96_clf_5cond_model_conf(conf.model_conf)
    return conf

def chP96_clf_2cond_model_conf(model_conf):
    model_conf.hidden_ch = 64
    model_conf.num_classes = 2
    model_conf.dropout = 0.2
    model_conf.final_dropout = 0.2
    model_conf.num_hidden_layers = 3
    model_conf.out_num = 0
    return model_conf


def chP96_clf_5cond_model_conf(model_conf):
    model_conf = chP96_clf_2cond_model_conf(model_conf)
    model_conf.num_classes = 5
    return model_conf


def get_chP96_clf_2cond_model_conf():
    model_conf = MSClfNetConfig()
    return chP96_clf_2cond_model_conf(model_conf)


def get_chP96_clf_5cond_model_conf():
    model_conf = MSClfNetConfig()
    return chP96_clf_5cond_model_conf(model_conf)
