from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Tuple, Union, Callable, Optional

import numpy as np
import torch
from torch.nn import Module

from src import ModelConfig
from src._types import TrainMode, ConfigData, \
    ModelName, ModelType, CheckpointConfig, LatentNetType, LossType
from src.models.dae.diffusion import SpacedDiffusionBeatGansConfig
from src.utils.config_functioanlities import ConfigFunctionalities
from src.utils.core import get_class
from src.metrics import setup_loss
from variables import DEVICE

"""
    Copyright (C) 2024 LeratLy - All Rights Reserved
    You may use, distribute and modify this code under the
    terms of the MIT license.
    You should have received a copy of the MIT license with
    this file.

    Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology.
    Code of VISTEC has been used and modified in this file under terms of the MIT license.
"""

LossFunction = Union[
    Module,
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor, ...], torch.Tensor]
]


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


@dataclass
class EvaluationConfig(ConfigFunctionalities):
    """
    :param best_loss: best validation loss
    :type best_loss: float, default np.inf
    :param eval_training_every_epoch: every n epochs, evaluate metrics should be called
    :type eval_training_every_epoch: int, default 5
    :param num_visual_samples: number of samples that will be generated when evaluating metrics (rendered visually)
    :type num_visual_samples: int, default 2 (one per class)
    :param num_reconstructions: number of reconstructions that will be generated when evaluating metrics (rendered visually)
    :type num_reconstructions: int, default 2 (one per class)
    :param num_evals: number of samples that will be evaluated when evaluating metrics (can be more than visually inspected)
    :type num_evals: int, default 10
    :param eval_epoch_metrics_val_samples: whether to use and collect validation samples for evaluation metrics
    :type eval_epoch_metrics_val_samples: bool, default False
    :param eval_epoch_metrics_val_samples: whether to use and collect validation samples for evaluation metrics
    :type eval_epoch_metrics_val_samples: bool, default False
    """
    best_loss: float = np.inf
    ema_every_samples: int = 0  # TODO remove deprecated
    every_samples: int = 0  # TODO remove deprecated
    eval_training_every_epoch: int = 5
    num_visual_samples: int = 2
    num_reconstructions: int = 2
    num_evals: int = 10
    eval_epoch_metrics_val_samples: bool = False


@dataclass
class TorchInstanceConfig(ConfigFunctionalities):
    """
    Configuration for torch optimizer or scheduler instances.
    """
    instance_type: str  # Type[Union[Optimizer, Scheduler, LRScheduler]]
    settings: list


@dataclass
class ClfConfig(ConfigFunctionalities):
    """
    Configuration for torch optimizer or scheduler instances.
    """
    load_path: Optional[str]
    classifier_name: ModelName
    classifier_conf: BaseConfig


# TODO add all arguments that are possible for the config and set to default setting
# TODO Also mark which variables need to be set in which config


@dataclass
class BaseConfig(ConfigFunctionalities):
    """
    Base configuration for Training Pipeline (training specific parameters + hyperparameters)
    :param add_running_metrics: list containing names of additional metrics that should be logged ot the tensorboard in evaluation mode (this metrics need to be returned in a dictionary from the forward pass of the model)
    :type add_running_metrics: list[str], default None
    """
    # Architecture/Data dependent params do not have default parameters and must be set!
    data: ConfigData = None
    model: str = None
    model_conf: ModelConfig = None
    model_conf_params: dict = None
    model_name: ModelName = None
    model_type: ModelType = None
    clf_conf: ClfConfig = None
    net_latent_net_type: LatentNetType = LatentNetType.skip
    diffusion_conf: SpacedDiffusionBeatGansConfig = None
    latent_diffusion_conf: SpacedDiffusionBeatGansConfig = None

    # Default values applicable for all autoencoders
    img_size: Union[Tuple[int, ...], int] = 96
    dims: int = 3
    accum_batches: int = 3
    batch_size: int = 32
    min_change: float = 0.0000001
    patience: int = 5
    min_epochs: int = 0
    device: str = DEVICE
    dropout: float = 0.1
    final_dropout: float = 0.0
    dtype: torch.dtype = torch.float32
    ema_decay: float = 0.99
    fp16: bool = False  # True
    grad_clip: float = 0
    log_interval: int = 30
    logging_dir: str = "logging"
    run_dir: str = "runs/"
    loss_type: str = LossType.bce
    loss_kwargs: dict = None
    loss_type_eval: str = None
    loss_kwargs_eval: dict = None
    loss_func: LossFunction = None
    loss_func_eval: LossFunction = None
    lr: float = 1e-4
    num_epochs: int = 2
    # So far only data_parallel
    data_parallel: bool = False
    # TODO: implement and verify DDP
    parallel: bool = False
    scheduler: TorchInstanceConfig = None
    optimizer: TorchInstanceConfig = None
    seed: int = 0
    shuffle: bool = True
    preprocess_img: str = 'crop'
    style_ch: int = 512
    train_mode: TrainMode = TrainMode.base
    use_early_stop: bool = False
    name: str = None
    create_checkpoint: bool = True
    clip_denoised: bool = False
    normalize_denoised: bool = False
    num_classes: int = None
    warmup_epochs: int = None
    sample_size: int = 1
    randomWeightedTrainSet: bool = False
    classes: dict = None
    use_transforms: bool = False
    add_running_metrics: list[str] = None
    latent_infer_path: str = None
    eval_max_batches: int = None

    # for ddp
    nodes: int = 1
    gpus: int = 1

    def __init__(self):
        # Set mutable default class attributes
        self.checkpoint = CheckpointConfig(dir="checkpoints", resume_optimizer=True, resume_scheduler=True)
        self.eval: EvaluationConfig = EvaluationConfig()

    def __post_init__(self):
        assert self.model_name is not None, "Please specify a model name before starting the training process."
        if self.name is None:
            self.name = self.model_name.value + "_" + self.data["name"]
        self.loss_func = setup_loss(self.loss_type)
        self.loss_func_eval = setup_loss(self.loss_type_eval)
        self.loss_kwargs = self.loss_kwargs if self.loss_kwargs is not None else dict()
        self.loss_kwargs_eval = self.loss_kwargs_eval if self.loss_kwargs_eval is not None else dict()

        if self.optimizer is None:
            self.optimizer: TorchInstanceConfig = TorchInstanceConfig(
                instance_type="torch.optim.AdamW",
                settings=[self.lr, (0.9, 0.999), 1e-8, 0.02]
            )
        if self.scheduler is None:
            print("Using standard reduceLROnPlateau scheduler")
            self.scheduler = TorchInstanceConfig(
                instance_type="torch.optim.lr_scheduler.ReduceLROnPlateau",
                settings=['min', 0.5, 10, 0.0001, 'rel', 0, 1e-8]
            )
        return self

    def setup_optimizer(self, params):
        """
        Sets up an optimizer based on config settings

        :param params: parameters of the model
        :return: the optimizer based on config
        """
        assert self.optimizer is not None
        optimizer = get_class(self.optimizer.instance_type)(params, *self.optimizer.settings)
        scheduler = None if self.scheduler is None else get_class(self.scheduler.instance_type)(
            *[optimizer, *self.scheduler.settings])
        return optimizer, scheduler

    # TODO figure out when and how to setup models in general (does the code needs some modifications?)
    def make_model(self):
        return get_class(self.model)(self.model_conf)


class BaseConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseConfig):
            return asdict(obj)
        return super().default(obj)
