from pathlib import Path

import torch
from torch import save, load
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from src._types import ModelName
from src.config import BaseConfig
from src.models.clf.ms_clf_wrapper import MSClfWrapperModel
from src.utils.logger import TrainingLogger

"""
    Copyright (C) 2024 LeratLy - All Rights Reserved
    You may use, distribute and modify this code under the
    terms of the MIT license.
    You should have received a copy of the MIT license with
    this file.
"""


def create_checkpoint(model, filename, optimizer=None, scheduler=None):
    save_object = create_save_object(model, optimizer, scheduler)
    save(save_object, filename)


def create_checkpoint_for_dict(save_object: dict, filename):
    save(save_object, filename)


def create_save_object(model, optimizer=None, scheduler=None):
    save_object = {
        "model": model.state_dict()
    }
    if hasattr(model, "no_checkpoint_vars"):
        for skip_key in model.no_checkpoint_vars:
            save_object["model"] = {k: v for k, v in save_object["model"].items() if not k.startswith(skip_key)}
    if optimizer is not None:
        save_object["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_object["scheduler"] = scheduler.state_dict()
    return save_object


def resume_checkpoint(model, filename, optimizer=None, scheduler=None, skip_keys=None):
    """
    Resumes a saved checkpoint
    :param model:
    :param filename:
    :param optimizer:
    :param scheduler:
    :return:
    """
    checkpoint = load(filename, weights_only=True)
    assert "model" in checkpoint, "Model is missing in checkpoint file, invalid checkpoint"

    state_dict = checkpoint['model']
    if skip_keys is not None:
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if not any([k.startswith(skip_key) for skip_key in skip_keys])
        }
        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])


def load_model(checkpoint_path: str, model_name: ModelName, conf: BaseConfig, logger: TrainingLogger,
               writer: SummaryWriter):
    if model_name == ModelName.ms_clf:
        model = MSClfWrapperModel(conf, logger, writer)
    else:
        raise NotImplementedError

    checkpoint = load(checkpoint_path, weights_only=True)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict, strict=False)
    return model


def load_model_directly(model: Module, checkpoint_path: str):
    checkpoint = load(checkpoint_path, weights_only=True)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict, strict=False)
    return model
