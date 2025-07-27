import copy
import random
from abc import abstractmethod
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src._types import TrainMode, Mode
from src.config import BaseConfig
from src.utils.logger import TrainingLogger
from src.utils.visualisation import plot_3d_data_cloud_ch_p

"""
    Copyright (C) 2024 LeratLy - All Rights Reserved
    You may use, distribute and modify this code under the
    terms of the MIT license.
    You should have received a copy of the MIT license with
    this file.

    Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology.
    Code of VISTEC has been used and modified in this file under terms of the MIT license.
"""


class BaseModel(nn.Module):
    """
    Model with functionalities to work with trainer and provides implementations and guidelines of which functionalities can be used
    """
    no_checkpoint_vars = []

    def __init__(self, conf: BaseConfig, logger: TrainingLogger, writer: SummaryWriter):
        """
        TODO add documentation
        :param conf:
        """
        super().__init__()
        self.batch_idx = 0
        self.epoch_idx = 0
        self.dataset_size = 0
        self.logger = logger
        self.writer = writer
        assert conf.train_mode != TrainMode.manipulate
        if conf.seed is not None:
            torch.manual_seed(conf.seed)
            np.random.seed(conf.seed)
            random.seed(conf.seed)
            if conf.device.startswith("cuda"):
                torch.cuda.manual_seed(conf.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.enabled = False

        # TODO: self.save_hyperparameters(conf.as_dict_jsonable())
        # as_dict(conf)
        self.conf = conf
        self.model = conf.make_model()

        # ema is exponential moving average, only used for evaluation and stabilize training process
        # exponentially weighted moving average to take history in evaluation into account
        self.ema_model = copy.deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self.ema_model.eval()

        # Calculate model size
        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print(f'Model params for {self.conf.model_name.value}: {model_size}')

        try:
            self._init_model()
        except NotImplementedError as e:
            if not isinstance(e, NotImplementedError):
                raise e
            self.logger.file_logger.warning(
                f"An _init_model() method is not implemented for {self.conf.model_name}, "
                f"no additional initialization steps are performed.")
        except Exception as e:
            self.logger.file_logger.error(
                f"An error occurred while initializing {self.conf.model_name}: {str(e)}")
            raise e

    @abstractmethod
    def _init_model(self):
        """
        Initialization of autoencoder that is model dependent
        """
        raise NotImplementedError

    @abstractmethod
    def _forward_model(self, model, target, features, mode: Mode):
        """
        Specific forward process of the autoencoder that is model dependent
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, features, ema=True, target=None, kwargs=None) -> torch.tensor:
        """
        Pass through model with random stochastic sub-code and semantic code encoded with features
        :return: predicted image
        """
        raise

    @abstractmethod
    def reconstruct(self, features, ema=True, target=None, kwargs=None) -> torch.tensor:
        """
        Pass through model with encoded stochastic sub-code and semantic code encoded with features
        :return: predicted image
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int = 1, ema=True, target=None, kwargs=None) -> torch.tensor:
        """
        Create sample based on sampled semantic sub-code from semantic ddim and random gaussian stochastic sub-code
        :return: predicted image, random noise, semantic sub-code
        """
        raise NotImplementedError

    @abstractmethod
    def epoch_metrics(self, output_list, target_list) -> (dict, Union[int, float]):
        """
        Calculate metrics for the epoch results and return those, as well as an optional update metric
        :return: metrics that have been calculated for data, update metric that is used for updating the model instead of the loss
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_training(self, output_list, target_list):
        """
        Evaluate training is called after each epoch if it is time to call evaluate training (defined in config)
        :return: metrics that have been calculated for data, update metric that is used for updating the model instead of the loss
        """
        raise NotImplementedError

    @abstractmethod
    def infer(self, dataloader: dict[Mode, DataLoader], mode: Mode = None, save_path: str = None):
        """
        Method which can be called to infer specific properties of the model and saves them
        """
        raise NotImplementedError

    @abstractmethod
    def infer_whole_folder(self, folder_path, exclude_folders=None, file_name="samples.npz", batch_size: int = 1, use_initial_labels=False):
        """
        Method which can be called to infer features of data in folder
        """
        raise NotImplementedError

    def switch_train_mode(self, mode: TrainMode):
        self.conf.train_mode = mode
        try:
            self._setup_train_mode()
        except NotImplementedError:
            self.logger.file_logger.warning(
                f"No method to setup train mode setup implemented fro model {self.conf.model_type.value}")

    @abstractmethod
    def _setup_train_mode(self):
        """
        Setup all important classes and settings that correlate with a changing train mode
        """
        raise NotImplementedError

    def forward(self, x, target, batch_idx: int, ema_model: bool = False, mode: Mode = Mode.train):
        """
        Either perform normal forward pass on model/eval_model or sample
        :param ema_model: whether ema_model should be used
        :param batch_idx: current batch index
        :param x: the features which are propagated
        :param target: the target(label) of the corresponding data
        :param mode: mode with which the forward pass has been performed
        :return:
        """
        self.batch_idx = batch_idx
        with (torch.autocast(self.conf.device, enabled=False)):
            if ema_model:
                model = self.ema_model.latent_net if self.conf.train_mode == TrainMode.latent_diffusion else self.ema_model
            else:
                model = self.model.latent_net if self.conf.train_mode == TrainMode.latent_diffusion else self.model

            # Use model specific forwarding pass
            try:
                gen = self._forward_model(model, x, target, mode)
            # Simple model forwarding step
            except NotImplementedError as e:
                gen = model.forward(x)

            eval_results = {}
            #  TODO LeratLy (25.05.25): create evaluation logic per batch
            # if mode == Mode.test or mode == Mode.val:
            #     try:
            #         eval_results = model.evaluate_results(x, target, gen)
            #     except NotImplementedError as e:
            #         pass
            return gen, eval_results

    def create_samples_per_class(self, num_samples, num_classes, batch_size=4):
        """
        Creates n samples per class
        :param num_samples: number of samples per class
        :param num_classes: number of classes
        :param batch_size: in which batches the samples are generated
        :return: tensor with samples and targets if not save_only_to_path is set, otherwise they will be saved to the path
        """
        assert num_samples % batch_size == 0, f"num_samples {num_samples} must be dividable by batch_size {batch_size}"
        samples = []
        targets = []
        for target in range(num_classes):
            batches = num_samples // batch_size
            target_tensor = torch.full((num_samples,), target, dtype=torch.int64, device=self.conf.device)
            for batch_idx in range(batches):
                s = self.sample(batch_size, target=target_tensor[batch_idx * batch_size:(batch_idx + 1) * batch_size])
                t = target_tensor[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                samples.append(s)
                targets.append(t)
        return torch.cat(samples), torch.cat(targets)

    def create_samples_for_class(self, num_samples, class_id, batch_size=4):
        """
        Creates n samples for given class
        :param num_samples: number of samples per class
        :param class_id: target id to create samples for
        :param batch_size: in which batches the samples are generated
        :return: list of samples and targets
        """
        assert num_samples % batch_size == 0, f"num_samples {num_samples} must be dividable by batch_size {batch_size}"
        samples = []
        targets = []
        target_tensor = torch.full((num_samples,), class_id, dtype=torch.int64, device=self.conf.device)
        for batch_idx in range(num_samples // batch_size):
            s = self.sample(batch_size, target=target_tensor[batch_idx * batch_size:(batch_idx + 1) * batch_size])
            t = target_tensor[batch_idx * batch_size:(batch_idx + 1) * batch_size].unsqueeze(1)
            samples.append(s)
            targets.append(t)
        return samples, targets

    def render_example(self, imgs, title=None):
        """
        Render given images to tensorboard
        """
        self.logger.file_logger.info(
            f"Starting to render new image for monitoring (total samples so far: {self.num_samples_total}) ...")
        kwargs = {}
        if title is not None:
            kwargs["title"] = title
        plot_3d_data_cloud_ch_p(imgs, self.epoch_idx, self.writer, **kwargs)
        self.logger.file_logger.info("...done rendering image (saved to tensorboard)")

    def to(self, device):
        self.model.to(device)
        self.ema_model.to(device)

    @property
    def num_samples(self):
        """
        Number of total samples that have been processed in this epoch
        :return:
        """
        return (self.batch_idx + 1) * self.conf.batch_size

    @property
    def num_samples_total(self):
        """
        Number of total samples that have been processed
        :return:
        """
        return self.epoch_idx * self.dataset_size + self.num_samples

    @property
    def num_batches_total(self):
        """
        Total number of batches (excluding current) within all epochs
        :return:
        """
        return self.epoch_idx * self.conf.batch_size + self.batch_idx + 1
