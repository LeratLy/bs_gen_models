import gc
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Union

import numpy as np
import torch
from ray import tune as ray_tune
from torch import distributed, cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src._types import ModelName, Mode, TrainMode, NoiseType, SaveTo
from src.config import BaseConfig
from src.data.dataloader import get_dataloader
from src.models.autoencoder_base import BaseModel
from src.models.clf.ms_clf_wrapper import MSClfWrapperModel
from src.models.dae.dae import DAE
from src.models.vae.vae import VAE
from src.utils.checkpoints import resume_checkpoint, create_checkpoint, create_save_object, create_checkpoint_for_dict
from src.utils.core import save_call, is_callable
from src.utils.logger import TrainingLogger
from src.utils.ray import is_ray_running
from src.utils.training import scheduler_step
from src.utils.visualisation import plot_3d_data_cloud
from src.utils.writer import add_scalar_dict
from variables import data_paths, split_csvs


class Trainer:
    """
    Class that can be used to train and evaluate a model for a given config
    """
    logger: TrainingLogger

    # TODO handle distributed computing based on gpus and nodes (Distributed Data Parallel) with DistributedSampler
    def __init__(self, conf: BaseConfig):
        self.best_model = None
        torch.cuda.empty_cache()
        self.dataloaders = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = torch.inf
        self.conf = conf

        if not self.conf.device:
            self.conf.device = "cuda" if cuda.is_available() else "cpu"
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_identifier = f"{self.conf.name}_{self.conf.data["name"]}_{self.timestamp}_{self.conf.train_mode.value}"

        # Logging directory set up
        if not os.path.exists(conf.logging_dir):
            os.makedirs(conf.logging_dir)
        self.logger = TrainingLogger(conf, self.timestamp)
        self.writer = SummaryWriter(os.path.join(self.conf.run_dir, self.model_identifier))

        self.logger.file_logger.info(f"Setting up training pipeline..."
                                     f"\nLogging setup for director : {conf.logging_dir}..."
                                     f"\nLoading model...")

        # Add correct model
        if ModelName.is_dae(self.conf.model_name):
            self._wrapper_model = DAE(conf, self.logger, self.writer)
        elif self.conf.model_name == ModelName.conditional_variational_autoencoder:
            self._wrapper_model = VAE(conf, self.logger, self.writer)
        elif self.conf.model_name == ModelName.ms_clf:
            self._wrapper_model = MSClfWrapperModel(conf, self.logger, self.writer)
        else:
            self._wrapper_model = BaseModel(conf, self.logger, self.writer)

        self.device = torch.device(conf.device)
        if self.conf.data_parallel and not self.conf.parallel:
            assert self.conf.device != "cpu" and not self.conf.parallel, "data parallelism is not possible using cpu device mode or when also trying to use ddp!"
            self._wrapper_model = torch.nn.DataParallel(self._wrapper_model)
            self.logger.file_logger.info("Using data parallelism with main gpu:" + self.conf.device)
        self._wrapper_model.to(self.device)
        self._setup_train_mode()

        self.logger.file_logger.info(f"Model loaded...\nLoading optimizer")
        self.optimizer, self.scheduler = conf.setup_optimizer(self.get_wrapper_model().model.parameters())
        self.logger.file_logger.info(f"Optimizer ({self.optimizer}) loaded...")

        # Checkpoint handling
        if not os.path.exists(conf.checkpoint["dir"]):
            os.makedirs(conf.checkpoint["dir"])
        if "name" in conf.checkpoint:
            self.resume_checkpoint(
                resume_optimizer=self.conf.checkpoint["resume_optimizer"],
                resume_scheduler=self.conf.checkpoint["resume_scheduler"],
                skip_keys=conf.checkpoint.get("resume_skip_keys")
            )

        self.logger.file_logger.info(f"Training pipeline is set up!")

    def get_wrapper_model(self) -> Union[torch.nn.Module, BaseModel]:
        if self.conf.data_parallel:
            return self._wrapper_model.module
        return self._wrapper_model

    def handler(self, signum, frame):
        print('Signal handler called with signal', signum)
        print('You pressed Ctrl+C! Saving best or current model...')
        if self.best_model is None:
            self.best_model = create_save_object(self.get_wrapper_model(), self.optimizer, self.scheduler)
        self.save_best_model()
        sys.exit(0)

    def train_batch(self, batch, batch_idx):
        """
        Train a batch
        :param batch: batch of data
        :param batch_idx: index of current batch
        :param ema_model: whether to use exponential moving average model or normal one
        :return: loss: the loss of the batch
        """
        (features, target, data_id) = batch
        # Set gradients to zero for each batch
        features = features.to(self.device).type(self.conf.dtype)
        target = target.to(self.device)
        # Forward pass
        if self.conf.fp16:
            with torch.autocast(device_type=self.conf.device, dtype=torch.bfloat16):
                outputs, _ = self._wrapper_model(features, target, batch_idx, mode=Mode.train)
        else:
            with torch.autocast(self.conf.device, enabled=False):
                outputs, _ = self._wrapper_model(features, target, batch_idx, mode=Mode.train)

        # Compute loss and gradients
        if type(outputs) is dict and outputs.get("loss") is not None:
            # Needed for data parallel
            if outputs["loss"].numel() > 1:
                loss = torch.mean(outputs["loss"])
            else:
                loss = outputs["loss"]
        else:
            loss = self.conf.loss_func(outputs, target, kwargs=self.conf.loss_kwargs)
        loss.backward()

        if self.conf.grad_clip > 0:
            params = [p for group in self.optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_value_(params, clip_value=self.conf.grad_clip)
        return loss

    def train_epoch(self, epoch_idx: int):
        """
        training of a model for an epoch
        :type epoch_idx: index of current epoch
        :return:
        """
        assert self.dataloaders.get(Mode.train) is not None
        for param_group in self.optimizer.param_groups:
            self.logger.file_logger.info(f"learning rate: {param_group['lr']}")

        # Set train mode
        if self.conf.train_mode == TrainMode.latent_diffusion:
            self.get_wrapper_model().model.eval()
            self.get_wrapper_model().model.latent_net.train()
        else:
            self.get_wrapper_model().model.train()
        running_epoch_loss = 0.

        # Choose correct model
        if self.conf.train_mode == TrainMode.latent_diffusion:
            # it trains only the latent hence change only the latent
            model = self.get_wrapper_model().model.latent_net
            ema_model = self.get_wrapper_model().ema_model.latent_net
        else:
            model = self.get_wrapper_model().model
            ema_model = self.get_wrapper_model().ema_model

        # Go through batches
        for batch_idx, batch in enumerate(self.dataloaders[Mode.train]):
            # Gather data and report
            loss = self.train_batch(batch, batch_idx)
            running_epoch_loss += loss.item()
            if self.is_last_accum(self.get_wrapper_model().num_batches_total):
                # only apply ema on the last gradient accumulation step,
                # if it is the iteration that has optimizer.step()
                self.optimizer.step()
                ema(model, ema_model, self.conf.ema_decay)
                self.optimizer.zero_grad()

            self.logger.on_batch_end(batch_idx, {'train_loss': loss})
        gc.collect()

        # Evaluate training
        if (self.conf.eval.eval_training_every_epoch not in [-1, 0]
                and is_time(self.get_wrapper_model().epoch_idx + 1, self.conf.eval.eval_training_every_epoch)):
            self.evaluate_training()
        return running_epoch_loss / len(self.dataloaders.get(Mode.train))

    def evaluate_training(self):
        """
        Evaluate the training process so far, based on predefined samples
        """
        save_call(self.get_wrapper_model(), "evaluate_training", self.val_samples, self.train_samples)

    def evaluate_epoch(self, epoch_idx: int, mode: Mode = Mode.val):
        """
        evaluation of a model for an epoch
        :param epoch_idx: index of current epoch
        :param mode: the evaluation mode (either 'val' or 'test')
        :return:
        """
        assert mode == Mode.val or mode == Mode.test
        dataloader = self.dataloaders.get(Mode.val) if mode == Mode.val else self.dataloaders.get(Mode.test)
        assert dataloader is not None

        self._wrapper_model.eval()
        running_epoch_loss = 0.
        all_preds = []
        all_targets = []
        add_running_metrics = {}
        with (torch.no_grad()):
            for batch_idx, (features, target, data_id) in enumerate(dataloader):
                if self.conf.eval_max_batches is not None and batch_idx > self.conf.eval_max_batches:
                    break
                features = features.to(self.device).type(self.conf.dtype)
                target = target.to(self.device)
                outputs, eval_results = self._wrapper_model(features, target, batch_idx, ema_model=True, mode=mode)
                # Compute loss
                if isinstance(outputs, dict) and outputs.get("loss"):
                    loss = outputs["loss"]
                    # add additional logs to dictionary to be able to also log them to tensorboard
                    self.maintain_add_running_metrics(
                        add_running_metrics,
                        outputs,
                        divide=None if batch_idx < len(dataloader) - 1 else len(dataloader))
                elif self.conf.loss_func_eval is None:
                    loss = self.conf.loss_func(outputs, target, kwargs=self.conf.loss_kwargs)
                else:
                    loss = self.conf.loss_func_eval(outputs, target, kwargs=self.conf.loss_kwargs_eval)
                running_epoch_loss += loss.item()

                if self.conf.eval.eval_epoch_metrics_val_samples:
                    all_preds.append(outputs.detach().cpu())
                    all_targets.append(target.detach().cpu())

                eval_results[f'{mode.value}_loss'] = loss.item()
                if self.conf.use_early_stop is not None and self.conf.use_early_stop > 0:
                    self.writer.add_scalar("idle_epochs", self.counter)
                self.logger.on_batch_end(batch_idx, eval_results)

        if self.conf.eval_max_batches is not None:
            mean_loss = running_epoch_loss / self.conf.eval_max_batches
        else:
            mean_loss = running_epoch_loss / len(dataloader)

        update_metric = None
        if is_callable(self.get_wrapper_model(), "epoch_metrics"):
            metrics, update_metric = self.get_wrapper_model().epoch_metrics(all_preds, all_targets)
            self.logger.file_logger.info(f"Evaluation metrics: {metrics}")
            add_scalar_dict(metrics, self.writer, f"{mode.value}", epoch_idx)
        if len(add_running_metrics) > 0:
            add_scalar_dict(add_running_metrics, self.writer, f"{mode.value}", epoch_idx)
        del all_targets, all_preds
        gc.collect()
        return mean_loss, update_metric

    def maintain_add_running_metrics(self, metrics: dict, new_values: dict, divide: int = None):
        """
        Maintains a dictionary with epoch running metrics (adds new data based on config and divides by given value if set)
        """
        if self.conf.add_running_metrics is not None:
            for add_metrics in self.conf.add_running_metrics:
                if add_metrics in new_values.keys():
                    metrics.setdefault(add_metrics, 0)
                    metrics[add_metrics] += new_values[add_metrics]
                    if divide is not None:
                        metrics[add_metrics] /= divide

    def save_results(self, epoch_idx, avg_train_loss=None, avg_val_loss=None, update_metric=None):
        """
        saves avg training and/or validation loss for an epoch
        Moreover it saves a checkpoint if the new model is better than the ones before
        :param epoch_idx: the index of the epoch related to the results
        :param avg_train_loss: the average training loss for the epoch
        :param avg_val_loss: the average validation loss for the epoch
        :return:
        """
        self.logger.file_logger.info(f"Saving results...")
        logs = {}
        name = ""
        if avg_train_loss is not None:
            logs["train_loss"] = avg_train_loss
            name = "_".join((name, 'TrainingLoss'))
        if self.dataloaders.get(Mode.val) is not None:
            logs["val_loss"] = avg_val_loss
            name = "_".join((name, 'ValidationLoss'))
        if bool(logs):
            self.writer.add_scalars(name, logs, epoch_idx + 1)
        self.logger.on_epoch_end(epoch_idx, logs)

        # Save model if performance is better than any other before
        if avg_train_loss is not None:
            compare_loss = update_metric if update_metric is not None else avg_val_loss if avg_val_loss is not None else avg_train_loss
            new_best_loss = self.save_checkpoint(compare_loss, epoch_idx)
            if self.conf.create_checkpoint and not new_best_loss:
                self.logger.file_logger.info(f"Keeping best model.")

            self.logger.file_logger.info(f"Average loss = {compare_loss}"
                                         f"\nBest loss = {self.best_loss}")

            if is_ray_running():
                ray_tune.report({
                    "idle_epochs": self.counter,
                    "loss": self.best_loss
                })
        self.writer.flush()

    def save_checkpoint(self, loss, epoch_idx):
        """
        Save the current model to a checkpoint if either
        a) time_save -> checkpoints are automatically created
        """
        time_save = ("save_every_epoch" in self.conf.checkpoint
                     and is_time(epoch_idx + 1, self.conf.checkpoint["save_every_epoch"]))
        best_save = loss is not None and loss < self.best_loss and (self.best_loss - loss) >= self.conf.min_change
        if best_save:
            self.best_loss = loss
            self.counter = 0
            if not self.conf.create_checkpoint:
                self.best_model = create_save_object(self.get_wrapper_model(), self.optimizer, self.scheduler)
        elif not self.conf.min_epochs or epoch_idx > self.conf.min_epochs:
            # No improvement
            if self.counter >= self.conf.patience:
                self.logger.file_logger.info(f"Early stopping at epoch {epoch_idx + 1}.")
                self.early_stop = True
            else:
                self.counter += 1
        if self.conf.create_checkpoint and (time_save or best_save):
            assert "dir" in self.conf.checkpoint, "No directory defined to save checkpoint to. Please define a directory in conf."
            names = []
            if time_save:
                names.append(f"{self.model_identifier}_{epoch_idx}")
            if best_save:
                names.append(f"{self.model_identifier}_best")
            for name in names:
                path = os.path.join(self.conf.checkpoint["dir"], name)
                create_checkpoint(
                    self.get_wrapper_model(),
                    path,
                    self.optimizer,
                    self.scheduler
                )
                self.conf.save(path + "_conf.json")
                self.logger.file_logger.info(f"Done saving results for model: {name}")
            self.conf.checkpoint["name"] = names[-1]
        return best_save

    def train(self):
        """
        Perform training and optional validation in each epoch
        :return:
        """
        assert self.dataloaders.get(Mode.train) is not None
        self.early_stop = False
        self.counter = 0
        self.get_wrapper_model().dataset_size = len(self.dataloaders.get(Mode.train))

        self.logger.file_logger.info(f"Training dataset...")
        for epoch_idx in range(self.conf.num_epochs):
            self.get_wrapper_model().epoch_idx = epoch_idx
            if self.conf.use_early_stop and self.early_stop:
                break
            self.logger.on_epoch_begin(epoch_idx)
            avg_train_loss = self.train_epoch(epoch_idx)
            self.logger.file_logger.info("Evaluating dataset...")
            avg_val_loss, update_metric = self.evaluate_epoch(epoch_idx) if self.dataloaders.get(
                Mode.val) is not None else (None, None)
            if self.scheduler is not None and epoch_idx > self.conf.min_epochs:
                step_metric = update_metric if update_metric is not None else avg_val_loss if avg_val_loss is not None else avg_train_loss
                scheduler_step(scheduler=self.scheduler, metric=step_metric, epoch_idx=epoch_idx)
                self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], epoch_idx)
            self.save_results(epoch_idx, avg_train_loss, avg_val_loss, update_metric)
        if not self.conf.create_checkpoint:
            self.save_best_model()
        self.logger.file_logger.info(f"Done Training dataset!")

    def save_best_model(self):
        self.logger.file_logger.info(
            f"Saving best model: {self.model_identifier} with following keys: {self.best_model["model"].keys()}")
        create_checkpoint_for_dict(self.best_model,
                                   os.path.join(self.conf.checkpoint["dir"], f"{self.model_identifier}_best"))
        self.conf.save(os.path.join(self.conf.checkpoint["dir"], f"{self.model_identifier}_best") + "_conf.json")

    def validate(self):
        """
        Perform validation only
        :return:
        """
        assert self.dataloaders.get(Mode.val) is not None
        self.get_wrapper_model().dataset_size = len(self.dataloaders.get(Mode.val))

        self.logger.file_logger.info(f"Validating dataset...")
        for epoch_idx in range(self.conf.num_epochs):
            self.get_wrapper_model().epoch_idx = epoch_idx
            self.logger.on_epoch_begin(epoch_idx)
            avg_val_loss, _ = self.evaluate_epoch(epoch_idx)
            self.save_results(epoch_idx, avg_val_loss=avg_val_loss)
        self.logger.file_logger.info(f"Done Validating dataset!")

    def test(self):
        """
        Perform validation only
        :return:
        """
        assert self.dataloaders.get(Mode.test) is not None
        self.get_wrapper_model().dataset_size = len(self.dataloaders.get(Mode.test))

        self.logger.file_logger.info(f"Testing dataset...")
        for epoch_idx in range(self.conf.num_epochs):
            self.logger.on_epoch_begin(epoch_idx)
            self.get_wrapper_model().epoch_idx = epoch_idx
            avg_test_loss, _ = self.evaluate_epoch(epoch_idx, Mode.test)
            logs = {"test_loss": avg_test_loss}
            self.writer.add_scalars("TestLoss", logs, epoch_idx + 1)
            self.logger.on_epoch_end(epoch_idx, logs)
            self.writer.flush()
        self.logger.file_logger.info(f"Done Testing dataset!")

    def infer_latents(self, dataloader_mode: Mode = None, save_path: str = None) -> torch.tensor:
        """
        Method which can be called to infer specific properties of the model
        :param dataloader_mode: if mode is not given all data will be inferred
        """
        if ModelName.is_dae(self.conf.model_name):
            self.get_wrapper_model().infer(self.dataloaders, dataloader_mode, save_path)
        else:
            raise NotImplementedError

    def is_last_accum(self, num_batches):
        """
        is it the last gradient accumulation loop?
        used with gradient_accum > 1 and to see if the optimizer will perform "step" in this iteration or not
        """
        return self.conf.accum_batches <= 0 or is_time(num_batches, self.conf.accum_batches)

    def resume_checkpoint(self, resume_optimizer=True, resume_scheduler=True, skip_keys=None):
        """
        Resumes a checkpoint based on configs checkpoint settings
        """
        resume_checkpoint(
            self.get_wrapper_model(),
            self.conf.checkpoint["name"],
            self.optimizer if resume_optimizer else None,
            self.scheduler if resume_scheduler else None,
            skip_keys
        )
        self.logger.file_logger.info(f"Resuming from checkpoint {self.conf.checkpoint['name']}...")

    def close(self):
        """
        Cleanup function for loggers and writers
        """
        if hasattr(self, 'writer'):
            self.writer.close()
        if hasattr(self, 'logger'):
            self.logger.file_logger.handlers.clear()

    def setup_dataloaders(self, data_path, data_split_csv):
        args = {
            "batch_size": self.conf.batch_size,
            "shuffle": self.conf.shuffle,
            "distributed": self.conf.parallel or distributed.is_initialized(),
            "preprocess_img": self.conf.preprocess_img,
            "do_normalize_gaussian": ModelName.is_dae(self.conf.model_name) and (
                    self.conf.diffusion_conf.noise_type == NoiseType.gaussian),
            "randomWeightedTrainSet": self.conf.randomWeightedTrainSet,
            "use_transforms": self.conf.use_transforms,
        }
        # Latent model with own Dataset
        if ModelName.is_dae(
                self.conf.model_name) and self.conf.model_conf.latent_net_conf and self.conf.train_mode.require_dataset_infer():
            assert self.conf.latent_infer_path is not None, \
                ("latent infer path must be set to train the semantic model. "
                 "Please run inference before and save results")
            self.dataloaders = get_dataloader(datasets=self.get_wrapper_model().cond_tensor_dict, **args)
        # Classical Loading and setup of Datasets
        else:
            self.dataloaders = get_dataloader(
                data_path=data_path,
                data_type=self.conf.data["type"],
                split_csv_paths=data_split_csv,
                img_size=self.conf.img_size,
                dims=self.conf.dims,
                **args
            )

    def setup_evaluation_samples(self):
        """
        Setup examples that are used for reconstruction and sampling
        """
        val_samples = None
        if self.dataloaders.get(Mode.val) is not None:
            val_samples = self.setup_evaluation_samples_for_dataloader(self.dataloaders[Mode.val])
        train_samples = self.setup_evaluation_samples_for_dataloader(self.dataloaders[Mode.train])
        return val_samples, train_samples

    def setup_evaluation_samples_for_dataloader(self, dataloader: DataLoader):
        """
        Extract reconstruction samples from the dataloader that are going to be used for evaluation metrics
        """
        assert self.conf.eval.eval_training_every_epoch <= 0 or self.conf.eval.num_reconstructions >= len(
            set(self.conf.classes.values())), f"You need at least one reconstruction per class ({len(set(self.conf.classes.values()))}), you defined {self.conf.eval.num_reconstructions}"
        num_samples = self.conf.eval.num_reconstructions
        sample_list = []
        target_list = []
        unique_targets = set()
        while num_samples > 0:
            for batch_idx, (features, targets, data_ids) in enumerate(dataloader):
                if num_samples == 0:
                    break
                if num_samples > self.conf.batch_size:
                    add_samples = self.conf.batch_size
                else:
                    add_samples = num_samples

                samples = features[:add_samples]
                targets = targets[:add_samples]
                unique_before = len(unique_targets)
                unique_targets.update(targets.tolist())
                if self.conf.classes is None or len(unique_targets) == len(
                        set(self.conf.classes.values())) or unique_before < len(unique_targets):
                    sample_list.append(samples.to(self.conf.device, dtype=self.conf.dtype))
                    target_list.append(targets.to(self.conf.device))
                    num_samples -= add_samples
        self.logger.file_logger.info(
            f"Created {self.conf.eval.num_reconstructions} samples: sample length={len(torch.cat(sample_list))}, target length={torch.cat(target_list)}")
        return torch.cat(sample_list), torch.cat(target_list)

    def switch_train_mode(self, mode: TrainMode):
        """
        Switch train mode but keep rest of model config
        Setup dataloaders again and as well as potential diffusion samplers
        Note: Avoid using this function but create a new model with new config and load previously saved model as pretrained one
        """
        self.conf.train_mode = mode
        self.get_wrapper_model().switch_train_mode(mode)
        self._setup_train_mode()

    def _setup_train_mode(self):
        # Get paths for data and optional data splits
        self.model_identifier = f"{self.conf.name}_{self.conf.train_mode.value}_{self.timestamp}"
        self.best_loss = self.conf.eval.best_loss

        # Dataloaders
        self.dataloaders = {}
        data_path = data_paths[self.conf.data["name"]]
        data_split_csv = split_csvs.get(self.conf.data["name"])
        self.setup_dataloaders(data_path, data_split_csv)

        self.val_samples = None
        self.train_samples = None
        if is_callable(self.get_wrapper_model(), "evaluate_training"):
            self.val_samples, self.train_samples = self.setup_evaluation_samples()

    def data_to_npz(self, path: str, save_single_imgs: bool = False):
        """
        Save data from train, validation and test set to file in numpy format for easier processing and loading
        :param save_single_imgs: save single images to different target folders
        :param path: base folder to save data to (will be saved in path_{mode} where mode is train, val or test)
        :type path: str
        """
        for mode, dataloader in self.dataloaders.items():
            data_path = os.path.join(f"{path}_{mode.value}")
            target_ids = defaultdict(int)
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            running_targets = []
            running_features = []
            for batch_idx, (features, targets, ids) in tqdm(enumerate(dataloader)):
                if save_single_imgs:
                    for j, target in enumerate(targets):
                        np_array = features[j].detach().cpu().numpy()
                        plot_3d_data_cloud(np_array[0], title=f"class_{target.item()}_{target_ids[target.item()]}", save_to=SaveTo.png, path=data_path)
                        target_ids[target.item()] += 1

                running_features.append(features)
                running_targets.append(targets)
            np.savez(os.path.join(data_path, "samples.npz"), images=torch.cat(running_features).detach().cpu().numpy(),
                     labels=torch.cat(running_targets).detach().cpu().numpy())


def ema(source, target, decay):
    """
    Exponential moving average model, integrates source and target model using decay
    :param source: new model
    :param target: old model
    :param decay: weight for old model, (1-decay) is weight for new model
    :return:
    """
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))
    target.load_state_dict(target_dict)


def is_time(num_samples: int, every_samples: int):
    """
    Calculates if it is time to evaluate
    :param num_samples: current total number of samples
    :param every_samples: whether every_samples is reached
    :return:
    """
    if every_samples > 0 and num_samples > 0:
        return num_samples % every_samples == 0
    else:
        return False
