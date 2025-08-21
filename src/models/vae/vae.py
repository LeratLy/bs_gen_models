import os

import numpy as np
import torch
from ray import tune as ray_tune
from sklearn.metrics import classification_report

from src.analysis.plots import create_confusion_matrix
from src.models.autoencoder_base import BaseModel
from src._types import TrainMode, Mode, SaveTo
from src.metrics import setup_loss, mse_loss_l1
from src.utils.checkpoints import load_model
from src.utils.ray import is_ray_running
from src.utils.writer import add_scalar_dict


class VAE(BaseModel):
    """
    A full conditional variational autoencoder model
    """

    # annealing_idx = -1

    def _init_model(self):
        """
        Initialize the diffusion autoencoder model my registering the stochastic sub code randomly, setting up samplers
        as well as diffusion noise setup.
        :return:
        """
        self.no_checkpoint_vars = ["clf."]
        if self.conf.clf_conf is not None and self.conf.clf_conf.load_path is not None:
            self.clf = load_model(self.conf.clf_conf.load_path, self.conf.clf_conf.classifier_name,
                                  self.conf.clf_conf.classifier_conf, self.logger, self.writer)
            self.clf.eval()
            self.clf.to(self.conf.device)
        else:
            self.clf = None
        self.loss_func = setup_loss(self.conf.loss_type)
        self._setup_train_mode()
        #
        # if self.conf.warmup_epochs and self.conf.cyclicl_annealing_plateau:
        #     self.annealing_idx = 0
        #     self.current_annealing_cycle = 0
        #     self.annealing_epochs = self.conf.num_epochs // self.conf.warmup_epochs

    def _setup_train_mode(self):
        pass

    def _forward_model(self, model, batch, target, mode: Mode):
        """
        Forward pass for the whole conditional variational autoencoder the config further defines what exactly this returns

        :param model: the model, used for the forward pass
        :param batch: current batch data
        :param target: targets o current batch data
        :return:
        """
        # forward
        if self.conf.train_mode == TrainMode.base:
            target = target if self.conf.model_conf.num_classes else None
            x_pred, mu, log_var, z = self.model(batch, target)
        else:
            raise NotImplementedError
        loss, kld, kld_weight, reconstruction = self.loss_function(x_pred, batch, mu, log_var,
                                                                   self.conf.model_conf.kld_weight)
        if (self.batch_idx + 1) % self.conf.log_interval == 0:
            self.logger.file_logger.info(
                f"kld loss: {kld}, reconstruction loss: {reconstruction}, 'kld_weight': {kld_weight}")
            self.writer.add_scalar(f'loss_{mode.value}', loss, self.num_samples_total)
        return {'loss': loss, "kld": kld, "reconstruction": reconstruction}

    def reconstruct(self, x, target=None, ema=True, kwargs=None):
        model = self.ema_model if ema else self.model
        t = target[:x.shape[0]] if target is not None else None
        z = model.encode(x, t, mu_only=True)
        return self.render(z, target=t, ema=ema, kwargs=kwargs)

    def sample(self, batch_size=1, ema=True, target=None, kwargs=None):
        """
        Sample an image with random gaussian noise and sampled semantic sub-code
        """
        z = torch.randn(batch_size, self.conf.model_conf.latent_size, device=self.conf.device)
        t = target[:batch_size] if target is not None else None
        return self.render(z, target=t, ema=ema, kwargs=kwargs)

    def render(self, z, target=None, ema=True, kwargs=None):
        model = self.model if not ema else self.ema_model
        t = target[:z.shape[0]] if target is not None else None
        return model.inference(z, t)

    def loss_function(self, x_pred, x, mu, log_var, kld_weight=1.0):
        if self.conf.warmup_epochs:
            final_kld_weight = self.kld_annealing(kld_weight)
        else:
            final_kld_weight = kld_weight
        reconstruction_loss = self.loss_func(x_pred, x, kwargs={'reduction': 'sum'}) / x.size(0)
        # Auto-Encoding Variational Bayes
        # https://arxiv.org/abs/1312.6114
        kld = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / x.size(0)
        return reconstruction_loss + final_kld_weight * kld, kld, final_kld_weight, reconstruction_loss

    def kld_annealing(self, kld_weight):
        # if self.conf.cyclicl_annealing_plateau:
        #     if ((self.epoch_idx + 1) % self.annealing_epochs) == 0:
        #         self.annealing_idx = 0
        #         self.current_annealing_cycle += 1
        #         return kld_weight
        #     elif self.annealing_idx < self.conf.cyclicl_annealing_plateau:
        #         self.annealing_idx += 1
        #         return kld_weight
        #     else:
        #         return ((self.epoch_idx + 1) + (self.current_annealing_cycle = -1
        #                                                     ) * self.conf.cyclicl_annealing_plateau) / (
        #                                        self.annealing_epochs * self.current_annealing_cycle + 1)
        return min(1, (self.epoch_idx + 1) / self.conf.warmup_epochs) * kld_weight

    def evaluate_training(self, val_samples: tuple[torch.tensor, torch.tensor],
                          train_samples: tuple[torch.tensor, torch.tensor]):
        """
        Evaluate training by sampling and reconstructing
        """
        # Sample adn evaluate samples
        self.logger.file_logger.info("\nStarted sampling...")
        num_classes = self.conf.clf_conf.classifier_conf.model_conf.num_classes if self.conf.clf_conf is not None else len(set(self.conf.classes.values()))
        sampled, sampled_targets = self.create_samples_per_class(self.conf.eval.num_evals,
                                                                 num_classes,
                                                                 min(self.conf.batch_size, self.conf.eval.num_evals))
        unique_values, counts = torch.unique(sampled_targets, return_counts=True)
        self.logger.file_logger.info(f"\nEvaluating {self.conf.eval.num_evals} samples -> {unique_values}: {counts}")
        self.logger.file_logger.info(f"Report samples:\n{self.evaluate_metrics(sampled, sampled_targets)}")
        # faster than passing it directly
        # self.render_example(sampled, f"ChP Sampled")

        # Sample and visually render samples
        sampled, sampled_targets = self.create_samples_per_class(self.conf.eval.num_visual_samples,
                                                                 num_classes,
                                                                 min(self.conf.eval.num_visual_samples,
                                                                     self.conf.batch_size))
        for i, img in enumerate(sampled):
            self.render_example(img[0], f"ChP Sampled/{i} - target: {sampled_targets[i]}")
        self.logger.file_logger.info("...done with sampling.")

        # Reconstruct and visually render
        for mode, samples in zip(["val", "train"], [val_samples, train_samples]):
            if samples is not None:
                reconstructed = self.evaluate_reconstruction(samples[0], samples[1], title_add="val")
                # faster than passing it directly
                for i, img in enumerate(reconstructed[0]):
                    self.render_example(img[0], f"ChP Reconstructions {mode}/{i}")

    def evaluate_reconstruction(self, imgs, targets=None, title_add=None):
        """
        Reconstruct samples for given images
        """
        kwargs = {}
        if title_add is not None:
            kwargs['title_add'] = title_add

        self.logger.file_logger.info("\nStarted reconstructing...")
        reconstructed = self.reconstruct(imgs, target=targets)
        self.logger.file_logger.info("...done with reconstruction.")

        l1_reconstruction_loss = mse_loss_l1(reconstructed, imgs)

        if is_ray_running():
            ray_tune.report({
                "reconstruction_loss": l1_reconstruction_loss.mean().item()
            })
        self.logger.file_logger.info(
            f"Mean l1 reconstruction loss {l1_reconstruction_loss.mean().item()}\n"
        )
        return reconstructed, targets

    def evaluate_metrics(self, features, targets, batch_size=4):
        """
        Compute f1, precision and recall for the given list of results
        """
        if self.clf is not None and features.shape[0] > 1:
            # Make predictions
            assert features.shape[0] // batch_size, "total number of samples must eb dividable by batch_size"
            n_batches = features.shape[0] // batch_size
            predictions = []
            targets = targets.detach().cpu().numpy()
            for batch_idx in range(n_batches):
                preds, probs = self.clf.classify(features[batch_idx * batch_size:(batch_idx + 1) * batch_size])
                predictions.append(preds.detach().cpu().numpy())
            # To cpu for evaluation
            predictions = np.concatenate(predictions)

            # Evaluate
            create_confusion_matrix(targets, predictions, save_to=SaveTo.tensorboard, step=self.epoch_idx,
                                    writer=self.writer, labels=self.conf.classes)
            report = classification_report(targets, predictions, output_dict=True, zero_division=0.0)

            if is_ray_running():
                ray_tune.report(report)

            add_scalar_dict(report["macro avg"], self.writer, "Evaluation metric", self.epoch_idx)
            return report
        return {}

    def infer_whole_folder(self, folder_path, exclude_folders=None, fil_name: str = "samples.npz", batch_size: int = 1, use_initial_labels=False):
        """
        Infer semantic sub-code for all samples in the dataloaders dataset
        :param fil_name:
        :param batch_size:
        :param use_initial_labels:
        :param folder_path: folder with numpy arrays that should be inferred
        :type folder_path: str
        :param exclude_folders: list of folders to exclude
        :type exclude_folders: list
        :return: semantic embeddings in same order as data is saved in dataset as tensor num_samples x encoding_size
        :rtype: torch.Tensor
        """
        self.ema_model.eval()
        cond_tensor = torch.tensor([]).to(self.conf.device)
        target_tensor = torch.tensor([]).to(self.conf.device)
        initial_targets_tensor = torch.tensor([]).to(self.conf.device)
        file_names_tensor = torch.tensor([]).to(self.conf.device)
        files = [file for file in os.listdir(folder_path) if file.endswith(fil_name)]
        for i, file in enumerate(files):
            if exclude_folders is not None and file in exclude_folders:
                continue
            with torch.no_grad():
                data = np.load(os.path.join(folder_path, file))
                features = torch.tensor(data['images'], device=self.conf.device, dtype=self.conf.dtype)
                targets = torch.tensor(data['initial_labels' if use_initial_labels else 'labels'], device=self.conf.device)
                num_batches = features.shape[0] // batch_size
                for j in range(num_batches):
                    start_idx = j * batch_size
                    end_idx = j + 1 * batch_size
                    cond = self.ema_model.encode(features[start_idx:end_idx], targets[start_idx:end_idx])
                    cond_tensor = torch.cat([cond_tensor, cond], dim=0)

                    if use_initial_labels:
                        targets = torch.tensor(data['labels'], device=self.conf.device)
                    target_tensor = torch.cat([target_tensor, targets[start_idx:end_idx].unsqueeze(1)], dim=0)
                    initial_targets = torch.tensor(data['initial_labels'], device=self.conf.device) if data.get('initial_labels') is not None else targets
                    initial_targets_tensor = torch.cat([initial_targets_tensor, initial_targets[start_idx:end_idx].unsqueeze(1)], dim=0)
                # Not working yet with .npz, pkl must be used instead
                # if data.get('file_names') is not None:
                #     file_names = torch.tensor(data['file_names'], device=self.conf.device)
                #     file_names_tensor = torch.cat([file_names_tensor, file_names.unsqueeze(1)], dim=0)
        return cond_tensor, target_tensor, initial_targets_tensor, file_names_tensor
