# Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology
# Copyright (c) 2024 LeratLy

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from ray import tune as ray_tune

from src.analysis.plots import create_confusion_matrix
from src.config import BaseConfig
from src.metrics import mse_loss_l1, bce_loss
from src.models.autoencoder_base import BaseModel
from run_models.model_templates import assign_diffusion_conf, make_t_sampler
from src._types import TrainMode, ModelType, Mode, SaveTo, NoiseType
from src.utils.checkpoints import load_model
from src.utils.preprocessing import unnormalize_gaussian
from src.utils.ray import is_ray_running
from src.utils.visualisation import plot_3d_data_cloud, plot_3d_cloud_norm
from src.data.datasets import load_tensor_dict_dataset
from src.utils.writer import add_scalar_dict


class DAE(BaseModel):
    """
    A full diffusion autoencoder model
    """
    cond_mean = None
    cond_tensor_dict = None
    cond_std = None
    cond_class_std = None
    cond_class_mean = None

    def _init_model(self):
        """
        Initialize the diffusion autoencoder model my registering the stochastic sub code randomly, setting up samplers
        as well as diffusion noise setup.
        :return:
        """
        self.no_checkpoint_vars = ["clf."]
        if self.conf.clf_conf is not None and self.conf.clf_conf.load_path is not None:
            self.logger.file_logger.info("Loading clf for evaluation...")
            self.clf = load_model(self.conf.clf_conf.load_path, self.conf.clf_conf.classifier_name,
                                  self.conf.clf_conf.classifier_conf, self.logger, self.writer)
            self.clf.eval()
            self.clf.to(self.conf.device)
            self.logger.file_logger.info(f"... done loading clf:\n {self.conf.clf_conf.load_path}")
        else:
            self.clf = None

        # Setup samplers
        self.sampler = assign_diffusion_conf(self.conf, self.conf.diffusion_conf.T).make_sampler()
        self.eval_sampler = assign_diffusion_conf(self.conf, self.conf.diffusion_conf.T_eval).make_sampler()
        # this is shared for both model and latent
        self.T_sampler = make_t_sampler(self.conf)

        self._setup_train_mode()

        # latent stats for sampling saved from previous
        if self.conf.model_conf.latent_net_conf and self.conf.latent_infer_path is not None:
            self.logger.file_logger.info('loading latent stats ...')
            state = torch.load(self.conf.latent_infer_path, weights_only=False)
            self.cond_tensor_dict = load_tensor_dict_dataset(state['cond_tensor_dict'])
            self.cond_mean = state['cond_mean'][None, :]
            self.cond_std = state['cond_std'][None, :]
            self.cond_class_mean = state.get('cond_class_mean')
            self.cond_class_std = state.get('cond_class_std')
            self.logger.file_logger.info(
                f'...done loading stats.\n con_mean = {self.cond_mean} and cond_std = {self.cond_std}')

    def _setup_train_mode(self):
        if self.conf.train_mode == TrainMode.latent_diffusion:
            self.latent_sampler = assign_diffusion_conf(self.conf,
                                                        self.conf.latent_diffusion_conf.T,
                                                        latent=True).make_sampler()
            self.eval_latent_sampler = assign_diffusion_conf(self.conf,
                                                             self.conf.latent_diffusion_conf.T_eval,
                                                             latent=True).make_sampler()
        else:
            self.latent_sampler = None
            self.eval_latent_sampler = None

    def _forward_model(self, model, batch, target, mode: Mode):
        """
        Forward pass for the whole diffusion autoencoder the config further defines what exactly this returns

        :param model: the model, used for the forward pass
        :param batch: current batch data
        :param target: targets o current batch data
        :return:
        """
        add_metrics = {}
        # forward
        if self.conf.train_mode == TrainMode.base:
            """
            Main training mode
            """
            target = target if self.conf.model_conf.num_classes else None
            x_start = batch
            # with numpy seed we have the problem that the sample t's are related!
            t, weight = self.T_sampler.sample(len(x_start), x_start.device)
            losses = self.sampler.training_losses(model=model, x_start=x_start, t=t, target=target)

            # Add reconstruction to
            if mode != Mode.train and self.conf.add_running_metrics is not None and "reconstruction" in self.conf.add_running_metrics and (
                    self.epoch_idx + 1) % self.conf.eval.eval_training_every_epoch == 0:
                sampled = self.render(x_start, target=target)
                add_metrics["reconstruction"] = bce_loss(x_start, sampled)
        elif self.conf.train_mode.require_dataset_infer():
            """
            Training semantic model (latent ddim)
            """
            target = target if self.conf.model_conf.latent_net_conf.num_classes else None
            # this mode as pre-calculated cond
            cond = batch

            if self.conf.model_conf.latent_net_conf.class_znormalize:
                assert target is not None and self.cond_class_mean is not None and self.cond_class_std is not None, "class dependent std and mean are not inferred"
                # Stack per-class stats for the batch
                class_mean = torch.stack([self.cond_class_mean[c.item()].to(cond.device) for c in target])
                class_std = torch.stack([self.cond_class_std[c.item()].to(cond.device) for c in target])

                cond = (cond - class_mean) / class_std
            elif self.conf.model_conf.latent_net_conf.znormalize:
                assert self.cond_mean is not None and self.cond_std is not None, "std and mean are not inferred"
                cond = (cond - self.cond_mean.to(
                    cond.device)) / self.cond_std.to(cond.device)

            # diffusion on the latent
            t, weight = self.T_sampler.sample(len(cond), cond.device)
            latent_losses = self.latent_sampler.training_losses(model=model, x_start=cond, t=t, target=target)
            # train only do the latent diffusion
            losses = {
                'latent': latent_losses['loss'],
                'loss': latent_losses['loss']
            }
        else:
            raise NotImplementedError()

        for key in ['loss', 'vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
            if key in losses:
                losses[key] = (weight * losses[key]).mean()

        self.writer.add_scalar(f'loss_{mode.value}', losses['loss'], self.num_samples_total)
        for key in ['vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
            if key in losses:
                self.writer.add_scalar(f'loss_{mode.value}/{key}', losses[key], self.num_samples_total)

        loss = losses['loss']
        # TODO REMOVE ONLY FOR DEBUGGING
        # if self.num_samples_total % 1200 == 0:
        #     plot_3d_cloud_norm(torch.abs(losses['model_pred'] - losses['noise'])[0][0].detach(),
        #                        f"noise_diff_{mode.value}", save_to=SaveTo.tensorboard, writer=self.writer,
        #                        step=self.num_samples_total)
        return {'loss': loss} | add_metrics

    def encode(self, x, ema=True, target=None):
        """
        Semantically encode the input data into the latent representation vector

        :param x: input data
        :return: latent representation of the input data
        """
        # assert self.conf.model_type == ModelType.autoencoder, "Do not use latent model for reconstruction, it contains latent data, no images"
        model = self.ema_model if ema else self.model
        cond = model.encoder.forward(x, label=target)
        return cond

    def encode_stochastic(self, x, cond, T=None, ema=True):
        """
        Stochastically encode the input data int a stochastic sub-code

        :param x: input data
        :param cond: TODO
        :param T: number if timesteps
        :type T: int
        :return: xT the stochastic code of the input data
        """
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = assign_diffusion_conf(self.conf, T).make_sampler()

        model = self.ema_model if ema else self.model

        out = sampler.ddim_reverse_sample_loop(model, x, model_kwargs={'cond': cond})
        return out['sample']

    def _render(self, x_start=None, cond=None, x_T=None, T=None, ema=True, kwargs=None) -> torch.tensor:
        """
        Render a given image by encoding x_start to semantic sub-code and using random noise as x_T if not defined.
        """
        assert x_start is not None or cond is not None, "For rendering either a condition or image must be given"

        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = assign_diffusion_conf(self.conf, T).make_sampler()

        # conditional sub-code (z_semantic)
        if cond is None:
            cond = self.encode(x_start, ema=ema)

        # random stochastic sub-code (z_stochastic)
        if x_T is None:
            x_T = create_random(sampler, len(cond), self.conf, kwargs)

        model = self.ema_model if ema else self.model
        sampler.writer = self.writer
        pred_img = sampler.sample(model=model,
                                  noise=x_T,
                                  model_kwargs={'cond': cond},
                                  clip_denoised=self.conf.clip_denoised,
                                  normalize_denoised=self.conf.normalize_denoised,
                                  kwargs=kwargs)
        if self.conf.diffusion_conf.noise_type == NoiseType.gaussian:
            pred_img = unnormalize_gaussian(pred_img)
        return pred_img

    def render(self, x_start, ema=True, target=None, kwargs=None):
        if kwargs is None:
            kwargs = {}
        return self._render(x_start=x_start, T=kwargs.get("T"), ema=ema, kwargs=kwargs)

    def reconstruct(self, x_start, target=None, ema=True, kwargs=None):
        if kwargs is None:
            kwargs = {}
        kwargs["deterministic"] = True
        T = kwargs.get("T")
        self.logger.file_logger.info(
            f"x_start stats: {x_start.min().item()}, {x_start.max().item()}, {x_start.mean().item()}")
        self.logger.file_logger.info("...extracting semantic sub-code")
        cond = self.encode(x_start, ema=ema, target=target)
        self.logger.file_logger.info(f"cond stats: {cond.min().item()}, {cond.max().item()}, {cond.mean().item()}")

        self.logger.file_logger.info("...extracting stochastic sub-code")
        x_T = self.encode_stochastic(x_start, cond, T, ema=ema)
        plot_3d_data_cloud(x_T[0][0], "x_T", save_to=SaveTo.tensorboard, writer=self.writer)
        self.logger.file_logger.info(f"x_T stats: {x_T.min().item()}, {x_T.max().item()}, {x_T.mean().item()}")
        self.logger.file_logger.info("...predicting image")

        return self._render(x_start=x_start, T=T, x_T=x_T, cond=cond, ema=ema, kwargs=kwargs)

    def sample(self, batch_size=1, ema=True, target=None, kwargs=None):
        """
        Sample an image with random gaussian noise and sampled semantic sub-code
        """
        if kwargs is None:
            kwargs = {}
        T = kwargs.get("T")
        clip_latent_noise = kwargs.get("clip_latent_noise", False)

        if T is None:
            sampler_latent = self.eval_latent_sampler
        else:
            sampler_latent = assign_diffusion_conf(self.conf, T, latent=True).make_sampler()

        if self.conf.train_mode == TrainMode.latent_diffusion:
            latent_noise = create_random_semantic_noise(batch_size, self.conf)
        else:
            raise NotImplementedError()

        if clip_latent_noise:
            latent_noise = latent_noise.clip(-1, 1)

        model = self.ema_model if ema else self.model
        cond = sampler_latent.sample(
            model=model.latent_net,
            noise=latent_noise,
            clip_denoised=self.conf.model_conf.latent_net_conf.clip_sample,
            target=target
        )

        if self.conf.model_conf.latent_net_conf.class_znormalize:
            assert target is not None and self.cond_class_mean is not None and self.cond_class_std is not None, "class dependent std and mean are not inferred"
            # Stack per-class stats for the batch
            class_means = torch.stack([self.cond_class_mean[c.item()].to(cond.device) for c in target])
            class_stds = torch.stack([self.cond_class_std[c.item()].to(cond.device) for c in target])

            cond = cond * class_stds + class_means
        elif self.conf.model_conf.latent_net_conf.znormalize:
            cond = cond * self.cond_std.to(cond.device) + self.cond_mean.to(cond.device)

        return self._render(cond=cond, T=T, ema=ema, kwargs=kwargs)

    def infer(self, dataloaders: dict[Mode, DataLoader], mode: Mode = None, save_path: str = None):
        """
        Infer semantic encoding for all datasets (train, val and test if defined in dataloaders)
        For each set save cond_mean and std
        """
        save_path = save_path if save_path is not None else f'{self.conf.checkpoint["dir"]}/{self.conf.name}/latent.pkl'
        modes = [mode] if mode is not None else [mode for mode in Mode]
        modes = [m for m in modes if m in dataloaders.keys()]
        self.logger.file_logger.info("Inferring whole dataset...")
        cond_tensor = []
        cond_tensor_by_class = defaultdict(list)
        cond_tensor_dict = {}
        for mode in modes:
            features, targets, ids = self.infer_whole_dataset(dataloaders.get(mode))
            cond_tensor_dict[mode] = {
                "features": features,
                "targets": targets,
                "ids": ids,
            }
            cond_tensor.append(features)

            for feature, target , id in zip(features, targets, ids):
                initial_id = dataloaders.get(mode).dataset.initial_labels[int(id)]
                cond_tensor_by_class[int(initial_id)].append(feature)
        cond_tensor_cat = torch.cat(cond_tensor)
        self.cond_mean = cond_tensor_cat.mean(dim=0)
        self.cond_std = cond_tensor_cat.std(dim=0)
        self.cond_class_mean = {}
        self.cond_class_std = {}

        for cls, feature_list in cond_tensor_by_class.items():
            cond_tensor_cls = torch.stack(feature_list)
            self.cond_class_mean[cls] = cond_tensor_cls.mean(dim=0)
            self.cond_class_std[cls] = cond_tensor_cls.std(dim=0)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        torch.save({
            'cond_tensor_dict': cond_tensor_dict,
            'cond_mean': self.cond_mean,
            'cond_std': self.cond_std,
            'cond_class_std': self.cond_class_std,
            'cond_class_mean': self.cond_class_mean,
        }, save_path)

        self.logger.file_logger.info(f"Done inferring whole dataset for modes '{modes}'")
        self.conf.latent_infer_path = save_path
        self.cond_tensor_dict = load_tensor_dict_dataset(cond_tensor_dict)
        return self.cond_tensor_dict, self.cond_mean, self.cond_std

    def infer_whole_dataset(self, dataloader, model=None):
        """
        Infer semantic sub-code for all samples in the dataloaders dataset

        :param dataloader: dataloader from which samples should be loaded
        :type dataloader: DataLoader
        :param model: model which should be used for encoding the semantic vectors
        :type model: BaseModel
        :return: semantic embeddings in same order as data is saved in dataset as tensor num_samples x encoding_size
        :rtype: torch.Tensor
        """
        if model is None:
            model = self.ema_model
        model.eval()
        cond_tensor = torch.tensor([]).to(self.conf.device)
        id_tensor = torch.tensor([]).to(self.conf.device)
        target_tensor = torch.tensor([]).to(self.conf.device)
        for (features, target, data_id) in tqdm(dataloader, total=len(dataloader), desc='infer'):
            with torch.no_grad():
                features = features.to(self.conf.device).type(self.conf.dtype)
                data_id = data_id.to(self.conf.device)
                target = target.to(self.conf.device)
                cond = model.encoder(features, label=target if self.conf.model_conf.num_classes else None)
                if cond.dim() == 3:
                    cond = cond.flatten(0, 1)
                cond_tensor = torch.cat([cond_tensor, cond], dim=0)
                target_tensor = torch.cat([target_tensor, target], dim=0)
                id_tensor = torch.cat([id_tensor, data_id], dim=0)
        sorted_indices = torch.argsort(id_tensor)
        # back to training mode

        if self.conf.train_mode == TrainMode.latent_diffusion:
            model.latent_net.train()
        else:
            model.train()
        return cond_tensor[sorted_indices].detach().cpu(), target_tensor[sorted_indices].detach().cpu(), id_tensor[sorted_indices].detach().cpu()

    def normalize(self, cond):
        """
        Standardize conditional vector to have 0 mean and std of 1

        :param cond: conditional vector
        :return: standardized conditional vector
        """
        cond = (cond - self.cond_mean.to(self.device)) / self.cond_std.to(
            self.device)
        return cond

    def denormalize(self, cond):
        """
        Reverse standardization of condition vector

        :param cond: conditional vector
        :return: non-standardized conditional vector in its original scale
        """
        cond = (cond * self.cond_std.to(self.device)) + self.cond_mean.to(
            self.device)
        return cond

    def evaluate_training(self, val_samples: tuple[torch.tensor, torch.tensor],
                          train_samples: tuple[torch.tensor, torch.tensor]):
        """
        Evaluate training by sampling and reconstructing
        """
        # 1. Sample and render visually using num_samples
        if self.conf.train_mode != TrainMode.latent_diffusion:
            assert val_samples is not None, "Please provide validation examples to pseudo samples images based on their latent code"
            self.logger.file_logger.info("Start sampling with semantic code from a validation base image...")
            sampled = self.render(val_samples[0][:self.conf.eval.num_visual_samples],
                                  target=val_samples[1][:self.conf.eval.num_visual_samples])
        else:
            self.logger.file_logger.info("Start sampling with inferred semantic and stochastic code...")
            sampled, _ = self.create_samples_per_class(self.conf.eval.num_visual_samples,
                                                       self.conf.clf_conf.classifier_conf.model_conf.num_classes,
                                                       min(self.conf.batch_size, self.conf.eval.num_visual_samples))
        for i, img in enumerate(sampled):
            self.render_example(img[0], f"ChP Sampled/{i}")
        self.logger.file_logger.info("...done with sampling.")

        # 2. Sample and evaluate metrics (done for more samples that visual inspection)
        self.logger.file_logger.info("Start evaluating more samples...")
        if self.conf.train_mode != TrainMode.latent_diffusion:
            sampled, sampled_targets = self.create_samples_for_images(val_samples[0], val_samples[1],
                                                                      self.conf.eval.num_evals,
                                                                      min(self.conf.batch_size,
                                                                          self.conf.eval.num_evals))
        else:
            sampled, sampled_targets = self.create_samples_per_class(self.conf.eval.num_evals,
                                                                     self.conf.clf_conf.classifier_conf.model_conf.num_classes,
                                                                     min(self.conf.batch_size,
                                                                         self.conf.eval.num_evals))
        self.logger.file_logger.info(f"Report samples:\n{self.evaluate_metrics(sampled, sampled_targets)}")

        # 3. Reconstruct (not possible for latent mode, dataloader does not has access to original data)
        if self.conf.train_mode != TrainMode.latent_diffusion:
            for mode, samples in zip(["val", "train"], [val_samples, train_samples]):
                if samples is not None:
                    reconstructed = self.evaluate_reconstruction(samples[0], samples[1], title_add="val")
                    # faster than passing it directly
                    for i, img in enumerate(reconstructed[0]):
                        self.render_example(img[0], f"ChP Reconstructions {mode}/{i}")

    def create_samples_for_images(self, samples, targets, num_samples, batch_size=1):
        """
        Since we can not sample directly for non-latent mode, we can only vary the stochastic subcode with encoded
        semantic one. It is done in batches and given samples are repeated until num_evals is matched.
        """
        assert samples.shape[0] % num_samples == 0 and samples.shape[
            0] % batch_size == 0, f"samples.shape[0]={samples.shape[0]}, num_samples={num_samples}, batch_size={batch_size}"
        n_iterations = samples.shape[0] // num_samples
        n_batches = samples.shape[0] // batch_size
        sampled = []
        sampled_targets = targets.repeat(n_iterations)
        for i in range(n_iterations):
            for batch_idx in range(n_batches):
                sampled.append(self.render(samples[batch_idx * batch_size:(batch_idx + 1) * batch_size],
                                           target=targets[batch_idx * batch_size:(batch_idx + 1) * batch_size]))
        sampled = torch.cat(sampled)
        return sampled, sampled_targets

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

    def evaluate_metrics(self, features, targets, batch_size=1):
        """
        Compute f1, precision and recall for the given list of results
        """
        if self.clf is not None and features.shape[0] > 1:
            # Make predictions
            assert features.shape[
                       0] // batch_size, f"total number of samples {features.shape[0]} must be dividable by batch_size {batch_size}"
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

    # TODO combine dae and vae code (can be brought together in base class)
    def infer_whole_folder(self, folder_path, exclude_folders=None, fil_name: str = "samples.npz", batch_size: int = 1, use_initial_labels=False):
        """
        Infer semantic sub-code for all samples in the dataloaders dataset
        :param use_initial_labels:
        :param batch_size:
        :param fil_name:
        :param folder_path: folder with numpy arrays that should be inferred
        :type folder_path: str
        :param exclude_folders: list of folders to exclude
        :type exclude_folders: list
        :return: semantic embeddings in same order as data is saved in dataset as tensor num_samples x encoding_size
        :rtype: torch.Tensor
        """
        self.ema_model.eval()
        cond_tensor = torch.tensor([])
        target_tensor = torch.tensor([])
        initial_targets_tensor = torch.tensor([])
        file_names_tensor = torch.tensor([])
        files = [file for file in os.listdir(folder_path) if file.endswith(fil_name)]
        for i, file in enumerate(files):
            if exclude_folders is not None and file in exclude_folders:
                continue
            with torch.no_grad():
                data = np.load(os.path.join(folder_path, file))
                features = torch.tensor(data['images'], device=self.conf.device, dtype=self.conf.dtype)
                targets = torch.tensor(data['labels'], device=self.conf.device)
                initial_targets = torch.tensor(data['initial_labels'], device=self.conf.device) if data.get('initial_labels') is not None else targets
                num_batches = features.shape[0] // batch_size
                for j in range (num_batches):
                    start_idx = j * batch_size
                    end_idx = j + 1 * batch_size
                    targets_batch = initial_targets[start_idx:end_idx] if use_initial_labels else targets[start_idx:end_idx]
                    cond = self.ema_model.encoder(features[start_idx:end_idx], label=targets_batch if self.conf.num_classes is not None else None)
                    if cond.dim() == 3:
                        cond = cond.flatten(0, 1)
                    cond_tensor = torch.cat([cond_tensor, cond.detach().cpu()], dim=0)

                    target_tensor = torch.cat([target_tensor, targets[start_idx:end_idx].unsqueeze(1).detach().cpu()], dim=0)
                    initial_targets_batch = initial_targets[start_idx:end_idx]
                    initial_targets_tensor = torch.cat([initial_targets_tensor, initial_targets_batch.unsqueeze(1).detach().cpu()], dim=0)
                # Not working yet with .npz, pkl must be used instead
                # if data.get('file_names') is not None:
                #     file_names = torch.tensor(data['file_names'], device=self.conf.device)
                #     file_names_tensor = torch.cat([file_names_tensor, file_names.unsqueeze(1)], dim=0)
        return cond_tensor, target_tensor, initial_targets_tensor, file_names_tensor


def create_random(sampler, batch_size: int, conf: BaseConfig, kwargs=None):
    """
    Create random gaussian noise for an image

    :param batch_size: batch size of noise tensor
    :type batch_size: int
    :param conf: defines channels, and shape of noise data
    :type conf: BaseConfig
    :return: tensor containing random gaussian noise
    :rtype: torch.Tensor
    """
    return sampler.get_noise_by_shape(shape=(
        batch_size,
        conf.model_conf.in_channels,
        *(conf.dims * [conf.img_size])
    ), device=conf.device, kwargs=kwargs)


def create_random_semantic_noise(batch_size: int, conf: BaseConfig):
    """
    Create random gaussian noise for an image

    :param batch_size: batch size of noise tensor
    :type batch_size: int
    :param conf: defines channels, and shape of noise data
    :type conf: BaseConfig
    :return: tensor containing random gaussian noise
    :rtype: torch.Tensor
    """
    return torch.randn(batch_size, conf.style_ch, device=conf.device)
