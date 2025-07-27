"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
It was also extended by a bernoulli diffusion behavior inspired by
https://github.com/takimailto/BerDiff.git
Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch.amp import autocast
from torch.nn import Module
from tqdm import tqdm

from src._types import GenerativeType, ModelType, ModelMeanType, ModelVarType, LossType, NoiseType, ModelGradType
from src.models.dae.architecture.latentnet import LatentNetReturn
from src.models.dae.architecture.nn import mean_flat
from src.utils.config_functioanlities import ConfigFunctionalities
from src.metrics import mse_loss_l2, setup_loss, contrastive_loss
from src.utils.preprocessing import normalize_01

"""
    Copyright (C) 2024 LeratLy - All Rights Reserved
    You may use, distribute and modify this code under the
    terms of the MIT license.
    You should have received a copy of the MIT license with
    this file.

    Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology.
    Code of VISTEC has been used and modified in this file under terms of the MIT license.
"""


@dataclass
class DiffusionBeatGansConfig(ConfigFunctionalities):
    fp16: bool = False
    device: str = "cpu"
    gen_type: GenerativeType = GenerativeType.ddim
    loss_type: LossType = LossType.mse
    loss_type_eps: LossType = None
    loss_type_x_start: LossType = None
    noise_type: NoiseType = NoiseType.gaussian
    model_mean_type: ModelMeanType = ModelMeanType.eps
    model_grad_type: ModelGradType = ModelGradType.eps
    model_var_type: ModelVarType = ModelVarType.fixed_large
    rescale_timesteps: bool = False
    train_pred_xstart_detach: bool = True

    T: int = 1_000
    T_eval: int = 1_000
    T_sampler: str = 'uniform'

    beta_scheduler: str = "cosine"
    betas: np.array = None
    model_type: ModelType = None
    deterministic: bool = None
    num_classes: int = None


class DiffusionBeatGans:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """
    writer = None

    def __init__(self, conf: DiffusionBeatGansConfig):
        self.conf = conf
        self.model_mean_type = conf.model_mean_type
        self.model_grad_type = conf.model_grad_type
        self.model_var_type = conf.model_var_type
        self.loss_type = conf.loss_type
        self.rescale_timesteps = conf.rescale_timesteps

        # Use float64 for accuracy.
        betas = conf.betas.astype('float64')
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        self.init_scheduler_variables()
        self.loss = setup_loss(self.loss_type)
        self.loss_eps = setup_loss(conf.loss_type_eps) if conf.loss_type_eps else None
        self.loss_x_start = setup_loss(conf.loss_type_x_start) if conf.loss_type_x_start else None

    def init_scheduler_variables(self):
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod -
                                                   1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) /
                                   (1.0 - self.alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = (self.betas *
                                     np.sqrt(self.alphas_cumprod_prev) /
                                     (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) *
                                     np.sqrt(self.alphas) /
                                     (1.0 - self.alphas_cumprod))

    def training_losses(self,
                        model: Module,
                        x_start: torch.Tensor,
                        t: torch.Tensor,
                        model_kwargs=None,
                        noise: torch.Tensor = None,
                        target: torch.Tensor = None):
        """
        Compute training losses for a batch of timesteps.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :param target: if specified, the label which should also be embedded into the timestep conditioning.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}

        if target is not None:
            model_kwargs["target"] = target

        if noise is None:
            noise = self.get_noise(x_start, t)

        # get noisy version of x_start
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {'x_t': x_t, 'noise': noise}

        # l1 loss is for latent and mse for standard DDIM
        if self.loss_type in [
            LossType.mse,
            LossType.l1,
            LossType.bce_logits,
            LossType.bce,
            LossType.l1_sum
        ]:
            with autocast(self.conf.device, dtype=torch.bfloat16 if self.conf.fp16 else x_t.dtype):
                # x_t is static wrt. to the diffusion process
                model_forward = model.forward(x=x_t.detach(),
                                              t=self._scale_timesteps(t),
                                              x_start=x_start.detach(),
                                              **model_kwargs)
            # prediction is the noise which x_t possibly contains
            model_output = model_forward.pred
            terms['model_pred'] = model_forward.pred

            _model_output = model_output
            if self.conf.train_pred_xstart_detach:
                _model_output = _model_output.detach()
            # get the pred xstart and mean and variance of the resulting posterior distribution
            p_mean_var = self.p_mean_variance(
                model=DummyModel(pred=_model_output),
                # gradient goes through x_t
                x=x_t,
                t=t,
                clip_denoised=False,
                normalize_denoised=False)
            terms['pred_xstart'] = p_mean_var['pred_xstart']

            # model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            pred_target_types = {
                ModelGradType.eps: noise,
                ModelGradType.x_start: x_start,
            }
            if model_kwargs.get("target") is not None and not isinstance(model_forward, LatentNetReturn):
                pred_target_types[ModelGradType.contrastive] = contrastive_loss(model_forward.cond, model_kwargs["target"])

            if self.loss is not None:
                if self.model_mean_type == ModelMeanType.eps:
                    # (n, c, h, w) => (n, )
                    if self.model_grad_type.value.startswith('mix'):
                        # use x_start and eps
                        if self.model_grad_type == ModelGradType.mix_start_eps:
                            loss_x_start = self.loss_x_start or self.loss
                            loss_eps = self.loss_eps or self.loss
                            terms[self.loss_type.value] = (1 / 2) * (
                                    loss_eps(model_output, pred_target_types[ModelGradType.eps])
                                    + loss_x_start(terms['pred_xstart'], pred_target_types[ModelGradType.x_start])
                            )
                        else:
                            raise NotImplementedError
                    else:
                        pred_target = pred_target_types[self.model_grad_type]
                        assert model_output.shape == pred_target.shape == x_start.shape
                        # use eps
                        if self.model_grad_type == ModelGradType.eps:
                            pred = model_output
                        # use x_start
                        elif self.model_grad_type == ModelGradType.x_start:
                            pred = self._predict_xstart_from_eps(x_t, t, model_output)
                        else:
                            raise NotImplementedError()
                        terms[self.loss_type.value] = self.loss(pred, pred_target)
                else:
                    raise NotImplementedError()

            # We do not learn the variance here
            if "vb" in terms:
                # if learning the variance also use the vlb loss
                terms["loss"] = terms[self.loss_type.value] + terms["vb"]
            else:
                terms["loss"] = terms[self.loss_type.value]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    @abstractmethod
    def get_noise(self, x_start, t=None):
        """
        Get initial random noise with the same shape as x_start
        """
        raise NotImplementedError

    @abstractmethod
    def get_noise_by_shape(self, shape, device, kwargs):
        """
        Get initial random noise with the same shape as x_start
        """
        raise NotImplementedError

    def sample(self,
               model: Module,
               shape=None,
               noise=None,
               cond=None,
               x_start=None,
               clip_denoised=True,
               model_kwargs=None,
               progress=True,
               normalize_denoised=False,
               target=None,
               kwargs=None):
        """
        Args:
            :param x_start: given for the autoencoder
            :param clip_denoised:
            :param model_kwargs:
            :param progress:
            :param shape:
            :param model:
            :param noise:
            :param cond:
        """
        if model_kwargs is None:
            model_kwargs = {}
            if self.conf.model_type == ModelType.autoencoder:
                model_kwargs['x_start'] = x_start
                model_kwargs['cond'] = cond
            if self.conf.num_classes is not None:
                model_kwargs['target'] = target
        if kwargs is None:
            kwargs = {}

        if self.conf.gen_type == GenerativeType.ddpm:
            return self.p_sample_loop(model,
                                      shape=shape,
                                      noise=noise,
                                      clip_denoised=clip_denoised,
                                      model_kwargs=model_kwargs,
                                      progress=progress,
                                      normalize_denoised=normalize_denoised,
                                      **kwargs)
        elif self.conf.gen_type == GenerativeType.ddim:
            return self.ddim_sample_loop(model,
                                         shape=shape,
                                         noise=noise,
                                         clip_denoised=clip_denoised,
                                         model_kwargs=model_kwargs,
                                         progress=progress,
                                         normalize_denoised=normalize_denoised,
                                         **kwargs)
        else:
            raise NotImplementedError()

    @abstractmethod
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        raise NotImplementedError

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0). = sqrt(alpha_t)*x_0 + sqrt(1-alpha_t) * noise

        :param x_start: the initial data batch.
        :type x_start: Array
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step. (a batch)
        :param noise: if specified, the split-out normal noise of noise is None a random noise is generated with the same dimensions as x_start.
        :type noise: Array
        :return: A noisy version of x_start.
        :rtype: Array
        """
        if noise is None:
            noise = self.get_noise(x_start, t)
        assert noise.shape == x_start.shape
        return self._q_sample(x_start, t, noise)

    @abstractmethod
    def _q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0). = sqrt(alpha_t)*x_0 + sqrt(1-alpha_t) * noise

        :param x_start: the initial data batch.
        :type x_start: Array
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step. (a batch)
        :param noise: if specified, the split-out normal noise of noise is None a random noise is generated with the same dimensions as x_start.
        :type noise: Array
        :return: A noisy version of x_start.
        :rtype: Array
        """
        raise NotImplementedError

    @abstractmethod
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior as defined in Ho et al:

            q(x_{t-1} | x_t, x_0)

        """
        raise NotImplementedError

    def p_mean_variance(self,
                        model: Module,
                        x,
                        t,
                        clip_denoised=True,
                        denoised_fn=None,
                        model_kwargs=None,
                        normalize_denoised=False):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x (x_0).

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t with noisy images of the current batch (N is the same as the length of t).
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        with autocast(self.conf.device, dtype=torch.bfloat16 if self.conf.fp16 else x.dtype):
            model_forward = model.forward(x=x,
                                          t=self._scale_timesteps(t),
                                          **model_kwargs)
        # noise prediction for the noisy images
        model_output = model_forward.pred

        # do not learn variance but use fixed one based on beta schedule and log variance
        model_variance, model_log_variance = None, None
        if self.model_var_type in [
            ModelVarType.fixed_large, ModelVarType.fixed_small
        ]:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.fixed_large: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(
                        np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.fixed_small: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t,
                                                      x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            # order is important clip_denoised > normalize_denoise (therefore, must be explicitly set to False)
            if clip_denoised:
                if self.conf.noise_type == NoiseType.gaussian:
                    return x.clamp(-1, 1)
                elif self.conf.noise_type == NoiseType.xor:
                    return x.clamp(0, 1)
            elif normalize_denoised:
                return normalize_01(x)
            return x

        # how should mean of the resulting gaussian be calculated
        if self.model_mean_type in [
            ModelMeanType.eps,
        ]:
            # get model mean based on epsilon hence predicted x_start, x_t and t
            if self.model_mean_type == ModelMeanType.eps:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t,
                                                  eps=model_output))
            else:
                raise NotImplementedError()
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t)
            # use pred start if t == 0
            if self.conf.noise_type == NoiseType.xor:
                model_mean = torch.where((t == 0)[:, *((None,) * (len(model_mean.shape) - 1))], pred_xstart, model_mean)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_log_variance is None or (model_mean.shape == model_log_variance.shape ==
                                              pred_xstart.shape == x.shape)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            'model_forward': model_forward,
        }

    @abstractmethod
    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        Predict the original image based on the given timestep and noise
        :param x_t: noisy image
        :param t:
        :param eps:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        raise NotImplementedError

    @abstractmethod
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        raise NotImplementedError

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            # scale t to be maxed out at 1000 steps
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (p_mean_var["mean"].float() +
                    p_mean_var["variance"] * gradient.float())
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
            self,
            model: Module,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            normalize_denoised=False,
            deterministic=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        raise NotImplementedError

    def p_sample_loop(
            self,
            model: Module,
            shape=None,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            normalize_denoised=False,
            deterministic=None,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                normalize_denoised=normalize_denoised,
                deterministic=deterministic
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
            self,
            model: Module,
            shape=None,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            normalize_denoised=False,
            deterministic=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            assert isinstance(shape, (tuple, list))
            img = self.get_noise_by_shape(*shape, device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * len(img), device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    normalize_denoised=normalize_denoised,
                    deterministic=deterministic
                )
                yield out
                img = out["sample"]

    @abstractmethod
    def ddim_sample(
            self,
            model: Module,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
            normalize_denoised=False,
            deterministic=None
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        raise NotImplementedError

    def ddim_reverse_sample(
            self,
            model: Module,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
            normalize_denoised=False,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        raise NotImplementedError

    def ddim_reverse_sample_loop(
            self,
            model: Module,
            x,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
            device=None,
            normalize_denoised=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        sample_t = []
        xstart_t = []
        T = []
        indices = list(range(self.num_timesteps))
        sample = x
        for i in tqdm(indices):
            t = torch.tensor([i] * len(sample), device=device)
            with torch.no_grad():
                out = self.ddim_reverse_sample(model,
                                               sample,
                                               t=t,
                                               clip_denoised=clip_denoised,
                                               denoised_fn=denoised_fn,
                                               model_kwargs=model_kwargs,
                                               eta=eta,
                                               normalize_denoised=normalize_denoised, )
                sample = out['sample']
                # [1, ..., T]
                sample_t.append(sample)
                # [0, ...., T-1]
                xstart_t.append(out['pred_xstart'])
                # [0, ..., T-1] ready to use
                T.append(t)

        return {
            #  xT "
            'sample': sample,
            # (1, ..., T)
            'sample_t': sample_t,
            # xstart here is a bit different from sampling from T = T-1 to T = 0
            # may not be exact
            'xstart_t': xstart_t,
            'T': T,
        }

    def ddim_sample_loop(
            self,
            model: Module,
            shape=None,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            normalize_denoised=False,
            deterministic=None,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        samples = self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            normalize_denoised=normalize_denoised,
            deterministic=deterministic,
        )
        # plot_3d_data_cloud(
        #     noise[0][0].detach(),
        #     "init_noise",
        #     save_to=SaveTo.tensorboard,
        #     writer=self.writer,
        #     step=0
        # )
        for i, sample in enumerate(samples):
            #     if self.writer:
            #         plot_3d_data_cloud(
            #             sample["sample"][0][0].detach(),
            #             "diffusion_steps",
            #             save_to=SaveTo.tensorboard,
            #             writer=self.writer,
            #             step=i+1)
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
            self,
            model: Module,
            shape=None,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=True,
            eta=0.0,
            normalize_denoised=False,
            deterministic=None
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            assert isinstance(shape, (tuple, list))
            self.get_noise_by_shape(*shape, device)

        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:

            if isinstance(model_kwargs, list):
                # index dependent model kwargs
                # (T-1, ..., 0)
                _kwargs = model_kwargs[i]
            else:
                _kwargs = model_kwargs

            t = torch.tensor([i] * len(img), device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=_kwargs,
                    eta=eta,
                    normalize_denoised=normalize_denoised,
                    deterministic=deterministic,
                )
                out['t'] = t
                yield out
                img = out["sample"]

    def _vb_terms_bpd(self,
                      model: Module,
                      x_start,
                      x_t,
                      t,
                      clip_denoised=True,
                      model_kwargs=None,
                      normalize_denoised=False):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(model,
                                   x_t,
                                   t,
                                   clip_denoised=clip_denoised,
                                   model_kwargs=model_kwargs,
                                   normalize_denoised=normalize_denoised)
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"],
                       out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"])
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {
            "output": output,
            "pred_xstart": out["pred_xstart"],
            'model_forward': out['model_forward'],
        }

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size,
                         device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean,
                             logvar1=qt_log_variance,
                             mean2=0.0,
                             logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self,
                      model: Module,
                      x_start,
                      clip_denoised=True,
                      model_kwargs=None,
                      normalize_denoised=False):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        assert self.conf.noise_type is not NoiseType.xor, "xor variational lower bound is not supported yet"
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with torch.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    normalize_denoised=normalize_denoised
                )
            vb.append(out["output"])
            xstart_mse.append(mse_loss_l2(out["pred_xstart"], x_start))
            eps = self._predict_eps_from_xstart(x_t, t_batch,
                                                out["pred_xstart"])
            mse.append(mse_loss_l2(eps, noise))

        vb = torch.stack(vb, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps) -> np.array:
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """

    # xor scheduler based on https://github.com/vkinakh/binary-diffusion-tabular/blob/main/binary_diffusion_tabular/diffusion.py
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start,
                           beta_end,
                           num_diffusion_timesteps,
                           dtype=np.float64)
    elif schedule_name == "cosine_xor":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            max_beta=0.02
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "const0.01":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.01] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.015":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.015] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.008":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.008] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0065":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0065] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0055":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0055] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0045":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0045] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0035":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0035] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0025":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0025] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0015":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0015] * num_diffusion_timesteps,
                        dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) +
                  ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (
            1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min,
                    torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


class DummyModel(torch.nn.Module):
    def __init__(self, pred):
        super().__init__()
        self.pred = pred

    def forward(self, *args, **kwargs):
        return DummyReturn(pred=self.pred)


class DummyReturn(NamedTuple):
    pred: torch.Tensor
