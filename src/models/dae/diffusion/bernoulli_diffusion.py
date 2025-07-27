from abc import ABC

import torch
from torch.distributions import Binomial
from torch.nn import Module

from src._types import SaveTo
from src.models.dae.diffusion.base import DiffusionBeatGans, _extract_into_tensor, DiffusionBeatGansConfig
from src.utils.visualisation import plot_3d_data_cloud

"""
Bernoulli diffusion 
Includes code from: https://github.com/takimailto/BerDiff.git
"""


def beta_weighted(x, alpha):
    """
    Linear interpolation between pure noise (0.5) and the original signal x
    :param x:
    :param alpha:
    :return:
    """
    return alpha * x + (1 - alpha) / 2


def extract_props(x, alpha):
    """
    TODO
    :param x:
    :param alpha:
    :return:
    """
    x_prob = beta_weighted(x, alpha)
    x_prob_rev = beta_weighted((1 - x), alpha)
    return x_prob, x_prob_rev


class BernoulliDiffusionBeatGans(DiffusionBeatGans, ABC):
    def __init__(self, conf: DiffusionBeatGansConfig):
        super().__init__(conf)
        self.model_var_type = None

    def get_noise(self, x_start, t=None):
        """
        Get initial random noise with the same shape as x_start
        """
        flip_prop = (1 - _extract_into_tensor(self.alphas_cumprod, t, x_start.shape)) / 2
        # sample mask with 1 for flip bit and 0 for no flip
        return torch.bernoulli(flip_prop).to(x_start.device)

    def get_noise_by_shape(self, shape, device, kwargs):
        """
        Get initial random noise with the same shape as x_start
        """
        base_probs = torch.ones(shape) / 2
        if kwargs and kwargs.get("deterministic"):
            return base_probs.to(device)
        return Binomial(1, base_probs).sample().to(device)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = beta_weighted(x_start, _extract_into_tensor(self.alphas_cumprod, t, x_start.shape))
        return mean, None, None

    def _q_sample(self, x_start, t, mask=None):
        """
        :param mask: equivalent to the gaussian noise parameter, a binary random noise mask (1 for flip 0 for no flip) this already depends on t
        """
        dtype = x_start.dtype
        if mask is None:
            mask = self.get_noise(x_start, t)
        # flip bits everywhere where mask is 1
        x_t = torch.bitwise_xor(x_start.int(), mask.int()).to(dtype)
        return x_t.to(x_start.device)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        posterior mean of q(x_{t-1} | x_t, x0)
        """
        alpha_t = _extract_into_tensor(self.alphas, t, x_t.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x_t.shape)
        x_t_prob, x_t_prob_rev = extract_props(x_t, alpha_t)
        x_t_prev, x_t_prev_rev = extract_props(x_t, alpha_bar_prev)
        numerator = x_t_prob * x_t_prev
        denominator = x_t_prob * x_t_prev + x_t_prob_rev * x_t_prev_rev
        mean = numerator / denominator
        return mean, None, None

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        Predict the original image based on the given timestep and noise
        Flip bits back (via xor is similar to subtracting the noise and taking abs)
        :param x_t: noisy image
        :param t:
        :param eps:
        :return:
        """
        assert x_t.shape == eps.shape
        return torch.abs(x_t - eps).to(device=t.device).float()

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        """
        TODO
        :param x_t:
        :param t:
        :param xprev:
        :return:
        """
        assert x_t.shape == xprev.shape
        alpha_t = _extract_into_tensor(self.alphas, t, x_t.shape)
        alpha_prev = _extract_into_tensor(self.alphas_cumprod, t - 1, x_t.shape)
        x_t_prob, x_t_prob_rev = extract_props(x_t, alpha_t)
        noise_prev = (1 - alpha_prev) / 2
        numerator = (x_t_prob_rev * noise_prev * xprev
                     + x_t_prob * noise_prev * (xprev - 1)
                     + x_t_prob_rev * xprev * alpha_prev)
        denominator = (x_t_prob + x_t_prob_rev * xprev - x_t_prob * xprev) * alpha_prev
        return numerator / denominator

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """
        TODO
        :param x_t:
        :param t:
        :param pred_xstart:
        :return:
        """
        assert x_t.shape == pred_xstart.shape
        return torch.abs(x_t - pred_xstart).to(device=x_t.device).float()

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
            deterministic=None,
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
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            normalize_denoised=normalize_denoised
        )

        sample = torch.bernoulli(out["mean"])  # out["mean"] if deterministic else torch.bernoulli(out["mean"])
        if t[0] != 0:
            return {"sample": sample.to(x.device), "pred_xstart": out["pred_xstart"]}
        else:
            return {"sample": out["mean"], "pred_xstart": out["pred_xstart"]}

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
            deterministic=None,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            normalize_denoised=normalize_denoised,
        )
        if t[0] != 0:
            return self._get_ddim_result(x, t, out["pred_xstart"], eta=eta, deterministic=deterministic)
        else:
            return {"sample": out["mean"], "pred_xstart": out["pred_xstart"]}

    def _get_ddim_result(self, x, t, x_start, eta=.0, deterministic=False):
        """
        Compute x_{t-1} from the model for given x, t and x_start

        Same usage as p_sample().
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar))
        mean = (
                sigma * x
                + (alpha_bar_prev - sigma * alpha_bar) * x_start
                + ((1 - alpha_bar_prev) - (1 - alpha_bar) * sigma)/2
                )
        sample = torch.bernoulli(mean)  # using mean does not make sense (since model is sued to {0,1})
        return {"sample": sample.to(x.device), "pred_xstart": x_start}

    def ddim_reverse_sample(
            self,
            model: Module,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
            normalize_denoised=False
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        !!! Note: Does not make sense for bernoulli but is implemented anyway for compatibility in usage !!!
        """
        assert eta == 0, "for deterministic path only"
        return {"sample": torch.full_like(x, 0.5, device=x.device), "pred_xstart": torch.full_like(x, 0.5, device=x.device)}
        # out = self.p_mean_variance(
        #     model,
        #     x,
        #     t,
        #     clip_denoised=clip_denoised,
        #     denoised_fn=denoised_fn,
        #     model_kwargs=model_kwargs,
        #     normalize_denoised=normalize_denoised,
        # )
        # alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        # # add noise based on beta schedule to get x_t+1
        # x_t_next = beta_weighted(out["pred_xstart"], alpha_bar_next)
        # # never use bernoulli here since we want to have static results
        # sample = x_t_next.to(x.device)
        # plot_3d_data_cloud(sample[0][0], f"ChP_pred_{t[0]}", save_to=SaveTo.png)
        # return {"sample": sample, "pred_xstart": out["pred_xstart"]}
