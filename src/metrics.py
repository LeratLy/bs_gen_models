import numpy as np
import torch
from torch import nn
from torcheval.metrics.functional import multiclass_f1_score

from src._types import LossType
from src.models.dae.architecture.nn import mean_flat


class CrossEntropyWeightedLoss(nn.Module):
    def __init__(self, device, weights: np.ndarray):
        super().__init__()
        assert weights is not None
        self.class_weights = torch.from_numpy(weights).to(device, dtype=torch.float)

    def forward(self, model_input, target, kwargs=None):
        if kwargs is None:
            kwargs = {}
        criterion = nn.CrossEntropyLoss(weight=self.class_weights, **kwargs)
        return criterion(model_input, target)

def contrastive_loss(embeddings: torch.tensor, labels: torch.tensor, tau=0.7):
    """
    Adapted contrastive loss computation from https://github.com/A-Ijishakin/Contrast-DiffAE/tree/main
    :param embeddings:
    :param labels:
    :param tau:
    :return:
    """
    def calc_cosine(vec1, vec2):
        return torch.nn.functional.normalize(vec1, p=2, dim=1) @ torch.nn.functional.normalize(vec2, p=2, dim=1).T

    shift_range, scale_range = 0.005, 2.3

    batch_size = embeddings.shape[0]
    # Generate random shifts
    shifts = torch.FloatTensor(batch_size, 512).uniform_(-0.001, 0.01).to(embeddings.device)

    # Generate random scales
    scales = torch.FloatTensor(batch_size, 512).uniform_(-2, 4).to(embeddings.device)

    # Apply shifts and scales to the input tensor
    embeddings = (embeddings + shifts) * scales

    # select the indices appropriately
    indices = torch.tensor(
        np.array([idx for (idx, label) in enumerate(list(labels.detach().cpu().numpy().astype(int))) if label in [1,2,3,4]])).to(embeddings.device)

    # get the negative ones
    neg_indices = torch.tensor(
        np.array([idx for (idx, label) in enumerate(list(labels.detach().cpu().numpy().astype(int))) if label == 0])).to(embeddings.device)

    # get the positive and negative embeddings
    if neg_indices.numel() > 0 and indices.numel() > 0:
        pos = embeddings[indices]
        neg = embeddings[neg_indices]

        # calculate the contrastive loss
        loss = - torch.log(
            torch.exp(calc_cosine(pos, pos) / tau) / torch.sum(torch.exp(calc_cosine(pos, neg) / tau))).mean()
    else:
        loss = 0
    return loss


def bce_logits_loss(model_input, target, kwargs=None):
    if kwargs is None:
        kwargs = {}
    criterion = nn.BCEWithLogitsLoss(**kwargs)
    return criterion(model_input, target)


def bce_loss(model_input, target, kwargs=None):
    if kwargs is None:
        kwargs = {}
    criterion = nn.BCELoss(**kwargs)
    return criterion(model_input, target)


def ce_loss(model_input, target, kwargs=None):
    if kwargs is None:
        kwargs = {}
    criterion = nn.CrossEntropyLoss(**kwargs)
    return criterion(model_input, target)


def weighted_ce_loss(model_input, target, kwargs=None):
    if kwargs is None:
        kwargs = {}
    criterion = CrossEntropyWeightedLoss(**kwargs)
    return criterion(model_input, target)


def mse_loss_l2(model_input, target, kwargs=None):
    return mean_flat((target - model_input) ** 2)


def mse_loss_l1(model_input, target, kwargs=None):
    return mean_flat((target - model_input).abs())


def sum_loss_l1(model_input, target, kwargs=None):
    return (target - model_input).abs().sum()

def macro_inverse_f1(model_input, target, kwargs=None):
    return 1 - multiclass_f1_score(model_input, target, average="macro", **kwargs)

def dice_coef(pred, target, kwargs=None):
    """
    Compute dice score for batches of predictions and targets
    :param pred: prediction with N batches with tensors with values in [0,1]
    :param target: targets with N batches with tensors with values in [0,1]
    :param kwargs:
    :return:
    """
    smooth = kwargs.get('smooth') if kwargs is not None and kwargs.get("smooth") is not None else 1
    # Calculate intersection and union
    reduce_axis = list(range(2, target.ndim))
    intersection = (pred * target).sum(axis=reduce_axis)
    union = pred.sum(axis=reduce_axis) + target.sum(axis=reduce_axis)

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()


def setup_loss(loss_type):
    """
    Setup loss based on loss_type.
    loss is a function that takes (model_input, target) and return the corresponding loss
    """
    if loss_type is None:
        return None
    if loss_type == LossType.bce:
        return bce_loss
    elif loss_type == LossType.bce_logits:
        return bce_logits_loss
    # simple loss (mean squared error between actual noise added and the noise predicted by the model
    elif loss_type == LossType.mse:
        return mse_loss_l2
    # simple loss for latent model (l1 error between actual noise added and the noise predicted by the model
    elif loss_type == LossType.l1:
        # (n, c, h, w) => (n, )
        return mse_loss_l1
    elif loss_type == LossType.l1_sum:
        # (n, c, h, w) => (n, )
        return sum_loss_l1
    elif loss_type == LossType.cel:
        return ce_loss
    elif loss_type == LossType.w_cel:
        return weighted_ce_loss
    elif loss_type == LossType.m_if1:
        return macro_inverse_f1
    else:
        raise NotImplementedError()


def get_class_weights(class_counts: np.array):
    """
    Return weights based on class counts (summing to 1)
    """
    raw_weights = 1 / class_counts
    return raw_weights / raw_weights.sum()

def harmonic_mean(a, b):
    """
    Harmonic mean of a and b
    :param a:
    :param b:
    :return:
    """
    if a+b == 0:
        return 0
    return (2*a*b)/(a+b)