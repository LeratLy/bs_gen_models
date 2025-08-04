import numpy as np
import torch
from torch import nn
from torcheval.metrics.functional import multiclass_f1_score

from src._types import LossType
from src.models.dae.architecture.nn import mean_flat

"""
Collection of different loss and score functions used by the models for training and for the analysis
"""


class CrossEntropyWeightedLoss(nn.Module):
    """
    Cross Entropy loss with class weights
    """

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
    :param embeddings: embeddings for which contrastive loss should be applied
    :param labels: labels of embeddings
    :param tau: temperature (lower temperature leads to increase punishment on hard negative samples
    :return:
    """

    def calc_cosine(vec1, vec2):
        return torch.nn.functional.normalize(vec1, p=2, dim=1) @ torch.nn.functional.normalize(vec2, p=2, dim=1).T

    batch_size = embeddings.shape[0]
    # Generate random shifts
    shifts = torch.FloatTensor(batch_size, 512).uniform_(-0.001, 0.01).to(embeddings.device)
    # Generate random scales
    scales = torch.FloatTensor(batch_size, 512).uniform_(-2, 4).to(embeddings.device)
    # Apply shifts and scales to the input tensor
    embeddings = (embeddings + shifts) * scales

    # select the indices appropriately
    indices = torch.tensor(
        np.array([idx for (idx, label) in enumerate(list(labels.detach().cpu().numpy().astype(int))) if
                  label in [1, 2, 3, 4]])).to(embeddings.device)

    # get the negative ones
    neg_indices = torch.tensor(
        np.array(
            [idx for (idx, label) in enumerate(list(labels.detach().cpu().numpy().astype(int))) if label == 0])).to(
        embeddings.device)

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
    """
    Binary Cross Entropy loss working with logits output from a model
    """
    if kwargs is None:
        kwargs = {}
    criterion = nn.BCEWithLogitsLoss(**kwargs)
    return criterion(model_input, target)


def bce_loss(model_input, target, kwargs=None):
    """
    Binary Cross Entropy loss working with normalised outputs from a model
    """
    if kwargs is None:
        kwargs = {}
    criterion = nn.BCELoss(**kwargs)
    return criterion(model_input, target)


def ce_loss(model_input, target, kwargs=None):
    """
    Cross Entropy loss working with normalised outputs from a model
    """
    if kwargs is None:
        kwargs = {}
    criterion = nn.CrossEntropyLoss(**kwargs)
    return criterion(model_input, target)


def weighted_ce_loss(model_input, target, kwargs=None):
    """
    Weighted Cross Entropy loss assigning higher and lower weights to losses based on predefined weights
    """
    if kwargs is None:
        kwargs = {}
    criterion = CrossEntropyWeightedLoss(**kwargs)
    return criterion(model_input, target)


def mse_loss_l2(model_input, target, kwargs=None):
    """
    Mean L2 loss
    """
    return mean_flat((target - model_input) ** 2)


def mse_loss_l1(model_input, target, kwargs=None):
    """
    Mean L1 loss
    """
    return mean_flat((target - model_input).abs())


def sum_loss_l1(model_input, target, kwargs=None):
    """
    Summed L1 loss
    """
    return (target - model_input).abs().sum()


def macro_inverse_f1(model_input, target, kwargs=None):
    """
    Macro weighted inverse f1 score
    """
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
    reduce_axis = list(range(2, target.ndim))
    intersection = (pred * target).sum(axis=reduce_axis)
    union = pred.sum(axis=reduce_axis) + target.sum(axis=reduce_axis)

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()


def setup_loss(loss_type):
    """
    Setup loss based on loss_type.
    loss is a function that takes (model_input, target) and return the corresponding loss
    # (n, c, h, w) => (n, )
    """
    if loss_type is None:
        return None
    if loss_type == LossType.bce:
        return bce_loss
    elif loss_type == LossType.bce_logits:
        return bce_logits_loss
    elif loss_type == LossType.mse:
        return mse_loss_l2
    elif loss_type == LossType.l1:
        return mse_loss_l1
    elif loss_type == LossType.l1_sum:
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
    :param a: first score
    :param b: second score
    :return: harmonic mean of a and b
    """
    if a + b == 0:
        return 0
    return (2 * a * b) / (a + b)
