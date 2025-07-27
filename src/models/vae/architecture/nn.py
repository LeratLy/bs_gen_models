from torch import nn
from torch.nn import init

from src._types import Activation


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(min(32, channels), channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def init_weights(modules, activation: Activation):
    for module in modules:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            if activation == Activation.relu:
                init.kaiming_normal_(module.weight,
                                     a=0,
                                     nonlinearity='relu')
            elif activation == Activation.lrelu:
                init.kaiming_normal_(module.weight,
                                     a=0.2,
                                     nonlinearity='leaky_relu')
            elif activation == Activation.silu:
                init.kaiming_normal_(module.weight,
                                     a=0,
                                     nonlinearity='relu')
            else:
                # leave it as default
                pass
