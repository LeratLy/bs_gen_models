from dataclasses import dataclass

from torch import nn

from src._types import Activation
from src.models.vae.architecture.nn import init_weights
from src.utils.config_functioanlities import ConfigFunctionalities


@dataclass
class MSClfNetConfig(ConfigFunctionalities):
    """
    A classification model that is used for testing the trianed models sampling ability
    """
    input_size = 96
    out_num = 0
    num_classes = 5
    input_ch = 1
    hidden_ch = 8
    dropout = 0.1
    final_dropout = 0.0
    num_hidden_layers = 0
    dims = 3


class MSClfNet(nn.Module):

    def __init__(self, conf: MSClfNetConfig):
        super().__init__()
        self.conf = conf
        layers = []

        ch_in = conf.input_ch
        ch_out = conf.hidden_ch
        for i in range(conf.num_hidden_layers):
            layers += [
                nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(ch_out),
                nn.LeakyReLU(),
                nn.MaxPool3d(2),
                nn.Dropout3d(self.conf.dropout),
            ]
            ch_in = ch_out
            ch_out = ch_in * 2

        self.middle_convs = nn.Sequential(*layers)
        self.out_conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(ch_out),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout3d(self.conf.dropout),
        )

        out_num = ch_out
        if self.conf.out_num > 0:
            out_num = self.conf.out_num
            self.lin1 = nn.Linear(ch_out, conf.out_num)

        self.lin2 = nn.Linear(out_num, conf.num_classes)
        init_weights(self.modules(), Activation.lrelu)

    def forward(self, x):
        x = self.middle_convs(x)
        x = self.out_conv(x)

        x = x.flatten(1)

        if self.conf.out_num > 0:
            x = self.lin1(x)
        x = self.lin2(x)
        return x
