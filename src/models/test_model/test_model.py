from dataclasses import dataclass

import torch
from torch import nn

from src.utils.config_functioanlities import ConfigFunctionalities


@dataclass
class SimpleModelConfig(ConfigFunctionalities):
    """
    A simple model that is used for testing
    """
    in_features = 21952
    out_features1 = 200
    out_features2 = 1
    img_size = 0


class SimpleModel(nn.Module):
    def __init__(self, conf: SimpleModelConfig):
        self.model_conf = conf
        super().__init__()
        self.linear1 = nn.Linear(conf.in_features, conf.out_features1)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(conf.out_features1, conf.out_features2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.linear1(x.flatten().type(torch.float))
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x