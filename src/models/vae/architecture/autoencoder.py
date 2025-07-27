from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn

from src._types import Activation
from src.models.vae.architecture.decoder import Decoder
from src.models.vae.architecture.encoder import Encoder
from src.utils.config_functioanlities import ConfigFunctionalities


@dataclass
class VAEModelConfig(ConfigFunctionalities):
    in_channels: int = 1
    ch: int = 32
    num_layers: int = 2
    latent_size: int = 512
    num_classes: int = 5
    num_target_emb_channels: int = 64
    img_size: int = 64
    dims: int = 3
    kld_weight: int = 1.0
    last_act: Activation = Activation.sigmoid
    activation: Activation = Activation.lrelu
    dropout: float = 0.0

class VAEModel(nn.Module):

    def __init__(self, conf: VAEModelConfig):
        super().__init__()
        self.model_conf = deepcopy(conf)

        self.latent_size = conf.latent_size

        self.encoder = Encoder(
            conf.in_channels,
            conf.ch,
            conf.num_layers,
            conf.latent_size,
            conf.num_classes,
            conf.num_target_emb_channels,
            conf.img_size,
            conf.dims,
            conf.activation,
            dropout=conf.dropout
        )
        self.decoder = Decoder(
            conf.in_channels,
            conf.num_layers,
            conf.latent_size,
            conf.num_classes,
            conf.dims,
            self.encoder.bottleneck_shape,
            activation=conf.activation,
            last_act=self.model_conf.last_act,
            dropout=conf.dropout
        )

    def forward(self, x, target=None):
        mu, log_var = self.encoder(x, target)
        # log_var = log_var.clamp(min=-20, max=3)
        z = self.reparameterize(mu, log_var)
        x_pred = self.decoder(z, target)

        return x_pred, mu, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, target=None):
        x_pred = self.decoder(z, target)
        return x_pred

    def encode(self, x, target=None, mu_only: bool = False):
        mu, log_var = self.encoder(x, target)
        if mu_only:
            return mu
        z = self.reparameterize(mu, log_var)
        return z
