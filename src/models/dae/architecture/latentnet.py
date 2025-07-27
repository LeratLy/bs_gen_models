from torch.nn import init
from .unet import *
from src._types import Activation, LatentNetType
from src.models.dae.architecture.nn import timestep_embedding
"""
    Copyright (C) 2024 LeratLy - All Rights Reserved
    You may use, distribute and modify this code under the
    terms of the MIT license.
    You should have received a copy of the MIT license with
    this file.

    Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology.
    Code of VISTEC has been used and modified in this file under terms of the MIT license.
"""


class LatentNetReturn(NamedTuple):
    pred: torch.Tensor = None


@dataclass
class MLPSkipNetConfig:
    """
    default MLP for the latent DPM in the paper!
    """
    num_channels: int = 512
    num_hid_channels: int = 1024
    num_layers: int = 10
    num_time_emb_channels: int = 64
    num_target_emb_channels: int = 64
    num_time_layers: int = 2
    num_target_layers: int = 2

    condition_bias: float = 0
    target_bias: float = 0
    dropout: float = 0

    activation: Activation = Activation.silu
    last_act: Activation = Activation.none

    time_last_act: bool = False
    target_last_act: bool = False
    use_norm: bool = False
    use_target: bool = False
    clip_sample: bool = False
    class_znormalize: bool = False
    znormalize = None

    skip_layers: Tuple[int, ...] = ()  # which layers are skip layers?
    latent_type: LatentNetType = LatentNetType.skip

    num_classes: int = None
    shift_target: bool = False
    scale_target_alpha: bool = False


class MLPSkipNet(nn.Module):
    """
    concat x to hidden layers

    default MLP for the latent DPM in the paper!
    """

    def __init__(self, conf: MLPSkipNetConfig):
        super().__init__()
        self.conf = conf

        self.time_embed = nn.Sequential(
            *self.create_conditional_layers(conf.num_time_layers, conf.num_time_emb_channels, conf.time_last_act)
        )
        cond_channels = conf.num_channels
        if self.conf.num_classes:
            self.target_embedding = nn.Embedding(self.conf.num_classes, self.conf.num_target_emb_channels)
            self.target_embed = nn.Sequential(
                *self.create_conditional_layers(conf.num_target_layers, conf.num_target_emb_channels,
                                                conf.target_last_act)
            )

        self.layers = nn.ModuleList([])
        for i in range(conf.num_layers):
            if i == 0:
                act = conf.activation
                norm = conf.use_norm
                cond = True
                a, b = conf.num_channels, conf.num_hid_channels
                dropout = conf.dropout
            elif i == conf.num_layers - 1:
                act = Activation.none
                norm = False
                cond = False
                a, b = conf.num_hid_channels, conf.num_channels
                dropout = 0
            else:
                act = conf.activation
                norm = conf.use_norm
                cond = True
                a, b = conf.num_hid_channels, conf.num_hid_channels
                dropout = conf.dropout

            if i in conf.skip_layers:
                a += conf.num_channels

            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=cond_channels,
                    use_cond=cond,
                    use_target=self.conf.use_target,
                    condition_bias=conf.condition_bias,
                    target_bias=conf.target_bias,
                    dropout=dropout,
                    shift_target=self.conf.shift_target,
                    scale_target_alpha=self.conf.scale_target_alpha,
                ))
        self.last_act = conf.last_act.get_act()

    def forward(self, x, t, target=None, **kwargs):
        assert not self.conf.num_classes or target is not None, ("You are using a target including mode, therefore also "
                                                                "pass the target to the forward function of the model.")
        t = timestep_embedding(t, self.conf.num_time_emb_channels)
        cond = self.time_embed(t)
        if self.conf.num_classes:
            target = target.to(dtype = torch.int32)
            target = self.target_embedding(target)
            target = self.target_embed(target)
        else:
            target = None
        h = x
        for i in range(len(self.layers)):
            if i in self.conf.skip_layers:
                # injecting input into the hidden layers
                h = torch.cat([h, x], dim=1)
            h = self.layers[i](x=h, cond=cond, target=target)
        h = self.last_act(h)
        return LatentNetReturn(h)

    def create_conditional_layers(self, num_layers: int, init_channels: int, last_act: bool):
        """
        Create linear embedding layers for time and label
        """
        layers = []
        for i in range(num_layers):
            if i == 0:
                a = init_channels
                b = self.conf.num_channels
            else:
                a = self.conf.num_channels
                b = self.conf.num_channels
            layers.append(nn.Linear(a, b))
            if i < num_layers - 1 or last_act:
                layers.append(self.conf.activation.get_act())
        return layers


class MLPLNAct(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            norm: bool,
            use_cond: bool,
            use_target: bool,
            activation: Activation,
            cond_channels: int,
            condition_bias: float = 0,
            target_bias: float = 0,
            dropout: float = 0,
            shift_target: bool = False,
            scale_target_alpha: bool = False,
    ):
        super().__init__()
        self.activation = activation
        self.condition_bias = condition_bias
        self.target_bias = target_bias
        self.use_cond = use_cond
        self.use_target = use_target
        self.shift_target = shift_target
        self.scale_target_alpha = scale_target_alpha

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = activation.get_act()
        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels)
            self.cond_layers = nn.Sequential(self.act, self.linear_emb)
            if self.use_target:
                self.target_scale_layers = nn.Sequential(self.act, nn.Linear(cond_channels, out_channels))
                if shift_target: self.target_shift_layers = nn.Sequential(self.act, nn.Linear(cond_channels, out_channels))
                if scale_target_alpha: self.target_scale_alpha_layers = nn.Sequential(self.act, nn.Linear(cond_channels, out_channels))
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == Activation.relu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                elif self.activation == Activation.lrelu:
                    init.kaiming_normal_(module.weight,
                                         a=0.2,
                                         nonlinearity='leaky_relu')
                elif self.activation == Activation.silu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x, cond=None, target=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            cond = (cond, None)

            # scale shift first
            x = x * (self.condition_bias + cond[0])
            if cond[1] is not None:
                x = x + cond[1]

            if self.use_target:
                target_scale = self.target_scale_layers(target)
                x *= (self.target_bias + target_scale)
                if self.shift_target:
                    target_shift = self.target_shift_layers(target)
                    x += target_shift

            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        if self.use_cond and self.use_target:
            if self.scale_target_alpha:
                x *= self.target_scale_alpha_layers(target)
        return x
