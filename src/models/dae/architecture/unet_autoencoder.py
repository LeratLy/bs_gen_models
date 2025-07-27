from copy import deepcopy
from typing import NamedTuple

import torch
from torch import nn, Tensor

from src.models.dae.architecture.blocks import ResBlock
from src.models.dae.architecture.latentnet import MLPSkipNet
from src.models.dae.architecture.nn import linear, timestep_embedding
from src.models.dae.architecture.unet import BeatGANsUNetModel, BeatGANsEncoderModel, BeatGANsAutoencoderConfig
from src._types import LatentNetType, Activation

"""
    Copyright (C) 2024 LeratLy - All Rights Reserved
    You may use, distribute and modify this code under the
    terms of the MIT license.
    You should have received a copy of the MIT license with
    this file.

    Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology.
    Code of VISTEC has been used and modified in this file under terms of the MIT license.
"""


class AutoencoderReturn(NamedTuple):
    pred: Tensor
    cond: Tensor = None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None
    # style only (but could depend on time)
    style: Tensor = None


class BeatGANsAutoencoderModel(BeatGANsUNetModel):
    """
    UNet Autoencoder of the DAE model
    """

    def __init__(self, conf: BeatGANsAutoencoderConfig):
        super().__init__(conf)
        self.model_conf = deepcopy(conf)

        # having only time, cond
        self.time_embed = TimeStyleSeparateEmbed(
            time_channels=self.model_conf.net.ch,
            time_out_channels=self.model_conf.net.embed_channels,
        )

        self.model_conf.net_enc.use_time_condition = False
        self.encoder = BeatGANsEncoderModel(self.model_conf)

        if self.model_conf.latent_net_conf.latent_type == LatentNetType.skip:
            self.latent_net = MLPSkipNet(self.model_conf.latent_net_conf)
        else:
            raise NotImplementedError

    def encode(self, x, target=None):
        cond = self.encoder.forward(x, label=target)
        return {'cond': cond}

    @property
    def style_space_sizes(self):
        """
        The style space sizes for each ResBlock

        :return: List of sizes of the style space for each ResBlock (The dimensionality of the conditioning embedding, hence the number of output features of the linear layer/dimensionality embedding transformation)
        :rtype: list[int]
        """
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        sizes = []
        for module in modules:
            if isinstance(module, ResBlock):
                linear = module.cond_emb_layers[-1]
                sizes.append(linear.weight.shape[0])
        return sizes

    def encode_style_space(self, x, return_vector: bool = True):
        """
        Encode an input to the style space
        """
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        # (n, c)
        cond = self.encoder.forward(x)
        S = []
        for module in modules:
            if isinstance(module, ResBlock):
                # (n, c')
                s = module.cond_emb_layers.forward(cond)
                S.append(s)

        if return_vector:
            # (n, sum_c)
            return torch.cat(S, dim=1)
        else:
            return S

    def forward(self,
                x,
                t,
                y=None,
                x_start=None,
                cond=None,
                style=None,
                noise=None,
                t_cond=None,
                **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """

        if t_cond is None:
            t_cond = t

        if noise is not None:
            # if the noise is given, we predict the cond from noise
            cond = self.noise_to_cond(noise)

        if cond is None:
            # cond is encoded image
            if x is not None:
                assert len(x) == len(x_start), f'{len(x)} != {len(x_start)}'

            tmp = self.encode(x_start, kwargs.get("target"))
            cond = tmp['cond']

        if t is not None:
            _t_emb = timestep_embedding(t, self.model_conf.net.ch)
            _t_cond_emb = timestep_embedding(t_cond, self.model_conf.net.ch)
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None

        if self.model_conf.net.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=cond,
                time_cond_emb=_t_cond_emb,
            )
        else:
            raise NotImplementedError()

        if self.model_conf.net.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None

        # override the style if given
        style = style or res.style

        assert (kwargs.get("target") is not None) == (
                self.model_conf.num_classes is not None
        ) or cond is not None, "must specify y if and only if the model is class-conditional"

        # Enable if desired
        # if self.model_conf.num_classes is not None:
        #     assert y.shape == (x.shape[0], )
        #     emb = emb + self.label_emb(y)

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond_emb
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb

        # hs = []
        hs = [[] for _ in range(len(self.model_conf.net.ch_mult))]

        if x is not None:
            h = x.type(self.dtype)

            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    h = self.input_blocks[k](h,
                                             emb=enc_time_emb,
                                             cond=enc_cond_emb)

                    # print(i, j, h.shape)
                    hs[i].append(h)
                    k += 1
            assert k == len(self.input_blocks)

            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.model_conf.net.ch_mult))]

        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)

                h = self.output_blocks[k](h,
                                          emb=dec_time_emb,
                                          cond=dec_cond_emb,
                                          lateral=lateral)
                k += 1

        pred = self.out(h)

        # For predicting noise for xor noise
        if self.last_act:
            pred = self.last_act(pred)
        return AutoencoderReturn(pred=pred, cond=cond)


class TimeStyleSeparateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)
