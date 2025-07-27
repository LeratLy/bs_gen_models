from dataclasses import dataclass, field
from typing import NamedTuple, Tuple, Optional

import torch
from torch import nn

from src.models.dae.architecture.blocks import TimestepEmbedSequential, AttentionBlock, Downsample, \
    Upsample, ResBlock, ResBlockConfig
from src.models.dae.architecture.latentnet import MLPSkipNetConfig
from src.models.dae.architecture.nn import (conv_nd, linear, normalization, timestep_embedding,
                                            zero_module)
from src._types import LatentNetType, Activation
from src.utils.config_functioanlities import ConfigFunctionalities


@dataclass
class BeatGANsNetConfig(ConfigFunctionalities):
    """
    Configuration parameters for the unet's submodules

    :param ch: Base channels, that will be multiplied in the unet. Originally this has been called mod_channels or model_channels. This should be dividable by 32 and should be greater than 32 (for group normalisation), defaults to 192.
    :type ch: int
    :param num_res_blocks: Number of repeating ResBlocks per resolution. The decoding side has one additional ResBlock.
    :type num_res_blocks: int
    :param attn: At what resolutions self-attention of the feature maps should be applied and was originally referred to as attention_resolution, default to (16,) ad for beatgans to (32, 16, 8). Attention generally imporved performance
    :type attn: Tuple[int, ...]
    :param ch_mult: For each resolution level there is one multiplication factor for the base channel number ``ch``, allowing th enetwork to increase the number of channels while the resolution decreases and the other way around for the upsampling part. It was originally referred to as channel_mult, default to (16,) ad for beatgans to (32, 16, 8).
    :type ch_mult: Tuple[int, ...]
    :param resblock_updown: Use ResBlocks fo Upscale and Downscale block (which is expensive, otherwise Downsample(Upsample is usd directly), default to True
    :type resblock_updown:  bool
    :param embed_channels: Number of time embedding channels and style channels (for latent representation of data), default to 512
    :type embed_channels: int
    :param attn_head: Number of attention heads for the attention block (originally called num_heads), default to 1
    :type attn_head: int
    :param num_head_channels: Specifying the number of channels per attention head, default to -1 (not used)
    :type num_head_channels: int
    :param resnet_two_cond: Enable the second conditioning pathway of the ResBlock, default to False
    :type resnet_two_cond: bool
    :param resnet_use_zero_module: Init the decoding conv layers with zero weights, this speeds up training, default to True
    :type resnet_use_zero_module: bool
    :param grad_checkpoint: Use activation checkpointing aka gradient checkpoints (originally called use_checkpoint) to reduce memory footprint, default to False
    :type grad_checkpoint: bool
    :param resnet_cond_channels: Optionally set the channels for the conditional embedding for the ResBlocks manually, default to None (use embed_channels)
    :type resnet_cond_channels: int, optional
    :param num_input_res_blocks: Number of ResBlocks only for the input default to None. Hence, ``num_res_blocks`` is used for input ResBlocks
    :type num_input_res_blocks: int, optional
    :param input_channel_mult: Channel multiplication for input only, defaults to None
    :type input_channel_mult: Tuple[int, ...], optional
    """
    attn_head: int = 1
    ch: int = 32
    embed_channels: int = 512
    num_head_channels: int = -1
    num_res_blocks: int = 2
    grad_checkpoint: bool = False
    resblock_updown: bool = True
    resnet_two_cond: bool = False
    resnet_use_zero_module: bool = True
    attn: Tuple[int, ...] = (16,)
    ch_mult: Tuple[int, ...] = (1, 2, 4, 8)

    # ------ Begin section (default: None) ------
    num_input_res_blocks: int = None
    input_channel_mult: Tuple[int, ...] = None
    resnet_cond_channels: int = None
    # ------ End section (default: None) --------


@dataclass
class BeatGANsUNetConfig(ConfigFunctionalities):
    """
    Configuration parameters for the unet

    :param net: Configuration for submodules of the UNet
    :type net: :class:`.BeatGANsNetConfig`
    :param in_channels: Number of input channels of the unet, default to 1 grey sclae use 3 for RGB
    :type in_channels: int
    :param out_channels: Number of output channels (originally called model_out_channels) of the UNet, default and suggested 1 same as in_channels, sicne we predict the mean per channel. You need to double this if you also model the variance of the noise prediction per channel. For RGB it would be 3 or 6, default to 1
    :type out_channels: int
    :param dims: Number of dimensions of the input 2 = 2d conv, 3 = 3d conv, default to 3
    :type dims: int
    :param num_heads_upsample: "what's this?" --> original comment they also did not know TODO, default to -1
    :type num_heads_upsample: int
    :param img_size: size of the image (we expect the image to have the same size per dimension, default to 192
    :type img_size: int
    :param conv_resample: Whether convolutional layer should be used for down and up-sampling, default to True
    :type conv_resample: bool
    :param use_new_attention_order: This has not been tried out yet TODO, default to False
    :type use_new_attention_order: bool
    :param attn_checkpoint: Gradient checkpoint the attention operation, default to False
    :type attn_checkpoint: bool
    :param dropout: Whether dropout should be used in the network, default to False
    :type dropout: bool
    :param num_classes: Number of classes should not be used and was marked as legacy from BeatGANs, default to None
    :type num_classes: int, optional
    :param time_embed_channels: Number of time embed channels, default to None
    :type time_embed_channels: int
    :param target_embed_channels: Number of target embed channels, default to None
    :type target_embed_channels: int
    """
    net: BeatGANsNetConfig = field(default_factory=BeatGANsNetConfig)

    dims: int = 3
    img_size: int = 192
    in_channels: int = 1
    num_heads_upsample: int = -1
    out_channels: int = 1
    attn_checkpoint: bool = False
    conv_resample: bool = True
    dropout: float = 0.1
    use_new_attention_order: bool = False
    use_efficient_attention: bool = False
    enc_merge_time_and_cond_embedding: bool = False
    latent_net_type: LatentNetType = LatentNetType.skip

    train_pred_xstart_detach: bool = True
    last_act: Activation = Activation.none

    # ------ Begin section (default: None) ------
    num_classes: int = None
    time_embed_channels: int = None
    # ------ End section (default: None) --------


@dataclass
class BeatGANSNetEncoderConfig(ConfigFunctionalities):
    """
    The configuration for the encoder of the UNet which uses Residual Block to turn an image into a latent representation

    :param pool: Type of pooling operation used at the end of the encoder to reduce spatial dimension, default to 'adaptivenonzero' (adaptive average pooling layers with non-zero ourput)
    :type pool: str
    :param num_res_blocks: The Number of residual blocks used at each resolution level (more ResBlocks more complex features), default to 2
    :type num_res_blocks: int
    :param out_channels: Number of output channels of the encoder (determining dimensionality of the latent representation, default to 512
    :type out_channels: int
    :param use_time_condition: Whether a time embedding is used for the encoder part of the UNet for the semantic encoder this is set to False, default True
    :type use_time_condition: bool
    :param ch_mult: Channel multiplier for each resolution level in the encoder, default ch_mult from :class:`.BeatGANsNetConfig`
    :type ch_mult: Tuple[int, ...], optional
    :param grad_checkpoint: Use checkpoints for gradient computation, default :class:`.BeatGANsNetBeatgansConfig`'s ``grad_checkpoint``
    :type grad_checkpoint: bool, optional
    :param attn: At what resolution self-attention should be applied on the feature maps, was originally called attention resolution
    :type attn: Tuple[int, ...], optional
    """
    pool: str = 'adaptivenonzero'
    num_res_blocks: int = 2
    out_channels: int = 512  # out_hid_channels is the same
    use_time_condition: bool = True

    # ------ Begin section (default: None) ------
    ch_mult: Tuple[int, ...] = None
    grad_checkpoint: bool = None
    attn: Tuple[int, ...] = None
    # ------ End section (default: None) --------

@dataclass
class BeatGANsAutoencoderConfig(BeatGANsUNetConfig):
    """
    The configuration for a diffusion autoencoder model

    :param net_enc: The configuration for the encoder part of the network
    :type net_enc: :class:`.BeatGANSNetEncoderConfig`
    """
    net_enc: BeatGANSNetEncoderConfig = field(default_factory=BeatGANSNetEncoderConfig)
    latent_net_conf: Optional[MLPSkipNetConfig] = None


class BeatGANsUNetModel(nn.Module):
    def __init__(self, model_conf: BeatGANsUNetConfig):
        """
        Initialization of BeatGANsUNetModel with up and down-sampling layers
        :param model_conf: please be aware that the encoder does NOT work on a copy of the config
        (Change this if you want to modify the config within)
        """
        super().__init__()
        self.model_conf = model_conf

        if model_conf.num_heads_upsample == -1:
            self.num_heads_upsample = model_conf.net.attn_head

        self.dtype = torch.float32

        # create time_embedding function for creating time embedding that has the same shape as the original conditional embedding
        self.time_emb_channels = model_conf.time_embed_channels or model_conf.net.ch
        self.time_embed = nn.Sequential(
            linear(self.time_emb_channels, model_conf.net.embed_channels),
            nn.SiLU(),
            linear(model_conf.net.embed_channels, model_conf.net.embed_channels))
        # Enable if desired
        if model_conf.num_classes is not None:
            self.label_emb = nn.Embedding(model_conf.num_classes,
                                          model_conf.net.embed_channels)
        else:
            self.label_emb = None

        ch = input_ch = int(model_conf.net.ch_mult[0] * model_conf.net.ch)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(model_conf.dims, model_conf.in_channels, ch, 3, padding=1))
        ])

        kwargs = dict(
            use_condition=True,
            two_cond=model_conf.net.resnet_two_cond,
            use_zero_module=model_conf.net.resnet_use_zero_module,
            # style channels for the resnet block
            cond_emb_channels=model_conf.net.resnet_cond_channels,
        )

        self._feature_size = ch

        # input_block_chans = [ch]
        input_block_chans = [[] for _ in range(len(model_conf.net.ch_mult))]
        input_block_chans[0].append(ch)

        # number of blocks at each resolution
        self.input_num_blocks = [0 for _ in range(len(model_conf.net.ch_mult))]
        self.input_num_blocks[0] = 1
        self.output_num_blocks = [0 for _ in range(len(model_conf.net.ch_mult))]

        resolution = model_conf.img_size
        for level, mult in enumerate(model_conf.net.input_channel_mult
                                     or model_conf.net.ch_mult):
            for _ in range(model_conf.net.num_input_res_blocks or model_conf.net.num_res_blocks):
                layers = [
                    ResBlock(ResBlockConfig(
                        channels=ch,
                        emb_channels=model_conf.net.embed_channels,
                        dropout=model_conf.dropout,
                        out_channels=int(mult * model_conf.net.ch),
                        dims=model_conf.dims,
                        use_checkpoint=model_conf.net.grad_checkpoint,
                        **kwargs,
                    ))
                ]
                ch = int(mult * model_conf.net.ch)
                if model_conf.net.attn is not None and resolution in model_conf.net.attn:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=model_conf.net.grad_checkpoint or model_conf.attn_checkpoint,
                            num_heads=model_conf.net.attn_head,
                            num_head_channels=model_conf.net.num_head_channels,
                            use_efficient_attention=model_conf.use_efficient_attention,
                            use_new_attention_order=model_conf.use_new_attention_order,
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                # input_block_chans.append(ch)
                input_block_chans[level].append(ch)
                self.input_num_blocks[level] += 1
                # print(input_block_chans)
            if level != len(model_conf.net.ch_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ResBlockConfig(
                            channels=ch,
                            emb_channels=model_conf.net.embed_channels,
                            dropout=model_conf.dropout,
                            out_channels=out_ch,
                            dims=model_conf.dims,
                            use_checkpoint=model_conf.net.grad_checkpoint,
                            down=True,
                            **kwargs,
                        )) if model_conf.net.resblock_updown
                        else Downsample(ch,
                                        model_conf.conv_resample,
                                        dims=model_conf.dims,
                                        out_channels=out_ch)))
                ch = out_ch
                # input_block_chans.append(ch)
                input_block_chans[level + 1].append(ch)
                self.input_num_blocks[level + 1] += 1
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ResBlockConfig(
                channels=ch,
                emb_channels=model_conf.net.embed_channels,
                dropout=model_conf.dropout,
                dims=model_conf.dims,
                use_checkpoint=model_conf.net.grad_checkpoint,
                **kwargs,
            )),
            AttentionBlock(
                ch,
                use_checkpoint=model_conf.net.grad_checkpoint or model_conf.attn_checkpoint,
                num_heads=model_conf.net.attn_head,
                num_head_channels=model_conf.net.num_head_channels,
                use_efficient_attention=model_conf.use_efficient_attention,
                use_new_attention_order=model_conf.use_new_attention_order,
            ),
            ResBlock(ResBlockConfig(
                channels=ch,
                emb_channels=model_conf.net.embed_channels,
                dropout=model_conf.dropout,
                dims=model_conf.dims,
                use_checkpoint=model_conf.net.grad_checkpoint,
                **kwargs,
            )),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(model_conf.net.ch_mult))[::-1]:
            for i in range(model_conf.net.num_res_blocks + 1):
                # print(input_block_chans)
                # ich = input_block_chans.pop()
                try:
                    ich = input_block_chans[level].pop()
                except IndexError:
                    # this happens only when num_res_block > num_enc_res_block
                    # we will not have enough lateral (skip) connections for all decoder blocks
                    ich = 0
                # print('pop:', ich)
                layers = [
                    ResBlock(ResBlockConfig(
                        # only direct channels when gated
                        channels=ch + ich,
                        emb_channels=model_conf.net.embed_channels,
                        dropout=model_conf.dropout,
                        out_channels=int(model_conf.net.ch * mult),
                        dims=model_conf.dims,
                        use_checkpoint=model_conf.net.grad_checkpoint,
                        # lateral channels are described here when gated
                        has_lateral=True if ich > 0 else False,
                        lateral_channels=None,
                        **kwargs,
                    ))
                ]
                ch = int(model_conf.net.ch * mult)
                if model_conf.net.attn is not None and resolution in model_conf.net.attn:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=model_conf.net.grad_checkpoint
                                           or model_conf.attn_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=model_conf.net.num_head_channels,
                            use_efficient_attention=model_conf.use_efficient_attention,
                            use_new_attention_order=model_conf.use_new_attention_order,
                        ))
                if level and i == model_conf.net.num_res_blocks:
                    resolution *= 2
                    out_ch = ch
                    layers.append(
                        ResBlock(ResBlockConfig(
                            channels=ch,
                            emb_channels=model_conf.net.embed_channels,
                            dropout=model_conf.dropout,
                            out_channels=out_ch,
                            dims=model_conf.dims,
                            use_checkpoint=model_conf.net.grad_checkpoint,
                            up=True,
                            **kwargs,
                        )) if (
                            model_conf.net.resblock_updown
                        ) else Upsample(ch,
                                        model_conf.conv_resample,
                                        dims=model_conf.dims,
                                        out_channels=out_ch))
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_num_blocks[level] += 1
                self._feature_size += ch

        # print(input_block_chans)
        # print('inputs:', self.input_num_blocks)
        # print('outputs:', self.output_num_blocks)

        if model_conf.net.resnet_use_zero_module:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(
                    conv_nd(model_conf.dims,
                            input_ch,
                            model_conf.out_channels,
                            3,
                            padding=1)),
            )
        else:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                conv_nd(model_conf.dims, input_ch, model_conf.out_channels, 3, padding=1),
            )

        if self.model_conf.last_act:
            self.last_act = self.model_conf.last_act.get_act()
        else:
            self.last_act = None

    def forward(self, x, t, y=None, **kwargs):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
                self.model_conf.num_classes is not None
        ) or self.label_emb is None, "must specify y if and only if the model is class-conditional"

        # hs = []
        hs = [[] for _ in range(len(self.model_conf.net.ch_mult))]
        emb = self.time_embed(timestep_embedding(t, self.time_emb_channels))

        # Enable if desired
        # if self.model_conf.num_classes is not None:
        #     assert y.shape == (x.shape[0], )
        #     emb = emb + self.label_emb(y)

        # new code supports input_num_blocks != output_num_blocks
        h = x.type(self.dtype)
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                h = self.input_blocks[k](h, emb=emb)
                # print(i, j, h.shape)
                hs[i].append(h)
                k += 1
        assert k == len(self.input_blocks)

        h = self.middle_block(h, emb=emb)
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
                h = self.output_blocks[k](h, emb=emb, lateral=lateral)
                k += 1

        h = h.type(x.dtype)
        pred = self.out(h)
        return Return(pred=pred)


class Return(NamedTuple):
    pred: torch.Tensor


class BeatGANsEncoderModel(nn.Module):
    """
    The first half of the UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(self, model_conf: BeatGANsAutoencoderConfig):
        """
        Initialization of BeatGANsEncoderModel
        :param model_conf: please be aware that the encoder does NOT work on a copy of the config
        (Change this if you want to modify the config within)
        """
        super().__init__()
        self.model_conf = model_conf
        self.dtype = torch.float32
        self.channel_mult = model_conf.net_enc.ch_mult or model_conf.net.ch_mult
        self.attention_resolutions = model_conf.net_enc.attn or model_conf.net.attn
        self.use_checkpoint = model_conf.net.grad_checkpoint or model_conf.net_enc.grad_checkpoint

        if model_conf.net_enc.use_time_condition or model_conf.num_classes:
            time_embed_dim = model_conf.net.ch * 4

            # Enable if desired
            if model_conf.num_classes:
                self.label_embed = nn.Embedding(model_conf.num_classes,
                                              time_embed_dim)
            else:
                self.label_embed = None

            self.time_embed = nn.Sequential(
                linear(model_conf.net.ch, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None

        ch = int(self.channel_mult[0] * model_conf.net.ch)

        # 1. Input blocks (later ResBlocks and AttentionBlocks are added to input blocks
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(model_conf.dims, model_conf.in_channels, ch, 3, padding=1))
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        resolution = model_conf.img_size
        for level, mult in enumerate(self.channel_mult):
            # Add ResBlock + optional AttnBlock as configured
            for _ in range(model_conf.net_enc.num_res_blocks):
                # Add ResBlock to layer
                out_channels = int(mult * model_conf.net.ch)
                layers = [
                    ResBlock(ResBlockConfig(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=model_conf.dropout,
                        out_channels=out_channels,
                        dims=model_conf.dims,
                        use_condition=model_conf.net_enc.use_time_condition or model_conf.num_classes is not None,
                        two_cond=model_conf.num_classes is not None, # new in contrast to before
                        use_checkpoint=self.use_checkpoint,
                    ))
                ]
                # Increase number of channels
                ch = int(mult * model_conf.net.ch)

                # Add attention block if corresponding level
                if self.attention_resolutions is not None and resolution in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=model_conf.net.attn_head,
                            num_head_channels=model_conf.net.num_head_channels,
                            use_efficient_attention=model_conf.use_efficient_attention,
                            use_new_attention_order=model_conf.use_new_attention_order,
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            # If not last level, then downsample
            if level != len(self.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ResBlockConfig(
                            channels=ch,
                            emb_channels=time_embed_dim,
                            dropout=model_conf.dropout,
                            out_channels=out_ch,
                            dims=model_conf.dims,
                            use_condition=model_conf.net_enc.use_time_condition or model_conf.num_classes is not None,
                            two_cond=model_conf.num_classes is not None, # new in contrast to before
                            use_checkpoint=self.use_checkpoint,
                            down=True,
                        )) if (
                            model_conf.net.resblock_updown
                        ) else Downsample(ch,
                                          model_conf.conv_resample,
                                          dims=model_conf.dims,
                                          out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch

        # 2. Middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ResBlockConfig(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=model_conf.dropout,
                dims=model_conf.dims,
                use_condition=model_conf.net_enc.use_time_condition or model_conf.num_classes is not None,
                use_checkpoint=self.use_checkpoint,
                two_cond=model_conf.num_classes is not None # new in contrast to before
            )),
            AttentionBlock(
                ch,
                use_checkpoint=self.use_checkpoint,
                num_heads=model_conf.net.attn_head,
                num_head_channels=model_conf.net.num_head_channels,
                use_efficient_attention=model_conf.use_efficient_attention,
                use_new_attention_order=model_conf.use_new_attention_order,
            ),
            ResBlock(ResBlockConfig(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=model_conf.dropout,
                dims=model_conf.dims,
                use_condition=model_conf.net_enc.use_time_condition or model_conf.num_classes is not None,
                two_cond=model_conf.num_classes is not None,
                use_checkpoint=self.use_checkpoint,
            )),
        )
        self._feature_size += ch

        # 3. Output layer: flattening the latent vector
        if model_conf.net_enc.pool == "adaptivenonzero":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)) if model_conf.dims == 3 else nn.AdaptiveAvgPool2d((1, 1)),
                conv_nd(model_conf.dims, ch, model_conf.net_enc.out_channels, 1),
                nn.Flatten(),
            )
        else:
            raise NotImplementedError(f"Unexpected {model_conf.net_enc.pool} pooling")

    def forward(self, x, t=None, return_2d_feature=False, label=None):
        """
        Apply the encoder model to an input batch to produce a latent representation h.
        If time conditioning is enabled, it also handles spatial pooling

        :param x: Inputs for the UNet.
        :type x: an [N x C x ...] Tensor
        :param timesteps: A sequence of timesteps which are embedded.
        :type timesteps: 1-D batch
        :return: latent representation of x and optionally a 2D feature map.
        :rtype: [N x K] Tensor
        """
        assert not self.model_conf.num_classes or label is not None or self.label_embed is None, "Label must be given if num_classes is set"
        # Enable if desired
        cond = self.label_embed(label) if self.model_conf.num_classes and self.label_embed is not None else None
        # emb = None
        # Only use conditional embedding in the time condition
        # pass through neural network to obtain embeddings
        time_emb = self.time_embed(timestep_embedding(t, self.model_conf.net.ch)) if self.model_conf.net_enc.use_time_condition else None
        if self.model_conf.enc_merge_time_and_cond_embedding:
            if cond is not None and time_emb is not None:
                time_emb += cond
            elif cond is not None and time_emb is None:
                time_emb = cond
                cond = None


        # results is used to store intermediate feature maps (if there is spatial pooling
        results = []
        h = x.type(self.dtype)

        # 1. Input blocks: Iterate through TimestepEmbedSequential input blocks with ResBlocks
        for module in self.input_blocks:
            h = module(h, emb=time_emb, cond=cond)
            if self.model_conf.net_enc.pool.startswith("spatial"):
                # Compute mean of feature map, reducing spatial size
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        # 2. Middle block: TimestepEmbedSequential block with ResBlocks
        h = self.middle_block(h, emb=time_emb, cond=cond)
        if self.model_conf.net_enc.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            # Concatenate feature maps along channel dimension
            h = torch.cat(results, axis=-1)
        else:
            h = h.type(x.dtype)
        # 3. Output layer: pass result through output layer to obtain latent representation
        h_2d = h
        h = self.out(h)

        if return_2d_feature:
            return h, h_2d
        else:
            return h

    def forward_flatten(self, x):
        """
        transform the last 2d feature into a flatten vector
        """
        h = self.out(x)
        return h
