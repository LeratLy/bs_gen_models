from enum import Enum
from typing import TypedDict, NotRequired

from torch import nn

"""
    Copyright (C) 2024 LeratLy - All Rights Reserved
    You may use, distribute and modify this code under the
    terms of the MIT license.
    You should have received a copy of the MIT license with
    this file.

    Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology.
    Code of VISTEC has been used and modified in this file under terms of the MIT license.
"""


class LatentNetType(str, Enum):
    # injecting inputs into the hidden layers
    skip = 'skip'


class ScaleAt(str, Enum):
    after_norm = 'afternorm'


class TrainMode(str, Enum):
    # manipulate mode = training the classifier
    manipulate = "manipulate"
    # default training mode!
    base = 'base'
    # default latent training mode!
    # fitting the ddim to a given latent
    latent_diffusion = 'latentdiffusion'
    infer = 'infer'

    def is_diffusion(self):
        return self in [
            TrainMode.base,
            TrainMode.latent_diffusion,
        ]

    def is_autoencoder(self):
        # the network possibly does auto-encoding
        return self in [
            TrainMode.base,
        ]

    def require_dataset_infer(self):
        """
        whether training in this mode requires the latent variables to be available?
        """
        # this will precalculate all the latents before hand
        # and the dataset will be all the predicted latents
        return self in [
            TrainMode.latent_diffusion,
            TrainMode.manipulate,
        ]


class ModelType(str, Enum):
    """
    Kinds of the backbone models
    """
    ddpm = 'ddpm'
    # unconditional ddpm
    latent_diffusion = 'latent_diffusion'
    # auto-encoding ddpm cannot do unconditional generation
    autoencoder = 'autoencoder'


class DataType(str, Enum):
    """
    Different types of data for loading (use extension as type)
    """

    np = "np"
    nii = "nii"


class Mode(str, Enum):
    """
    Different data
    """
    train = "train"
    val = "val"
    test = "test"


class ModelName(str, Enum):
    """
    List of all supported model classes
    """

    beatgans_ddpm = 'beatgans_ddpm'
    beatgans_autoencoder = 'beatgans_autoencoder'
    simple_model = 'simple_model'
    base_nii = 'base_nii'
    ms_clf = 'ms_clf'
    conditional_variational_autoencoder = 'conditional_variational_autoencoder'

    def is_dae(self):
        return self in [ModelName.beatgans_autoencoder]


class ModelMeanType(str, Enum):
    """
    Which type of output the model predicts.
    """

    eps = 'eps'  # the model predicts epsilon


class ModelGradType(str, Enum):
    """
    Which type of output the model predicts.
    """
    eps = 'eps'  # the model predicts epsilon
    x_start = 'x_start'
    contrastive = 'contrastive'
    mix_start_eps = 'mix_tart_eps'
    mix_contrastive_eps = 'mix_contrastive_eps'


class ModelVarType(str, Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    # posterior beta_t
    fixed_small = 'fixed_small'
    # beta_t
    fixed_large = 'fixed_large'


class LossType(str, Enum):
    mse = 'mse'  # use raw MSE loss (and KL when learning variances)
    l1 = 'l1'
    l1_sum = 'l1_sum'
    bce = 'bce'  # Binary-Cross-Entropy Loss (for binary flipping noise)
    bce_logits = 'bce_logits'
    cel = "cel"
    w_cel = "weighted_cel"
    m_if1 = "macro_inv_f1_loss"


class NoiseType(str, Enum):
    gaussian = 'gaussian'  # typical dae setting
    xor = 'xor'  # incorporate xor noise (random flipping)


class GenerativeType(str, Enum):
    """
    How's a sample generated
    """

    ddpm = 'ddpm'
    ddim = 'ddim'


class OptimizerType(str, Enum):
    adam = 'adam'
    adam_w = 'adam_w'


class Activation(str, Enum):
    none = 'none'
    relu = 'relu'
    lrelu = 'lrelu'
    silu = 'silu'
    tanh = 'tanh'
    sigmoid = 'sigmoid'

    def get_act(self):
        if self == Activation.none:
            return nn.Identity()
        elif self == Activation.relu:
            return nn.ReLU()
        elif self == Activation.lrelu:
            return nn.LeakyReLU(negative_slope=0.2)
        elif self == Activation.silu:
            return nn.SiLU()
        elif self == Activation.tanh:
            return nn.Tanh()
        elif self == Activation.sigmoid:
            return nn.Sigmoid()
        else:
            raise NotImplementedError()


class ManipulateLossType(str, Enum):
    bce = 'bce'
    mse = 'mse'


class SaveTo(str, Enum):
    svg = 'svg'
    png = 'png'
    tensorboard = 'tensorboard'
    none = 'none'
    show = 'show'


class SavedModelTypes(str, Enum):
    """
    List of all supported and saved models
    """

    cvae = 'cvae'
    clf = 'clf'
    bdae = 'bdae'

    cvae_16_1 = 'cvae_16_1'
    cvae_64_1 = 'cvae_64_1'
    cvae_16_20 = 'cvae_16_20'
    cvae_64_20 = 'cvae_64_20'
    cvae_16_50 = 'cvae_16_50'
    cvae_64_50 = 'cvae_64_50'

    bdae_10_2048_5_shift = 'bdae_10_2048_5_shift'
    bdae_10_2048_5_class_norm = 'bdae_10_2048_5_class_norm'
    bdae_10_2048_5_alpha_shift = 'bdae_10_2048_5_alpha_shift'
    bdae_10_2048_5_cond_encoder = 'bdae_10_2048_5_cond_encoder'

    bdae_10_1024_5_class_znormalize = 'bdae_10_1024_5_class_znormalize'
    bdae_20_1024_5_class_znormalize = 'bdae_20_1024_5_class_znormalize'
    bdae_10_2048_5_class_znormalize = 'bdae_10_2048_5_class_znormalize'
    bdae_20_2048_5_class_znormalize = 'bdae_20_2048_5_class_znormalize'
    bdae_20_2048_5_class_znormalize_alpha = 'bdae_20_2048_5_class_znormalize_alpha'

    bdae_10_1024_5_cond_encoder_shift_scale = "bdae_10_1024_5_cond_encoder_shift_scale"
    bdae_20_1024_5_cond_encoder_shift_scale = "bdae_20_1024_5_cond_encoder_shift_scale"
    bdae_10_2048_5_cond_encoder_shift_scale = "bdae_10_2048_5_cond_encoder_shift_scale"
    bdae_20_2048_5_cond_encoder_shift_scale = "bdae_20_2048_5_cond_encoder_shift_scale"

    bdae_10_1024_5_cond_encoder_shift_scale_alpha = "bdae_10_1024_5_cond_encoder_shift_scale_alpha"
    bdae_20_1024_5_cond_encoder_shift_scale_alpha = "bdae_20_1024_5_cond_encoder_shift_scale_alpha"
    bdae_10_2048_5_cond_encoder_shift_scale_alpha = "bdae_10_2048_5_cond_encoder_shift_scale_alpha"
    bdae_20_2048_5_cond_encoder_shift_scale_alpha = "bdae_20_2048_5_cond_encoder_shift_scale_alpha"
    bdae_20_2048_5_cond_encoder_shift_scale_alpha_full = "bdae_20_2048_5_cond_encoder_shift_scale_alpha_full"

    bdae_10_1024_5_cond_encoder_scale = "bdae_10_1024_5_cond_encoder_scale"
    bdae_20_1024_5_cond_encoder_scale = "bdae_20_1024_5_cond_encoder_scale"
    bdae_10_2048_5_cond_encoder_scale = "bdae_10_2048_5_cond_encoder_scale"
    bdae_20_2048_5_cond_encoder_scale = "bdae_20_2048_5_cond_encoder_scale"

    bdae_10_1024_5_cond_encoder_scale_alpha = "bdae_10_1024_5_cond_encoder_scale_alpha"
    bdae_20_1024_5_cond_encoder_scale_alpha = "bdae_20_1024_5_cond_encoder_scale_alpha"
    bdae_10_2048_5_cond_encoder_scale_alpha = "bdae_10_2048_5_cond_encoder_scale_alpha"
    bdae_20_2048_5_cond_encoder_scale_alpha = "bdae_20_2048_5_cond_encoder_scale_alpha"
    def is_cvae(self):
        return self in [
            SavedModelTypes.cvae,
            SavedModelTypes.cvae_16_1,
            SavedModelTypes.cvae_64_1,
            SavedModelTypes.cvae_16_20,
            SavedModelTypes.cvae_64_20,
            SavedModelTypes.cvae_16_50,
            SavedModelTypes.cvae_64_50,
        ]

    def is_bdae(self):
        return self in [
            SavedModelTypes.bdae_10_2048_5_shift,
            SavedModelTypes.bdae_10_2048_5_class_norm,
            SavedModelTypes.bdae_10_2048_5_alpha_shift,
            SavedModelTypes.bdae_10_2048_5_cond_encoder,

            SavedModelTypes.bdae_10_1024_5_class_znormalize,
            SavedModelTypes.bdae_20_1024_5_class_znormalize,
            SavedModelTypes.bdae_10_2048_5_class_znormalize,
            SavedModelTypes.bdae_20_2048_5_class_znormalize,
            SavedModelTypes.bdae_20_2048_5_class_znormalize_alpha,

            SavedModelTypes.bdae_10_1024_5_cond_encoder_shift_scale,
            SavedModelTypes.bdae_20_1024_5_cond_encoder_shift_scale,
            SavedModelTypes.bdae_10_2048_5_cond_encoder_shift_scale,
            SavedModelTypes.bdae_20_2048_5_cond_encoder_shift_scale,

            SavedModelTypes.bdae_10_1024_5_cond_encoder_shift_scale_alpha,
            SavedModelTypes.bdae_20_1024_5_cond_encoder_shift_scale_alpha,
            SavedModelTypes.bdae_10_2048_5_cond_encoder_shift_scale_alpha,
            SavedModelTypes.bdae_20_2048_5_cond_encoder_shift_scale_alpha,
            SavedModelTypes.bdae_20_2048_5_cond_encoder_shift_scale_alpha_full,

            SavedModelTypes.bdae_10_1024_5_cond_encoder_scale,
            SavedModelTypes.bdae_20_1024_5_cond_encoder_scale,
            SavedModelTypes.bdae_10_2048_5_cond_encoder_scale,
            SavedModelTypes.bdae_20_2048_5_cond_encoder_scale,

            SavedModelTypes.bdae_10_1024_5_cond_encoder_scale_alpha,
            SavedModelTypes.bdae_20_1024_5_cond_encoder_scale_alpha,
            SavedModelTypes.bdae_10_2048_5_cond_encoder_scale_alpha,
            SavedModelTypes.bdae_20_2048_5_cond_encoder_scale_alpha,

        ]

    def get_subtype(self, param_1: int, param_2: int, wrs: bool = False):
        """
        Only temporary for testing different model configs
        :param wrs:
        :param param_1:
        :param param_2:
        :return:
        """
        if self.is_cvae():
            if param_1 == 1:
                if param_2 == 64:
                    return SavedModelTypes.cvae_64_1
                elif param_2 == 16:
                    return SavedModelTypes.cvae_16_1
                else:
                    raise NotImplementedError()
            elif param_1 == 20:
                if param_2 == 64:
                    return SavedModelTypes.cvae_64_20
                elif param_2 == 16:
                    return SavedModelTypes.cvae_16_20
                else:
                    raise NotImplementedError()
            elif param_1 == 50:
                if param_2 == 64:
                    return SavedModelTypes.cvae_64_50
                elif param_2 == 16:
                    return SavedModelTypes.cvae_16_50
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        elif self.is_bdae():
            if param_1 == 10:
                if param_2 == 1024:
                    return SavedModelTypes.wrs_bdae_10_1024 if wrs else SavedModelTypes.bdae_10_1024
                elif param_2 == 2048:
                    return SavedModelTypes.wrs_bdae_10_2048 if wrs else SavedModelTypes.bdae_10_2048
                else:
                    raise NotImplementedError()
            elif param_1 == 20:
                if param_2 == 1024:
                    return SavedModelTypes.wrs_bdae_20_1024 if wrs else SavedModelTypes.bdae_20_1024
                elif param_2 == 2048:
                    return SavedModelTypes.wrs_bdae_20_2048 if wrs else SavedModelTypes.bdae_20_2048
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()


# ----------------------- Begin section BaseConfig ---------------------------------
class ConfigData(TypedDict):
    """
    TODO add documentation
    """
    name: str
    type: DataType


class CheckpointConfig(TypedDict):
    """
    TODO add documentation
    """
    resume_optimizer: bool
    resume_scheduler: bool
    dir: str
    resume_skip_keys: NotRequired[list[str]]
    save_every_epoch: NotRequired[int]
    name: NotRequired[str]
# ----------------------- End section BaseConfig -------------------------------
