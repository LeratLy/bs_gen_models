from typing import Union

from src.models.clf.ms_clf import MSClfNetConfig, MSClfNet
from src.models.dae.architecture.unet import BeatGANsUNetModel, BeatGANsAutoencoderConfig, BeatGANsUNetConfig
from src.models.dae.architecture.unet_autoencoder import BeatGANsAutoencoderModel
from src.models.test_model.test_model import SimpleModelConfig
from src.models.vae.architecture.autoencoder import VAEModelConfig

ModelConfig = Union[BeatGANsAutoencoderConfig, BeatGANsUNetConfig, SimpleModelConfig, VAEModelConfig, MSClfNetConfig]
Model = Union[BeatGANsUNetModel, BeatGANsAutoencoderModel, SimpleModelConfig, VAEModelConfig, MSClfNet]