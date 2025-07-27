import unittest

import torch
from tqdm import tqdm

from src._types import ModelType, Mode, SaveTo, NoiseType, LossType
from src.models.dae.architecture.blocks import ResBlock, ResBlockConfig
from src.models.dae.architecture.unet import BeatGANsAutoencoderConfig, BeatGANsEncoderModel, BeatGANsUNetConfig, \
    BeatGANsUNetModel
from src.models.dae.diffusion.resample import UniformSampler
from run_models.model_templates import assign_diffusion_conf
from src.models.trainer import Trainer
from src.utils.visualisation import plot_3d_data_cloud
from tests.test_debugging import overfit_conf, autoencoder_conf


class DAETestCase(unittest.TestCase):

    def setUp(self):
        # 20 timesteps in total
        self.conf = {
            "T": 20
        }
        # batch size = 2, data = 32x32x32 array
        self.x = torch.randn((2, 1, 32, 32, 32))

    def test_uniform_sampler(self):
        t, weight = UniformSampler(self.conf.get("T")).sample(len(self.x), self.x.device)
        # as many sampled timesteps as batches
        self.assertEqual(len(t), self.x.shape[0])
        # since it is uniform everyone gets the same weight
        self.assertListEqual(weight.tolist(), [1] * len(weight))

    def test_BeatGANsUNetModel(self):
        """
        The UNet model is used as base model for the DAE and is used for the DDPM
        An image/the data and timesteps are passed through the UNet
        :return:
        """
        t, weight = UniformSampler(self.conf.get("T")).sample(len(self.x), self.x.device)
        self.model_type = ModelType.ddpm
        self.model_conf = BeatGANsUNetConfig()
        self.model_conf.net.embed_channels = 512
        self.model_conf.net.ch = 64
        encoder = BeatGANsUNetModel(self.model_conf)
        model_output = encoder.forward(self.x, t)
        self.assertEqual(model_output.pred.shape, self.x.shape)

    def test_BeatGANsEncoderModel(self):
        config = BeatGANsAutoencoderConfig()
        config.net_enc.use_time_condition = False
        config.net.ch = 64
        encoder = BeatGANsEncoderModel(config)
        model_output = encoder.forward(self.x)
        self.assertEqual(model_output.shape[0], self.x.shape[0])
        # 512 this is the latent vector
        self.assertEqual(model_output.shape[1], config.net_enc.out_channels)

    def test_ResBlock(self):
        # ResBlock with same num out as in channels
        model_output = ResBlock(ResBlockConfig(
            channels=1,
            dropout=0.1,
            dims=3,
            use_condition=False,
            use_checkpoint=False,
        )).forward(self.x)
        self.assertEqual(model_output.shape, self.x.shape)

        # ResBlock with more out_channels
        model_output = ResBlock(ResBlockConfig(
            channels=1,
            dropout=0.1,
            dims=3,
            out_channels=3,
            use_condition=False,
            use_checkpoint=False,
        )).forward(model_output)
        self.assertEqual(model_output.shape[1], 3)

        # Down sampling with ResBlock
        model_output = ResBlock(ResBlockConfig(
            channels=3,
            dropout=0.1,
            dims=3,
            out_channels=3,
            use_condition=False,
            use_checkpoint=False,
            down=True
        )).forward(model_output)
        self.assertEqual(model_output.shape[2], self.x.shape[2] / 2)

        # Up sampling and reducing number of channels
        model_output = ResBlock(ResBlockConfig(
            channels=3,
            dropout=0.1,
            dims=3,
            out_channels=1,
            use_condition=False,
            use_checkpoint=False,
            up=True
        )).forward(model_output)
        self.assertEqual(model_output.shape, self.x.shape)

    def test_forward_diffusion_gaussian(self):
        self.conf = get_overfit_config()
        self.conf.__post_init__()
        self.forward_diffusion(self.conf, "forward_gaussian")

    def test_forward_diffusion_xor(self):
        self.conf = get_overfit_config()
        self.conf.diffusion_conf.noise_type = NoiseType.xor
        self.conf.diffusion_conf.loss_type = LossType.bce
        self.conf.diffusion_conf.beta_scheduler = "cosine_xor"
        self.conf.__post_init__()
        self.forward_diffusion(self.conf, "forward_xor")

    def forward_diffusion(self, conf, title, normalized=False):
        self.trainer = Trainer(conf)
        img, target, index = next(iter(self.trainer.dataloaders.get(Mode.train)))
        sampler = assign_diffusion_conf(conf, 1000).make_sampler()
        show_diff_step_at = 20
        x_start = img.to(conf.device)
        title += ("_normalized" if normalized else "_clamped")
        plot_3d_data_cloud(x_start[0][0], title, step=0, save_to=SaveTo.tensorboard,
                           writer=self.trainer.writer)
        for t in tqdm(range(1, 301)):
            torch_t = torch.tensor([t], device=conf.device, dtype=torch.int64)
            out = sampler.training_losses(model=self.trainer.wrapperModel.model, x_start=x_start, t=torch_t)
            x = out["x_t"]
            if t % show_diff_step_at == 0:
                if normalized:
                    x = (x - x.min()) / (x.max() - x.min())
                print("Logging normalized x_t to tensorboard iteration: ", t)
                print(x[0][0].min(), x[0][0].max())
                plot_3d_data_cloud(x[0][0], title, step=t, save_to=SaveTo.tensorboard, writer=self.trainer.writer)


class TrainedDAETestCase(unittest.TestCase):
    def setUp(self):
        self.conf = get_overfit_config()
        self.conf.__post_init__()
        self.trainer = None

    def tearDown(self):
        if self.trainer is not None:
            self.trainer.close()

    def test_reverse_diffusion_gaussian(self):
        self.trainer = Trainer(self.conf)
        self.reverse_diffusion("gaussian")

    def test_reverse_diffusion_xor(self):
        self.conf.diffusion_conf.noise_type = NoiseType.xor
        self.conf.diffusion_conf.loss_type = LossType.bce
        self.conf.__post_init__()
        self.trainer = Trainer(self.conf)
        self.reverse_diffusion("xor")

    def reverse_diffusion(self, title):
        img, target, index = list(self.trainer.dataloaders.get(Mode.train))[0]
        sampler = self.trainer.wrapperModel.sampler
        show_diff_step_at = 100
        x = img.to(self.conf.device)
        T = 1000
        for t in tqdm(range(0, T)):
            torch_t = torch.tensor([t], device=self.conf.device, dtype=torch.int64)
            eps = sampler.get_noise(img, torch_t).to(self.conf.device)
            x_t = sampler.q_sample(x, torch_t, noise=eps)
            x = sampler._predict_xstart_from_eps(x_t, torch_t, eps)
            if t % show_diff_step_at == 0:
                plot_3d_data_cloud(eps[0][0], title + "_noise", step=t, save_to=SaveTo.tensorboard,
                                   writer=self.trainer.writer)
                plot_3d_data_cloud(x_t[0][0], title + "_x_t", step=t, save_to=SaveTo.tensorboard,
                                   writer=self.trainer.writer)
                plot_3d_data_cloud(x[0][0], title + "_x_denoised", step=t, save_to=SaveTo.tensorboard,
                                   writer=self.trainer.writer)


def get_overfit_config():
    conf = autoencoder_conf()
    overfit_conf(conf)
    conf.name = "chP3D_overfit_gaussian"
    return conf


if __name__ == '__main__':
    unittest.main()
