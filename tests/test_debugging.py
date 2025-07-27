import unittest
import torch

from src.config import BaseConfig
from run_models.model_templates import assign_model_config
from src.models.trainer import Trainer
from src._types import ModelName, DataType, GenerativeType, TrainMode, Mode, SaveTo, LossType, NoiseType, ModelGradType
from src.utils.visualisation import plot_3d_data_cloud


def standard_conf():
    conf = BaseConfig()
    conf.log_interval = 2
    conf.data = {
        "name": 'chP3D_test',
        "type": DataType.nii,
    }
    conf.img_size = 96  # 192
    conf.batch_size = 1  # 32
    conf.num_epochs = 20  # 30
    conf.model_name = ModelName.beatgans_autoencoder
    conf.model = "src.models.dae.architecture.unet_autoencoder.BeatGANsAutoencoderModel"
    conf.eval.eval_training_every_epoch = 5
    conf.diffusion_conf.T_eval = 20
    conf.diffusion_conf.T = 1000
    conf.diffusion_conf.gen_type = GenerativeType.ddim
    conf.net_enc_pool = 'adaptivenonzero'
    conf.eval.eval_training_every_epoch = -1
    return conf


def model_conf(conf):
    conf.model_conf.net.resnet_two_cond = True
    conf.model_conf.net.grad_checkpoint = True
    conf.model_conf.attn_checkpoint = True
    conf.model_conf.net_enc.enc_channel_mult = (1, 2, 4, 8, 8)
    conf.model_conf.net.ch_mult = (1, 2, 4, 8)
    conf.model_conf.net.attn_head = 1
    conf.model_conf.in_channels = 1
    conf.model_conf.net.ch = 32


def autoencoder_conf():
    conf = standard_conf()
    assign_model_config(conf)
    model_conf(conf)
    conf.__post_init__()
    return conf

def overfit_conf(conf):
    conf.shuffle = False
    conf.num_epochs = 500
    conf.eval.eval_training_every_epoch = 50
    conf.ema_decay = 0.9
    conf.batch_size = 1
    conf.img_size = 96
    conf.accum_batches = 32
    conf.log_interval = 16
    conf.clip_denoised = True
    conf.checkpoint["save_every_epoch"] = 100
    conf.data["name"] = "chP3D_overfit"


class ExperimentAutoencoder(unittest.TestCase):
    """
    Testcase to test whole model pipeline on an autoencoder
    """

    def setUp(self):
        self.conf = autoencoder_conf()
        self.trainer = None

    def tearDown(self):
        if self.trainer is not None:
            self.trainer.close()

    def test_train_autoencoder(self):
        # 1. Train main model in with mode TrainingMode.base (standard)
        self.conf.data["name"] = "chP3D_test"
        self.conf.__post_init__()
        self.trainer = Trainer(self.conf)
        self.trainer.train()
        self.assertIsInstance(self.trainer, Trainer)
        self.assertIsNotNone(self.trainer.conf.checkpoint["name"])

        # 2. Save semantic encoding for training the latent DDIM
        self.trainer.infer_latents()
        latent_path = self.conf.latent_infer_path
        self.assertIsNotNone(latent_path)

        # 3. Train the latent DDIM
        self.trainer.close()
        self.conf.train_mode = TrainMode.latent_diffusion
        self.trainer = Trainer(self.conf)
        self.assertEqual(TrainMode.latent_diffusion, self.trainer.conf.train_mode)
        self.assertEqual(TrainMode.latent_diffusion, self.trainer.wrapperModel.conf.train_mode)
        self.trainer.train()

    def test_changing_gradients(self):
        self.conf.shuffle = False
        self.conf.num_epochs = 10
        self.conf.batch_size = 1
        self.conf.accum_batches = 1
        self.conf.data["name"] = "chP3D_overfit"
        self.conf.__post_init__()
        self.trainer = Trainer(self.conf)
        img, target, index = list(self.trainer.dataloaders.get(Mode.train))[0]
        result = self.trainer.reconstruct(img)
        plot_3d_data_cloud(result[0][0], "ChP_initial", save_to=SaveTo.png)
        self.trainer.train()
        result2 = self.trainer.reconstruct(img)
        plot_3d_data_cloud(result2[0][0], "ChP_after_training", save_to=SaveTo.png)
        # gradients changed, therefore the results should not be the same
        self.assertFalse(torch.equal(result, result2))

    def test_overfit_gaussian(self):
        overfit_conf(self.conf)
        self.conf.checkpoint["name"] = "./overfit_checkpoints/chP3D_overfit_gaussian_base_20250505_182246_best"
        self.conf.num_epochs = 20
        self.conf.batch_size = 2
        self.conf.diffusion_conf.T_eval = 1000
        self.conf.accum_batches = 1
        self.conf.model_conf.net.ch = 32
        self.conf.eval.eval_training_every_epoch = -1
        self.conf.lr = 0.0001
        self.conf.ema_decay = 0.0
        self.conf.name = "chP3D_overfit_gaussian"
        self.conf.diffusion_conf.loss_type = LossType.l1
        self.conf.diffusion_conf.model_grad_type = ModelGradType.eps
        self.conf.diffusion_conf.loss_type_eps = LossType.l1
        self.conf.diffusion_conf.loss_type_x_start = LossType.bce
        self.conf.__post_init__()
        self.trainer = Trainer(self.conf)
        img, target, index = next(iter(self.trainer.dataloaders.get(Mode.train)))
        plot_3d_data_cloud(img[0][0], "ChP_real", save_to=SaveTo.png)
        # self.trainer.train()
        print(self.trainer.conf.checkpoint)
        result2 = self.trainer.reconstruct(img, ema=True)
        plot_3d_data_cloud(result2[0][0], "ChP_reconstructed", save_to=SaveTo.png)

    def test_overfit_xor(self):
        overfit_conf(self.conf)
        self.conf.name = "chP3D_overfit_xor"
        self.conf.diffusion_conf.beta_scheduler = "linear"
        self.conf.diffusion_conf.loss_type = LossType.bce_logits
        self.conf.diffusion_conf.model_grad_type = ModelGradType.eps
        self.conf.diffusion_conf.noise_type = NoiseType.xor
        self.conf.num_epochs = 40
        self.conf.diffusion_conf.T = 1000
        self.conf.batch_size = 1
        self.conf.accum_batches = 2
        self.conf.eval.eval_training_every_epoch = 10
        self.conf.lr = 0.0001
        self.conf.ema_decay = 0.0
        self.conf.__post_init__()
        self.trainer = Trainer(self.conf)
        img, target, index = list(self.trainer.dataloaders.get(Mode.train))[0]
        plot_3d_data_cloud(img[0][0], "ChP_real_xor", save_to=SaveTo.png)
        self.trainer.train()
        print(self.trainer.conf.checkpoint)
        result2 = self.trainer.reconstruct(img)
        plot_3d_data_cloud(result2[0][0], "ChP_reconstructed_xor", save_to=SaveTo.png)

    def test_loss(self):
        self.conf.shuffle = False
        self.conf.num_epochs = 30
        self.conf.batch_size = 1
        self.conf.img_size = 96
        self.conf.accum_batches = 1
        self.conf.log_interval = 1
        self.conf.data["name"] = "chP3D_overfit"
        self.conf.__post_init__()
        self.trainer = Trainer(self.conf)
        self.trainer.train_epoch(0)
        for batch_index, (features, label, index) in enumerate(self.trainer.dataloaders.get(Mode.val)):
            features = features.to(self.conf.device).type(self.conf.dtype)
            loss, _ = self.trainer.wrapperModel(features, batch_index, True)
            print(loss)


class ExperimentSemanticDDIM(unittest.TestCase):
    def setUp(self):
        self.conf = autoencoder_conf()
        self.conf.checkpoint[
            "name"] = "../testing_checkpoints/beatgans_autoencoder_chP3D_test_chP3D_test_20250402_112239"
        self.conf.__post_init__()
        self.trainer = Trainer(self.conf)
        self.trainer.infer_latents()

    def tearDown(self):
        self.trainer.close()

    def test_infer_latents(self):
        self.assertIsNotNone(self.trainer.conf.latent_infer_path)
        semantic_codes = torch.load(self.trainer.conf.latent_infer_path)
        self.assertIn("cond_tensor_dict", semantic_codes)
        self.assertIn("cond_mean", semantic_codes)
        self.assertIn("cond_std", semantic_codes)

    def test_semantic_ddim(self):
        self.trainer.switch_train_mode(TrainMode.latent_diffusion)
        self.trainer.train()


if __name__ == '__main__':
    unittest.main()
