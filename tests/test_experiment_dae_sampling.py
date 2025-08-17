import unittest

import torch

from src._types import Mode, TrainMode
from run_models.model_templates import chp96_diffae_xor_conf, chp96_diffae_latent_conf, \
    chp96_diffae_latent_training_conf, chp96_diffae_gaussian_conf
from src.models.trainer import Trainer


class ExperimentAutoencoder(unittest.TestCase):
    """
    Testcase to test whole model pipeline on an autoencoder
    """

    def setUp(self):
        self.trainer = None

    def tearDown(self):
        if self.trainer is not None:
            self.trainer.close()

    # ---- xor training ---------
    def test_train_xor(self):
        # 1. train main xor diffae
        self.conf = chp96_diffae_xor_conf()
        self.conf.num_epochs = 1
        self.conf.name ="xor_deep_32"
        self.training()

    def test_semantic_xor(self):
        # 2. infer semantic embedding for whole dataset
        self.conf = chp96_diffae_xor_conf()
        self.conf.num_epochs = 1
        # self.conf.checkpoint["name"] = "add_checkpoint"
        # self.semantic_encoding()

    def test_sampling_xor(self):
        # 3. sample without latent
        self.conf = chp96_diffae_xor_conf()
        self.conf.num_epochs = 1
        # self.conf.checkpoint["name"] = "add_checkpoint"
        # self.sample()

    def test_semantic_training_xor(self):
        # 4. train latent ddim
        self.conf = chp96_diffae_xor_conf()
        self.conf = chp96_diffae_latent_conf(self.conf)
        self.conf = chp96_diffae_latent_training_conf(self.conf)
        self.conf.num_epochs = 1
        self.conf.latent_infer_path = "add_latent_infer_path"
        self.conf.checkpoint["name"] = "add_checkpoint"
        # self.semantic_training()

    def test_sampling_xor_latent(self):
        # 5. create completely new samples
        self.conf = chp96_diffae_xor_conf()
        self.conf = chp96_diffae_latent_conf(self.conf)
        self.conf = chp96_diffae_latent_training_conf(self.conf)
        self.conf.latent_infer_path = "add_latent_infer_path"
        self.conf.num_epochs = 1
        # self.conf.checkpoint["name"] = "add_checkpoint"
        # self.sample(latent=True)

    # ---------------------------

    # ---- gaussian training ----

    def test_train_autoencoder_gaussian(self):
        # 1. Train main model in with mode TrainingMode.base (standard)
        self.conf = chp96_diffae_gaussian_conf()
        self.conf.num_epochs = 1
        self.training()

    # ---------------------------

    # ---- general helpers ----
    def training(self):
        """
        1. Train the base diffae
        """
        self.conf.__post_init__()
        self.conf.num_epochs = 1
        self.trainer = Trainer(self.conf)

        self.trainer.train()

        self.assertIsNotNone(self.trainer)
        self.assertIsNotNone(self.trainer.conf.checkpoint.get("name"))

    def semantic_encoding(self):
        """
        2. Save semantic encoding for training the latent DDIM
        """
        self.conf.__post_init__()
        self.conf.num_epochs = 1
        self.trainer = Trainer(self.conf)
        self.trainer.infer_latents()
        latent_path = self.conf.latent_infer_path
        self.assertIsNotNone(latent_path)

    def semantic_training(self):
        """
        3. Train latent DDIM based on (2)
        """
        self.conf.__post_init__()
        self.conf.num_epochs = 1
        self.trainer = Trainer(self.conf)
        self.assertEqual(TrainMode.latent_diffusion, self.trainer.conf.train_mode)
        self.assertEqual(TrainMode.latent_diffusion, self.trainer.get_wrapper_model().conf.train_mode)
        self.trainer.train()

    def sample(self, latent=False):
        """
        Create samples with model
        """
        self.conf.__post_init__()
        self.trainer = Trainer(self.conf)

        img, target, index = next(iter(self.trainer.dataloaders.get(Mode.train)))
        if not latent:
            sampled1 = self.trainer.get_wrapper_model().sample(img if not latent else None)
            sampled2 = self.trainer.get_wrapper_model().sample(img if not latent else None)
            print(abs(sampled1-sampled2).mean())
            self.assertFalse(torch.equal(sampled1, sampled2), "Sampled from same base are equal")

        else:
            sampled1_1 = self.trainer.get_wrapper_model().sample(target=torch.tensor([1], device=self.conf.device))
            sampled1_2 = self.trainer.get_wrapper_model().sample(target=torch.tensor([1], device=self.conf.device))
            sampled0_1 = self.trainer.get_wrapper_model().sample(target=torch.tensor([0], device=self.conf.device))
            sampled0_2 = self.trainer.get_wrapper_model().sample(target=torch.tensor([0], device=self.conf.device))
            self.assertFalse(torch.equal(sampled1_1, sampled1_2), "Sampled from same base are equal")
            self.assertFalse(torch.equal(sampled0_1, sampled0_2), "Sampled from same base are equal")
            self.assertFalse(torch.equal(sampled1_1, sampled0_2), "Sampled from same base are equal")
            self.assertFalse(torch.equal(sampled0_1, sampled1_2), "Sampled from same base are equal")

        if not latent:
            reconstructed1 = self.trainer.get_wrapper_model().reconstruct(img)
            reconstructed2 = self.trainer.get_wrapper_model().reconstruct(img)
            print(abs(reconstructed1-reconstructed2).mean())

            self.assertTrue(torch.equal(reconstructed1, reconstructed2), "Reconstructed images should be equal")

    # -------------------------
