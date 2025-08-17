import unittest

import torch

from src._types import Mode
from run_models.model_templates import chp96_cvae_bernoulli_conf
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

    def test_train(self):
        self.conf = chp96_cvae_bernoulli_conf()
        self.conf.num_epochs = 1
        self.conf.checkpoint["save_every_epoch"] = -1
        # self.conf.checkpoint["name"] = "enable_and_add_checkpoint_name_for_testing"
        self.conf.eval.eval_training_every_epoch = 1

        self.conf.name ="cvae"
        self.training()

    def test_sampling(self):
        # 3. sample without latent
        self.conf = chp96_cvae_bernoulli_conf()
        self.sample()


    # ---- general helpers ----
    def training(self):
        """
        1. Train the base diffae
        """
        self.conf.__post_init__()
        self.trainer = Trainer(self.conf)
        self.trainer.train()

        self.assertIsNotNone(self.trainer)
        self.assertIsNotNone(self.trainer.conf.checkpoint.get("name"))


    def sample(self):
        """
        Create samples with model
        """
        self.conf.__post_init__()
        self.trainer = Trainer(self.conf)

        img, target, index = next(iter(self.trainer.dataloaders.get(Mode.train)))
        img = img.to(self.trainer.conf.dtype)

        sampled1 = self.trainer.get_wrapper_model().sample(target=torch.tensor([1], device=self.conf.device))
        sampled2 = self.trainer.get_wrapper_model().sample(target=torch.tensor([0], device=self.conf.device))

        reconstructed1 = self.trainer.get_wrapper_model().reconstruct(img, target=target)
        reconstructed2 = self.trainer.get_wrapper_model().reconstruct(img, target=target)
        self.assertFalse(torch.equal(sampled1, sampled2), "Sampled from same base are equal")
        self.assertTrue(torch.equal(reconstructed1, reconstructed2), "Reconstructions should be equal")
    # -------------------------
