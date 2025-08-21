import unittest
import torch

from tqdm import tqdm

from src._types import SaveTo, Mode
from run_models.model_templates import chp96_diffae_gaussian_conf, chp96_diffae_xor_conf
from src.models.trainer import Trainer
from src.utils.visualisation import plot_3d_data_cloud


class DiffusionTestCase(unittest.TestCase):
    """
    Test cases and class to visualize the diffusion process for xor and gaussian models
    """

    def setUp(self):
        self.trainer = None

    def tearDown(self):
        if self.trainer is not None:
            self.trainer.close()

    def test_diffusion_gaussian(self):
        self.conf = chp96_diffae_gaussian_conf()
        self.conf.eval.eval_training_every_epoch = -1
        self.conf.batch_size = 1
        self.conf.__post_init__()
        self.trainer = Trainer(self.conf)
        self.diffuse("gaussian diffuse x_start")

    def test_ddim_sample_loop_gaussian(self):
        self.conf = chp96_diffae_gaussian_conf()
        self.conf.eval.eval_training_every_epoch = -1
        self.conf.batch_size = 1
        self.conf.__post_init__()
        self.trainer = Trainer(self.conf)
        self.ddim_sample_loop("gaussian ddim_loop x_start")

    def test_diffusion_xor(self):
        self.conf = chp96_diffae_xor_conf()
        self.conf.eval.eval_training_every_epoch = -1
        self.conf.batch_size = 1
        self.conf.__post_init__()
        self.trainer = Trainer(self.conf)
        self.diffuse("xor diffuse x_start")

    def test_ddim_sample_loop_xor(self):
        self.conf = chp96_diffae_xor_conf()
        self.conf.eval.eval_training_every_epoch = -1
        self.conf.batch_size = 1
        self.conf.__post_init__()
        self.trainer = Trainer(self.conf)
        self.ddim_sample_loop("xor ddim_loop x_start")

    def diffuse(self, title):
        img, target, index = next(iter(self.trainer.dataloaders.get(Mode.train)))
        sampler = self.trainer.get_wrapper_model().sampler
        show_diff_step_at = 50
        x_start = img.to(self.conf.device)
        T = 1000
        for t in tqdm(range(0, T)):
            torch_t = torch.tensor([t], device=self.conf.device, dtype=torch.int64)
            eps = sampler.get_noise(img, torch_t).to(self.conf.device)
            x_t = sampler.q_sample(x_start, torch_t, noise=eps)
            x_start_pred = sampler._predict_xstart_from_eps(x_t, torch_t, eps)
            if t % show_diff_step_at == 0:
                plot_3d_data_cloud(eps[0][0], title + "_noise", step=t, save_to=SaveTo.tensorboard,
                                   writer=self.trainer.writer)
                plot_3d_data_cloud(x_t[0][0], title + "_x_t", step=t, save_to=SaveTo.tensorboard,
                                   writer=self.trainer.writer)
                plot_3d_data_cloud(x_t[0][0], f"{title}_x_t_{t}", step=t, save_to=SaveTo.svg,
                                   writer=self.trainer.writer, ax_limits=[(20,80), (20,80),(20,80)])
                plot_3d_data_cloud(x_start_pred[0][0], title + "_x_start_pred", step=t, save_to=SaveTo.tensorboard,
                                   writer=self.trainer.writer)

    def ddim_sample_loop(self, title):
        img, target, index = next(iter(self.trainer.dataloaders.get(Mode.train)))
        sampler = self.trainer._wrapper_model.sampler
        show_diff_step_at = 100
        x_start = img.to(self.conf.device)
        T = 999
        torch_t = torch.tensor([T], device=self.conf.device, dtype=torch.int64)
        x_t = sampler.get_noise(x_start, torch_t).to(self.conf.device)
        for t in tqdm(reversed(range(0, T))):
            torch_t = torch.tensor([t], device=self.conf.device, dtype=torch.int64)
            eps = sampler._predict_eps_from_xstart(x_t, torch_t, x_start)  # perfect x_start and eps
            x = sampler._get_ddim_result(x_t, torch_t, x_start)["sample"]
            if t % show_diff_step_at == 0:
                plot_3d_data_cloud(eps[0][0], title + " -> noise_ddim_loop", step=T - t, save_to=SaveTo.tensorboard,
                                   writer=self.trainer.writer)
                plot_3d_data_cloud(x_t[0][0], title + " -> x_t_ddim_loop", step=T - t, save_to=SaveTo.tensorboard,
                                   writer=self.trainer.writer)
            x_t = x
