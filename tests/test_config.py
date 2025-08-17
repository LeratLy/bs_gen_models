import os
import unittest

from src import SimpleModelConfig, VAEModelConfig
from src.config import BaseConfig
from run_models.model_templates import assign_model_config, _chp96_diffae_base_conf
from src.models.test_model.test_model import SimpleModel
from src._types import DataType, ModelName
from variables import ROOT_DIR, DATA_DIR


class ConfigUnittest(unittest.TestCase):
    """
    Testcases to test the configs and whether they can be used to create the correct models
    For the first time please run in order
    """

    def setUp(self):
        self.main_path = ROOT_DIR
        self.data_path = os.path.join(DATA_DIR, "test_data")
        self.conf = BaseConfig()
        self.conf.model = "src.models.test_model.test_model.SimpleModel"
        self.conf.model_name = ModelName.simple_model
        self.conf.batch_size = 1,
        self.conf.shuffle = True,
        self.conf.data = {
            "type": DataType.np,
            "name": None,
        }

    def test_make_model_config(self):
        # Created correct model config
        assign_model_config(self.conf)
        self.assertIsNotNone(self.conf.model, "Model config has not been assigned")
        self.assertIsInstance(self.conf.model_conf, SimpleModelConfig, "The wrong model config has been created")

        # Created correct model with config and
        model = self.conf.make_model()
        self.assertIsNotNone(self.conf.model_conf, "Model has not been created")
        self.assertIsInstance(model, SimpleModel, "The wrong model hs been created")

    def test_dae_base_conf(self):
        conf = _chp96_diffae_base_conf()
        self.assertIsInstance(conf, BaseConfig)

    def test_save_and_load_config(self):
        self.conf.model_conf = VAEModelConfig()
        self.conf.save("./test_files/test_config.json")
        self.conf = BaseConfig()
        self.conf.load("./test_files/test_config.json")
        self.conf.__post_init__()


if __name__ == '__main__':
    unittest.main()
