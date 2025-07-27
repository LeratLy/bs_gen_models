import unittest
from src.config import BaseConfig
from run_models.model_templates import assign_model_config
from src.models.trainer import Trainer
from src._types import ModelName, DataType, ConfigData


class ExperimentSimpleModel(unittest.TestCase):
    """
    Testcase to test whole model pipeline on a SimpleModel
    """

    def setUp(self):
        self.conf = BaseConfig()
        self.conf.name = 'simple_model'
        self.conf.model_name = ModelName.simple_model
        self.conf.batch_size = 1
        self.conf.img_size = 128
        self.conf.log_interval = 100
        self.conf.num_epochs = 1
        self.conf.model = "src.models.test_model.test_model.SimpleModel"
        self.conf.data = ConfigData(name='test3D', type=DataType.np)

    def test_simple_model_pipeline(self):
        assign_model_config(self.conf)
        self.conf.__post_init__()
        trainer = Trainer(self.conf)
        trainer.train()
        self.assertIsNotNone(trainer.conf.checkpoint["name"])
        trainer.close()

    def test_simple_model_pipeline_with_checkpoint(self):
        assign_model_config(self.conf)
        self.conf.checkpoint["name"] = "./testing_checkpoints/simple_model_test3D_20250317_095137_0"
        self.conf.__post_init__()
        trainer = Trainer(self.conf)
        trainer.train()
        self.assertIsNotNone(trainer.conf.checkpoint["name"])
        trainer.close()


if __name__ == '__main__':
    unittest.main()
