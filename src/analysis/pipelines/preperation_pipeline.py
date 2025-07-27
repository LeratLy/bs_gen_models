import os.path
from enum import Enum
from typing import Union

from src.analysis.latents import infer_latents
from src.analysis.pipelines.pipeline import Pipeline, SavedModel
from src.analysis.samples import create_samples, data_to_npz

class PreparationSteps(Enum):
    """
    Pipeline that can be used to evaluate the models performance
    """
    sample_model = "sample_model"
    sample_original = "sample_original"
    infer = "infer"

class PreparationPipeline(Pipeline):
    supported_steps = [PreparationSteps.sample_model, PreparationSteps.sample_original, PreparationSteps.infer]
    num_model_samples: Union[list[int], int] = 200
    save_single_images: bool = False

    def __init__(self, supported_steps: list[str] = None, num_new_samples: Union[list[int], int] = None, **kwargs):
        super().__init__(**kwargs)
        if supported_steps is not None:
            self.supported_steps = supported_steps
        if num_new_samples is not None:
            self.num_model_samples = num_new_samples

    def run(self, saved_model: SavedModel = None):
        if saved_model is not None:
            self.update_model(saved_model)
        samples_path = os.path.join(self.base_path, f"samples_{self.saved_model.model_name.value}")

        if PreparationSteps.sample_model  in self.supported_steps:
            self.logger.info(f"Creating {self.num_model_samples} new samples...")
            create_samples(self.model, self.num_model_samples, batch_size=1, path=os.path.join(self.base_path, f"samples_{self.saved_model.model_name.value}"), num_classes=self.num_classes)

        if PreparationSteps.sample_original in self.supported_steps:
            # TODO: so far needs to be run with main labels in dataset
            self.logger.info(f"Saving original data to folder...")
            path = os.path.join(self.base_path, f"original_samples")
            data_to_npz(self.dataloaders, path, self.save_single_images)

        if PreparationSteps.infer in self.supported_steps:
            self.logger.info(f"Inferring latents for all samples...")
            self.model.conf.batch_size = 1
            folders = [samples_path] + self.original_folders
            infer_latents(self.model, self.base_path, file_name=self.latent_file_name, alternative_folders=folders, use_initial_labels=self.use_hidden_initial_labels)
