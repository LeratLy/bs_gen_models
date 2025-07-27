import dataclasses
import logging
from typing import Union

from src._types import SavedModelTypes, DataType
from src.analysis.utils import load_model_with_checkpoint
from src.data.dataloader import get_dataloader
from src.models.dae.dae import DAE
from src.models.vae.vae import VAE
from variables import data_paths, split_csvs


@dataclasses.dataclass
class SavedModel:
    checkpoint_path: str
    model_name: SavedModelTypes
    config_kwargs: dict = None

class Pipeline:
    supported_steps: list[str]
    logger: logging.Logger
    base_path: str
    model: Union[DAE, VAE] = None
    saved_model: SavedModel = None
    data_name: str = "chP3D_tune_70" #  !!! Note please always use the base class here, since this is what is saved in labels (init labels contains specifc class info=
    latent_file_name = "features.pkl"
    prototypes_file_name = "prototypes.pkl"
    samples_file_name = "samples.npz"
    prototypes_3D_file_name = "prototypes_3D.npz"
    prototypes_initial_3D_file_name = "prototypes_initial_3D.npz"
    device = "cpu"
    num_classes: int = 2
    dataloader_args = {
        "batch_size": 1,
        "shuffle": False,
        "distributed": False,
        "preprocess_img": "crop",
        "do_normalize_gaussian": False,  # we do not use this for gaussian
        "randomWeightedTrainSet": False,
        "use_transforms": False,
    }
    original_folders = ["original_samples_train", "original_samples_test", "original_samples_val"]
    # use hidden labels indicates that the model has been trained with more labels (those labels are saved in initial labels. The model needs to use those for sampling)
    # Nevertheless, labels will be used to aggregate the result (since the classifier and main comparison should only be performed on the main types. Not for each subtype
    use_hidden_initial_labels: bool = False

    def __init__(self, base_path: str, saved_model: SavedModel, device: str = None, **kwargs):
        self.dataloaders: dict = {}
        self.logger = logging.Logger("PipelineLogger", level=logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.base_path = base_path
        if device is not None:
            self.device = device
        if saved_model is None:
            self.logger.warning("No saved model has been passed, be aware that only original data can be evaluated")
        else:
            self.logger.info("Setting up model..")
            self.update_model(saved_model)
            self.setup_dataloaders()
            self.logger.info("Done setting up model!")

    def setup_dataloaders(self):
        data_path = data_paths[self.data_name]
        data_split_csv = split_csvs.get(self.data_name)

        self.dataloaders = get_dataloader(
            data_path=data_path,
            data_type=DataType.nii,
            split_csv_paths=data_split_csv,
            img_size=self.model.conf.img_size,
            dims=self.model.conf.dims,
            **self.dataloader_args
        )

    def update_model(self, saved_model: SavedModel):
        """
        Update model which is used for pipeline
        :param saved_model: config for loading model
        :return:
        """
        self.model = load_model_with_checkpoint(saved_model.model_name, saved_model.checkpoint_path, saved_model.config_kwargs)
        self.model.to(self.device)
        self.saved_model = saved_model
        self.latent_file_name = f"features_{self.saved_model.model_name.value}.pkl"
        self.prototypes_file_name = f"prototypes_{self.saved_model.model_name.value}.pkl"
        self.prototypes_3D_file_name = f"prototypes_3D_{self.saved_model.model_name.value}.npz"
        self.prototypes_initial_3D_file_name = f"prototypes_initial_3D_{self.saved_model.model_name.value}.npz"
        self.setup_dataloaders()

    @staticmethod
    def run(self):
        raise NotImplementedError()
