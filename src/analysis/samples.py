import os
from collections import defaultdict
from typing import Union

import numpy as np
import torch
from tqdm import tqdm

from run_models.model_templates import chp96_cvae_bernoulli_conf
from src._types import SavedModelTypes, SaveTo
from src.analysis.latents import infer_latents
from src.analysis.utils import load_model_with_checkpoint
from src.models.dae.dae import DAE
from src.models.trainer import Trainer
from src.models.vae.vae import VAE
from src.utils.visualisation import plot_3d_data_cloud
from variables import MS_MAIN_TYPE, MS_TYPES_TO_MAIN_TYPE


def create_samples(model: Union[DAE, VAE], num: Union[list[int], int], path: str, num_classes: int = 2, batch_size=32):
    """
    Create samples for classes
    :param model: model used for generating samples (should inherit autoencoder_base model)
    :type model: Union[DAE,VAE]
    :param num: list with number of samples per class or one number of samples for all classes
    :type num: Union[list[int], int], default
    :param num_classes: number of classes to generate samples for (0-num_classes)
    :type num_classes: int, default 2
    :param batch_size: batch size to generate samples, num should ideally be dividable by batch_size, if num is smaller than batch_size, num is used as batch_size
    :type batch_size: int, default 32
    :param path: absolute path, where samples should be saved
    :type path: str
    :returns: tensors with samples and classes
    """
    file_name = os.path.join(path, "samples.npz")
    if not os.path.exists(path):
        os.makedirs(os.path.join(path))
    if type(num) is int:
        assert model.conf.num_classes == num_classes, "Create equal amount of samples per class mode does not support differing classes between mode.\nPlease use explicit list for samples per class if you want to create differing samples!"
        assert num is not None, "You need to specify number of classes when num is an integer"
        samples, targets = model.create_samples_per_class(num, num_classes, min(num, batch_size))
        np.savez(file_name, images=samples.detach().cpu().numpy(), labels=targets.detach().cpu().numpy())

    else:
        assert len(
            num) == model.conf.num_classes, f"one num must be defined for each class, number of classes is {model.conf.num_classes}, but {len(num)} numbers are given"
        running_samples, running_targets = [], []
        running_initial_targets = []
        # create samples per class the model knows
        for i in range(model.conf.num_classes):
            samples, targets = model.create_samples_for_class(num[i], i, min(num[i], batch_size))
            running_samples += samples
            if num_classes < model.conf.num_classes:
                running_initial_targets += targets
            else:
                running_targets += targets

        # the model was trained with 5 classes (more than 2 main classes and therefore those should be used as initial targets)
        if len(running_initial_targets) > 0:
            initial_targets = torch.cat(running_initial_targets).squeeze(1).detach().cpu().numpy()
            targets = np.vectorize(MS_TYPES_TO_MAIN_TYPE.get)(initial_targets)
            np.savez(file_name, images=torch.cat(running_samples).detach().cpu().numpy(),
                     labels=targets, initial_labels=initial_targets)
        # the model only works on main classes
        else:
            np.savez(file_name, images=torch.cat(running_samples).detach().cpu().numpy(),
                     labels=torch.cat(running_targets).squeeze(1).detach().cpu().numpy())


def npy_to_npz(folder_path: str, output_file: str):
    """
    Loop through all npy files in folder_path and convert them to numpy arrays
    :param folder_path: to search for .npy files
    :param output_file: to save numpy arrays as .npz files
    :return:
    """
    data_dict = {}
    # Loop through all .npy files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            key_name = os.path.splitext(filename)[0]  # Use filename (without extension) as key
            data_dict[key_name] = np.load(file_path)

    # Save all arrays to a single .npz file
    np.savez(os.path.join(folder_path, output_file), **data_dict)

    print(f"Saved {len(data_dict)} arrays to {output_file}")


def original_to_npz(base_path):
    """
    Save data of trainer to given path (different sub folder will be created, per set)
    :param base_path: path to folder where original samples should be saved
    :return:
    """
    conf = chp96_cvae_bernoulli_conf()  # also any other model can be used, it is only important that the data is loaded correctly
    conf.classes = MS_MAIN_TYPE
    conf.__post_init__()
    trainer = Trainer(conf)
    original_path = os.path.join(base_path, "original")
    trainer.data_to_npz(os.path.join(original_path), True)
    trainer.close()


def sample(sampled_path, model_name, checkpoint_dir, num: Union[list[int], int]):
    """
    Create samples for classes based on model type and checkpoint and saves them to a samples.npz file
    :param sampled_path: folder, where samples should be saved
    :param model_name: name of model used for generating samples
    :param checkpoint_dir: checkpoint folder, containing checkpoint with parameters used to initialize model
    :param num: number of samples per class or one number of samples for all classes
    :return:
    """
    print("Starting sample generation")
    model = load_model_with_checkpoint(model_name, checkpoint_dir)
    create_samples(model, num, batch_size=2, path=sampled_path)


def infer(sampled_path, model_name: SavedModelTypes, checkpoint: str, device="cpu"):
    """
    Infer latents for all npz files in the given folder structure
    :param checkpoint:
    :param sampled_path: folder containing npz files and to store features.pkl
    :param model_name: name of model used to infer latents
    :param device: to work on
    :return:
    """
    print("Starting to infer latents")
    model = load_model_with_checkpoint(model_name, checkpoint)
    model.conf.batch_size = 1
    model.conf.device = device
    infer_latents(model, sampled_path, file_name="features.pkl")


def data_to_npz(dataloaders: dict, path: str, save_single_imgs: bool = False):
    """
    !!!! NOTE: Datalaoders must not have shuffle enabled !!!!
    Save data from train, validation and test set to file in numpy format for easier processing and loading
    :param dataloaders: with data that should be saved
    :param save_single_imgs: save single images to different target folders
    :param path: base folder to save data to (will be saved in path_{mode} where mode is train, val or test)
    :type path: str
    """
    for mode, dataloader in dataloaders.items():
        data_path = f"{path}_{mode.value}"
        target_ids = defaultdict(int)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        running_targets = []
        running_features = []
        for batch_idx, (features, targets, ids) in tqdm(enumerate(dataloader)):
            if save_single_imgs:
                for j, target in enumerate(targets):
                    np_array = features[j].detach().cpu().numpy()
                    plot_3d_data_cloud(np_array[0], title=f"class_{target.item()}_{target_ids[target.item()]}",
                                       save_to=SaveTo.png, path=data_path)
                    target_ids[target.item()] += 1

            running_features.append(features)
            running_targets.append(targets)
        np.savez(
            os.path.join(data_path, "samples.npz"),
            images=torch.cat(running_features).detach().cpu().numpy(),
            labels=torch.cat(running_targets).detach().cpu().numpy(),
            file_names=dataloader.dataset.image_paths,
            initial_labels=dataloader.dataset.initial_labels
        )
