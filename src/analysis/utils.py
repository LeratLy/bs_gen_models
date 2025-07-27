import os
from time import time
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from run_models.clf_templates import get_chP96_clf_2cond_model_conf, get_chP96_clf_2cond_conf
from run_models.dae.tune_hyperparams_latent import get_base_config_latent
from run_models.model_templates import chp96_cvae_bernoulli_conf, get_final_base_config_latent
from src import MSClfNet
from src._types import SavedModelTypes, LossType, NoiseType, GenerativeType, Activation, ModelName
from src.config import ClfConfig
from src.data.datasets import SimpleDataset
from src.models.dae.dae import DAE
from src.models.vae.vae import VAE
from src.utils.checkpoints import load_model_directly
from src.utils.logger import TrainingLogger


def load_latents_from_multiple_folders(paths: list[str], file_name: str = "features.pkl", kwargs_load: dict = None):
    """
    Load latent tensors
    :param paths: list of paths to folders, where the latent files can be found
    :param file_name: name of file to load (should be in paths)
    :return: latent tensors for embeddings and labels
    """
    embeddings_list = []
    labels_list = []
    initial_labels_list = []
    for path in paths:
        _embeddings, _labels, _initial_labels = load_latents(path, file_name=file_name, kwargs_load=kwargs_load)
        embeddings_list.append(_embeddings)
        labels_list.append(_labels)
        initial_labels_list.append(_initial_labels)
    return torch.cat(embeddings_list, dim=0), torch.cat(labels_list, dim=0), torch.cat(initial_labels_list, dim=0)


def load_latents(base_path: str, file_name: str = "features.pkl", kwargs_load: dict = None):
    """
    Load latent tensors
    :param base_path: path to folder, where the latent file can be found
    :param file_name: name of file to load (should be in base_path)
    :return: latent tensors for embeddings and labels
    """
    features_path = os.path.join(base_path, file_name)
    if kwargs_load is None:
        kwargs_load = {}
    features = torch.load(features_path, **kwargs_load)
    embeddings = features["cond"]

    labels = features["labels"].to(dtype=torch.int32)
    initial_labels = features["initial_labels"].to(dtype=torch.int32)

    return embeddings, labels.squeeze(), initial_labels.squeeze()


def load_np_latents(base_path: str, file_name: str = "features.pkl", alternative_paths: list[str] = None,
                    kwargs_load: dict = None):
    """
    Load latent numpy arrays
    :param alternative_paths: list of paths to folders, where the latent files can be found, when given use them instead of base_path
    :param base_path: path to folder, where the latent file can be found
    :param file_name: name of file to load (should be in base_path)
    :return: latent numpy arrays for embeddings and labels
    """
    if alternative_paths is not None:
        embeddings, labels, initial_labels = load_latents_from_multiple_folders(alternative_paths, file_name,
                                                                                kwargs_load=kwargs_load)
    else:
        embeddings, labels, initial_labels = load_latents(base_path, file_name, kwargs_load=kwargs_load)
    return embeddings.detach().cpu().numpy(), labels.squeeze().detach().cpu().numpy(), initial_labels.squeeze().detach().cpu().numpy()


def load_npz_dataset(data_path: Union[list[str], str]):
    if isinstance(data_path, list):
        images_list = []
        labels_list = []
        initial_labels_list = []

        for path in data_path:
            np_array = np.load(path)
            images_list.append(np_array["images"])
            labels_list.append(np_array["labels"])
            if np_array.get("initial_labels") is not None:
                initial_labels_list.append(np_array.get("initial_labels"))

        images = np.concatenate(images_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        initial_labels = np.concatenate(initial_labels_list, axis=0) if len(initial_labels_list) > 0 else labels

    else:
        np_array = np.load(data_path)
        images = np_array["images"]
        labels = np_array["labels"]
        initial_labels = np_array.get("initial_labels") if np_array.get("initial_labels") is not None else labels

    return SimpleDataset(images, labels, initial_labels=initial_labels)


def npz_dataloader(data_path: str = None, batch_size: int = 32, shuffle: bool = False,
                   alternative_paths: list[str] = None):
    assert data_path is not None or alternative_paths is not None, "Either data or alternative paths must be given"
    return DataLoader(load_npz_dataset(alternative_paths if alternative_paths is not None else data_path),
                      batch_size=batch_size, shuffle=shuffle)


def load_model_with_checkpoint(model_name: SavedModelTypes, checkpoint_dir, add_kwargs=None) -> Union[VAE, DAE]:
    if model_name.is_cvae():
        conf = chp96_cvae_bernoulli_conf()
        if add_kwargs is not None:
            conf.model_conf.ch = add_kwargs["ch"]
            conf.model_conf.kld_weight = add_kwargs["kld"]
            if add_kwargs.get("num_classes") is not None:
                conf.model_conf.num_classes = add_kwargs["num_classes"]
                conf.num_classes = add_kwargs["num_classes"]
        conf.__post_init__()
        logger = TrainingLogger(conf, time())
        model = VAE(conf, logger, SummaryWriter())
    elif model_name.is_bdae():
        conf = get_final_base_config_latent()
        conf.clf_conf = ClfConfig(
            "path/to//data/final_models/checkpoints/analysis_final_ms_clf_base_20250711_101044_best",
            ModelName.ms_clf,
            get_chP96_clf_2cond_conf(),
        )
        conf.model_conf.latent_net_conf.num_layers = add_kwargs["layers"]
        conf.model_conf.latent_net_conf.skip_layers = list(range(1, add_kwargs["layers"]))
        conf.model_conf.latent_net_conf.num_hid_channels = add_kwargs["hidden_ch"]
        if add_kwargs.get("num_classes") is not None:
            conf.num_classes = add_kwargs["num_classes"]
        if add_kwargs.get("model_conf_num_classes") is not None:
            conf.model_conf.num_classes = add_kwargs["model_conf_num_classes"]
        if add_kwargs.get("latent_net_conf_num_classes") is not None:
            conf.model_conf.latent_net_conf.num_classes = add_kwargs["latent_net_conf_num_classes"]
        if add_kwargs.get("class_znormalize") is not None:
            conf.model_conf.latent_net_conf.class_znormalize = add_kwargs["class_znormalize"]
            conf.model_conf.latent_net_conf.znormalize = not add_kwargs["class_znormalize"]
        if add_kwargs.get("scale_target_alpha") is not None:
            conf.model_conf.latent_net_conf.scale_target_alpha = add_kwargs["scale_target_alpha"]
        if add_kwargs.get("shift_target") is not None:
            conf.model_conf.latent_net_conf.shift_target = add_kwargs["shift_target"]
        if add_kwargs.get("enc_merge_time_and_cond_embedding") is not None:
            conf.model_conf.enc_merge_time_and_cond_embedding = add_kwargs["enc_merge_time_and_cond_embedding"]
        conf.latent_infer_path = add_kwargs["latent_infer_path"]

        conf.__post_init__()
        logger = TrainingLogger(conf, time())
        model = DAE(conf, logger, SummaryWriter())
    else:
        raise NotImplementedError

    model = load_model_directly(model, checkpoint_dir)
    model.to(conf.device)
    return model


def load_clf_model(model_name: SavedModelTypes, checkpoint_path: str, device: str = "cpu") -> Union[MSClfNet]:
    if model_name == SavedModelTypes.clf:
        model_conf = get_chP96_clf_2cond_model_conf()
        model = MSClfNet(model_conf)
        checkpoint = torch.load(checkpoint_path)
        ema_state_dict = {}
        for k, v in checkpoint["model"].items():
            if k.startswith('ema_model.'):
                new_key = k[len('ema_model.'):]  # strip prefix
                ema_state_dict[new_key] = v
        model.load_state_dict(ema_state_dict)
    else:
        raise NotImplementedError()
    model.to(device)
    return model


def perform_for_all_folders(path: str, func, kwargs: dict = None):
    """
    Call function for all folders in given directory, passing folder as "path" argument
    :param path: to search for subfolders
    :type path: str
    :param func: apply function within each subfolder (add path variable accordingly)
    :type func: function
    :param kwargs: arguments for func
    :type kwargs: dict
    :return:
    """
    if kwargs is None:
        kwargs = {}
    for file in os.listdir(path):
        abs_path = os.path.join(path, file)
        if os.path.isdir(abs_path):
            print(f"Applying function for {abs_path}")
            kwargs["path"] = abs_path
            func(**kwargs)
