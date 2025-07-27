from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset  # , DistributedSampler

from src._types import DataType, Mode
from src.data.datasets import SimpleDataset, NiiDataset
# from src.utils.multiprcoessing import DistributedWeightedSampler

"""
    Copyright (C) 2024 LeratLy - All Rights Reserved
    You may use, distribute and modify this code under the
    terms of the MIT license.
    You should have received a copy of the MIT license with
    this file.
"""


def get_dataloader(data_path: str = None,
                   batch_size: int = 32,
                   shuffle: bool = False,
                   data_type: DataType = None,
                   distributed: bool = False,
                   split_csv_paths: dict[Mode, str] = None,
                   img_size: Union[Tuple[int, ...], int] = None,
                   dims: int = None,
                   datasets: dict[Mode, Dataset] = None,
                   preprocess_img: str = 'crop',
                   do_normalize_gaussian: bool = True,
                   randomWeightedTrainSet: bool = False,
                   use_transforms:bool = False) -> dict[str, DataLoader]:
    """
    Creates dataloaders for the given configuration
    :param distributed: whether a distributed Sampler should be initialized and used
    :param data_type: the data type, to use correct loading function
    :param data_path: path where dataloader can find the data
    :param batch_size: batch size for the loader
    :param shuffle: whether the data should be shuffled
    :param split_csv_paths: file with information about splits to load (not mandatory for np loader so far)
    :param img_size: the size the image should be cropped to
    :type img_size: Tuple[int, ...]
    :param dims: number of dimensions, used if image is simple integer
    :type img_size: int
    :param datasets: alternatively give dataset for which a loader should be created dictionary with entries per mode
    :type datasets: dict[Mode, Dataset]
    :param preprocess_img: what preprocessing method should be used, default 'crop'
    :type preprocess_img: str, [None,'crop','scale']
    :param randomWeightedTrainSet: should the training set be randomly sampled based on label frequency
    :type randomWeightedTrainSet: bool
    :return:
    """
    _shuffle = shuffle
    if datasets is None:
        if data_type == DataType.np:
            datasets = npz_loader(data_path)
        elif data_type == DataType.nii:
            datasets = nii_loader(
                data_path,
                split_csv_paths,
                img_size,
                dims,
                preprocess_img=preprocess_img,
                do_normalize_gaussian=do_normalize_gaussian,
                use_transforms=use_transforms
            )
        else:
            raise NotImplementedError

    dataloaders = {}
    for mode in Mode:
        if mode in datasets:
            sampler = None
            if mode == Mode.train and randomWeightedTrainSet:
                sampler_weights = compute_weights_for_sampler(datasets[mode])
                # Note! Not tested yet
                # if distributed:
                #     sampler = DistributedWeightedSampler(datasets[mode], weights=sampler_weights)
                # else:
                sampler = WeightedRandomSampler(torch.from_numpy(sampler_weights), len(sampler_weights))
                _shuffle = False
            # elif distributed:
            #     sampler = DistributedSampler(datasets[mode], shuffle=shuffle, drop_last=True)
            dataloaders[mode] = DataLoader(datasets[mode], batch_size=batch_size, shuffle=_shuffle, sampler=sampler)
    return dataloaders


def npz_loader(data_path) -> dict[str, SimpleDataset]:
    """
    Loads numpy data into a dataset from a specific path containing already split data
    :param data_path: exact path where to find the .npz file
    :return: a dictionary with SimpleDatasets loaded for different splits (train, val, test)
    """
    assert type(data_path) is str, f"Path is wrong, should be string but is {type(data_path)}"
    np_array = np.load(data_path)
    datasets = {}
    for mode in Mode:
        datasets[mode] = SimpleDataset(np_array[f"{mode.value}_images"], np_array[f"{mode.value}_labels"])
    return datasets


def nii_loader(data_path: str,
               split_csv_paths: dict[Mode, str],
               img_size: Union[Tuple[int, ...], int] = None,
               dims: int = None,
               preprocess_img: str = "crop",
               do_normalize_gaussian: bool = True,
               use_transforms: bool = False,) \
        -> dict[str, NiiDataset]:
    """
    Loads nifti data into a dataset from a specific path and for a specific split
    :param data_path: where to find nifti data
    :param split_csv_paths: a dictionary with paths to the split files containing the labels and filenames of the data
    :param img_size: the size the image should be cropped to (either per dimension or one integer for all dimensions)
    :type img_size:  img_size: Union[Tuple[int, ...], int]
    :param dims: number of dimensions, used if image is simple integer
    :type img_size: int
    :param preprocess_img: what preprocessing method should be used, default 'crop'
    :type preprocess_img: str, [None,'crop','scale']
    :return: a dictionary with NiiDatasets loaded for different splits (train, val, test)
    """
    assert data_path is not None, "A data_path has not bee found or was not defined, please check again."
    datasets = {}
    for mode in Mode:
        if mode == Mode.train:
            use_transforms = use_transforms
        else:
            use_transforms = False
        if mode in split_csv_paths:
            datasets[mode] = NiiDataset(
                data_path,
                split_csv_paths[mode],
                img_size,
                dims,
                preprocess_img,
                use_transforms=use_transforms,
                do_normalize_gaussian=do_normalize_gaussian
            )
    return datasets


def compute_weights_for_sampler(dataset: Union[NiiDataset, SimpleDataset, TensorDataset]):
    """
    Compute weights for weighted random sampler based on label frequency
    Labels must be saved in a laebls property or as second argument in TensorDataset
    :param dataset:
    :return:
    """
    labels = dataset.tensors[1].numpy().astype(np.int32) if type(dataset) == TensorDataset else dataset.labels
    class_sample_count = np.unique(labels, return_counts=True)[1]
    weight = 1. / class_sample_count
    return weight[labels]