import json
import os
from typing import Union, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio.transforms as transforms
from torch.utils.data import Dataset, TensorDataset

from src.utils.logger import TrainingLogger
from src.utils.preprocessing import downsample_image, crop_image, crop_image_to_indices, normalize_gaussian


class LoggableDataset(Dataset):
    logger: TrainingLogger = None
    labels: Union[list, np.ndarray] = None

    def add_logger(self, logger: TrainingLogger):
        self.logger = logger


class SimpleDataset(LoggableDataset):
    """
    A simple npz dataset with preloaded images and labels
    """

    def __init__(self, images, labels, transform=None, target_transform=None, initial_labels=None) -> None:
        """
        Initialize dataset
        :param images: the images of the dataset
        :param labels: labels of the images (in same order as images)
        :param transform: a transform that is applied to the images
        :param target_transform: a transform that is applied to the labels
        """
        self.images = images
        self.labels = labels
        self.initial_labels = initial_labels if initial_labels is not None else labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.labels.size

    def __getitem__(self, idx):
        torch_array = torch.from_numpy(self.images[idx])
        label = self.labels[idx]
        index_as_tensor = torch.from_numpy(np.array(idx))
        if self.transform is not None:
            torch_array = self.transform(torch_array)
        if self.target_transform:
            label = self.target_transform(label)
        return torch_array, label.squeeze(0), index_as_tensor


class NiiDataset(LoggableDataset):
    """
    A dataset that loads nifti data lazily
    """

    def __init__(self, data_path: str, split_csv_name: str, img_size: Union[Tuple[int, ...], int] = None,
                 dims: int = None, preprocess_img: str = "crop", do_normalize_gaussian: bool = True,
                 use_transforms: bool = False):
        """
        Initialize dataset

        :param data_path: path where the data can be found
        :param split_csv_name: csv with file names and labels
        :param img_size: the size the image should be cropped to (either tuple with one value per dimension or integer used for all dimensions)
        :type img_size: Union[Tuple[int, ...], int]
        :param dims: number of dimensions, used if image is simple integer
        :type img_size: int
        """

        def read_crop_indices(extra_path: str = None):
            """
            Read dataset.json from data_dir (possibly extended by the 'extra_path' and save crop indices in dictionary
            """
            dataset_path = data_path if extra_path is None else os.path.join(data_path, extra_path)
            with open(os.path.join(dataset_path, 'dataset.json')) as f:
                d = json.load(f)
                self.crop_indices |= d["min_bb_indices_per_file"]

        split = pd.read_csv(split_csv_name, sep=";")
        self.preprocess_img = preprocess_img
        self.data_path = data_path
        self.initial_labels =  split["initial_label"].to_numpy() if split.get("initial_label") is not None else None
        self.image_paths = split["file_name"].to_numpy()
        self.labels = split["label"].to_numpy()
        self.data_len = len(self.image_paths)
        self.img_sizes = np.array(img_size) if img_size else None
        self.crop_warning_disabled = False
        self.crop_indices = {}

        self.do_normalize_gaussian = do_normalize_gaussian
        # get crop indices for cropping images
        separated = split["file_name"].str.replace("\\", "/").str.rsplit(r"/", expand=True, n=1)
        if len(separated.columns) > 1:
            separated.rename(columns={0: "folder_name", 1: "file_name"}, inplace=True)
            folder_names = separated["folder_name"].unique()
            for folder_name in folder_names:
                read_crop_indices(folder_name)
        else:
            read_crop_indices()

        # get image size array, indicating shape of items
        if isinstance(img_size, int):
            assert dims is not None, "Invalid parameters: Dimensions must be defined, when img size is an integer"
            self.img_sizes = np.full(dims, img_size)

        if use_transforms:
            self.transforms = transforms.Compose([
                transforms.RandomAffine(translation=(0, int(self.img_sizes[0]//10)), degrees=(0,15))
            ])
        else:
            self.transforms = None

        if self.preprocess_img == "crop_and_scale":
            sizes = tuple(map(int, self.img_sizes))
            self.scale_transform = transforms.Compose([
                transforms.Resize(sizes[0])]
            )
        else:
            self.scale_transform = None

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        try:
            # read a single nii
            img_nii = nib.load(os.path.join(self.data_path, img_path))
            img = img_nii.get_fdata()

            if self.img_sizes is not None and self.preprocess_img is not None:
                img = self._resize_image(img, img_path)
            img_as_tensor = torch.from_numpy(img).unsqueeze(0)

            label_as_tensor = torch.from_numpy(np.array(self.labels[index]))
            index_as_tensor = torch.from_numpy(np.array(index))
        except Exception as e:
            print("img_path: ", img_path)
            raise e

        if self.do_normalize_gaussian:
            img_as_tensor = normalize_gaussian(img_as_tensor)

        if self.scale_transform is not None:
            img_as_tensor = self.scale_transform(img_as_tensor)
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_tensor)
        return img_as_tensor, label_as_tensor, index_as_tensor

    def _resize_image(self, img, img_path):
        if self.preprocess_img == "crop" or self.preprocess_img == "crop_and_scale":
            cropped_img = crop_image_to_indices(img, self.crop_indices[img_path])
            if self.preprocess_img == "crop_and_scale":
                img = cropped_img
            elif  np.any(self.img_sizes - cropped_img.shape < 0):
                if not self.crop_warning_disabled:
                    warning = ("Size to crop to is smaller than the cropped image. "
                               "Fallback cropping mode is used (crop patch from middle)")
                    if self.logger is not None:
                        self.logger.file_logger.debug(warning)
                    else:
                        print(warning)
                    self.crop_warning_disabled = True
                img = crop_image(img, self.img_sizes)
            else:
                img = self.add_zero_padding(cropped_img)
        elif self.preprocess_img == "scale":
            img = downsample_image(img, scale_factor=1.5)

        return img


    def __len__(self):
        return self.data_len

    def add_zero_padding(self, img):
        assert np.all(self.img_sizes - img.shape >= 0), (f"Invalid values: The image sizes per dimension "
                                                         f"{self.img_sizes} must be greater or equal to the "
                                                         f"corresponding shape {img.shape}")
        total_padding = self.img_sizes - img.shape
        pad_left = total_padding // 2
        pad_right = total_padding - pad_left
        return np.pad(img, tuple(zip(pad_left, pad_right)), mode="constant", constant_values=0)


def load_tensor_dict_dataset(tensor_dict):
    return {
        k: TensorDataset(v["features"], v["targets"], v["ids"]) for k, v in tensor_dict.items()
    }
