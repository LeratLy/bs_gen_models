import os.path
import unittest

import numpy as np
import random
from torch.utils.data import DataLoader

from src.config import BaseConfig
from src.data.dataloader import get_dataloader, nii_loader, npz_loader
from src._types import Mode, DataType, SaveTo, ConfigData
from src.data.datasets import NiiDataset, SimpleDataset
from src.utils.helpers import get_filenames_by
from variables import ROOT_DIR, DATA_DIR

from src.utils.visualisation import plot_3d_data_cloud

"""
    Copyright (C) 2024 LeratLy - All Rights Reserved
    You may use, distribute and modify this code under the
    terms of the MIT license.
    You should have received a copy of the MIT license with
    this file.
"""


class DataloaderHelperUnittest(unittest.TestCase):
    """
    Testing various helper functions which are independent of specifc data
    """

    def setUp(self):
        self.main_path = ROOT_DIR

    def test_get_filenames_by(self):
        files = get_filenames_by(r".*\.npz$", os.path.join(DATA_DIR, "test_data"))

        result_files = ["test3D.npz", "synapsemnist3d_2.npz", "synapsemnist3d.npz"]

        # Found correct files
        self.assertEqual(3, len(files))
        files.sort()
        result_files.sort()
        self.assertListEqual(files, result_files)


class NiiDataloaderUnittest(unittest.TestCase):
    """
    Unittest class for the dataloader.py which handles loading of nii data and creating corresponding datasets and
    dataloaders
    """

    def setUp(self):
        self.main_path = ROOT_DIR
        self.data_path = DATA_DIR
        self.conf = BaseConfig()
        self.conf.batch_size = 1
        self.conf.shuffle = True
        self.conf.data = {
            "type": DataType.nii
        }

    def test_nii_dataset_loader(self):
        data_splits = {
            Mode.train: os.path.join(self.data_path, "train_split_nii.csv")
        }
        datasets = nii_loader(self.data_path, data_splits)

        # Train dataset is created
        self.assertIn(Mode.train, datasets)
        self.assertNotIn(Mode.test, datasets)
        self.assertNotIn(Mode.val, datasets)

        # Train dataset is an NiiDataset
        self.assertIsInstance(datasets[Mode.train], NiiDataset)

    def test_nii_dataloader(self):
        data_splits = {
            Mode.train: os.path.join(self.data_path, "train_split_nii.csv")
        }
        dataloaders = get_dataloader(
            self.data_path, 1, self.conf.shuffle, self.conf.data["type"], False, data_splits
        )

        self.assertIsInstance(dataloaders[Mode.train], DataLoader)

    def test_nii_dataset(self):
        nii_dataset = NiiDataset(self.data_path, os.path.join(self.data_path, "train_split_nii.csv"))
        self.assertIsInstance(nii_dataset, NiiDataset)

        item, target, data_id = nii_dataset.__getitem__(0)
        dataset_length = len(nii_dataset)

        # Correct length of dataset and shape of slices
        self.assertEqual(545, dataset_length)
        self.assertEqual(256, item.shape[2])
        self.assertEqual(256, item.shape[3])
        # Plot -> only enable this for manual testing since windows does not close automatically yet
        plot_3d_data_cloud(np.asarray(item[0]), "Segmented Choroid Plexus")

    def test_nii_dataset_cropped_single(self):
        img_size = 96
        nii_dataset = NiiDataset(self.data_path, os.path.join(self.data_path, "train_split_nii.csv"), img_size, 3)
        self.assertIsInstance(nii_dataset, NiiDataset)

        item, target, data_id = nii_dataset.__getitem__(0)
        dataset_length = len(nii_dataset)

        # Correct length of dataset and shape of slices
        self.assertEqual(545, dataset_length)
        self.assertEqual(img_size, item.shape[1])
        self.assertEqual(img_size, item.shape[2])
        self.assertEqual(img_size, item.shape[3])
        item[0][item[0] == 1] = random.uniform(0, 0.1)
        plot_3d_data_cloud(np.asarray(item[0]), "Segmented Choroid Plexus", SaveTo.svg)

    def test_nii_dataset_cropped_smaller(self):
        img_size = (96, 20, 20)
        nii_dataset = NiiDataset(self.data_path, os.path.join(self.data_path, "train_split_nii.csv"), img_size)
        self.assertIsInstance(nii_dataset, NiiDataset)

        item, target, data_id = nii_dataset.__getitem__(0)
        dataset_length = len(nii_dataset)

        # Correct length of dataset and shape of slices
        self.assertEqual(545, dataset_length)
        self.assertEqual(img_size[0], item.shape[1])
        self.assertEqual(img_size[1], item.shape[2])
        self.assertEqual(img_size[2], item.shape[3])


class NpzDataloaderUnittest(unittest.TestCase):
    """
    Unittest class for the dataloader.py which handles loading of npz data and creating corresponding datasets and
    dataloaders
    """

    def setUp(self):
        self.main_path = ROOT_DIR
        self.data_path = os.path.join(DATA_DIR, "test_data/test3D.npz")
        self.conf = BaseConfig()
        self.conf.batch_size = 1,
        self.conf.shuffle = True,
        self.conf.data = ConfigData(name='test3D', type=DataType.np)
        self.num_samples = 2

    def test_simple_dataset(self):
        images = np.repeat(np.full((1, 20, 20, 20), 2, ), self.num_samples, 0)
        targets = np.ones(self.num_samples)
        simple_dataset = SimpleDataset(images, targets, transform=lambda x: x * 2, target_transform=lambda x: x + 1)

        self.assertIsInstance(simple_dataset, SimpleDataset)

        item, target, data_id = simple_dataset.__getitem__(0)
        dataset_length = len(simple_dataset)

        # Correct length of dataset and shape of slices
        self.assertEqual(self.num_samples, dataset_length)
        self.assertEqual(20, item.shape[1])
        self.assertEqual(20, item.shape[2])
        # Correct transformations
        self.assertEqual(4, item[0, 0, 0])
        self.assertEqual(2, target)

    def test_np_dataloader(self):
        dataloaders = get_dataloader(
            self.data_path, 1, self.conf.shuffle, self.conf.data["type"], False
        )

        self.assertIsInstance(dataloaders[Mode.train], DataLoader)

    def test_np_dataset_loader(self):
        datasets = npz_loader(self.data_path)

        # Train dataset is created
        self.assertIn(Mode.train, datasets)
        self.assertIn(Mode.test, datasets)
        self.assertIn(Mode.val, datasets)

        # Train dataset is an NiiDataset
        self.assertIsInstance(datasets[Mode.train], SimpleDataset)


if __name__ == '__main__':
    unittest.main()
