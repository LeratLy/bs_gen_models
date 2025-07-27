import json
import os

import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm

from src.utils.helpers import get_filenames_by
from variables import DATA_DIR


# from src.utils.visualisation import plot_3d_data

def find_bounding_box_multiple(paths: list[str], squared=True, save_div_by=32, save_path: str = None):
    """
    Find the minimal bounding box that is needed to include all non-zero values of nii.gz files in the given directories

    :param save_div_by: save additional the size ceiled up to be dividable by this value
    :type save_div_by: int
    :param paths: names of the folders with nii.gz files
    :type paths: [str]
    :param squared: whether the bounding box should be squared or is allowed to be an arbitrary rectangle
    :type squared: bool
    :return: min_bb_total - min bounding_box size per dimension, min_bbs, min bounding box per image and dimension (in indices)
    :rtype: tuple(np.array, np.array([
    [[img_0_min_dim0, img_0_max_dim0], ..., [img_0_min_dimN, img_0_max_dimN]],
     ...,
    [[img_M_min_dim0, img_M_max_dim0], ..., [img_M_min_dimN, img_M_max_dimN]]
    ]))
    """
    min_bb_overall = 0
    min_bbs_overall = {}
    for i, path in enumerate(paths):
        join = i > 0
        min_bb_total, min_bbs = find_bounding_box(path, squared, save_div_by, save_path, join=join)
        if np.any(min_bb_overall < min_bb_total):
            min_bb_overall = min_bb_total
        min_bbs_overall = min_bbs_overall | min_bbs
    return min_bb_overall, min_bbs_overall


def find_bounding_box(folder_name: str, squared=True, save_div_by=32, save_path=None, join: bool = False):
    """
    Find the minimal bounding box that is needed to include all non-zero values of nii.gz files in the given directory

    :param save_div_by: save additional the size ceiled up to be dividable by this value
    :type save_div_by: int
    :param folder_name: name of the folder with nii.gz files
    :type folder_name: str
    :param squared: whether the bounding box should be squared or is allowed to be an arbitrary rectangle
    :type squared: bool
    :return: min_bb_total - min bounding_box size per dimension, min_bbs, min bounding box per image and dimension (in indices)
    :rtype: tuple(np.array, np.array([
    [[img_0_min_dim0, img_0_max_dim0], ..., [img_0_min_dimN, img_0_max_dimN]],
     ...,
    [[img_M_min_dim0, img_M_max_dim0], ..., [img_M_min_dimN, img_M_max_dimN]]
    ]))
    """
    data_path = os.path.join(DATA_DIR, folder_name)
    if save_path is None:
        save_path = data_path
    filenames = get_filenames_by(r".*\.nii.gz", data_path)

    print("Searching bounding box...")
    min_bbs = {}
    min_bb_total = None
    num_samples_above_64 = 0

    for file_name in tqdm(filenames):
        img_nii = nib.load(os.path.join(data_path, file_name))
        img = img_nii.get_fdata()
        current_min_bb = min_bb_indices_for_img(img)
        min_bbs[os.path.join(folder_name, file_name)] = current_min_bb.tolist()

        cropped_img = crop_image_to_indices(img, current_min_bb)

        if np.any(cropped_img.shape > (64, 64, 64)):
            num_samples_above_64 +=1
        if min_bb_total is None or np.any(min_bb_total < cropped_img.shape):
            min_bb_total = np.full(cropped_img.ndim, max(cropped_img.shape)) if squared else cropped_img.shape

    json_save_path = os.path.join(save_path, 'dataset.json')
    if not os.path.exists(json_save_path):
        with open(json_save_path, 'w') as f:
            json.dump({}, f)
    with open(json_save_path) as f:
        data = json.load(f)

    data['min_bb_' + folder_name] = min_bb_total.tolist()
    if save_div_by is not None:
        data[f'min_bb_{save_div_by}_{folder_name}'] = make_dividable_by(min_bb_total, save_div_by).tolist()
    data['min_bb_indices_per_file'] = min_bbs

    # Write the modified data to a new JSON file
    with open(json_save_path, 'w') as f:
        json.dump(data, f)
    print(num_samples_above_64, "samples exist with bounding box that needs to be bigger than 64x64x64")

    return min_bb_total, min_bbs


def crop_image_to_indices(img: np.array, img_indices: np.array) -> np.array:
    """
    Crop the image to given indices

    :param img: The image that should be cropped
    :type img: Array
    :param img_indices: The different sizes for each dimension of the image
    :type img_indices: Array
    :return: The cropped image
    :rtype: Array
    """
    slices = tuple(slice(dim[0], dim[1]) for dim in img_indices)
    return img[slices]


def crop_image(img: np.array, img_size: np.array) -> np.array:
    """
    Crop the image to a given tuple of sizes (per dimension) It cuts pixels from the beginning and end of the
    dimension equally

    :param img: The image that should be cropped
    :type img: Array
    :param img_size: The different sizes for each dimension of the image
    :type img_size: Array
    :return: The cropped image
    :rtype: Array
    """
    assert img.ndim == len(img_size), \
        "The image size array should have the same length as the number of dimensions of the image"
    slices = []
    for i, size in enumerate(img_size):
        assert img.shape[i] >= size, \
            "The given image size for the dimension should not be smaller greater than the original size"
        crop_num_pixels = (img.shape[i] - size) // 2
        end = img.shape[i] - crop_num_pixels
        slices.append(slice(crop_num_pixels, end))
    return img[tuple(slices)]


def make_dividable_by(to_be_div: np.array, div_by):
    """
    Makes all elements in an array dividable by a specified value, by ceiling it
    :param to_be_div: the array that should be dividable
    :type to_be_div: np.array
    :param div_by: the value the array should be dividable by
    :type div_by: int
    :return:
    """
    return np.array(np.ceil(to_be_div / float(div_by)) * div_by).astype(int)


def min_bb_indices_for_img(img: np.array):
    """
    Return the indices of the min bounding box for a given numpy array (non-zero values)
    :param img:
    :return:
    """
    indices = np.nonzero(img)
    min_values = np.array([dim_indices.min() for dim_indices in indices])
    max_values = np.array([dim_indices.max() for dim_indices in indices])
    return np.column_stack((min_values, max_values)).astype(int)


def normalize_01(x):
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)


def normalize_gaussian(x):
    """
    Normalize to [-1,1]
    """
    return (x - 0.5) / 0.5


def unnormalize_gaussian(x):
    """
    Normalize back from [-1,1] to [0,1]
    """
    return (x + 1) / 2


def downsample_image(image, scale_factor=2.0):
    """
    Zooming in the image with order 1 (linear interpolation)
    Args:
        image: image that should be down-sampled
        scale_factor: scale-factor for sampling

    Returns: down-sampled array of [H//scale_factor,W//scale_factor,D/scale_factor]
    """
    return ndimage.zoom(image, 1.0 / scale_factor, order=1)


def upsample_image(image, scale_factor=2.0):
    """
    Zoom in the image with order 1 (linear interpolation)
    Args:
        image: image that should be up-sampled
        scale_factor: scale-factor for sampling

    Returns: up-sampled disparity matrix of [H*scale_factor,W*scale_factor,D*scale_factor]

    """
    return ndimage.zoom(image, scale_factor, order=1)


def resample_nifti(input_path: str, scale_factor: float = 1.5, direction: str = "down") -> nib.Nifti1Image:
    """
    Scale nifit image and save to putput path
    """
    img = nib.load(input_path)
    data = img.get_fdata()

    if direction == "up":
        data = upsample_image(data, scale_factor=scale_factor)
    elif direction == "down":
        data = downsample_image(data, scale_factor=scale_factor)
    else:
        raise NotImplementedError("Direction not implemented")
    return nib.Nifti1Image(data, img.affine)


def save_resampled_nifit(data_path):
    for file_path in tqdm(os.listdir(data_path)):
        if file_path.endswith(".nii.gz"):
            img = resample_nifti(os.path.join(data_path, file_path))
            nib.save(img, os.path.join(data_path + "_small", file_path))


