import os
from typing import Union

import numpy as np
import pandas as pd
import torch
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.measure import regionprops_table
from skimage.morphology import skeletonize
from tqdm import tqdm

from src.analysis.utils import npz_dataloader

"""
A collection of 3D shape metrics that are used to analyse generated 3D choroid plexus shapes
"""


#
# def approx_3d_curvature(skeleton, k:int = 5):
#     """
#     Approximates curvature of given points pute curvature, based on 3 nearest enigbours
#     :param k:
#     :param mask:
#     :return:
#     """
#     curvatures = []
#     nbrs  = NearestNeighbors(n_neighbors=k).fit(skeleton)
#     distances, indices = nbrs.kneighbors(skeleton)
#     always take 3 points and lay circumcircle around it (but not making sence if no proper line but hey maybe still acurate (display skelton before)
#     return np.array(curvatures)

def approx_medial_axis_length(mask):
    """
    Approximate medial axis length as skeleton of given shape
    """
    skeleton = skeletonize(mask, method="lee")
    # plot_3d_data_cloud(skeleton.astype(int), "Skeleton", SaveTo.png, "x", "y", "z")
    return np_volume(skeleton)


def np_volume(np_array):
    """
    Volume of given shape
    """
    return np.count_nonzero(np_array)


def surface_area(mask):
    """
    Compute the surface area of a 3D mask via marching_cubes and mesh_surface_area
    :param mask:
    :return:
    """
    verts, faces, _, _ = marching_cubes(mask, allow_degenerate=False, level=0.5)
    areas = mesh_surface_area(verts, faces)
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # mesh = Poly3DCollection(verts[faces])
    # mesh.set_edgecolor('k')
    # ax.add_collection3d(mesh)
    # plt.savefig(os.path.join(PLOT_DIR, "surface_area.png"))
    return areas


def volume(mask: torch.Tensor):
    """
    Return amount of nonzero pixels per element in a batch
    """
    return torch.count_nonzero(mask, dim=tuple(range(1, mask.ndim)))


def aspect_ratio(minor_ax_length, major_ax_length):
    """
    :param minor_ax_length: length of minor axis of ellipse surrounding the shape
    :param major_ax_length: length of major axis of ellipse surrounding the shape
    :return: Ratio of the major axis to the minor axis (of ellipse along eigenvectors)
    """
    return major_ax_length / minor_ax_length


def min_bb_shape(mask: torch.Tensor):
    """
    Computes min bounding box for a tensor and returns shape for each dimension
    :param mask: Tensor to get bb shape for (BxCxDxHxW), ndims=5D
    :type mask: torch.Tensor
    :return: min bounding boxes -> BATCH_SIZE x [min_d, min_h, min_w, max_d, max_h, max_w]
    """
    num_batches = mask.shape[0]
    device = mask.device

    # total num nonzero x num_dims
    coords = torch.nonzero(mask, as_tuple=False)
    bounding_boxes = torch.zeros((num_batches, 3), device=device, dtype=torch.int64)

    # For each batch compute min bounding box
    for b in range(num_batches):
        b_coords = coords[coords[:, 0] == b]  # get all coords for current batch
        b_coords = b_coords[:, 2:]  # drop batch index and channel
        if b_coords.numel() == 0:
            continue
        min_coords = b_coords.min(dim=0).values
        max_coords = b_coords.max(dim=0).values
        bounding_boxes[b] = max_coords - min_coords

    return bounding_boxes


def extract_feature_table(np_array):
    """
    Extract regionprops from a given array (should have no channel or batch dimension)
    Following properties are supported:
    area,
    area_convex,
    bbox,
    local_centroid,
    feret_diameter_max,
    solidity,
    minor_axis_length,
    aspect_ratio,
    surface_area,
    hull_volume_ratio

    :param np_array: array without channel and batch dimensions
    :type np_array: torch.Tensor (dtype int)
    :return: pandas table with regionporps for the given segmentation mask
    """
    try:
        table = regionprops_table(np_array, properties=(
            "area", "bbox", "local_centroid", "feret_diameter_max", "solidity", "minor_axis_length", "image_convex",
            "major_axis_length"))
        table["aspect_ratio"] = aspect_ratio(table["minor_axis_length"], table["major_axis_length"])
        table["surface_area"] = surface_area(np_array)
        table["approx_medial_axis_length"] = approx_medial_axis_length(np_array)
        table["volume"] = np_volume(np_array)
        table["surface_volume_ratio"] = table["surface_area"] / table["volume"]
        table["height"] = table["bbox-3"] - table["bbox-0"]
        table["width"] = table["bbox-4"] - table["bbox-1"]
        table["depth"] = table["bbox-5"] - table["bbox-2"]
        table["hull_volume_ratio"] = np.sum(np.sum(table["image_convex"])) / table["volume"]
        del table["image_convex"]
    except Exception as e:
        table = None
        print("Error:", str(e))
    return table


def extract_features(mask_tensor):
    """
    Extract geometric features from the given mask tensor
    :param mask_tensor:
    :return:
    """
    mask_np = mask_tensor.type(torch.int).detach().cpu().numpy()
    regions = extract_feature_table(mask_np[0][0])
    return torch.tensor(torch.concatenate([torch.tensor(v.flatten()) for v in regions.values()]),
                        device=mask_tensor.device)


def geometrics_for_folder(path: Union[str, list[str]], threshold: int = 0.5, file_name: str = "samples.npz",
                          save_path: str = None):
    """
    Read samples.npz file from given path and extract geometric features or corresponding arrays
    :param file_name:
    :param threshold: threshold used to create binary mask of data
    :param path: path to folder that contains a samples.npz file
    :type path: str
    :return:
    """
    assert save_path is not None or isinstance(path,
                                               str), "either save_path or path must be string, indicating, where to save csv to"
    if save_path is None:
        save_path = path
    if isinstance(path, list):
        file_paths = [os.path.join(p, file_name) for p in path]
    else:
        file_paths = [os.path.join(path, file_name)]
    dataloader = npz_dataloader(batch_size=1, alternative_paths=file_paths)
    rows = []
    print(f"Start extracting geometric features for {file_paths}...")
    for i, (features, targets, ids) in tqdm(enumerate(dataloader)):
        feature_table = compute_metrics_for_sample(features, ids, targets,
                                                   dataloader.dataset.initial_labels[ids.item()])
        if feature_table is not None:
            rows.append(feature_table)
    print("Done extracting geometric features.")

    df = pd.DataFrame(rows).set_index("idx")
    save_geometric_data_to_file(df, save_path)


def save_geometric_data_to_file(df: pd.DataFrame, save_path: str, file_prefix: str = ""):
    """
    Save geometrics to save path adding a file extension (separate csv are added for means)
    :param df:
    :param save_path:
    :param file_prefix:
    :return:
    """
    df.to_csv(os.path.join(save_path, file_prefix + "geometric_features.csv"))

    mean_df = df.groupby(['label']).mean().reset_index()
    mean_df.to_csv(os.path.join(save_path, file_prefix + "geometric_features_mean_per_class.csv"))
    mean_df = df.groupby(['initial_label']).mean().reset_index()
    mean_df.to_csv(os.path.join(save_path, file_prefix + "geometric_features_mean_per_class_initial_labels.csv"))
    print(f"Saved features to {os.path.join(save_path, file_prefix + "geometric_features.csv")}")


def compute_metrics_for_sample(features, ids=None, targets=None, initial_labels=None, threshold=0.5,
                               raw_features: bool = False):
    """
    Extract geometric features for a given sample
    :param features:
    :param ids:
    :param targets:
    :param initial_labels:
    :param threshold:
    :return:
    """
    numpy_features = features[0][0].detach().cpu().numpy() if not raw_features else features
    feature_mask = (numpy_features > threshold).astype(int)
    feature_table = extract_feature_table(feature_mask)
    if feature_table is not None:
        feature_table = {k: v.item() if hasattr(v, 'item') else (v[0] if type(v) == list else v) for k, v in
                         feature_table.items()}
        if ids is not None:
            feature_table["idx"] = ids.item()
        if targets is not None:
            feature_table["label"] = targets.item()
        if initial_labels is not None:
            feature_table["initial_label"] = initial_labels.item()
    return feature_table
