import os

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src._types import SaveTo
from src.utils.helpers import get_filenames_by
from src.utils.visualisation import plot_3d_data_cloud
from variables import DATA_DIR


def plot_all_of_folder(folder_name):
    """
    Plot all images in a given folder (folder needs to be located in DATA_PATH directory)
    """
    full_data_path = os.path.join(DATA_DIR, folder_name)
    filenames = get_filenames_by(r".*\.nii.gz", full_data_path)
    for file_name in tqdm(filenames):
        img_nii = nib.load(os.path.join(full_data_path, file_name))
        img = img_nii.get_fdata()
        plot_3d_data_cloud(img, title=file_name, save_to=SaveTo.png)


def plot_reduced_clusters(tsne_result, labels: np.array, base_path: str = None, file_name: str = "tsne.svg",
                          colors: list[str] = None, classes: list[str] = None, markers: list[str] = None):
    """
    Plot results of t-SNE and save them to a svg
    :param markers:
    :param colors:
    :param classes:
    :param file_name: resulting svg file name
    :param tsne_result: results that should be plotted
    :param labels: labels per data point (name)
    :param base_path: where to save "tsne.svg" to
    :return:
    """
    print("Creating plot and saving figure...")
    plt.figure(figsize=(6, 4), facecolor="white")
    classes = classes if classes is not None else sorted(np.arange(len(np.unique(labels))))

    if colors is None:
        cmap = plt.get_cmap("Spectral", len(classes))
    for i, class_name in enumerate(classes):
        idx = labels == i
        zorder = 1 if i == 3 else 2
        color = colors[i] if colors is not None else cmap[i]
        plt.scatter(
            tsne_result[idx, 0], tsne_result[idx, 1],
            color=color,
            marker=markers[i] if markers is not None else 'o',
            label=class_name,
            alpha=0.5,
            s=40,
            zorder=zorder,
        )
    plt.legend(title="Classes")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.title('UMAP Projection of Latent Embeddings')
    save_path = os.path.join(base_path, file_name) if base_path is not None else file_name
    plt.savefig(save_path)
    print(f"Saved figure to {save_path}")
