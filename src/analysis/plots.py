import os.path
from collections import defaultdict

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.ndimage import center_of_mass, shift
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from variables import MS_TYPES_FILTERED, DATA_DIR, MS_TYPE_COLORS
from src._types import SaveTo
from src.utils.visualisation import render_plot, add_cloud_plot_to_axs


def create_confusion_matrix(
        target,
        prediction,
        labels=MS_TYPES_FILTERED,
        save_to: SaveTo = None,
        step: int = None,
        writer: SummaryWriter = None,
        title: str = "Confusion Matrix",
):
    label_num = len(set(val for val in labels.values()))
    conf_mat = confusion_matrix(target, prediction)
    ticks = defaultdict(list)
    for key, val in sorted(labels.items()):
        ticks[val].append(key)
    fig = plt.figure(figsize=(label_num, label_num))
    ax = sns.heatmap(conf_mat, vmin=-1, cmap='coolwarm', annot=True)
    ax.set_yticklabels(ticks)
    ax.set_xticklabels(ticks)

    render_plot(fig, title=title, save_to=save_to, step=step, writer=writer)
    return conf_mat


def center_array(array, threshold=0.5) -> np.ndarray:
    binary = (array > threshold).astype(np.uint8)
    com = center_of_mass(binary)
    center = np.array(array.shape) / 2
    shift_vector = center - com
    centered = shift(array, shift=shift_vector, order=0, mode='constant', cval=0)
    return centered


def shape_difference_plot(array1: np.array, array2: np.array, threshold: float = 0.5, center_arrays: bool = False):
    """
    Plot shape differences (pixels that are added from array1 to array2 in green and removed pixels in red)
    """
    fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='3d'))
    _array1 = center_array(array1) if center_arrays else array1
    _array2 = center_array(array2) if center_arrays else array2
    a1 = (_array1 > threshold).astype(np.uint8)
    a2 = (_array2 > threshold).astype(np.uint8)
    added = (a2 & ~a1)
    removed = (a1 & ~a2)
    # add_cloud_plot_to_axs(array1, ax=ax, threshold=0, alpha=0.1, color="ref")
    add_cloud_plot_to_axs(added, ax=ax[0], threshold=0, alpha=0.1, color="green", edgecolors=None,
                          ax_limits=3 * [(20, 80)])
    add_cloud_plot_to_axs(removed, ax=ax[1], threshold=0, alpha=0.1, color="red", edgecolors=None,
                          ax_limits=3 * [(20, 80)])


# if __name__ == '__main__':
#     file_name = "prototypes_3D.npz"
#     file_path = os.path.join(DATA_DIR, "analysis_data", "original_samples_train")
#     npz_data = np.load(os.path.join(file_path, file_name))
#     labels = npz_data["labels"]
#     arrays = npz_data["images"]
#
#     arrays_0 = arrays[0][0]
#     array_1 = arrays[1][0]
#     shape_difference_plot(array_1, arrays_0, list(MS_TYPE_COLORS.values()))
#     plt.savefig(os.path.join(file_path, f"shape_difference.svg"))
