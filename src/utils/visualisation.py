import os.path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
import matplotlib
from matplotlib import colors
from torch.utils.tensorboard import SummaryWriter

from src._types import SaveTo
from variables import PLOT_DIR, MS_TYPE_COLORS

matplotlib.use('Agg')


def plot_3d_data_interactive(values: np.array, auto_close: bool = False):
    """
    Visualize 3D data in a separate interactive window
    :param s: how long plot should be shown, if None do not close automatically
    :param values: numpy array with 3D data of shape (D,H,W)
    Note!:The interactive plot is going to be opened in a new window, when sing on server, enable x11 forwarding
    """
    # Set up plot
    grid = pv.ImageData()
    grid.dimensions = values.shape
    grid.point_data["values"] = values.flatten(order="F")

    p = pv.Plotter()
    p.add_volume(grid, cmap=colors.ListedColormap(['white', 'teal']))

    # Show plot
    p.show(auto_close=auto_close)


def plot_3d_cloud_norm(values: Union[np.ndarray, torch.tensor],
                       title: str = None,
                       save_to: SaveTo = None,
                       x_label: str = None,
                       y_label: str = None,
                       z_label: str = None,
                       writer: SummaryWriter = None,
                       step: int = None,
                       threshold: float = 0.1):
    """
    Plot 3d dara to point clouds (scatter plot)
    """
    if torch.is_tensor(values):
        values = values.to('cpu')
    norm_diff = (values - values.min()) / (values.max() - values.min() + 1e-8)
    plot_3d_base(norm_diff, title, save_to, x_label, y_label, z_label, writer, step, threshold)


def plot_3d_data_cloud(values: Union[np.ndarray, torch.tensor],
                       title: str = "Point cloud",
                       save_to: SaveTo = SaveTo.png,
                       x_label: str = "coronal axis",
                       y_label: str = "sagittal axis",
                       z_label: str = "vertical axis",
                       writer: SummaryWriter = None,
                       step: int = None,
                       ax_limits = None,
                       threshold: float = 0.5,
                       path: str = None,
                       color: str = "cornflowerblue",
                       edgecolor: str = "royalblue"):
    """
    Plot 3d dara to point clouds (scatter plot)
    """
    if torch.is_tensor(values):
        values = values.to('cpu')
    values = np.asarray(values).clip(0, 1)
    plot_3d_base(values, title, save_to, x_label, y_label, z_label, writer, step, threshold, ax_limits, path, color, edgecolor)


def plot_3d_subplots_base(values, title, save_to: SaveTo, x_label, y_label, z_label, writer, step, threshold):
    """
    Plot 3d cloud plots (with thresholding) to a subplot (column width is at least 2)
    """
    plain_img_data = len(values.shape) == 3
    n_samples = 1 if plain_img_data else values.shape[0]
    n_cols = min(2, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, subplot_kw=dict(projection='3d'))
    for i in range(n_samples):
        row_idx = i // n_cols
        column_idx = i % n_cols
        current_values = values if plain_img_data else values[i][0]
        if n_rows - 1 > 0:
            ax = axs[row_idx][column_idx]
        elif n_cols - 1 > 0:
            ax = axs[column_idx]
        else:
            ax = axs
        add_cloud_plot_to_axs(current_values, threshold=threshold, ax=ax, x_label=x_label,
                              y_label=y_label, z_label=z_label)

    if title is not None:
        fig.suptitle(title)

    render_plot(fig, f'3D Pointcloud {title}', save_to, step, writer)


def plot_3d_base(values, title, save_to: SaveTo, x_label, y_label, z_label, writer, step, threshold, ax_limits=None,
                 path: str = None, color: str = "cornflowerblue", edgecolor: str = "royalblue"):
    plain_img_data = len(values.shape) == 3
    n_samples = 1 if plain_img_data else values.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if title is not None:
        fig.suptitle(title)
    for i in range(n_samples):
        current_values = values if plain_img_data else values[i][0]
        add_cloud_plot_to_axs(current_values, threshold=threshold, ax=ax, x_label=x_label, y_label=y_label,
                              z_label=z_label, ax_limits=ax_limits, color=color, edgecolors=edgecolor)

        render_plot(fig, f'3D Pointcloud {title}_{i}', save_to, step, writer, path)


def add_cloud_plot_to_axs(values: Union[np.ndarray, torch.tensor], threshold, ax, x_label="coronal axis", y_label="vertical axis", z_label="sagittal axis",
                          ax_limits=None, edgecolors="royalblue", color="cornflowerblue", alpha: float = None,
                          cmap: str = None, colorbar_title: str = None, alpha_mult: float = 1.0):
    x, y, z = np.where(values > threshold)
    c = values[values > threshold]
    if len(c) > 0:
        if ax_limits is not None:
            ax.set_xlim(ax_limits[0] if len(ax_limits) > 0 else None)
            ax.set_ylim(ax_limits[1] if len(ax_limits) > 1 else None)
            ax.set_zlim(ax_limits[2] if len(ax_limits) > 2 else None)

        kwargs = {
            "c": c if cmap is not None else color,
            "cmap": cmap,
            "alpha": alpha if alpha is not None else c*alpha_mult,
        }

        scatter = ax.scatter(x, y, z, marker='s', edgecolors=edgecolors, axlim_clip=True, depthshade=True, **kwargs)

        if colorbar_title is not None:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(colorbar_title)

        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if z_label is not None:
            ax.set_zlabel(z_label)


def plot_3d_data_cloud_ch_p(values: Union[np.ndarray, torch.tensor], step: int, writer: SummaryWriter,
                            title=f"Segmented Choroid Plexus"):
    """
    Calls point cloud plotter with standard config
    """
    assert None not in [values, step,
                        writer], "Values, id and summary writer must be given for using this plotting mode!"
    plot_3d_data_cloud(values, title, save_to=SaveTo.tensorboard, writer=writer, step=step)


def render_plot(fig, title=None, save_to: SaveTo = None, step: int = None, writer: SummaryWriter = None,
                path: str = None):
    if save_to is not None:
        if save_to in [SaveTo.svg, SaveTo.png]:
            if path is None:
                plt.savefig(os.path.join(PLOT_DIR, f"{title}.{save_to.value}"))
            else:
                plt.savefig(os.path.join(path, f"{title}.{save_to.value}"))
        elif save_to == SaveTo.tensorboard:
            assert writer is not None, "Writer is not allowed to be None when using tensorboard"
            writer.add_figure(title, fig, global_step=step)
            writer.flush()
        elif save_to == SaveTo.show:
            plt.show()

    if save_to != SaveTo.none:
        plt.close(fig)

def ms_subtypes_visualization():
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 11,
        'axes.labelsize': 11,
        # 'xtick.labelsize': 11,
        # 'ytick.labelsize': 11,
        'legend.fontsize': 11
    })
    plt.figure(figsize=(10, 6))

    # SPMS
    time = [0, 1, 2, 3, 4, 5]
    disease = [0, 1, 0.5, 1.5, 1.0, 2.0]
    plt.step(time, disease, where='post', label='SPMS', linewidth=2.5, color=MS_TYPE_COLORS[4], alpha=0.75)
    time = [5, 6, 7, 8, 9, 10]
    disease = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    plt.plot(time, disease, color=MS_TYPE_COLORS[4], linewidth=2.5, alpha=0.75)

    # RRMS
    time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    disease = [0, 1, 0.5, 1.5, 1.0, 2.0, 1.8, 2.5, 2.2, 3.0, 3.0]
    plt.step(time, disease, where='post', label='RRMS', linewidth=2.5, color=MS_TYPE_COLORS[3], alpha=0.75)

    # PPMS
    time = [0, 10]
    disease = [0, 4.2]
    plt.plot(time, disease, label='PPMS', linewidth=2.5, color=MS_TYPE_COLORS[2], alpha=0.75)

    # CIS
    time = [0, 0.5, 1, 10]
    disease = [0, 1, 0, 0]
    plt.plot(time, disease, label='CIS', linewidth=2.5, color=MS_TYPE_COLORS[1], alpha=0.75)

    plt.xlabel("Time")
    plt.ylabel("Clinical Disability")
    plt.title("Multiple Sclerosis Types")
    plt.legend()

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(os.path.join(PLOT_DIR, "ms_subtypes.svg"))

# if __name__ == "__main__":
#     ms_subtypes_visualization()