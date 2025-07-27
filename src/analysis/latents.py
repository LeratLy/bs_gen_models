import os
import time
from typing import Union

import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from src.analysis.utils import load_latents, load_np_latents, load_latents_from_multiple_folders
from src.analysis.visualizations import plot_reduced_clusters
from src.models.dae.dae import DAE
from src.models.vae.vae import VAE
from variables import PLOT_DIR


def infer_latents(model: Union[VAE, DAE], data_path: str, file_name: str = "features",
                  alternative_folders: list[str] = None, use_initial_labels: bool = False):
    """
    Infer latents for data in a folder with given model
    :param use_initial_labels:
    :param alternative_folders: that should be used for inference instead of all folders in the base path
    :param model: used for inferring latents
    :param data_path: folder in which samples are saved (containing subfolders with named accordingly to the class_id)
    :param file_name: name of the file to which the latents should be saved
    :return:
    """
    if alternative_folders is None:
        class_folders = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]
    else:
        class_folders = alternative_folders
    for folder in tqdm(class_folders):
        cond, labels, initial_labels, file_names = model.infer_whole_folder(os.path.join(data_path, folder),
                                                                            use_initial_labels=use_initial_labels)
        final_dict = {"cond": cond, "labels": labels,
                      "initial_labels": initial_labels if len(initial_labels) > 0 else labels}
        if len(file_names) > 0:
            final_dict["file_names"] = file_names
        torch.save(final_dict, os.path.join(data_path, folder, file_name))


def infer_latents_multiple(model: Union[VAE, DAE], data_folders: list[str], base_folder: str,
                           file_name: str = "features.pkl"):
    """
    Infer latents for multiple data folders with given model
    :param base_folder: full path of folder in which to search for data_folders
    :param model: used for inferring latents
    :param data_folders: folders in which samples are saved (containing subfolders with named accordingly to the class_id)
    :param file_name: name of the file to which the latents should be saved
    :return:
    """
    for folder in data_folders:
        data_path = os.path.join(base_folder, folder)
        infer_latents(model, data_path, file_name=file_name)
        print("Computed features for", folder)


def reduced_clustering(embeddings: np.array, kwargs: dict = None, mode: str = "tsne"):
    """
    Performs t-SNE for given embeddings
    :param mode: mode for reduction and clustering either 'tsne' or 'umap'
    :type mode: str, default 'tsne'
    :param embeddings: that should be clustered
    :type embeddings: np.array
    :param kwargs: arguments passed to t-SNE
    :type kwargs: dict
    :return:
    """
    print(f"Starting {mode}...")
    if kwargs is None:
        kwargs = {}
    time_start = time.time()
    if mode == 'tsne':
        reducer = TSNE(**kwargs)
    elif mode == 'umap':
        reducer = umap.UMAP(**kwargs)
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")
    reducer_result = reducer.fit_transform(embeddings)
    print(f'Reduction with {mode} done! Time elapsed: {time.time() - time_start} seconds')
    return reducer_result


def run_reducer(embeddings: np.array, labels: np.array, path: str, save_name: str = None, mode: str = "tsne",
                initial_labels: np.array = None, kwargs: dict = None, initial_colors: list[str] = None,
                colors: list[str] = None, classes: list[str] = None, initial_classes: list[str] = None,
                markers: list[str] = None, initial_markers: list[str] = None, ):
    """
    Runs dimensionality reduction and clustering on data
    :param markers:
    :param initial_markers:
    :param initial_classes:
    :param colors:
    :param classes:
    :param initial_labels: original labels of the dataset, if given an additional plot is generated, marking the corresponding data points accordingly
    :param embeddings: feature vectors that should be clustered
    :param labels: corresponding to embeddings
    :param path: where to save file
    :param save_name: base name of the resulting .svg
    :param mode: 'tsne' or 'umap'
    :param kwargs: arguments passed to the constructors of tsne or umap
    :return:
    """
    if kwargs is None:
        kwargs = {}
    result = reduced_clustering(embeddings, kwargs=kwargs, mode=mode)
    plot_reduced_clusters(result, labels, classes=classes, colors=colors, markers=markers,
                          file_name=os.path.join(path, f"{save_name if save_name is not None else mode}.svg"))
    if initial_labels is not None:
        plot_reduced_clusters(result, initial_labels, classes=initial_classes, colors=initial_colors,
                              markers=initial_markers,
                              file_name=os.path.join(path,
                                                     f"{save_name if save_name is not None else mode}_initial_labels.svg"))
    return result


def create_cluster_means(embeddings, reducer_result, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(reducer_result)

    plt.figure(figsize=(6, 4), facecolor="white")
    scatter = plt.scatter(reducer_result[:, 0], reducer_result[:, 1], c=cluster_labels, cmap='viridis', s=40, alpha=0.5)
    plt.title('UMAP Projection with KMeans Clusters')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    colors = [scatter.cmap(scatter.norm(i)) for i in range(kmeans.n_clusters)]
    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f"{i}")
                      for i, color in enumerate(colors)]
    plt.legend(handles=legend_handles, title='Clusters')
    plt.savefig(os.path.join(PLOT_DIR, "kmeans_of_umap.png"))
    plt.savefig(os.path.join(PLOT_DIR, "kmeans_of_umap.svg"))

    cluster_data = []
    for cluster_id in range(kmeans.n_clusters):
        cluster_data_points = embeddings[cluster_labels == cluster_id]
        cluster_data.append(cluster_data_points.mean(axis=0))
    return np.array(cluster_data), cluster_labels

def run_tsne(path: str, perplexities: list[str]):
    """
    Run tsne for different perplexities
    :param path: where to load latents from and save corresponding plots
    :param perplexities: values which should be used for perplexity
    :return:
    """
    embeddings, labels, _ = load_np_latents(path)
    for i in perplexities:
        run_reducer(embeddings, labels, path, f"tsne_{i}", mode="tsne", kwargs={"perplexity": int(i)})


def k_nn_clustering(embeddings, kwargs: dict = None):
    if kwargs is None:
        kwargs = {}
    nbrs = NearestNeighbors(**kwargs).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)


def create_prototypes(embeddings, labels, num_classes: int = 2):
    """
    Create prototypes for given embeddings (one per class)
    :param embeddings: to create prototypes for
    :param labels: corresponding to embeddings, same order as embeddings
    :param num_classes: total number of classes to create prototypes for
    :return:
    """
    sums = torch.zeros((num_classes, embeddings.shape[1]), dtype=embeddings.dtype, device=embeddings.device)
    sums.index_add_(0, labels, embeddings)
    counts = torch.bincount(labels, minlength=num_classes).clamp(min=1).unsqueeze(-1)
    for _ in range(len(embeddings.shape) - 2):
        counts = counts.unsqueeze(-1)

    prototypes = sums / counts  # per-class mean

    return prototypes


def create_prototype_latents(base_path: str, file_name: str = "prototypes.pkl",
                             latent_file_name: str = "features.pkl", use_initial_labels: bool = False,
                             alternative_paths: list[str] = None):
    """
    Creating and saving prototypes latents.
    :param alternative_paths: for the case that alternative_paths are given, base_path is only used for saving the prototypes and alternative paths are used to load one or multiple latents
    :param latent_file_name: file name of the latent pkl
    :param file_name: file name where prototypes are saved
    :param base_path: folder containing latents and where prototypes will be saved
    :param use_initial_labels: whether to use main labels or original ones for prototype generation (must be equal to num_classes)
    :return:
    """
    if alternative_paths is not None:
        embeddings, labels, initial_labels = load_latents_from_multiple_folders(alternative_paths,
                                                                                file_name=latent_file_name)
    else:
        embeddings, labels, initial_labels = load_latents(base_path, file_name=latent_file_name)
    _labels = initial_labels if use_initial_labels else labels
    num_classes = _labels.unique().shape[0]
    prototypes = create_prototypes(embeddings, _labels, num_classes)
    torch.save(prototypes, os.path.join(base_path, file_name))

    # The model can later only deal with the labels it was trained with (not initial_labels)
    if use_initial_labels:
        main_labels = []
        for initial_label in sorted(torch.unique(initial_labels)):
            corresponding_label = labels[initial_labels == initial_label]
            main_labels.append(torch.unique(corresponding_label)[0])  # should be only one
    else:
        main_labels = torch.arange(0, num_classes)

    return prototypes, torch.tensor(main_labels)
