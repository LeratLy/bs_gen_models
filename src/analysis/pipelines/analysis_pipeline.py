import dataclasses
import os
import time
from enum import Enum

import numpy as np
import torch
from matplotlib import cm, pyplot as plt
from tqdm import tqdm

from src._types import SaveTo
from src.analysis.latents import create_prototype_latents, run_reducer, create_cluster_means, create_prototypes
from src.analysis.pipelines.pipeline import Pipeline, SavedModel
from src.analysis.shape_metrics import geometrics_for_folder
from src.analysis.utils import load_np_latents
from src.utils.visualisation import plot_3d_data_cloud, add_cloud_plot_to_axs
from variables import MS_TYPE_DETAILS, \
    MS_MAIN_TYPE_DETAILS, PLOT_DIR, LABEL_TO_MAIN_NAME, MS_TYPE_MAIN_COLORS, MS_TYPE_MAIN_EDGECOLORS


class AnalysisPipelineSteps(Enum):
    """
    Pipeline that can be used to evaluate the models performance
    """
    proto = "prototype_generation"
    clusters = "clustering"
    cluster_prototypes = "clustering_prototypes"
    original_geometrics = "original_geometrics"
    model_geometrics = "model_geometrics"


@dataclasses.dataclass
class ClusterConfig:
    mode: str = "tsne"
    config: dict = None


class AnalysisPipeline(Pipeline):
    supported_steps = [AnalysisPipelineSteps.proto, AnalysisPipelineSteps.clusters,
                       AnalysisPipelineSteps.cluster_prototypes, AnalysisPipelineSteps.model_geometrics,
                       AnalysisPipelineSteps.original_geometrics]
    cluster_config: ClusterConfig = ClusterConfig()
    initial_class_details = MS_TYPE_DETAILS
    main_class_details = MS_MAIN_TYPE_DETAILS

    def __init__(self, original_folders: list[str] = None, **kwargs):
        super().__init__(**kwargs)
        if original_folders is not None:
            self.original_folders = original_folders

    def run(self, saved_model: SavedModel = None):
        start_time_run = time.time()
        if saved_model is not None:
            self.update_model(saved_model)
        original_folders = [os.path.join(self.base_path, folder) for folder in self.original_folders]
        if self.saved_model is not None:
            samples_path = os.path.join(self.base_path, f"samples_{self.saved_model.model_name.value}")

        if AnalysisPipelineSteps.proto in self.supported_steps:
            self.logger.info(f"Starting to create prototypes...")
            self.create_prototypes(samples_path)
            self.create_prototypes(samples_path, True)

        if AnalysisPipelineSteps.clusters in self.supported_steps:
            self.logger.info(f"Starting to create clusters...")
            self.create_clusters(samples_path, original_folders)

        if AnalysisPipelineSteps.original_geometrics in self.supported_steps:
            self.logger.info(f"Starting to compute original geometrics...")
            geometrics_for_folder(original_folders, save_path=self.base_path)

        if AnalysisPipelineSteps.model_geometrics in self.supported_steps:
            self.logger.info(f"Starting to compute model geomterics...")
            geometrics_for_folder(samples_path)

        self.logger.info(
            f"Completed analysis timeline (duration: {time.time() - start_time_run})!\nAll results are saved to the corresponding folders.")

    def create_prototypes(self, path: str, use_initial_labels: bool = False):
        """
        Create prototypes using all original samples based on their latent code
        :param path: base path where prototypes are saved
        :param use_initial_labels: whether to use the original label to create prototypes or not
        :return:
        """
        if not os.path.exists(path):
            os.makedirs(path)
        prototype_latents, prototype_model_labels = create_prototype_latents(
            path,
            latent_file_name=self.latent_file_name,
            file_name=self.prototypes_file_name,
            use_initial_labels=use_initial_labels,
            alternative_paths=[os.path.join(self.base_path, folder) for folder in self.original_folders]
        )
        labels = torch.arange(0, prototype_latents.shape[0])
        if self.saved_model.model_name.is_bdae():
            prototype_tensor = self.model._render(cond=prototype_latents.to(device=self.device), ema=True)
        else:
            prototype_tensor = self.model.render(prototype_latents,
                                                 target=prototype_model_labels.to(device=self.device))

        np.savez(os.path.join(path,
                              self.prototypes_initial_3D_file_name if use_initial_labels else self.prototypes_3D_file_name),
                 images=prototype_tensor.detach().cpu().numpy(),
                 labels=labels.detach().numpy()
                 )
        for i in range(prototype_tensor.shape[0]):
            plot_3d_data_cloud(prototype_tensor[i][0],
                               title=f'Prototype for {self.initial_class_details["names"][i] if use_initial_labels else self.main_class_details["names"][i]}',
                               save_to=SaveTo.svg,
                               path=path,
                               color=self.initial_class_details["colors"][i] if use_initial_labels else
                               self.main_class_details["colors"][i],
                               edgecolor=self.initial_class_details["edgecolors"][i] if use_initial_labels else
                               self.main_class_details["edgecolors"][i],
                               ax_limits=[(20, 80), (20, 80), (20, 80)]
                               )

    def create_clusters(self, save_path: str, original_folders: list[str], num_clusters=8):
        """
        Create clusters based on original samples and their latent space
        :return:
        """
        embeddings, labels, initial_labels = load_np_latents(
            save_path,
            file_name=self.latent_file_name,
            alternative_paths=[os.path.join(self.base_path, folder) for folder in self.original_folders]
        )
        kwargs = self.cluster_config.config if self.cluster_config.config is not None else {}
        reducer_result = run_reducer(embeddings, labels, save_path,
                                     save_name=f"Clusters ({self.cluster_config.mode})",
                                     colors=self.main_class_details["colors"],
                                     classes=self.main_class_details["names"],
                                     initial_colors=self.initial_class_details["colors"],
                                     initial_classes=self.initial_class_details["names"],
                                     markers=self.main_class_details["markers"],
                                     initial_markers=self.initial_class_details["markers"],
                                     mode=self.cluster_config.mode, initial_labels=initial_labels, kwargs=kwargs
                                     )

        if AnalysisPipelineSteps.cluster_prototypes in self.supported_steps:
            cluster_means, cluster_labels = create_cluster_means(embeddings, reducer_result, n_clusters=num_clusters)
            cluster_means = torch.from_numpy(cluster_means).to(self.device)
            labels = torch.arange(0, cluster_means.shape[0])
            if self.saved_model.model_name.is_bdae():
                prototype_tensor = self.model._render(cond=cluster_means, ema=True)
            else:
                raise NotImplementedError("Decoding prototypes without class label is not possible for CVAE or your model")
                # prototype_tensor = self.model.render(cluster_means, target=labels.to(device=self.device))

            numpy_prototypes = prototype_tensor.detach().cpu().numpy()
            np.savez(os.path.join(save_path, "cluster_prototypes.npz"),
                     images=numpy_prototypes,
                     labels=labels.detach().numpy()
                     )
            np.savez(os.path.join(save_path, "clustered_labels.npz"),
                     cluster_labels=cluster_labels,
                     )

            cmap = cm.viridis
            colors = [cmap(i / (num_clusters - 1)) for i in range(num_clusters)]
            print(numpy_prototypes.shape[0])
            dark_colors = [tuple([max(0, x * 0.5) for x in color[:3]]) + (color[3],)
                           for color in colors]

            variance_data = self.get_variance_data(original_folders, cluster_labels)
            self.plot_prototypes(numpy_prototypes, "Cluster Prototypes", names=list(map(str, range(8))),
                                 colors=colors, edgecolors=dark_colors, save_to=".png", individually=True,
                                 overlay_array=variance_data)
            self.plot_prototypes(numpy_prototypes, "Cluster Prototypes", names=list(map(str, range(8))),
                                 colors=colors, edgecolors=dark_colors, save_to=".svg", individually=True,
                                 overlay_array=variance_data)

    def plot_prototypes(self, prototype_array, title, colors=MS_TYPE_MAIN_COLORS, edgecolors=MS_TYPE_MAIN_EDGECOLORS,
                        names=LABEL_TO_MAIN_NAME, save_to: str = None, individually: bool = False,
                        overlay_array: np.ndarray = None):
        cols = prototype_array.shape[0] if not individually else 1
        rows = 1
        fig = plt.figure(figsize=(cols * 6, rows * 6))

        for i in range(prototype_array.shape[0]):
            if individually:
                fig = plt.figure(figsize=(cols * 6, rows * 6))
                ax = fig.add_subplot(1, 1, 1, projection="3d")
            else:
                ax = fig.add_subplot(rows, cols, i + 1, projection="3d")

            if overlay_array is not None:
                add_cloud_plot_to_axs(
                    overlay_array[i],
                    threshold=0.0,
                    ax=ax,
                    ax_limits=[(0, 100), (0, 100), (0, 100)],
                    color=colors[i],
                    alpha_mult=0.03,
                    edgecolors=edgecolors[i],
                )

            add_cloud_plot_to_axs(
                prototype_array[i][0],
                threshold=0.5,
                ax=ax,
                ax_limits=[(0, 100), (0, 100), (0, 100)],
                color=colors[i],
                edgecolors=edgecolors[i],
            )

            ax.set_facecolor('white')
            if individually:
                if save_to is not None:
                    plt.savefig(os.path.join(PLOT_DIR, names[i] + "_" + title + save_to))
                plt.close(fig)

        if not individually:
            if save_to is not None:
                plt.savefig(os.path.join(PLOT_DIR, title + save_to))

    def get_variance_data(self, original_folders, prototype_labels):
        means_samples = np.zeros((8, 96, 96, 96))
        num_samples = np.zeros(8)

        for folder in original_folders:
            print("Retrieving data from", folder)
            raw_data = np.load(os.path.join(folder, "samples.npz"))
            for i, image in tqdm(enumerate(raw_data["images"])):
                means_samples[prototype_labels[i]] += image[0]
                num_samples[prototype_labels[i]] += 1

        means_samples /= np.expand_dims(num_samples, axis=(1, 2, 3))
        return means_samples
