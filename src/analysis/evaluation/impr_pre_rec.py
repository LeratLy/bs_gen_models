# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Implementation from https://github.com/kynkaat/improved-precision-and-recall-metric
# Adapted for pytorch
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""k-NN precision and recall."""
import os
from logging import Logger
from time import time

import numpy as np
import torch

from src.analysis.evaluation.eval_metrics import batch_pairwise_distances
from src.utils.logger import setup_general_logger


# ----------------------------------------------------------------------------

class ManifoldEstimator:
    """Estimates the manifold of given feature vectors."""

    def __init__(self, features, row_batch_size=25000, col_batch_size=50000,
                 nhood_sizes=None, clamp_percentile=None, eps=1e-5, device='cpu'):
        """Estimate the manifold of given feature vectors.

            Args:
                features (np.array/torch.Tensor): Matrix of feature vectors to estimate their manifold.
                row_batch_size (int): Row batch size to compute pairwise distances
                    (parameter to trade-off between memory usage and performance).
                col_batch_size (int): Column batch size to compute pairwise distances.
                nhood_sizes (list): Number of neighbors used to estimate the manifold.
                clamp_to_percentile (float): Prune hyperspheres that have radius larger than
                    the given percentile.
                eps (float): Small number for numerical stability.
        """
        if nhood_sizes is None:
            nhood_sizes = [3]
        num_images = features.shape[0]
        self.device = device
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).to(device).float()
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        self.distances = torch.zeros((num_images, self.num_nhoods), dtype=torch.float32, device=device)
        for begin1 in range(0, num_images, row_batch_size):
            end1 = min(begin1 + row_batch_size, num_images)
            row_batch = features[begin1:end1]
            dists = []

            for begin2 in range(0, num_images, col_batch_size):
                end2 = min(begin2 + col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                dists.append(batch_pairwise_distances(row_batch, col_batch))

            # Find the k-nearest neighbor from the current batch.
            sorted_distances, _ = torch.cat(dists, dim=1).sort(dim=1)
            for idx, k in enumerate(nhood_sizes):
                self.distances[begin1:end1, idx] = sorted_distances[:, k-1]

        if clamp_percentile is not None:
            max_distances = torch.quantile(self.distances, clamp_percentile / 100.0, dim=0)
            self.distances = torch.where(self.distances > max_distances, torch.tensor(0.0, device=device), self.distances)

    def evaluate(self, eval_features, return_realism=False, return_neighbors=False):
        """Evaluate if new feature vectors are at the manifold."""
        if isinstance(eval_features, np.ndarray):
            eval_features = torch.from_numpy(eval_features).float()
        eval_features = eval_features.to(self.device)
        num_eval_images = eval_features.shape[0]
        num_ref_images = self._ref_features.shape[0]

        batch_predictions = torch.zeros((num_eval_images, self.num_nhoods), dtype=torch.float32, device=self.device)
        max_realism_score = torch.zeros((num_eval_images, ), dtype=torch.float32, device=self.device)
        nearest_indices = torch.zeros((num_eval_images, ), dtype=torch.int32, device=self.device)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]
            distances = []

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = self._ref_features[begin2:end2]
                distances.append(batch_pairwise_distances(feature_batch, ref_batch))

            all_distances = torch.cat(distances, dim=1)
            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then
            # the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            for k_idx, k in enumerate(self.nhood_sizes):
                radius_k = self.distances[:, k_idx].unsqueeze(0)
                batch_predictions[begin1:end1, k_idx] = (all_distances <= radius_k).any(dim=1).type(torch.float32)

            min_distances, nearest_idx = torch.min(all_distances + self.eps, dim=1)
            realism = self.distances[nearest_idx, 0] / min_distances
            max_realism_score[begin1:end1] = realism
            nearest_indices[begin1:end1] = nearest_idx

        if return_realism and return_neighbors:
            return batch_predictions, max_realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, max_realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices

        return batch_predictions


# ----------------------------------------------------------------------------

def knn_precision_recall_features(ref_features, eval_features, nhood_sizes=None,
                                  row_batch_size=10000, col_batch_size=50000, device='cpu'):
    """Calculates k-NN precision and recall for two sets of feature vectors.

        Args:
            ref_features (np.array/tf.Tensor): Feature vectors of reference images.
            eval_features (np.array/tf.Tensor): Feature vectors of generated images.
            nhood_sizes (list): Number of neighbors used to estimate the manifold.
            row_batch_size (int): Row batch size to compute pairwise distances
                (parameter to trade-off between memory usage and performance).
            col_batch_size (int): Column batch size to compute pairwise distances.
            device (str): Device to which the tensors are moved

        Returns:
            State (dict): Dict that contains precision and recall calculated from
            ref_features and eval_features.
    """
    if nhood_sizes is None:
        nhood_sizes = [3]
    state = dict()
    num_images = ref_features.shape[0]

    # Initialize DistanceBlock and ManifoldEstimators.
    ref_manifold = ManifoldEstimator(ref_features, row_batch_size, col_batch_size, nhood_sizes, device=device)
    eval_manifold = ManifoldEstimator(eval_features, row_batch_size, col_batch_size, nhood_sizes, device=device)

    # Evaluate precision and recall using k-nearest neighbors.
    print('Evaluating k-NN precision and recall with %i samples...' % num_images)
    start = time()

    # Precision: How many points from eval_features are in ref_features manifold.
    precision = ref_manifold.evaluate(eval_features.to(device))
    state['precision'] = torch.mean(precision, dim=0)

    # Recall: How many points from ref_features are in eval_features manifold.
    recall = eval_manifold.evaluate(ref_features.to(device))
    state['recall'] = torch.mean(recall, dim=0)

    print('Evaluated k-NN precision and recall in: %gs' % (time() - start))

    return state

# ----------------------------------------------------------------------------

def calc_impr_prec_rec(original_path: str, sampled_path: str, logger: Logger, device: str = "cpu"):
    """
    Calculate k-NN precision and recall
    :param original_path:
    :param sampled_path:
    :param logger:
    :return:
    """
    it_start = time()
    original_features = torch.load(original_path)['cond'].to(device)
    sampled_features = torch.load(sampled_path)['cond'].to(device)
    state = knn_precision_recall_features(original_features, sampled_features, nhood_sizes=[3], device=device)

    # Print progress.
    logger.info('Precision: %0.3f' % state['precision'][0])
    logger.info('Recall: %0.3f' % state['recall'][0])
    logger.info('Iteration time: %gs\n' % (time() - it_start))

def get_impr_prec_rec_npz(original_path: str, sampled_path: str, logger: Logger, device: str='cpu', map_to_labels: bool = False):
    """
    Calculate k-NN precision and recall per class
    :param map_to_labels:
    :param device:
    :param original_path:
    :param sampled_path:
    :param logger:
    :return:
    """
    def get_smaller_size(a: torch.tensor, b:torch.tensor):
        return a.shape[0] if a.shape[0] < b.shape[0] else b.shape[0]
    it_start = time()
    original_features = torch.load(original_path)['cond'].to(device)
    original_labels = torch.load(original_path)['labels'].to(device).squeeze(1)
    sampled_features = torch.load(sampled_path)['cond'].to(device)
    sampled_labels = torch.load(sampled_path)['labels'].to(device).squeeze(1)
    if map_to_labels:
        initial_original_labels = torch.load(original_path)['initial_labels'].to(device).squeeze(1)
        initial_sampled_labels = torch.load(sampled_path)['initial_labels'].to(device).squeeze(1)

    num_classes = original_labels.unique().numel()
    metrics_per_class = {}
    for label in range(num_classes):
        original_mask = (original_labels == label)
        _original_features = original_features[original_mask]
        # Complicated mode, where we draw n samples per hidden initial label class from the sampled data under the unhidden labels class
        if map_to_labels:
            _initial_original_labels = initial_original_labels[original_mask]
            samples = []
            # for each initial label draw corresponding initial sampled features (same amount or all)
            for class_label in _initial_original_labels.unique():
                original_mask = (initial_original_labels == class_label)
                _initial_original_features = original_features[original_mask]
                sampled_mask = (initial_sampled_labels == class_label)
                _initial_sampled_features = sampled_features[sampled_mask]
                num_samples = get_smaller_size(_initial_sampled_features, _initial_original_features)
                samples.append(_initial_sampled_features[:num_samples])
            _sampled_features = torch.cat(samples)
        # Simple mode where we can directly draw the first n samples for each class of the sampled data
        else:
            sampled_mask = (sampled_labels == label)
            _sampled_features = sampled_features[sampled_mask]
            num_samples = get_smaller_size(_sampled_features, _original_features)
            _sampled_features = sampled_features[:num_samples]

        state = knn_precision_recall_features(_original_features, _sampled_features, nhood_sizes=[3], device=device)
        metrics_per_class[label] = (state['precision'][0], state['recall'][0])

        # Print progress.
        logger.info(f"Results for class {label}:")
        logger.info('Precision: %0.3f' % state['precision'][0])
        logger.info('Recall: %0.3f' % state['recall'][0])
        logger.info('Iteration time: %gs\n' % (time() - it_start))

    return metrics_per_class


def calc_impr_prec_rec_multiple(original_folders: list[str], sampled_folders: list[str], base_folder: str = None,
                                latent_file_name="features.pkl", device: str = "cpu"):
    """
    Calculate improved precision and recall for the given folder pairs
    :param device:
    :param original_folders: folders containing original features, should have same length as sampled_folders (they are mapped one by one)
    :param sampled_folders: folders containing samples features, should have same length as sampled_folders (they are mapped one by one)
    :param base_folder: folder in which to search for folders
    :param latent_file_name: filename of the feature pkl
    :return:
    """
    logger = setup_general_logger("improved_precision_recall", os.getcwd(), time())
    for original, sampled in zip(original_folders, sampled_folders):
        print("Calculating improved precision recall for", original, "and", sampled)
        original_path = os.path.join(base_folder, original)
        sampled_path = os.path.join(base_folder, sampled)
        class_folders = [name for name in os.listdir(original_path) if os.path.isdir(os.path.join(original_path, name))]
        for folder in class_folders:
            print("Class:", folder)
            calc_impr_prec_rec(os.path.join(original_path, folder, latent_file_name),
                               os.path.join(sampled_path, folder, latent_file_name), logger=logger, device=device)
