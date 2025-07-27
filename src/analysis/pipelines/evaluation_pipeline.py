import os.path
import time
from collections import defaultdict
from enum import Enum

import pandas as pd
import torch
from torcheval.metrics.functional import multiclass_f1_score
from tqdm import tqdm

from src._types import SaveTo
from src.analysis.evaluation.eval_metrics import pairwise_inv_dice_coef
from src.analysis.evaluation.impr_pre_rec import get_impr_prec_rec_npz
from src.analysis.pipelines.pipeline import Pipeline, SavedModel
from src.analysis.plots import create_confusion_matrix
from src.analysis.shape_metrics import compute_metrics_for_sample, save_geometric_data_to_file
from src.analysis.utils import npz_dataloader, load_clf_model
from src.metrics import bce_loss, harmonic_mean
from src.utils.visualisation import plot_3d_data_cloud
from variables import MS_MAIN_TYPE, MS_TYPES_TO_MAIN_TYPE


class EvaluationPipelineSteps(Enum):
    """
    Pipeline that can be used to evaluate the models performance
    """
    rec_loss = "reconstruction_loss"
    clf_score = "clf_performance"
    prec_rec = "improved_precision_recall"
    div_score = "diversity_score"

class EvaluationPipeline(Pipeline):
    supported_steps = [EvaluationPipelineSteps.rec_loss, EvaluationPipelineSteps.clf_score, EvaluationPipelineSteps.prec_rec, EvaluationPipelineSteps.div_score]
    clf_model = None
    saved_clf_model = None
    batch_size = 8
    evaluate_on = ["original_samples_test"]
    classes = MS_MAIN_TYPE
    save_reconstruction_images = False

    def __init__(self, original_folders: list[str] = None, saved_clf: SavedModel = None, **kwargs):
        super().__init__(**kwargs)
        if original_folders is not None:
            self.original_folders = original_folders
        if saved_clf is not None:
            self.setup_clf_model(saved_clf)

    def run(self, saved_model: SavedModel = None):
        start_time_run = time.time()
        if saved_model is not None:
            self.update_model(saved_model)
        samples_path = os.path.join(self.base_path, f"samples_{self.saved_model.model_name.value}")

        metrics = {
            "model_name": [self.saved_model.model_name.value],
            "checkpoint_name": [self.saved_model.checkpoint_path]
        }

        if EvaluationPipelineSteps.rec_loss in self.supported_steps:
            self.logger.info(f"Starting to compute reconstruction loss...")
            for folder in self.original_folders:
                if folder in self.evaluate_on:
                    loss = self.compute_reconstruction_loss(os.path.join(self.base_path, folder), samples_path)
                    metrics[f"reconstruction_loss_{folder}"] = [loss]
                    self.logger.info(f"Reconstruction loss {folder}: {loss}")

        if EvaluationPipelineSteps.clf_score in self.supported_steps:
            assert self.clf_model is not None, "Clf must be specified to evaluate clf performance"
            self.logger.info(f"Evaluating clf performance on generated samples...")
            score = self.compute_classification_score(samples_path)
            self.logger.info(f"Classification score: {score.item()}")
            metrics[f"classification_score"] = [score.item()]

        if EvaluationPipelineSteps.prec_rec in self.supported_steps:
            self.logger.info(f"Computing improved precision and recall between samples and dataset...")
            for folder in self.original_folders:
                if folder in self.evaluate_on:
                    self.logger.info(f"Computing for {folder}...")
                    metrics_pre_class = get_impr_prec_rec_npz(
                        os.path.join(self.base_path, folder, self.latent_file_name),
                        os.path.join(samples_path, self.latent_file_name),
                        logger=self.logger,
                        device=self.device,
                        map_to_labels=self.use_hidden_initial_labels,
                    )
                    for k,v in metrics_pre_class.items():
                        metrics[f"improved_precision_{folder}_class_{k}"] = [v[0].item()]
                        metrics[f"improved_recall_{folder}_class_{k}"] = [v[1].item()]
                        metrics[f"improved_recall_{folder}_class_{k}_f1"] = [harmonic_mean(v[0].item(), v[1].item())]

        if EvaluationPipelineSteps.div_score in self.supported_steps:
            self.logger.info(f"Computing mean diversity score as inverse eman dice...")
            file_path = os.path.join(samples_path, self.samples_file_name)
            dataloader = npz_dataloader(file_path, self.batch_size)
            diversity_score = pairwise_inv_dice_coef(dataloader, device=self.device)
            self.logger.info(f"Mean diversity score: {diversity_score}")
            metrics[f"diversity"] = [diversity_score.item()]

        self.save_metrics(metrics)
        self.logger.info(f"Completed evaluation timeline (duration: {time.time()-start_time_run})!")

    def save_metrics(self, metrics: dict):
        """
        Save metrics to df, if file already exists add as column
        :param metrics: 
        :return: 
        """
        file_path = os.path.join(self.base_path, "evaluation_metrics.csv")
        try:
            df = pd.read_csv(file_path)
            old_entry = df[df['checkpoint_name'] == metrics["checkpoint_name"][0]]
            if len(old_entry) > 0:
                old_df = df.set_index(['checkpoint_name'])
                new_metrics = pd.DataFrame(metrics).set_index(['checkpoint_name'])
                old_df.update(new_metrics)
                df = old_df.reset_index()
            else:
                df = pd.concat([df, pd.DataFrame(metrics)], ignore_index=True)
        except FileNotFoundError:
            df = pd.DataFrame(metrics)

        df.to_csv(file_path, index=False)

    def setup_clf_model(self, saved_model: SavedModel):
        """
        Update clf model which is used for pipeline
        :param saved_model: config for loading model
        :return:
        """
        self.clf_model = load_clf_model(saved_model.model_name, saved_model.checkpoint_path, self.device)
        self.saved_clf_model = saved_model

    def compute_reconstruction_loss(self, data_path: str, samples_path: str):
        """
        Compute the reconstruction loss in form of bce loss for all items in dataloader
        :return:
        """
        assert os.path.isfile(os.path.join(data_path,self.samples_file_name)), f"{self.samples_file_name} not found in path {data_path}"
        file_path = os.path.join(data_path, self.samples_file_name)
        dataloader = npz_dataloader(file_path, 1)
        running_loss = 0.0
        target_ids = defaultdict(int)
        rows = []
        for (features, targets, ids) in tqdm(dataloader):
            features = features.to(self.device).to(self.model.conf.dtype)
            targets = targets.to(self.device)
            if self.use_hidden_initial_labels:
                initial_targets = [dataloader.dataset.initial_labels[i.item()] for i in ids]
                targets = torch.tensor(initial_targets, device=self.device)

            reconstructed = self.model.reconstruct(features, targets)
            rows.append(compute_metrics_for_sample(reconstructed, ids, targets, dataloader.dataset.initial_labels[ids.item()]))
            if self.save_reconstruction_images:
                for j, id in enumerate(ids):
                    np_array = features[j].detach().cpu().numpy()
                    label = dataloader.dataset.initial_labels[id.item()]
                    plot_3d_data_cloud(np_array[0], title=f"class_{label}_{target_ids[label]}",
                                       save_to=SaveTo.png, path=data_path)
                    target_ids[label] += 1
            running_loss += bce_loss(reconstructed, features).item()
        df = pd.DataFrame(rows).set_index("idx")
        save_geometric_data_to_file(df, samples_path, "reconstructed_")
        return running_loss / len(dataloader)

    def compute_classification_score(self, data_path: str):
        """
        Compute classification score for npz samples in given folder
        :param data_path:
        :return:
        """
        file_path = os.path.join(data_path, self.samples_file_name)
        dataloader = npz_dataloader(file_path, self.batch_size)
        pred_list = []
        target_list = []
        for (features, targets, ids) in tqdm(dataloader):
            features = features.to(self.device).to(self.model.conf.dtype)
            x = self.clf_model(features)
            probs = torch.softmax(x, dim=1)
            pred = torch.argmax(probs, dim=1)
            pred_list.append(pred.detach().cpu())
            target_list.append(targets.detach().cpu())

        predictions = torch.cat(pred_list)
        targets = torch.cat(target_list)
        create_confusion_matrix(
            targets, predictions,
            title=f"Confusion Matrix ({self.saved_model.model_name.value})",
            save_to=SaveTo.svg,
            labels=self.classes,
        )

        return multiclass_f1_score(predictions, targets, average="macro", num_classes=self.clf_model.conf.num_classes)
