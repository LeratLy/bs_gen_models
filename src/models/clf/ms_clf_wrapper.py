from abc import ABC

import torch
from sklearn.metrics import classification_report
from torcheval.metrics.functional import multiclass_f1_score, multiclass_precision, multiclass_recall

from src._types import SaveTo
from src.analysis.plots import create_confusion_matrix
from src.models.autoencoder_base import BaseModel


class MSClfWrapperModel(BaseModel, ABC):
    """
    A full conditional variational autoencoder model
    Only works with binary data {0,1} therefore threshold values to 0,1
    """

    def classify(self, img, ema_model:bool=True):
        output, eval_results = self.forward(torch.where(img > 0.5, 1.0, 0.0), None, batch_idx=-1, ema_model=ema_model)
        pobs = torch.softmax(output, dim=1)
        pred = torch.argmax(pobs, dim=1)

        return pred, pobs.gather(1, pred.unsqueeze(1)).squeeze(1)

    def epoch_metrics(self, output_list, target_list):
        """
        Compute f1, precision and recall for the given list of results
        """
        targets = torch.cat(target_list)
        predictions_logits = torch.cat(output_list)

        f1 = multiclass_f1_score(predictions_logits, targets, num_classes=self.conf.num_classes, average="macro")
        precision = multiclass_precision(predictions_logits, targets, num_classes=self.conf.num_classes,average="macro")
        recall = multiclass_recall(predictions_logits, targets, num_classes=self.conf.num_classes, average="macro")

        pobs = torch.sigmoid(predictions_logits)
        predictions = torch.argmax(pobs, dim=1)
        create_confusion_matrix(
            targets, predictions,
            save_to=SaveTo.tensorboard,
            step=self.num_samples_total,
            writer=self.writer,
            labels=self.conf.classes,
        )
        self.logger.file_logger.info(classification_report(targets, predictions))
        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }, (1-f1).item()

