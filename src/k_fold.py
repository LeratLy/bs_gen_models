from tqdm import tqdm

from src.config import BaseConfig
from src.models.trainer import Trainer

"""
Running function for a k-fold cross validation of a Trainer class with given base config
"""

def run_k_fold(conf: BaseConfig, k: int = 10):
    """
    Perform k-fold cross-validation with trainer (use corresponding dataset by adding fold number to dataset name)
    :param conf: configuration for initializing the trainer class with a model
    :param k: number of folds
    :return: mean loss of k folds
    """
    print("Starting K-Fold cross-validation")
    sum_loss = 0
    for i in tqdm(range(k)):
        print(f"Started fold {i}")

        data_path = conf.data["name"].split("_")
        last_idx = data_path.pop()
        if int(last_idx) > i:
            continue
        data_path.append(str(i))
        conf.data["name"] = "_".join(data_path)

        trainer = Trainer(conf)
        trainer.train()
        sum_loss += trainer.best_loss
        print(f"Done training. Loss is {trainer.best_loss}")
        trainer.close()

    mean_loss = sum_loss / k
    print("Final loss is {}".format(mean_loss))
    return mean_loss
