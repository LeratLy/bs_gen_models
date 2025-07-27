from __future__ import annotations

import logging
import os
import random
import sys
from time import time

"""
    Copyright (C) 2024 LeratLy - All Rights Reserved
    You may use, distribute and modify this code under the
    terms of the MIT license.
    You should have received a copy of the MIT license with
    this file.
"""


# https://www.geeksforgeeks.org/monitoring-model-training-in-pytorch-with-callbacks-and-logging/#implementing-callbacks-and-logging-in-pytorch
class TrainingLogger:
    training_logs = []

    def __init__(self, conf, timestamp):
        self.epoch_start_time = None
        self.log_interval = conf.log_interval
        self.file_logger = setup_logger(conf, timestamp)

    def on_epoch_begin(self, epoch):
        self.epoch_start_time = time()
        self.file_logger.info(f"\n----------------------------------")
        self.file_logger.info(f"Epoch {epoch + 1} starting.")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time() - self.epoch_start_time
        self.file_logger.info(f"Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds.")
        if logs is not None:
            logs['epoch_time'] = elapsed_time  # Add epoch time to logs
            self.training_logs.append(logs)  # Collect training logs
        self.file_logger.info(f"----------------------------------\n")

    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % self.log_interval == 0:
            logging_str = f"Batch {batch + 1}:"
            if "val_loss" in logs:
                logging_str += f"Validation Loss = {logs['val_loss']:.7f}"
                del logs["val_loss"]
            if "train_loss" in logs:
                logging_str += f"Training Loss = {logs['train_loss']:.7f}"
                del logs["train_loss"]
            if "test_loss" in logs:
                logging_str += f"Testing Loss = {logs['test_loss']:.7f}"
                del logs["test_loss"]

            if logs:
                logging_str += f"\n{logs}"
            self.file_logger.info(logging_str)


def get_logger(name: str, level: int | str = logging.INFO, filename: str = "outputs.log", filemode: str = 'a'):
    logging.basicConfig(filename=filename,
                        filemode='a',
                        format='%(asctime)s:%(levelname)s:%(name)s:%(module)s.py: %(message)s',
                        datefmt='%d.%m.%Y %H:%M:%S',
                        level=level)
    logger = logging.getLogger(name)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def setup_logger(conf, file_id=random.random()):
    conf.logging_dir = os.path.join(os.getcwd(), conf.logging_dir)
    if not os.path.exists(conf.logging_dir):
        os.mkdir(conf.logging_dir)
    logger = get_logger("experiment", logging.INFO,
                        os.path.join(conf.logging_dir, f"{conf.name}_{conf.data["name"]}_{file_id}.log"))
    logger.info("Setup logger with config for '" + conf.name + "'")
    return logger


def setup_general_logger(name: str, logging_dir: str, file_id=random.random()):
    """
    Sets up a logger with a given name for a given file path and id
    """
    logging_dir = os.path.join(os.getcwd(), logging_dir)
    logger = get_logger(name, logging.INFO,
                        os.path.join(logging_dir, f"_{name}_{file_id}.log"))
    logger.info("Setup logger with config for '" + name + "'")
    return logger
