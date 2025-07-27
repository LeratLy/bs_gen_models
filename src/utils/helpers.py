from __future__ import annotations

import json
import os
import re

from pathlib import Path

"""
    Copyright (C) 2024 LeratLy - All Rights Reserved
    You may use, distribute and modify this code under the
    terms of the MIT license.
    You should have received a copy of the MIT license with
    this file.
"""


def path_resolver(hierarchies, module_path):
    path = Path(os.getcwd())
    if os.path.isfile(module_path) or os.path.isdir(module_path):
        return module_path
    for hierarchy in range(hierarchies):
        full_path = os.path.join(path, module_path)
        if os.path.isfile(full_path) or os.path.isdir(full_path):
            return full_path
        path = path.parent
    return None


def save_to_json(data, conf):
    with open(os.path.join(conf.logging_dir, 'data.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def as_dict(conf):
    train_conf_dict = {}
    for k, v in conf.__dict__.items():
        train_conf_dict[k] = v
    return train_conf_dict


def get_filenames_by(rx: str, folder: str) -> list[str]:
    """
    Get all files corresponding that match your pattern in a directory
    :param rx: regex for choosing files in the directory
    :param folder: folder in which files should be found
    :return: list of file names
    """
    assert len(rx) > 0 and len(folder) > 0
    regex = re.compile(rx)
    filtered_filenames = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if regex.match(file):
                filtered_filenames.append(file)
    return filtered_filenames
