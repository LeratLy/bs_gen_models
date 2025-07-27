import os

import numpy as np
import torch

from src._types import SaveTo
from src.utils.visualisation import plot_3d_data_cloud


def save_to_file(tensor: torch.Tensor, file_path: str, target: torch.Tensor, index: int):
    """
    Save all elements of the given tensor to a file in the file_path in a target subdirectory
    """
    for i in range(tensor.shape[0]):
        base_path = os.path.join(file_path, str(target[i].item()))
        os.makedirs(base_path, exist_ok=True)
        np_array = tensor[i].detach().cpu().numpy()
        np.save(os.path.join(base_path, f'{index + i}.npy'), np_array)
        plot_3d_data_cloud(np_array[0], title=f'{index + i}', save_to=SaveTo.png, path=base_path)
        del np_array
