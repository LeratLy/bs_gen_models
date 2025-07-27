import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import dice_coef


def batch_pairwise_distances(U, V):
    """Compute pairwise distances between two batches of feature vectors."""
    # Squared norms of each row in U and V.
    norm_u = (U ** 2).sum(dim=1).unsqueeze(1)
    norm_v = (V ** 2).sum(dim=1).unsqueeze(0)

    # Pairwise squared Euclidean distances.
    dist = norm_u - 2 *  U @ V.t() + norm_v
    return torch.clamp(dist, min=0.0)

def pairwise_inv_dice_coef(dataloader: DataLoader, device: str = "cpu"):
    """
    Compute pairwise inverse dice coefficient for all elements in the dataset
    :param dataloader:
    :param device:
    :return:
    """
    running_dice_coef = 0
    num_batches = len(dataloader)
    for i, batch1 in tqdm(enumerate(dataloader)):
        for j, batch2 in enumerate(dataloader):
            array1 = batch1[0].to(device)
            array2 = batch2[0].to(device)
            if j < i:
                continue
            running_dice_coef += dice_coef(array1, array2)
    return 1 - (running_dice_coef / ((num_batches * num_batches + 1) / 2))