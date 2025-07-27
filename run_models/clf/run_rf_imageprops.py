import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src._types import Mode
from src.data.dataloader import get_dataloader
from src.data.datasets import load_tensor_dict_dataset
from src.analysis.shape_metrics import extract_features
"""
This has been an alternative approach to fit a classifier based on geometric properties with random forest, 
an alternative could also be fitting a support vector machine
"""

def infer(dataloaders: dict[Mode, DataLoader]):
    """
    """
    save_path = f'./image_props.pkl'
    modes = [m for m in Mode if m in dataloaders.keys()]
    print("Inferring whole dataset...")
    features_tensor_dict = {}
    for mode in reversed(modes):
        features, targets, ids = infer_whole_dataset(dataloaders.get(mode))
        features_tensor_dict[mode] = {
            "features": features,
            "targets": targets,
            "ids": ids,
        }

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save({
        'cond_tensor_dict': features_tensor_dict,
    }, save_path)

    return features_tensor_dict


def infer_whole_dataset(dataloader):
    """
    """
    feature_tensor = torch.tensor([])
    id_tensor = torch.tensor([])
    target_tensor = torch.tensor([])
    for (masks, target, data_id) in tqdm(dataloader, total=len(dataloader), desc='infer'):
        with torch.no_grad():
            masks = masks.type(torch.int)
            feature = extract_features(masks).unsqueeze(0)
            feature_tensor = torch.cat([feature_tensor, feature], dim=0)
            target_tensor = torch.cat([target_tensor, target], dim=0)
            id_tensor = torch.cat([id_tensor, data_id], dim=0)
    sorted_indices = torch.argsort(id_tensor)
    return feature_tensor[sorted_indices], target_tensor[sorted_indices], id_tensor[sorted_indices]


def get_data(dataloader):
    features, targets, ids = zip(*list(iter(dataloader)))
    features = np.stack([np.array(f).squeeze() for f in features])
    targets = np.array([np.array(t).squeeze() for t in targets])
    ids = np.array([np.array(i).squeeze() for i in ids])
    return features, targets, ids

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    args = {
        "batch_size": 1,
        "shuffle": True,
        "distributed": False,
        "preprocess_img": "crop",
        "do_normalize_gaussian": False,
        "randomWeightedTrainSet": False,
        "use_transforms": False,
    }
    # For each mask in your dataset:
    # data_splits = split_csvs["chP3D_tune"]
    # dataloaders = get_dataloader(
    #     DATA_DIR, 1, True, distributed= False, split_csv_paths=data_splits, data_type=DataType.nii, img_size=96, dims=3
    # )
    state = torch.load(f'./image_props.pkl')
    cond_tensor_dict = load_tensor_dict_dataset(state['cond_tensor_dict'])
    dataloaders = get_dataloader(datasets=cond_tensor_dict, **args)

    # features = infer(dataloaders)
    rf = RandomForestClassifier(n_estimators=10, class_weight='balanced')
    features, targets, _ = get_data(dataloaders.get(Mode.train))
    rf.fit(features, targets)
    features, targets, _ = get_data(dataloaders.get(Mode.val))
    probs = rf.predict_proba(features)[:, 1]  # Probability of class 0
    y_pred = np.where(probs >= 0.5, 1, 0)
    cm = confusion_matrix(targets, y_pred)
    print(cm)
