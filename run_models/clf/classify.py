import os
from pathlib import Path

import torch

from src import MSClfNet, MSClfNetConfig
from src._types import Mode, DataType
from src.data.dataloader import get_dataloader
from variables import ROOT_DIR, DATA_DIR


def load_model(checkpoint_path: str):
    model_conf = MSClfNetConfig()
    model_conf.hidden_ch = 32
    model_conf.num_classes = 2
    model_conf.dropout = 0.2
    model_conf.final_dropout = 0.5
    model_conf.num_hidden_layers = 3
    model_conf.out_num = 64
    model = MSClfNet(model_conf)
    checkpoint = torch.load(Path(checkpoint_path))
    ema_state_dict = {}
    for k, v in checkpoint["model"].items():
        if k.startswith('ema_model.'):
            new_key = k[len('ema_model.'):]  # strip prefix
            ema_state_dict[new_key] = v
    model.load_state_dict(ema_state_dict)
    return model

if __name__ == '__main__':
    main_path = ROOT_DIR
    data_path = DATA_DIR
    data_splits = {
        Mode.train: os.path.join(data_path, "train_split_nii.csv")
    }
    dataloaders = get_dataloader(
        data_path, 5, False, DataType.nii, False, data_splits, do_normalize_gaussian=False, img_size=96, dims=3
    )
    model = load_model("path/to//data/checkpoints_final_clf/clf_2_classes_min_lr0.0001_outnum64_hiddenl3_hiddench32_f1_base_20250616_143248_best")
    device = "cuda:4"
    model.to(device)
    model.eval()
    preds = []
    probs = []
    for i, (features, targets, ids) in enumerate(dataloaders[Mode.train]):
        x = model(features.to("cuda:4", dtype=torch.float32))
        pred = torch.softmax(x, dim=1)
        prob = torch.argmax(pred, dim=1)
        preds.append(pred.detach().cpu())
        probs.append(prob.detach().cpu())
        if i == 10:
            break
    print(preds, probs)