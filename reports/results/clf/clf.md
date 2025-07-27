# Clf Training

An additional classifier based on skimage regionprops has been fit but the performance was not good enough, therefore he CNN classifier is used.
Everything with seed 0.

## Hyperparameter Tuning

* **out_num**: Setting > 0 does not increase mf1 score, setting it to 0 seems to work best
* **hidden_ch**: very small not that good, following results for 16, 32, and 64 
* **weighted ce loss**: not as good as weighted random sampling (moreover additional overhead for computing weights and only using them for training)
* **random sampling**: better than weighted loss (faster convergence + better mF1)
* **dropout**: using 3D dropout to avoid overfitting (0.1 in general and 0.3 in final layer)
* **augmentations**: needed to have some correct classifications on newly generated samples

## Final Training session with 10-fold cross validation
We trained the best two configs with 10-fold cross validation on the whole dataset

### 3 layer, 64 ch, 0 outnum, 0.2 final dropout

#### Best inverse mF1 Scores per fold
1. 0.40259742736816406 (0: 0.91 (56 samples), 1: 0.29 (6 samples))
1. 0.24155843257904053 (0: 0.95, 1: 0.57)
1. 0.24979519844055176 (0. 0.95, 1: 0.55)
1. 0.0 (0: 1.0, 1: 1.0)
1. 0.21524345874786377 (0: 0.95, 1: 0.62)
1. 0.34826523065567017  (0: 0.87, 1: 0.43)
1. 0.29990166425704956 (0: 0.96, 1:0.44)
1. 0.44363635778427124 (0.84, 1:0.27)
1. 0.4013158082962036 (0: 0.95, 1: 0.25)
1. 0.28773581981658936 (0: 0.92, 1: 0.50)

* 2.890049397945404 (sum) / 10 = 0.2890049397945404
* 1 - 0.2890049397945404 = 0.7109950602054596 (average mF1)
mean f1 class 0: 0.93, mean f1 class 1: 0.492
### 4 layer, 64 ch, 0 outnum, 0.2 final dropout

#### Best inverse mF1 Scores per fold
1. 0.3249475955963135
1. 0.25995808839797974
1. 0.2392156720161438
1. 0.16137564182281494
1. 0.42762887477874756
1. 0.3227512836456299
1. 0.26604360342025757
1. 0.40343916416168213
1. 0.4013158082962036
1. 0.28773581981658936

* 3.094411551952362 (sum) / 10 = 0.3094411551952362
* 1 - 0.3094411551952362 = 0.6905588448047638 (average mF1)

### 3 layer, 16 ch, 0 outnum, 0.2 final dropout

1. 0.3491552472114563
1. 0.3220779299736023
1. 0.30134087800979614
1. 0.0
1. 0.2420635223388672
1. 0.3420560956001282
1. 0.3428717851638794
1. 0.4316037893295288
1. 0.4675523638725281
1. 0.30134087800979614

* 3.1000624895095825 (sum) / 10 =  0.31000624895095824
* 1 -  0.31000624895095824 =  0.6899937510490417 (average mF1)
#### Best inverse mF1 Scores per fold

### 3 layer, 32 ch, 0 outnum, 0.2 final dropout

1. 0.38669437170028687
1. 0.2767857313156128
1. 0.26754385232925415
1. 0.0
1. 0.24979519844055176
1. 0.3428717851638794
1. 0.30134087800979614
1. 0.4316037893295288
1. 0.4013158082962036
1. 0.30134087800979614

* 2.9592922925949097 (sum) / 10 = 0.29592922925949094
* 1 - 0.29592922925949094 = 0.704070770740509 (average mF1)
#### Best inverse mF1 Scores per fold

#### Best inverse mF1 Scores per fold
 

Trained on 5 classes:
Keeping best model.
Average loss = 0.37617072004538316
Best loss = 0.33525370844663716
Done Training dataset!

### Best Config
This config is then used to train the classifier on the whole dataset.
Directories and data paths need to be adjusted.
```
{
  "checkpoint": {
    "dir": "path/to//data/checkpoints_k_fold_clf",
    "resume_optimizer": true,
    "resume_scheduler": true,
    "save_every_epoch": -1
  },
  "eval": {
    "best_loss": Infinity,
    "ema_every_samples": 0,
    "every_samples": 0,
    "eval_training_every_epoch": -1,
    "num_visual_samples": 2,
    "num_reconstructions": 2,
    "num_evals": 10,
    "eval_epoch_metrics_val_samples": true,
    "eval_epoch_metrics_val": true
  },
  "name": "clf_2_classes_min_lr0.001_hiddenl3_hiddench64_out_num0_randomSamplingTruef1_maxpool",
  "model_name": "ms_clf",
  "classes": {
    "HC": 1,
    "CIS": 0,
    "PPMS": 0,
    "RRMS": 0,
    "SPMS": 0
  },
  "num_classes": 2,
  "num_epochs": 250,
  "create_checkpoint": false,
  "batch_size": 16,
  "img_size": 96,
  "log_interval": 30,
  "model": "src.models.clf.ms_clf.MSClfNet",
  "device": "cuda",
  "lr": 0.001,
  "accum_batches": 1,
  "data": {
    "name": "chP_k_fold_0",
    "type": "nii"
  },
  "ema_decay": 0.95,
  "loss_type": "cel",
  "model_conf": {
    "input_size": 96,
    "num_classes": 2,
    "dropout": 0.1,
    "dims": 3,
    "img_size": 96,
    "num_hidden_layers": 3,
    "out_num": 0,
    "final_dropout": 0.2,
    "hidden_ch": 64
  },
  "preprocess_img": "crop",
  "use_transforms": true,
  "use_early_stop": true,
  "loss_type_eval": "macro_inv_f1_loss",
  "loss_kwargs_eval": {
    "num_classes": 2
  },
  "logging_dir": "path/to//data/logging_k_fold_clf",
  "run_dir": "path/to//data/runs_k_fold_clf",
  "scheduler": {
    "instance_type": "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "settings": [
      "min",
      0.5,
      10,
      0.0001,
      "rel",
      0,
      1e-08
    ]
  },
  "randomWeightedTrainSet": true,
  "loss_kwargs": {},
  "patience": 50,
  "optimizer": {
    "instance_type": "torch.optim.AdamW",
    "settings": [
      0.001,
      [
        0.9,
        0.999
      ],
      1e-08,
      0.02
    ]
  }
}
```