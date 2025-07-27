# XOR Training 
gaussian did not look good when visually isnpected therefore we only have results for xor diffusion autoencoder.

## Base Model
### Hyperparameter Tuning

* **ch_mult**: (1, 2, 2, 4) and (1, 2, 2, 4, 4) better than (1, 2, 4, 8) with (1, 2, 4, 8, 8) and (1, 2, 4, 8, 8) with (1, 2, 4, 8, 8, 8)
* **ch**: 32 is better than 64 and 16
* **lr**: 1e-4 with decay since 1e-5 leads to better results but takes forever with 1e-3 training had a spike and did not fully recover from it
* **dropout**: using 3D dropout to avoid overfitting (0.1)


### Best Base Config
This config is then used to train the classifier on the whole dataset.
Directories and data paths need to be adjusted.
 (1, 2, 2, 4) and (1, 2, 2, 4, 4) ch_mult, 32 ch, 1e-4 lr, 0.1 dropout

final validation loss: 0.0022272222457737053
With random sampling on train set: 0.0022483442482315813
```

  "checkpoint": {
    "dir": "path/to//data/checkpoints_last_dae",
    "resume_optimizer": true,
    "resume_scheduler": true,
    "save_every_epoch": -1,
    "name": "tune_xor_base_20250701_115418_best"
  },
  "eval": {
    "best_loss": Infinity,
    "ema_every_samples": 0,
    "every_samples": 0,
    "eval_training_every_epoch": 10,
    "num_visual_samples": 2,
    "num_reconstructions": 4,
    "num_evals": 2,
    "eval_epoch_metrics_val_samples": false,
    "num_samples": 2
  },
  "log_interval": 10,
  "data": {
    "name": "chP3D_tune_70",
    "type": "nii"
  },
  "img_size": 96,
  "batch_size": 2,
  "lr": 0.001,
  "accum_batches": 4,
  "num_epochs": 450,
  "patience": 40,
  "use_early_stop": true,
  "device": "cuda",
  "preprocess_img": "crop",
  "create_checkpoint": true,
  "logging_dir": "path/to//data/logging_last_dae",
  "run_dir": "path/to//data/runs_last_dae",
  "model_name": "beatgans_autoencoder",
  "model": "src.models.dae.architecture.unet_autoencoder.BeatGANsAutoencoderModel",
  "add_running_metrics": [
    "reconstruction"
  ],
  "classes": {
    "HC": 1,
    "CIS": 0,
    "PPMS": 0,
    "RRMS": 0,
    "SPMS": 0
  },
  "clip_denoised": true,
  "diffusion_conf": {
    "fp16": false,
    "device": "cuda",
    "gen_type": "ddim",
    "loss_type": "bce",
    "loss_type_eps": null,
    "loss_type_x_start": "bce",
    "noise_type": "xor",
    "model_mean_type": "eps",
    "model_grad_type": "eps",
    "model_var_type": "fixed_large",
    "rescale_timesteps": false,
    "train_pred_xstart_detach": true,
    "T": 1000,
    "T_eval": 20,
    "T_sampler": "uniform",
    "beta_scheduler": "cosine",
    "model_type": "autoencoder",
    "deterministic": null,
    "num_classes": null
  },
  "model_type": "autoencoder",
  "model_conf": {
    "net": {
      "attn_head": 1,
      "ch": 32,
      "embed_channels": 512,
      "num_head_channels": -1,
      "num_res_blocks": 2,
      "grad_checkpoint": false,
      "resblock_updown": true,
      "resnet_two_cond": true,
      "resnet_use_zero_module": true,
      "attn": [
        12
      ],
      "ch_mult": [
        1,
        2,
        2,
        4
      ],
      "num_input_res_blocks": null,
      "input_channel_mult": null,
      "resnet_cond_channels": null
    },
    "dims": 3,
    "img_size": 96,
    "in_channels": 1,
    "num_heads_upsample": -1,
    "out_channels": 1,
    "attn_checkpoint": false,
    "conv_resample": true,
    "dropout": 0.1,
    "use_new_attention_order": false,
    "use_efficient_attention": false,
    "latent_net_type": "skip",
    "train_pred_xstart_detach": true,
    "last_act": "sigmoid",
    "num_classes": null,
    "time_embed_channels": null,
    "net_enc": {
      "pool": "adaptivenonzero",
      "num_res_blocks": 2,
      "out_channels": 512,
      "use_time_condition": true,
      "ch_mult": [
        1,
        2,
        2,
        4,
        4
      ],
      "grad_checkpoint": null,
      "attn": [
        12
      ]
    }
  },
  "ema_decay": 0.9,
  "data_parallel": false,
  "name": "tune_xor",
  "loss_func_eval": null,
  "loss_kwargs": {},
  "loss_kwargs_eval": {},
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
  },
  "scheduler": {
    "instance_type": "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "settings": [
      "min",
      0.1,
      10,
      0.0001,
      "rel",
      0,
      1e-08
    ]
  }
}
```

## Latent Model

## Hyperparameter Tuning

* **num layers**: 10 vs 20, 
* **num_hidden_ch**: 1024, 2048
* **lr**: 1e-4 with decay since 1e-5 leads to better results but takes forever
* **dropout**: using 3D dropout to avoid overfitting (0.1)
* **loss**: LossType.l1

### Results

Without random weighted sampling in xor and latent
* 10, 1024: 0.048621
* 10, 2048: 0.0428626 --> Best config
* 20, 1024: 0.0439688
* 20, 2048: 0.0439497

0.040309795488913856 -> 10, 2048 with 5 classes and class conditional normalization
0.06317401056488355 -> 10, 2048 with 5 classes class conditional normaliaziobn and conditionally encoded semantic vectors

With random weighted sampling in xor and without in latent:
* 10, 1024: 0.0397388
* 10, 2048: 0.0375783 --> Best config
* 20, 1024: 0.0396302
* 20, 2048: 0.0375835



## Trained based on final_latents_cond_encoder_class_spec_norm

Trial train_diffae_f82ab_00003 completed after 1320 iterations at 2025-07-22 11:29:50. Total running time: 1hr 13min 5s
(func pid=930363) Done Training dataset!


Trial status: 4 TERMINATED
Current time: 2025-07-22 11:29:50. Total running time: 1hr 13min 5s
Logical resource usage: 0/24 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:A40)
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                 status         num_layers     num_hid_channels   loss_type       iter     total time (s)     idle_epochs        loss │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_diffae_f82ab_00000   TERMINATED             10                 1024   LossType.l1     1673           1093.92              160   0.0354842 │
│ train_diffae_f82ab_00001   TERMINATED             10                 2048   LossType.l1     1320            896.639             160   0.0357928 │
│ train_diffae_f82ab_00002   TERMINATED             20                 1024   LossType.l1     1398           1136.89              160   0.0378846 │
│ train_diffae_f82ab_00003   TERMINATED             20                 2048   LossType.l1     1320           1144.58              160   0.0371323 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

2025-07-22 11:29:50,538	INFO tune.py:1009 -- Wrote the latest version of all result files and experiment state to 'path/to/project/ray_results/train_diffae_2025-07-22_10-16-45' in 0.0403s.
Best trial config: {'num_layers': 10, 'num_hid_channels': 1024, 'loss_type': <LossType.l1: 'l1'>, 'latent_infer_path': 'path/to//data/final_models/checkpoints/final_latents_cond_encoder_class_spec_norm.pkl'}
Best trial final validation loss: 0.035484179854393005

with alpha on best (10 ,1024):

Trial status: 1 TERMINATED
Current time: 2025-07-23 17:00:36. Total running time: 15min 4s
Logical resource usage: 0/24 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:A40)
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                 status         num_layers     num_hid_channels   loss_type       iter     total time (s)     idle_epochs        loss │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_diffae_7207f_00000   TERMINATED             10                 1024   LossType.l1     1320            884.754             160   0.0373781 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Best trial config: {'num_layers': 10, 'num_hid_channels': 1024, 'loss_type': <LossType.l1: 'l1'>, 'latent_infer_path': 'path/to//data/final_models/checkpoints/cond_encoder_shift_scale/final_latents_cond_encoder_class_spec_norm.pkl'}
Best trial final validation loss: 0.037378097573916115
(func pid=3947154) Done Training dataset!

Process finished with exit code 0
## Trained based on tune_xor_base_20250710_215919_plus_dropout_base_20250721_130617_best + latents_class_spec_norm

Trial status: 4 TERMINATED
Current time: 2025-07-22 15:31:50. Total running time: 1hr 53min 27s
Logical resource usage: 0/24 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:A40)
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                 status         num_layers     num_hid_channels   loss_type       iter     total time (s)     idle_epochs        loss │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_diffae_22f75_00000   TERMINATED             10                 1024   LossType.l1      946            1099.9              160   0.046137  │
│ train_diffae_22f75_00001   TERMINATED             10                 2048   LossType.l1      946            1616.39             160   0.0455489 │
│ train_diffae_22f75_00002   TERMINATED             20                 1024   LossType.l1      946            2714.31             160   0.0459863 │
│ train_diffae_22f75_00003   TERMINATED             20                 2048   LossType.l1      993            1065.11             160   0.0445043 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

 conf.checkpoint["name"] = os.path.join(DATA_DIR, "final_models", "checkpoints", "tune_xor_base_20250710_215919_plus_dropout_base_20250721_130617_best")
 conf.name = f"final_latent_5_class_norm_{conf.model_conf.latent_net_conf.num_layers}_hidden{conf.model_conf.latent_net_conf.num_hid_channels}"
 conf.latent_infer_path = config["latent_infer_path"]
 conf.create_checkpoint = False

 conf.model_conf.latent_net_conf.class_znormalize = True
 conf.model_conf.latent_net_conf.znormalize = False
 conf.model_conf.latent_net_conf.use_target = True
 conf.model_conf.latent_net_conf.shift_target = False
 conf.model_conf.latent_net_conf.scale_target_alpha = False
 conf.model_conf.latent_net_conf.num_classes = 5
 conf.num_classes = 5

and the best with additional alpha scaling:
conf.model_conf.latent_net_conf.shift_target = True
conf.model_conf.latent_net_conf.scale_target_alpha = True

Trial status: 1 TERMINATED
Current time: 2025-07-23 13:20:02. Total running time: 16min 11s
Logical resource usage: 0/24 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:A40)
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                 status         num_layers     num_hid_channels   loss_type       iter     total time (s)     idle_epochs        loss │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_diffae_7a572_00000   TERMINATED             20                 2048   LossType.l1      993            934.297             160   0.0399683 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Best trial config: {'num_layers': 20, 'num_hid_channels': 2048, 'loss_type': <LossType.l1: 'l1'>, 'latent_infer_path': 'path/to//data/final_models/checkpoints/bdae_class_norm/latents_class_spec_norm.pkl'}
Best trial final validation loss: 0.03996827701727549
(func pid=3575339) Done Training dataset!
 
# Trained on tune_xor_base_20250710_215919_plus_dropout_cond_encoder_scale_only and 

Early stopping at epoch 41.
Keeping best model.
Average loss = 0.0018802798043432867
Best loss = 0.0017008229145215844
Done Training dataset!


Trial status: 3 TERMINATED | 1 ERROR
Current time: 2025-07-23 08:30:01. Total running time: 38min 3s
Logical resource usage: 0/24 CPUs, 0/1 GPUs (0.0/1.0 accelerator_type:A40)
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                 status         num_layers     num_hid_channels   loss_type       iter     total time (s)     idle_epochs        loss │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_diffae_e8d4f_00000   TERMINATED             10                 1024   LossType.l1     1055            639.922             160   0.0378682 │
│ train_diffae_e8d4f_00001   TERMINATED             10                 2048   LossType.l1     1055            675.406             160   0.0378392 │
│ train_diffae_e8d4f_00002   TERMINATED             20                 1024   LossType.l1     1055            883.105             160   0.0393518 │
│ train_diffae_e8d4f_00003   TERMINATED             20                 2048   LossType.l1                                               0.0361062 │

with alpha on best (20 ,2048):

Trial status: 1 TERMINATED
Current time: 2025-07-23 17:03:50. Total running time: 13min 38s
Logical resource usage: 0/24 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:A40)
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                 status         num_layers     num_hid_channels   loss_type       iter     total time (s)     idle_epochs        loss │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_diffae_1954d_00000   TERMINATED             20                 2048   LossType.l1      838            795.667             160   0.0377387 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

(func pid=3959442) Done Training dataset!
2025-07-23 17:03:50,643	INFO tune.py:1009 -- Wrote the latest version of all result files and experiment state to 'path/to/project/ray_results/train_diffae_2025-07-23_16-50-11' in 0.0081s.
Best trial config: {'num_layers': 20, 'num_hid_channels': 2048, 'loss_type': <LossType.l1: 'l1'>, 'latent_infer_path': 'path/to//data/final_models/checkpoints/cond_encoder_scale/final_latents_cond_encoder_scale_class_spec_norm.pkl'}
Best trial final validation loss: 0.037738699465990067

Process finished with exit code 0
Trial status: 4 TERMINATED
Current time: 2025-07-23 19:23:57. Total running time: 39min 31s
Logical resource usage: 0/24 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:A40)
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                 status         num_layers     num_hid_channels   loss_type       iter     total time (s)     idle_epochs        loss │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_diffae_0e905_00000   TERMINATED             10                 1024   LossType.l1      714            573.436             160   0.0408373 │
│ train_diffae_0e905_00001   TERMINATED             10                 2048   LossType.l1     1055           1058.09              160   0.0365811 │
│ train_diffae_0e905_00002   TERMINATED             20                 1024   LossType.l1      714            630.137             160   0.0409782 │
│ train_diffae_0e905_00003   TERMINATED             20                 2048   LossType.l1                                                         │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

(func pid=31297) Skipping diffae 20 2048
Best trial config: {'num_layers': 10, 'num_hid_channels': 2048, 'loss_type': <LossType.l1: 'l1'>, 'latent_infer_path': 'path/to//data/final_models/checkpoints/cond_encoder_scale/final_latents_cond_encoder_scale_class_spec_norm.pkl'}
Best trial final validation loss: 0.0365811288356781

Process finished with exit code 0
