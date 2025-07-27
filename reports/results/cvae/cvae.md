# CVAE Training 


### Hyperparameter Tuning
save one model for mf1 and one for validation loss
cyclic annealing or simple annealing (for now only simple annealing)
random sampling is not applied (does not seem to have an effect))
num_layers does not make a huge difference
latent size (512 better than 128, so keep 512, this is also consistent with dae)
multiple layers and less channels seems to be a valid choice.

* **ch_mult**: (1, 2, 2, 4) and (1, 2, 2, 4, 4) better than (1, 2, 4, 8) with (1, 2, 4, 8, 8) and (1, 2, 4, 8, 8) with (1, 2, 4, 8, 8, 8)
* **ch**: 16, 64
* **lr**: at least 1e-4 since otherwise gradients are exploding
* **dropout**: using 3D dropout to avoid overfitting (0.2)
* **num_target_emb_channels**: 1 seems to be enough ( does not learn a mapping for encoder)
* **num_layers** 3, 4 (2 did not lead to good results, 5 is too much)

### Results kld = 1, 4 layers
* 16ch: 3660,3855
* 64ch:  4214,0098 

### Results kld = 20, 4 layers
* 16ch: 3825,9167
* 64ch: 4321,9194 

### Results kld = 50, 4 layers
* 16ch: 5279,9756
* 64ch: 4768,5278 

For each kld the best config will be used as best validation model:
kld1: 16ch
kld20: 16ch
kld50: 64ch
We will validate all models, since classifier performance differed between them


When creating 200 per class vs verteilte stichproben
verteilte stichproben
path/to//data/final_models/checkpoints/final_cvae_5_classes_ch64_kld20_base_20250719_112945_best	cvae_64_20_5_classes	0.00398561	0.550023437	0.6506024	0.34939757	0.454637809	1	0.666666687	0.800000014	0.342373908
200 samples
path/to//data/final_models/checkpoints/final_cvae_5_classes_ch64_kld20_base_20250719_112945_best	cvae_64_20_5_classes	0.00398561	0.641313434	0.542168677	0.373493969	0.442295494	1	0.444444448	0.615384619	0.342373908



Final training
│───────────────────────────────────────────────┤
The ticket has been successfully renewed.                                                                                     ││ train_75a6d_00005   RUNNING        64              4                        1      1e-08         0.1             50       11 
                                                                                                                              │           556.544               0   226822    │
                                                                                                                              ││ train_75a6d_00000   TERMINATED     16              4                        1      1e-08         0.1              1      714 
                                                                                                                              │         17561.4                80     3810.68 │
                                                                                                                              ││ train_75a6d_00002   TERMINATED     16              4                        1      1e-08         0.1             20      787 
                                                                                                                              │         18368.5                80     4125.19 │
                                                                                                                              ││ train_75a6d_00004   TERMINATED     16              4                        1      1e-08         0.1             50      713 
                                                                                                                              │         27976.8                80     5224.71 │
                                                                                                                              ││ train_75a6d_00001   ERROR          64              4                        1      1e-08         0.1              1          
                                                                                                                              │                                               │
                                                                                                                              ││ train_75a6d_00003   ERROR          64              4                        1      1e-08         0.1             20          
                                                                                                                              │                                               │
                                                                                                                              │╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
                                                                                                                              │───────────────────────────────────────────────╯