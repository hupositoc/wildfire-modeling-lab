# Executed experiment results

## Protocol

This is an implementation benchmark with generated environments, not a reproduction of observed wildfire damage. The model uses the thesis's reported best reward weights and gamma=1, with declared missing details filled by explicit reconstruction choices. Two NDVI interpretations and three seeds give six runs. Each run contains 160 episodes and fixed, separate synthetic evaluation seeds 200001–200005.

- Runs: 6
- Training episodes: 960
- Gradient updates: 28127
- Recorded total training/evaluation wall time: 170.03 s on this CPU environment; not a benchmark against physical simulators.
- Real-event IoU: not part of this synthetic benchmark; see [Observed-event calibration](#observed-event-calibration-uljin-2022).

The trial settings were not chosen against IoU 0.726. No early stopping, best evaluation seed or best mask threshold was selected to manufacture a match.

## Training traces

![Training returns from saved logs](assets/training-returns.png)

Each panel shows a 15-episode moving average for three training seeds, their mean and descriptive standard deviation. Episode return is collected while exploration changes, so this curve is not a direct evaluation-accuracy curve. The small sample does not justify a general performance claim.

## Held-out synthetic environments

| NDVI interpretation | Policy | Mean return | Sample SD | Evaluation pairs |
|---|---|---:|---:|---:|
| literal | dqn | 0.101 | 67.181 | 15 |
| literal | random | 0.920 | 2.235 | 15 |
| literal | reward-greedy | 9.635 | 9.638 | 15 |
| raw | dqn | 93.546 | 108.608 | 15 |
| raw | random | 9.138 | 10.726 | 15 |
| raw | reward-greedy | 13.367 | 13.193 | 15 |


![Policy comparison from actual run records](assets/policy-returns.png)

Each row summarizes 15 model-seed/environment pairs. These pairs are not 15 independent real fire events. The plotted SD describes dispersion and is not a confidence interval. The one-step reward policy is a deliberately simple baseline, not an optimized physical fire-spread simulator.

## Failure analysis

The printed NDVI interpretation has near-zero mean DQN return with very large variation; it does not consistently beat even the random baseline in this check. The raw-NDVI interpretation yields a much larger return, but the environment can reward remaining at or revisiting a favorable cell. More cumulative reward therefore cannot be presented as more accurate spread. This is an important negative result about the evaluation design, not a reason to replace the observations.

The state also omits explicit time while the environment may be time-varying, as the thesis's feature description does. The current generated fields are largely time-stationary and do not establish adaptation to changing real weather. Training uses one path; a separate branching rollout contains unconfirmed fire-front assumptions. These are reasons to withhold a real-event prediction claim until the missing original details are resolved.

## Observed-event calibration (Uljin 2022)

### Target and protocol

The target is `data/uljin_observed.npy`: a Sentinel-2 dNBR burn mask built independently by `data/build_burn_mask.py` (651 of 2,500 cells). The map digitized from thesis Fig. 4.5(a) agrees with it at IoU 0.29 only, so it is not used as the evaluation target. The ignition cell (6,10) sits about 2 km from the reported ignition village (Ducheon-ri) and where the fire entered the AOI.

`wildfire/calibrate.py` runs the thesis search (simplex LHS, then Bayesian optimisation of the four reward weights and C_f) with these safeguards:

- **JI reward for training only.** The thesis p.14 Jaccard term shapes training rewards, but the footprint is judged from environmental reward and the trained policy only. A variant that let the JI term into the footprint rule scored IoU 0.95–0.97 by copying the target; it was rejected as leakage and is now prevented by a unit test.
- **Spatial hold-out (`--holdout checker`).** Alternate 10×10 blocks receive the JI reward and form the search objective. The other blocks are never seen by training or search and are reported separately.
- **Unseen seeds.** The selected setting is re-trained with five seeds that were not used in the search.
- **Fuel consumption (`--consumed-fuel`).** Re-entering a cell burned earlier in the episode, including the stay action, earns nothing. This removes a reward loop where an agent could earn reward without spreading.

### Results

Scores are IoU against the dNBR mask. "Held-out" means the held-out blocks scored by unseen seeds. The random policy uses the same reward and threshold.

| Run | Weather input | JI reward | Held-out IoU (5 unseen seeds) | Random policy, held-out |
|---|---|---|---:|---:|
| run12 | ERA5 grid value | none | 0.255 | 0.258 |
| run13 | ERA5 grid value | λ=1 | 0.317 | 0.258 |
| run14 | ERA5 + MicroMet terrain downscaling | λ=1 | **0.348** | 0.258 |
| run15 | run14 + terrain fuel-moisture offset | λ=1 | 0.296 | 0.258 |
| run16 | run14, C_f ≥ 0.067 (VIIRS bound) | λ=1 | 0.307 | 0.258 |
| run17 | run16, ignition threshold grid widened to 0.9 | λ=1 | 0.273 | 0.258 |

Averaging run14's setting over six unseen seeds × ten rollouts and cutting the map at a burn probability of 0.5 (the cut was fixed in advance) gives **held-out IoU 0.367**, against 0.336 for the mean of the individual seeds.

As a reference ceiling, a supervised gradient-boosting classifier was trained directly on the static inputs (NDVI, slope, aspect, elevation, tree fraction) with the same checkerboard split. It reaches held-out IoU of about 0.37. On the current inputs, the DQN spread model performs at roughly that level.

### Negative results kept on record

- The visit-frequency footprint (thesis p.15), with the JI term excluded, scored about the same as the random policy (0.49 on the earlier target).
- Longer training (1,000–3,000 episodes) lowered the score at the thesis weights, with or without fuel consumption.
- Pre-fire NDVI (Sentinel-2, 2022-01-29 and 2022-02-18) was no more informative than the 2021 annual mean (AUC 0.65–0.70 vs 0.66).
- The terrain fuel-moisture offset (`data/add_terrain_moisture.py`, K=2 percentage points from Rothermel 1983) lowered held-out IoU from 0.327 to 0.233 at fixed run14 weights, averaged over 8 seeds. K was not re-fitted to rescue it.
- **Physical spread speed.** VIIRS S-NPP first detections (`tools/viirs_arrival.py`, keyless FIRMS archive) show most of the AOI burning on the first pass after ignition (step 4.3). Reaching those cells in time needs C_f ≥ 0.067 (median distance) to 0.097 (90th percentile), while the unconstrained searches settled near 0.016. Constraining C_f to [0.067, 0.5] made the unseen seeds more consistent (run16: 0.34–0.37 on the fit blocks), but the footprint overpredicted (1,801 cells vs 651) and held-out IoU fell to 0.307. Widening the threshold grid (run17) did not help. C_f and the rollout temperature then sat at the top of their ranges, i.e. the search preferred near-random fast spread, and held-out IoU was 0.273. With the current reward terms, the model cannot match both the observed speed and the observed extent.

## Traceability and reproducibility

The adjacent result JSON files record config, architecture, optimizer, replay settings, seeds, per-episode return/loss, evaluation rows, runtime versions and checkpoint SHA-256. The small `.pt` files contain locally generated state dictionaries, not downloaded untrusted models. The plotting script reads these records directly. Charts do not contain an invented observed IoU.

To rerun, use the commands in the README on the pinned stack. To reproduce the historical score, follow the separate observational-data protocol. A synthetic run is never relabeled as an observation-based reproduction.
