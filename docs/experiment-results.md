# Executed experiment results

## Protocol

This is an implementation benchmark with generated environments, not a reproduction of observed wildfire damage. The model uses the thesis's reported best reward weights and gamma=1, with declared missing details filled by explicit reconstruction choices. Two NDVI interpretations and three seeds give six runs. Each run contains 160 episodes and fixed, separate synthetic evaluation seeds 200001–200005.

- Runs: 6
- Training episodes: 960
- Gradient updates: 28127
- Recorded total training/evaluation wall time: 170.03 s on this CPU environment; not a benchmark against physical simulators.
- Real-event IoU: unavailable; no observational target mask was supplied.

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

## Traceability and reproducibility

The adjacent result JSON files record config, architecture, optimizer, replay settings, seeds, per-episode return/loss, evaluation rows, runtime versions and checkpoint SHA-256. The small `.pt` files contain locally generated state dictionaries, not downloaded untrusted models. The plotting script reads these records directly. Charts do not contain an invented observed IoU.

To rerun, use the commands in the README on the pinned stack. To reproduce the historical score, follow the separate observational-data protocol. A synthetic run is never relabeled as an observation-based reproduction.
