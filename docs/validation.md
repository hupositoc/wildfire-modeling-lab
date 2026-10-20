# Validation record

## Executed

- Ten initial model/environment unit tests passed before the recorded benchmark.
- Six CPU DQN runs completed: two NDVI interpretations, three seeds, 160 episodes each.
- Raw training/evaluation JSON records and state-dictionary checkpoints were saved.
- Result plots and tables were generated from those saved records.

The final test count, including added provenance/rollout checks, is recorded in the repository change summary after the final run.

## Not executed or not established

- Original thesis code/checkpoint recovery.
- Authenticated Earth Engine export of observational inputs.
- Reconstruction of the original satellite classification/mask.
- Reproduction of Uljin IoU 0.726.
- Frozen independent Uiseong–Andong or other real-event evaluation.
- Runtime comparison with FARSITE or another physical simulator.

The historical thesis score is labeled thesis-reported everywhere. High synthetic reward is not a substituted score. The old generated MATLAB actual_fire_data file is not treated as an observed event.
