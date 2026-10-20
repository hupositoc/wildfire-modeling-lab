# Thesis reproduction audit

## Objective and evidence levels

The target is the author's 2025 master's thesis, *Wildfire Spread Prediction Using Reinforcement Learning and Environmental Variables*. Its objective is to model the spatial progression of an already-ignited wildfire using environmental reward functions and a DQN policy. The thesis reports an Uljin Jaccard index of **0.726** for one reward-weight combination and describes an additional Uiseong–Andong case.

The entire 55-page PDF was extracted for review. The implementation/equation and experiment sections were inspected in detail; key equations were also checked against rendered page images because extracted mathematical text can be misleading. The private PDF, signatures, full text and rendered pages are not republished. PDF page references below are physical PDF pages; printed page numbers are noted where useful.

There are three distinct evidence levels:

1. **Thesis-reported:** statements and values printed in the historical thesis.
2. **Locally verified:** the code, input validation, unit checks and synthetic DQN runs executed in this repository.
3. **Not yet reproduced:** the real-event 0.726 score, which needs the original or explicitly reconstructed observational inputs, target mask and unresolved implementation choices.

These categories must not be merged into a single “validated model” claim.

## Section-by-section review

| Thesis section | What it establishes | Reproduction consequence |
|---|---|---|
| Abstract and introduction, PDF 6–17 | Uljin 2022 and Uiseong–Andong 2025 case studies; environmental variables; reported overlap. | Use the historical score as a comparison target, not a newly measured result. |
| Model comparison, PDF 18–22 | Contrasts physical models, cellular automata, data-driven forecasting and reinforcement learning. | A DQN return must not be called a calibrated fire-spread probability or physical rate of spread. |
| Grid/environment, PDF 23–28 | 50×50 cells, eight adjacent moves plus staying, multiple ignition sources, environmental rewards. | Coordinate origin, adjacency, obstacles, branching and terminal behavior need explicit contracts. |
| Scalar rewards, PDF 29–31 | Hailwood–Horrobin EMC with a 20% normalization cap; NDVI-derived fuel reward. | Implement the printed expression and alternative scaling as separate modes until the author confirms the intended scale. |
| Directional rewards, PDF 31–34 | Cosine alignment with wind and slope directions; circular averaging for terrain aspect. | Distinguish meteorological wind-from from mathematical wind-to and use circular rather than arithmetic angle means. |
| Diffusion and DQN, PDF 34–37 | Wind/time/resolution diffusion budget; gamma=1; replay buffer, target network and MSE. | Preserve specified components; document missing architecture and optimizer settings rather than pretending they are known. |
| Data/experiments, PDF 38–47 | Uljin AOI, ignition, timing, source datasets, weight search, reported score and transfer case. | Freeze input provenance, distinguish calibration from held-out evaluation, and identify the burn-mask extraction procedure. |
| Conclusion and references, PDF 48–55 | Limits, suggested extensions and cited data/model literature. | Present a research reconstruction; do not claim a deployed emergency-response system. |

## Mathematical specification and implementation map

The environmental reward is

\[
R = w_f F + w_c C + w_a A + w_w W,
\quad w_f+w_c+w_a+w_w=1.
\]

The best reported row in Table 4.2 uses `(moisture, vegetation, slope, wind) = (0.264, 0.113, 0.140, 0.483)`. These are fixed historical values in the implementation check, not weights selected to improve a held-out mask score.

| Component | Thesis specification | Current source |
|---|---|---|
| Moisture | `F = 1 - EMC/20`, using temperature and relative humidity | `wildfire/model.py::emc_percent` and `Dataset.reward` |
| Raw NDVI | `(NIR - Red)/(NIR + Red)` and spatial means | Source reconstruction helper in `data/export_earth_engine.js` |
| Printed vegetation expression | `C = mean(NDVI)/50 - 1`, equation 3.7, PDF 30 | `ndvi_mode="literal"` |
| Alternative interpretation | Use raw mean NDVI directly; **not silently asserted to be the original intent** | `ndvi_mode="raw"` |
| Wind/slope | Cosine of movement–direction angle | `angular_alignment` |
| Diffusion | `D = floor(C_f * v * dt / delta)`, initial `C_f=0.1` | `diffusion_steps`, explicit floor/ceil setting |
| Policy | Argmax of the learned action values | `QNetwork` and rollout action selection |
| Bellman target | `r + gamma*(1-done)*max Q_target(next_state)`, gamma=1 | `wildfire/experiment.py` |
| Loss | Mean squared TD error; replay and target-network synchronization | `wildfire/experiment.py` |
| Footprint score | Intersection over union of predicted and observed burned areas | `iou`, `evaluate_observed`, `wildfire.verify_thesis` |

The reconstructed network has eight inputs and nine outputs, with two 64-unit ReLU hidden layers. Hidden-layer widths, Adam at 0.001, replay capacity 20,000, batch size 64, target synchronization every 100 updates and gradient clipping at 10 are **declared reconstruction choices**. They were not specified in the reviewed thesis pages and are not presented as the recovered original configuration.

## Discrepancies requiring author confirmation

### A. NDVI units and equation 3.7

The printed equation is unambiguously `mean(NDVI)/50 - 1` when viewed as an image. Applying it to raw NDVI in [-1,1] yields a value near -1, which conflicts with nearby prose describing a useful -1..1 vegetation reward. A 0..100 encoded index could have been intended, but the original encoding has not been recovered. The code therefore keeps the printed expression and raw-NDVI interpretation separate. The latter is an assumption, not an undocumented correction.

### B. Floor symbol versus ceiling definition

Equation 3.12 uses the floor symbol, while the accompanying prose defines the smallest integer greater than or equal to the input—a ceiling operation. At 3.9 m/s, 1800 s, 100 m and `C_f=0.1`, these produce 7 and 8 microsteps respectively. The benchmark uses floor and records that choice. The exact historical behavior still requires the author’s code or confirmation.

### C. Time count and endpoints

The reported interval is March 4, 2022 at 11:00 to March 6 at 12:00, with 30-minute sampling and 100 steps. These endpoints are 49 hours apart: 98 half-hour intervals, or 99 inclusive samples. One hundred inclusive samples from 11:00 end at 12:30. The source helper exports a wider raw hourly bracket, and the preparation command requires an explicit 99/100 choice. It does not silently invent another observation.

### D. Weather resolution and hindsight

ERA5-Land is hourly and approximately 11 km in the cited catalog. Regridding to 50×50 and interpolating to 30 minutes do not create independent 100 m observations or genuinely observed half-hour weather. Using reanalysis over the entire event is a historical reconstruction/hindcast; it is not, by itself, a real-time forecast using only information available at ignition.

### E. Branching and repeated rewards

The thesis describes multiple agents spreading in parallel and each ignited cell becoming another source, but does not fully specify ignition multiplicity, burn residence time, duplicate-cell rewards, source retirement or the precise visit threshold for a final footprint. The code includes a separate branching rollout and explicitly records that ignited sources remain active. Training uses one path at a time. This distinction is a reconstruction gap, not an assertion of exact equivalence.

Staying is one of the nine actions. If a cell gives a positive reward repeatedly, an agent can earn a high return without spreading. Accordingly, cumulative return is reported only as an implementation diagnostic. Spatial accuracy must be evaluated separately against an independently sourced mask.

### F. Calibration and evaluation independence

The thesis describes maximizing Jaccard through weight optimization and also mentions Jaccard as an extra reward. Without the original split and optimization record, the reported Uljin score cannot be characterized as untouched held-out performance. The historical fitted score can still be reproduced as a historical experiment, but an independent generalization claim needs a frozen model evaluated on a separate event or reserved time window.

### G. Original target-mask procedure

The text mentions VIIRS-derived damage and Sentinel-2/GEE ISODATA classification. Exact scenes, temporal compositing windows, class selection, thresholds, clustering settings, masks and raster alignment are missing. Choosing a different burned-area product or tuning a threshold against the desired 0.726 would be a different experiment. The repository will not label a synthetic CA raster as observed damage.

## Local-data provenance finding

The older MATLAB folder contains `actual_fire_data.mat` with a 100×100×50 array. The accompanying `actual_fire_data_code.m` generates it through stochastic NDVI/wind rules and saves the result. The file name does **not** establish observational ground truth, and its dimensions differ from the thesis's 50×50×100 description. It is therefore excluded from real-event verification.

The older MATLAB scripts remain in the repository as historical exploration. They are not the recovered final thesis implementation. A local 2025 award document supports the existence of the prior research activity, but no private award scan or institution identity is published here.

## What is needed to verify 0.726

1. Recover the original final code, trained checkpoint and hyperparameter/seed record, or agree to an explicitly labeled reconstruction.
2. Recover the Uljin environmental tensors, their source scenes/products, coordinate mapping and timing convention.
3. Recover the observed final mask and its classification/alignment recipe.
4. Resolve NDVI scale, floor/ceil, branching, repeated rewards, cell-size interpretation and final-footprint threshold.
5. Freeze model, mask, weights and preprocessing; compute and retain IoU, prediction raster, hashes and configuration.
6. Report the observed score, including a lower score or failed reproduction. Then evaluate an independently reserved event before making a generalization claim.

No dataset is selected or modified to manufacture the target score. New observational exports are labeled reconstructed inputs; generated environments are labeled synthetic.

## Primary references for source reconstruction

- [ERA5-Land Hourly catalog](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY)
- [Harmonized Sentinel-2 surface reflectance](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- [JAXA ALOS AW3D30 v4.1](https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_AW3D30_V4_1)
- [Rothermel, 1972, USDA Forest Service](https://research.fs.usda.gov/treesearch/32533) — background comparison, not the equations implemented by this DQN reconstruction.
- [MathWorks DQN documentation](https://www.mathworks.com/help/reinforcement-learning/ug/dqn-agents.html) — algorithm terminology, not evidence of the author's original network configuration.
