# Data reconstruction and evaluation protocol

## Data types are not interchangeable

- **Historical original:** the exact data and mask used by the thesis. These have not been recovered.
- **Reconstructed observational input:** a newly exported, provenance-bearing source product with declared processing choices. It is not automatically the original input.
- **Synthetic input:** generated environments for testing software and reward behavior. They are not observations.
- **Observed evaluation mask:** a separately sourced final burned-area target, with event/window/grid/classifier metadata. It is never generated to match a desired score.

## Source exports

Run `export_earth_engine.js` in the user's authorized Google Earth Engine Code Editor. The script has not been executed against an authenticated project in this environment.

It exports 2021 annual mean Sentinel-2 NDVI for the stated Uljin AOI, a scene manifest, cell-mean terrain information and raw hourly ERA5-Land temperature/dewpoint/u/v fields. The source collections and assumptions are documented in the thesis audit. Current harmonized/terrain versions are reconstruction choices. The helper does not implement or invent the missing ISODATA burned-area classification.

Weather remains coarse hourly reanalysis. Sampling it on a fine grid and interpolating to half-hour intervals does not create new independent observations. ERA5-Land reanalysis over the future part of the event is hindsight input for historical reconstruction, not an operational weather forecast.

## Prepare an input tensor

After actual exports exist, choose 99 or 100 time samples only after resolving the thesis endpoint/count mismatch:

```powershell
python data/prepare_dataset.py --static static.csv --weather weather.csv --scenes scenes.csv --steps 100 --output data/uljin.npz
```

The output contains six named `[time,y,x]` fields, with a JSON sidecar carrying source hashes, collections, origin, units and processing choices. The converter rejects duplicate/missing cells and nonfinite values, requires the weather interval to bracket all requested timestamps, and does not extrapolate or fill missing observations automatically. The JSON example for a mask is deliberately marked as an example and is rejected by scoring.

## Train without opening the evaluation mask

```powershell
python -m wildfire.experiment --dataset data/uljin.npz --output results/uljin-reconstruction --episodes 160 --seeds 7 19 43
```

This trains the declared reconstruction, not a recovered original network. The example episode count is an explicit parameter, not a known original thesis value. Resolve the audit assumptions before calling any experiment an exact reproduction. Keep calibration, model selection and independent final evaluation separate.

## Evaluate a frozen model

Provide a boolean or binary `.npy` mask and complete observational provenance with the same lower-left grid:

```powershell
python -m wildfire.verify_thesis --dataset data/uljin.npz --mask data/uljin_observed.npy --mask-provenance data/uljin_observed.json --checkpoint results/uljin-reconstruction/dqn_raw_seed7.pt --training-record results/uljin-reconstruction/result_raw_seed7.json --output results/uljin-evaluation.json
```

The selected checkpoint must be fixed before using the final mask. Do not select a best evaluation seed, classify pixels, change the AOI or tune thresholds to reach 0.726. Report all declared seeds and a separate held-out event when making generalization claims. The evaluator stores the actual IoU, its difference from the historical score and file hashes. A numerical match under changed assumptions is not by itself proof of original-code reproduction.
