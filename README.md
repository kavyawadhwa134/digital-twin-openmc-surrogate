# Physics-Informed Digital Twin Surrogate

This project builds a small proof of concept for accelerated Monte Carlo neutron-transport workflows:

1. Run OpenMC pin-cell sweeps and train a multi-output reactor-response surrogate.
2. Generate LSTM-ready reactor-state sequences for short-horizon forecasting and anomaly detection.
3. Keep cross-section acceleration as an explicit GPU experiment, with a direct macroscopic XS path for A100 testing.

The installed nuclear data library is expected at:

```text
/Users/kavyawadhwa/Documents/Digital Twin/nuclear_data/endfb-viii.0-hdf5/cross_sections.xml
```

Run scripts from the project root after:

```bash
conda activate openmc
```

## First Pipeline

```bash
bash scripts/run_first_pipeline.sh
python scripts/train_state_forecaster.py
python scripts/run_pincell_sweep.py
python scripts/train_pincell_surrogate.py
```

The old microscopic ENDF/OpenMC cross-section surrogate is disabled in the default pipeline because CPU inference was slower than OpenMC's vectorized lookup. Run it only as an explicit experiment:

```bash
python scripts/extract_xs_dataset.py --energy-points 1200
python scripts/train_xs_surrogate.py --experimental --max-rows 100000
```

Recommended A100 experiment for the XS acceleration claim:

```bash
bash scripts/run_a100_macro_xs_pipeline.sh
```

This trains a direct macroscopic surrogate for `Sigma_r(E, T, material)` and benchmarks one batched neural forward pass against OpenMC HDF5 macroscopic accumulation. See `GPU_A100_README.md` before using it as a speed claim.

To train a separate 10-second state forecaster without overwriting the 5-second model:

```bash
python scripts/train_state_forecaster.py --horizon 10 --name state_forecaster_h10
```

Expanded pin-cell dataset used for the current poster-grade response surrogate:

```bash
python scripts/run_pincell_sweep.py \
  --sample-mode lhs \
  --n-cases 500 \
  --batches 20 \
  --inactive 5 \
  --particles 1600 \
  --threads 4 \
  --output data/processed/pincell_lhs500_openmc.csv

python scripts/train_pincell_surrogate.py \
  --dataset data/processed/pincell_lhs500_openmc.csv \
  --name pincell_lhs500_engineered
```

Higher-stat validation dataset for the strongest accuracy claim:

```bash
python scripts/run_pincell_sweep.py \
  --sample-mode lhs \
  --n-cases 120 \
  --batches 50 \
  --inactive 10 \
  --particles 4000 \
  --threads 4 \
  --seed 914 \
  --run-subdir pincell_lhs120_highstat \
  --output data/processed/pincell_lhs120_highstat_openmc.csv

python scripts/train_pincell_surrogate.py \
  --dataset data/processed/pincell_lhs120_highstat_openmc.csv \
  --name pincell_lhs120_highstat_engineered
```

The pin-cell trainer is now OpenMC-uncertainty aware by default:

- tree models use inverse-variance sample weights when OpenMC label standard deviations are available;
- additional Gaussian-process candidates use per-sample OpenMC uncertainty as known noise;
- model selection reports and lightly penalizes error relative to OpenMC Monte Carlo uncertainty.

Key outputs:

```text
models/state_forecaster_mlp.joblib
models/pincell_response_surrogate_best.joblib
models/pincell_lhs500_response_surrogate_best.joblib
models/pincell_lhs120_highstat_engineered_response_surrogate_best.joblib
data/processed/state_sequences_lstm_ready.npz
data/processed/state_forecaster_validation_summary.csv
figures/lstm_anomaly_demo.png
figures/state_forecaster_anomaly_detection.png
figures/state_forecaster_validation_summary.png
models/state_forecaster_h10_metrics.json
figures/state_forecaster_h10_anomaly_detection.png
figures/state_forecaster_h10_validation_summary.png
figures/pincell_surrogate_vs_openmc.png
figures/pincell_multioutput_surrogate_vs_openmc.png
figures/pincell_lhs500_surrogate_vs_openmc.png
figures/pincell_lhs500_multioutput_surrogate_vs_openmc.png
figures/pincell_lhs120_highstat_engineered_surrogate_vs_openmc.png
figures/pincell_lhs120_highstat_engineered_multioutput_surrogate_vs_openmc.png
```

Experimental microscopic XS status:

- The old sklearn XS surrogate remains available behind `--experimental`.
- It must not be used as a speedup result unless the benchmark reports `speed_claim_allowed = true`.
- The direct macroscopic XS surrogate is now the preferred A100 test because it predicts material response in one batched neural call.
- Even if it succeeds on A100, claim only restricted-domain material-response acceleration, not universal replacement of OpenMC or XSBench.

State forecaster snapshot:

- 20 s history predicts next 5 s
- monitored signals: neutron flux, reactor power, coolant outlet temperature, fuel temperature, and reactivity
- fuel-temperature RMSE: about `1.22 K`
- `keff` RMSE: about `28 pcm`
- anomaly begins at `280 s`; confirmed alarm at `282 s`
- detector: hybrid forecast-residual plus physics-consistency classifier
- selected forecaster: `regularized_mlp` from a 3-candidate search
- mixed held-out faults, precision/recall/F1: `0.997 / 0.983 / 0.990`
- held-out unseen anomaly families, precision/recall/F1: `0.999 / 0.959 / 0.978`
- weak-transient stress test, precision/recall/F1: `0.996 / 0.932 / 0.963`
- held-out normal false-positive rate in this synthetic test: about `0.019%`
- median alarm delay: `2 s`; weak-transient p90 alarm delay: `6.1 s`
- caveat: these are simulated physics-informed trajectories, not real plant data

10-second endpoint forecaster snapshot:

- command: `python scripts/train_state_forecaster.py --horizon 10 --name state_forecaster_h10`
- selected forecaster: ensemble of `regularized_mlp`, `balanced_mlp`, and `wide_mlp`
- reported endpoint metrics are for the true `+10 s` target, not the first future step
- normal-dynamics `+10 s` endpoint fuel-temperature RMSE: about `1.21 K`
- normal-dynamics `+10 s` endpoint `keff` RMSE: about `18 pcm`
- mixed held-out faults, precision/recall/F1: `0.999 / 0.979 / 0.989`
- held-out unseen anomaly families, precision/recall/F1: `0.999 / 0.945 / 0.971`
- weak-transient stress test, precision/recall/F1: `0.996 / 0.875 / 0.931`
- interpretation: the model forecasts normal reactor evolution accurately; during abnormal evolution, deviation from the normal forecast is the anomaly signal

Pin-cell response surrogate snapshot:

- broad OpenMC sweep cases: `500`
- high-stat validation cases: `120`
- input features: fuel temperature, enrichment, moderator density, moderator temperature, fuel radius, pin pitch, cladding thickness, boron concentration, plus engineered geometry features
- targets: `keff`, fuel flux, moderator flux, fission rate, fuel capture rate, moderator capture rate, total capture rate, power-density proxy
- best model: selected independently per target
- high-stat `keff` random-split MAE: about `279 pcm`
- high-stat `keff` tolerance accuracy: `83.3%` within `500 pcm`, `100%` within `1000 pcm`
- held-out hot-fuel-regime `keff` MAE: about `349 pcm`
- held-out wide-pitch-geometry `keff` MAE: about `349 pcm`
- mean relative MAE across predicted reactor-response targets: about `0.32%`
- apparent speedup over high-stat OpenMC pin-cell runs: about `1.6e4x`

The broader 500-case run has noisier labels, with mean `keff` uncertainty around `631 pcm`; its `keff` MAE is about `627 pcm`. That is why the high-stat 120-case validation is better for the headline accuracy claim. The previous 216-case, 3-input pin-cell grid gave about `444 pcm` `keff` MAE and `90.9%` within `1000 pcm`.

Safe wording: within the sampled pin-cell design space, the ML model is a fast response surrogate for OpenMC-computed `keff`, flux, fission rate, capture rate, and power-density proxy. It does not replace general OpenMC particle transport outside the trained domain.

The power-density target is a normalized `kappa-fission` energy-deposition proxy per fuel volume. It is suitable for comparing parameter-response trends; absolute power density requires normalizing to a specified reactor power.
