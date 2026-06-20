# Current Build Notes

## Scientific Target

This build is a kernel-level proof of concept for a Generation IV reactor digital twin:

- Pin-cell surrogate: learns OpenMC response quantities from geometry and operating inputs.
- Cross-section surrogate: experimental only; the current sklearn path is not a speed result.
- Digital-twin forecasting data: creates LSTM-ready state windows for short-horizon anomaly monitoring.

## GEN-IV Element

The extracted nuclides are chosen for Generation IV relevance:

- `Pu239`, `U238`, `U235`: actinides for fast and advanced reactor fuels.
- `Na23`: sodium coolant relevance for SFRs.
- `C12`: graphite moderator relevance for HTGRs.
- `F19`, `Li7`: molten-salt carrier relevance for MSRs.
- `Fe56`, `O16`: structural and oxide/coolant constituents.

## XSBench Position

The current sklearn surrogate is **not faster** than OpenMC's vectorized HDF5 lookup on CPU, so it has been removed from the default pipeline.

That is an important result, not a failure. The honest next benchmark is:

- run it only with `python scripts/train_xs_surrogate.py --experimental`;
- compare against an XSBench-style many-nuclide/material lookup kernel;
- evaluate batched neural inference;
- add GPU inference after installing PyTorch;
- report accuracy/speed trade-offs by energy region and reaction channel.

Safe poster claim:

> Microscopic cross-section acceleration remains future work. The current poster result should focus on the OpenMC pin-cell response surrogate and the digital-twin anomaly forecaster.

## Validation Snapshot

Current grouped-validation numbers from the physics-descriptor model:

- Random interpolation split, `hgb`: `R2 = 0.9929`, median relative error `5.0%`
- Held-out nuclide split on `Pu239`, `hgb`: `R2 = 0.9127`, median relative error `34.3%`
- Held-out temperature split on `1200 K`, `hgb`: `R2 = 0.9940`, median relative error `5.9%`

That is the right story for the poster:

- interpolation inside known nuclide/temperature regimes is strong
- transfer to an unseen nuclide is harder and reveals the model boundary
- temperature transfer is manageable with these features

## Reactor-State Forecasting Snapshot

The current state forecaster is a trained scikit-learn MLP sequence model, not an LSTM. It uses the same framing intended for the LSTM:

- input: previous `20 s` of state vectors
- output: next `5 s` of state vectors
- monitored signals shown in the digital-twin figure: neutron flux, reactor power, coolant outlet temperature, fuel temperature, and reactivity
- fuel-temperature validation RMSE: about `1.22 K`
- `keff` validation RMSE: about `28 pcm`
- injected anomaly begins at `280 s`
- confirmed alarm is reported at `282 s` using a 2-second confirmation convention
- alarm detector: hybrid forecast-residual plus physics-consistency classifier
- selected forecaster: `regularized_mlp` from a 3-candidate search
- mixed held-out faults, precision/recall/F1: `0.997 / 0.983 / 0.990`
- held-out unseen anomaly families, precision/recall/F1: `0.999 / 0.959 / 0.978`
- weak-transient stress test, precision/recall/F1: `0.996 / 0.932 / 0.963`
- held-out normal false-positive rate in this synthetic test: about `0.019%`
- median alarm delay: `2 s`; weak-transient p90 alarm delay: `6.1 s`

This is trained on simulated physics-informed trajectories, so it demonstrates the monitoring workflow rather than validating real plant dynamics.

The 10-second endpoint forecaster is trained separately with:

```bash
python scripts/train_state_forecaster.py --horizon 10 --name state_forecaster_h10
```

Its saved demo and metrics now report the true `+10 s` forecast target, not the first future point in the horizon. The best model is an ensemble of `regularized_mlp`, `balanced_mlp`, and `wide_mlp`, selected by a validation score weighted toward endpoint accuracy.

- normal-dynamics `+10 s` endpoint fuel-temperature RMSE: about `1.21 K`
- normal-dynamics `+10 s` endpoint `keff` RMSE: about `18 pcm`
- mixed held-out faults, precision/recall/F1: `0.999 / 0.979 / 0.989`
- held-out unseen anomaly families, precision/recall/F1: `0.999 / 0.945 / 0.971`
- weak-transient stress test, precision/recall/F1: `0.996 / 0.875 / 0.931`

Interpretation: the model forecasts normal reactor evolution accurately. During abnormal evolution, the growing difference between the normal forecast and the observed state is treated as the anomaly signal, so poor agreement after fault onset is expected and useful rather than a failed prediction.

## Pin-Cell Response Surrogate Snapshot

The current poster-grade pin-cell response surrogate has two complementary OpenMC datasets:

- broad coverage: 500-case Latin-hypercube sweep with lower-cost Monte Carlo labels
- high-stat validation: 120-case Latin-hypercube sweep with higher particle histories and lower `keff` label uncertainty

The high-stat dataset is the one to emphasize for the accuracy claim. The broad 500-case set is useful for showing coverage of the input space. Both datasets use the same physical input set:

- fuel temperature
- enrichment
- moderator density
- moderator temperature
- fuel radius
- pin pitch
- cladding thickness
- soluble boron concentration, approximated as trace natural boron
- engineered geometry features such as moderator-to-fuel area ratio and pitch-to-fuel diameter

- targets: `keff`, fuel flux, moderator flux, fission rate, fuel capture rate, moderator capture rate, total capture rate, and power-density proxy
- best model: selected independently per target
- high-stat `keff` random-split `R2`: about `0.994`
- high-stat `keff` random-split MAE: about `279 pcm`
- high-stat `keff` tolerance accuracy: `83.3%` within `500 pcm`, `100%` within `1000 pcm`
- high-stat mean relative MAE across reactor-response targets: about `0.32%`
- held-out hot-fuel-regime `keff` MAE: about `349 pcm`, with `100%` within `1000 pcm`
- held-out wide-pitch-geometry `keff` MAE: about `349 pcm`, with `95.8%` within `1000 pcm`
- OpenMC mean time per high-stat case: about `5.09 s`
- apparent speedup for reactor-response inference: about `1.6e4x`

The broader 500-case dataset has noisier labels: mean `keff` uncertainty is about `631 pcm`, and the model `keff` MAE is about `627 pcm`. That is useful evidence that the earlier error was limited by Monte Carlo label noise, not only surrogate quality.

The earlier 216-case, 3-input grid remains a useful baseline: `keff` MAE was about `444 pcm` and `90.9%` of held-out cases were within `1000 pcm`. The newer 500-case result is more conservative and more believable because it includes geometry and coolant-condition variation.

Safe claim:

> Within the sampled pin-cell design space, the ML model acts as a fast response surrogate for OpenMC-computed quantities such as `keff`, flux, fission rate, capture rate, and power-density proxy.

Boundary:

> The surrogate does not replace OpenMC particle transport in general; it replaces repeated OpenMC response evaluations only inside the trained geometry/material/operating domain.

## Current Commands

```bash
conda activate openmc
cd "/Users/kavyawadhwa/Documents/Digital Twin"
bash scripts/run_first_pipeline.sh
```

## Current Outputs

```text
models/xs_surrogate_disabled.json
models/state_forecaster_metrics.json
models/state_forecaster_h10_metrics.json
models/pincell_surrogate_metrics.json
models/pincell_lhs500_surrogate_metrics.json
models/pincell_lhs500_engineered_surrogate_metrics.json
models/pincell_lhs120_highstat_engineered_surrogate_metrics.json
data/processed/state_sequences_lstm_ready.npz
data/processed/state_forecaster_validation_summary.csv
figures/lstm_anomaly_demo.png
figures/state_forecaster_anomaly_detection.png
figures/state_forecaster_validation_summary.png
figures/state_forecaster_h10_anomaly_detection.png
figures/state_forecaster_h10_validation_summary.png
figures/pincell_surrogate_vs_openmc.png
figures/pincell_multioutput_surrogate_vs_openmc.png
figures/pincell_lhs500_surrogate_vs_openmc.png
figures/pincell_lhs500_multioutput_surrogate_vs_openmc.png
figures/pincell_lhs120_highstat_engineered_surrogate_vs_openmc.png
figures/pincell_lhs120_highstat_engineered_multioutput_surrogate_vs_openmc.png
figures/pincell_surrogate_highstat_vs_largedata.png
```
