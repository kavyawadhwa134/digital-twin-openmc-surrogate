# A100 GPU Run Guide

This repo has two separate XS paths:

1. `benchmark_xs_surrogate.py`
   Microscopic neural XS surrogate. This is scientifically useful, but on CPU it was slower than OpenMC vectorized lookup and should not be used as the main speed claim.

2. `extract_macro_xs_dataset.py` + `train_macro_xs_surrogate_torch.py` + `benchmark_macro_xs_surrogate.py`
   Direct macroscopic XS surrogate. This is the recommended A100 experiment because one batched neural forward pass predicts the material response directly.

## Required Data

The ENDF/B-VIII.0 OpenMC HDF5 library is expected at:

```text
nuclear_data/endfb-viii.0-hdf5/cross_sections.xml
```

The nuclear data itself is not committed to GitHub because it is too large. Copy or download it on the GPU machine before running the pipeline.

## Create Environment

```bash
conda env create -f environment.yml
conda activate digital-twin-openmc-gpu
```

If your GPU machine already has a CUDA-enabled PyTorch environment, you can use that environment as long as `openmc`, `torch`, `numpy`, `scipy`, `pandas`, and `scikit-learn` are installed.

Quick CUDA check:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## Recommended A100 XS Experiment

```bash
bash scripts/run_a100_macro_xs_pipeline.sh
```

For a larger run:

```bash
N_ENERGY=50000 EPOCHS=80 BATCH_SIZE=524288 EVAL_BATCH_SIZE=1048576 N_QUERIES=5000000 \
bash scripts/run_a100_macro_xs_pipeline.sh
```

Key output files:

```text
data/processed/macro_xs_dataset.npz
models/macro_xs_surrogate.pt
models/macro_xs_surrogate_metrics.json
models/macro_xs_benchmark_a100.json
```

## What Result Would Be Credible?

Use the macro XS result only if the benchmark reports both:

- `surrogate_speedup_vs_openmc > 1`
- acceptable error in `error_metrics.all`, especially median and p95 relative error

Good poster wording if it succeeds:

> On a restricted material/reaction/temperature envelope, a direct GPU-batched macroscopic XS surrogate achieved faster material-response evaluation than OpenMC HDF5 interpolation while retaining quantified approximation error.

Do not claim:

> The model replaces OpenMC or XSBench generally.

That would require a full XSBench-compatible problem definition, broader nuclide coverage, uncertainty gating, and transport-level validation.

## Pin-Cell and Digital-Twin Results

The pin-cell response surrogate and digital-twin anomaly detector remain the main poster-ready results. The XS A100 benchmark is best framed as the acceleration pathway being actively tested.
