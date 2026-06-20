#!/usr/bin/env bash
set -euo pipefail

# Recommended GPU experiment for the XS part of the poster.
# Run from the project root after activating an environment with OpenMC and PyTorch.

DEVICE="${DEVICE:-cuda}"
N_ENERGY="${N_ENERGY:-25000}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-262144}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-524288}"
N_QUERIES="${N_QUERIES:-1000000}"
CHUNK_SIZE="${CHUNK_SIZE:-1000000}"

python scripts/extract_macro_xs_dataset.py \
  --n-energy "${N_ENERGY}" \
  --output data/processed/macro_xs_dataset.npz

python scripts/train_macro_xs_surrogate_torch.py \
  --dataset data/processed/macro_xs_dataset.npz \
  --device "${DEVICE}" \
  --amp \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --name macro_xs

python scripts/benchmark_macro_xs_surrogate.py \
  --model models/macro_xs_surrogate.pt \
  --device "${DEVICE}" \
  --amp \
  --n-queries "${N_QUERIES}" \
  --chunk-size "${CHUNK_SIZE}" \
  --out models/macro_xs_benchmark_a100.json
