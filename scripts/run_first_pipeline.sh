#!/usr/bin/env bash
set -euo pipefail

python scripts/generate_state_sequences.py
echo "Skipping microscopic XS surrogate training by default; use train_xs_surrogate.py --experimental for XSBench/GPU experiments."
