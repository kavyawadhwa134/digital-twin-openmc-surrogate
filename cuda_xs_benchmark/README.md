# CUDA XS-lookup benchmark — self-contained handoff

Benchmarks the macroscopic cross-section lookup kernel (the XSBench bottleneck) on GPU,
comparing three **bit-exact** methods: per-nuclide binary search, the unionized grid, and a
hash-indexed unionized lookup (O(1) address + fixed-W correction, divergence-free → GPU-friendly).

This folder is **everything you need**. No OpenMC, no nuclear-data library required — the real
ENDF/B-VIII.0 energy grids are baked into the small `.npz` files.

## Contents
- `xs_lookup_gpu.py`            — the benchmark (run this)
- `xslookup_pincell.npz`  (+ `.names.json`)  — 6-nuclide PWR pin-cell, 190k grid pts
- `xslookup_assembly.npz` (+ `.names.json`)  — 16-nuclide assembly, 373k grid pts
- `xslookup_nuc30.npz`    (+ `.names.json`)  — 30-nuclide, 682k grid pts (best GPU stress)

Each `.npz` MUST stay paired with its `.names.json` sidecar.

## Setup on the CUDA box
```bash
python -m pip install numpy
python -m pip install cupy-cuda12x     # CUDA 12.x  (use cupy-cuda11x for CUDA 11.x)
```
If cupy is missing it falls back to CPU and says so on the `device:` line.

## Run
```bash
# from inside this folder. use a LARGE query count to saturate the GPU.
python xs_lookup_gpu.py --data-dir . --material assembly --n-queries 50000000
python xs_lookup_gpu.py --data-dir . --material pincell  --n-queries 50000000
python xs_lookup_gpu.py --data-dir . --material nuc30    --n-queries 50000000

# consumer GPU (RTX/GeForce, weak FP64): also test FP32 throughput
python xs_lookup_gpu.py --data-dir . --material assembly --n-queries 50000000 --fp32
```

## What to check
- `device:` line names the GPU → confirms it's engaged (not the CPU fallback).
- `max_rel` should be `0.0e+00` in fp64 → bit-exact vs binary search (the correctness guarantee).
- **Key comparison:** `hash_union / unionized` on GPU vs ~1.2–1.5× on CPU. Prediction: the gap
  *widens* on GPU (binary search suffers more from warp divergence + scattered reads), and raw
  lookups/s jumps ~10×+. Timing already includes GPU synchronization, so numbers are real.

## Reference CPU numbers (Apple M-series, NumPy, this same kernel)
| material | nuclides | unionized/baseline | hash/baseline | exact |
|---|---|---|---|---|
| pincell  | 6  | 2.40× | 3.52× | yes (0.0) |
| assembly | 16 | 3.49× | 4.57× | yes (0.0) |
| nuc30    | 30 | 4.18× | 5.03× | yes (0.0) |

Send the GPU tables back and they can be folded into a CPU-vs-GPU poster panel.
